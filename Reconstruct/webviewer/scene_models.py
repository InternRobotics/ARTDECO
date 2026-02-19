#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import Namespace
import gc
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import json
import math
import threading
import time
import warnings
import cv2
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
from typing import Optional
import shutil
from plyfile import PlyData

import lpips
from fused_ssim import fused_ssim
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from simple_knn._C import distIndex2
from Reconstruct.poses.feature_detector import DescribedKeypoints
from Reconstruct.scene.optimizers import SparseGaussianAdam
from Reconstruct.scene.keyframe import Keyframe
from Reconstruct.webviewer.anchors import Anchor
from Reconstruct.utils import (
    radial_decay_kernel,
    project_points_to_image,
    save_ply,
    RGB2SH,
    pointmap2points,
    depth2points,
    focal2fov,
    get_lapla_norm,
    getProjectionMatrix,
    inverse_sigmoid,
    align_poses,
    make_torch_sampler,
    psnr,
    rotation_distance,
    save_poses_as_pyramid_ply, sample, getProjectionMatrix2
)
from dataloaders.utils_IO import write_model, rotmat2qvec
from pathlib import Path

class SceneModel:
    """
    Scene Model class that contains the scene's Gaussians, anchors, keyframes, and methods for rendering and optimization.
    """

    def __init__(
            self,
            width: int,
            height: int,
            K,
            args: Namespace,
            inference_mode: bool = False,
            device = "cuda:0"
    ):
        """
        Args:
            width: Width of the image.
            height: Height of the image.
            args: Arguments for the scene model. Should always have anchor_overlap, and training parameters if inference_mode is False.
            matcher: Matcher for the scene model. Defaults to None (if inference_mode is True).
            inference_mode: Whether we load the scene for visualization. Defaults to False.
        """
        self.device = device
        self.width = width
        self.height = height
        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device=self.device)
        self.anchor_overlap = args.anchor_overlap
        self.optimization_thread = None
        self.args = args

        try:
            import sys
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            warnings.filterwarnings("ignore")
            self.lpips = lpips.LPIPS(net="vgg").to(self.device)
            sys.stdout = original_stdout
        except:
            self.lpips = None

        if not inference_mode:
            self.num_prev_keyframes_check = args.num_prev_keyframes_check
            self.active_sh_degree = args.sh_degree
            self.max_sh_degree = args.sh_degree
            self.lambda_dssim = args.lambda_dssim
            self.init_proba_scaler = args.init_proba_scaler
            self.max_active_keyframes = args.max_active_keyframes
            self.use_last_frame_proba = args.use_last_frame_proba
            self.num_active_frames_cpu = 0
            # self.guided_mvs = GuidedMVS(args)
            self.scaling_reg_factor = args.scaling_reg_factor
            self.rad_decay = args.rad_decay if args.rad_decay != -1 else 1.0
            self.debug_iter = 0

            self.lr_dict = {
                "xyz": {
                    "lr_init": args.position_lr_init,
                    "lr_decay": args.position_lr_decay,
                }
            }

            ## Initialize Gaussian parameters
            self.gaussian_params = {
                "xyz": {
                    "val": torch.empty(0, 3, device=self.device),
                    "lr": args.position_lr_init,
                },
                "f_dc": {
                    "val": torch.empty(0, 1, 3, device=self.device),
                    "lr": args.feature_lr,
                },
                "f_rest": {
                    "val": torch.empty(
                        0,
                        (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
                        3,
                        device=self.device,
                    ),
                    "lr": args.feature_lr / 20.0,
                },
                "scaling": {
                    "val": torch.empty(0, 3, device=self.device),
                    "lr": args.scaling_lr,
                },
                "rotation": {
                    "val": torch.empty(0, 4, device=self.device),
                    "lr": args.rotation_lr,
                },
                "opacity": {
                    "val": torch.empty(0, 1, device=self.device),
                    "lr": args.opacity_lr,
                },
                "id": {
                    "val": torch.empty(0, 1, device=self.device, dtype=torch.long)
                }
                
            }
            self.keyframe2pid = {}
            self.active_anchor = Anchor(self.gaussian_params)
            self.anchors = [self.active_anchor]
            ## Initialize optimizer
            self.reset_optimizer()

        self.keyframes: list[Keyframe] = []
        self.anchor_weights = [1.0]
        self.K = K
        self.f = K[0, 0].item()
        self.init_intrinsics()

        self.approx_cam_centres = None
        self.gt_Rts = torch.empty(0, 4, 4, device=self.device)
        self.gt_Rts_mask = torch.empty(0, device=self.device, dtype=bool)
        self.gt_f = self.f
        self.cached_Rts = torch.empty(0, 4, 4, device=self.device)
        self.valid_Rt_cache = torch.empty(0, device=self.device, dtype=torch.bool)
        self.sorted_frame_indices = None
        self.last_trained_id = 0
        self.valid_keyframes = torch.empty(0, dtype=torch.bool)
        self.lock = threading.Lock()
        self.inference_mode = inference_mode
        self.debug_info = []

        ## Initialize helpers for Gaussian initialization
        radius = 3
        self.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1).to(self.device)
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        x = x.to(self.device)
        y = y.to(self.device)
        self.disc_kernel[0, 0, torch.sqrt(x ** 2 + y ** 2) <= radius + 0.5] = 1
        self.disc_kernel = self.disc_kernel / self.disc_kernel.sum()

        self.uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, width), torch.arange(0, height), indexing="xy"
                ),
                dim=-1,
            )
            .float()
            .to(self.device)
        )


    def reset_optimizer(self):
        for key in self.gaussian_params:
            if key == 'id':
                continue
            if not self.gaussian_params[key]["val"].requires_grad:
                self.gaussian_params[key]["val"].requires_grad = True
        self.optimizer = SparseGaussianAdam(
            self.gaussian_params, (0.5, 0.99), lr_dict=self.lr_dict, device=self.device
        )

    @property
    def xyz(self):
        return self.gaussian_params["xyz"]["val"]

    @property
    def f_dc(self):
        return self.gaussian_params["f_dc"]["val"]

    @property
    def f_rest(self):
        return self.gaussian_params["f_rest"]["val"]

    @property
    def scaling(self):
        return torch.exp(self.gaussian_params["scaling"]["val"])

    @property
    def rotation(self):
        return F.normalize(self.gaussian_params["rotation"]["val"])

    @property
    def opacity(self):
        return torch.sigmoid(self.gaussian_params["opacity"]["val"])

    @property
    def id(self):
        return self.gaussian_params["id"]["val"]

    @property
    def n_active_gaussians(self):
        return self.xyz.shape[0]

    def mask_by_id(self, keyframe_id):
        return self.id == keyframe_id

    @classmethod
    def from_scene(cls, scene_dir: str, args):
        with open(os.path.join(scene_dir, "metadata.json")) as f:
            metadata = json.load(f)

        width = metadata["config"]["width"]
        height = metadata["config"]["height"]
        scene_model = cls(width, height, args, inference_mode=True)
        scene_model.active_sh_degree = metadata["config"]["sh_degree"]
        scene_model.max_sh_degree = metadata["config"]["sh_degree"]
        scene_model.f = metadata["config"]["f"]

        # Load anchors
        scene_model.anchors = []
        for i in range(len(metadata["anchors"])):
            scene_model.anchors.append(
                Anchor.from_ply(
                    os.path.join(scene_dir, "point_clouds", f"anchor_{i}.ply"),
                    torch.tensor(metadata["anchors"][i]["position"]),
                    metadata["config"]["sh_degree"],
                )
            )

        scene_model.active_anchor = scene_model.anchors[0]

        # Load keyframes
        for i in range(len(metadata["keyframes"])):
            keyframe = Keyframe.from_json(metadata["keyframes"][i], i, width, height)
            scene_model.add_keyframe(keyframe)

        return scene_model

    @property
    def first_active_frame(self):
        return self.active_anchor.keyframe_ids[0]

    @property
    def last_active_frame(self):
        return self.active_anchor.keyframe_ids[-1]

    @property
    def n_active_keyframes(self):
        return self.last_active_frame - self.first_active_frame + 1

    def get_training_id(self):
        while True:
            keyframe_id = np.random.randint(
                self.first_active_frame, self.last_active_frame + 1
            )
            if self.keyframes[keyframe_id].device.type == "cuda":
                return keyframe_id

    def optimization_step(self, is_important=True, finetuning=False):
        if len(self.xyz) == 0:
            return
        # Select which keyframe to train on
        # We train on the latest keyframe with self.use_last_frame_proba probability or a random keyframe otherwise
        if (
                np.random.rand() > self.use_last_frame_proba
                or self.last_trained_id == -1
                or finetuning
        ):
            keyframe_id = self.get_training_id()
        else:
            keyframe_id = -1
        keyframe: Keyframe = self.keyframes[keyframe_id]
        lvl = keyframe.pyr_lvl

        # Zero gradients
        keyframe.zero_grad()
        self.optimizer.zero_grad()

        # Render image and depth
        render_pkg = self.render_from_id(keyframe_id, pyr_lvl=lvl, bg=torch.rand(3, device=self.device))
        image = render_pkg["render"]
        invdepth = render_pkg["invdepth"]
        scale = render_pkg["scale"]
        # if self.debug_iter % 500 == 0: 
        #     save_image(image, f'results/debug/scaffoldgs/image/{self.debug_iter}.png')
        #     save_depth_colormap(1.0 / invdepth, f'results/debug/scaffoldgs/depth/{self.debug_iter}.png')
        #     print(f"[DEBUG] {self.debug_iter}")

        gt_image = keyframe.image_pyr[lvl]
        mono_idepth = keyframe.get_mono_idepth(lvl)
        # save_image(image, 'results/debug/image.png')
        # save_depth_colormap(1.0 / invdepth, 'results/debug/depth.png')

        # Loss
        c, h, w = image.shape
        rdk = radial_decay_kernel(h, w, self.rad_decay).cuda()
        if not is_important:
            error_map = (rdk * (image - gt_image).abs())
            alpha_mask = torch.bitwise_or(torch.bitwise_or(error_map[0] > 0.2, error_map[1] > 0.2), error_map[1] > 0.2)
            alpha_mask = ~alpha_mask
            image = image * alpha_mask
            gt_image = gt_image * alpha_mask
            invdepth = invdepth * alpha_mask
            mono_idepth = mono_idepth * alpha_mask
        l1_loss = (rdk * (image - gt_image).abs()).mean()
        ssim_loss = 1 - fused_ssim(image[None], gt_image[None])
        depth_loss = (rdk * (invdepth - mono_idepth).abs()).mean()
        # gs_mask = self.mask_by_id(keyframe_id)
        # gs_indices = render_pkg['mainGaussID']
        # gs_indices = gs_indices[gs_indices != -1]
        # binary_opacity_entropy_loss = 0
        # if len(gs_indices) > 0 and True:
        #     kf_opacity = torch.clamp(self.opacity[gs_indices], min=1e-3, max=1-1e-3)
        #     binary_opacity_entropy_loss = (-kf_opacity * torch.log(kf_opacity) - (1 - kf_opacity) * torch.log(1 - kf_opacity)).mean()

        scaling_reg = scale.prod(dim=1).mean()
        loss = (
            self.lambda_dssim * ssim_loss
            + (1 - self.lambda_dssim) * l1_loss
            + keyframe.depth_loss_weight * depth_loss
            + self.scaling_reg_factor * scaling_reg
            # + 0.01 * binary_opacity_entropy_loss
        )
        loss.backward()

        # Optimizers
        with torch.no_grad():
            # Pose optimization
            keyframe.step()

            # Skip the scene optimization if the current keyframe is a test keyframe
            if not keyframe.is_test:
                # Scene Gaussian optimization
                self.optimizer.step(
                    render_pkg["visibility_filter"], render_pkg["radii"].shape[0]
                )

            keyframe.latest_invdepth = render_pkg["invdepth"].detach()

        self.valid_Rt_cache[keyframe_id] = False
        self.last_trained_id = keyframe_id

    def optimization_loop(self, n_iters: int, is_important: bool, run_until_interupt: bool = False):
        """
        Runs at least n_iters optimization steps.
        If run_until_interupt, also runs until join_optimization_thread is called (Useful to run the optimization until the next keyframe is added in streaming mode).
        """
        self.interupt_optimization = False
        i = 0
        while i < n_iters or (run_until_interupt and not self.interupt_optimization):
            self.optimization_step(is_important=is_important)
            i += 1

    def join_optimization_thread(self):
        """
        Interupts the optimization loop and waits for the thread to finish.
        """
        if self.optimization_thread is not None:
            self.interupt_optimization = True
            self.optimization_thread.join()
            self.optimization_thread = None

    def optimize_async(self, n_iters: int, is_important: bool):
        """
        Starts an optimization thread that runs at least n_iters optimization steps.
        """
        self.join_optimization_thread()
        self.optimization_thread = threading.Thread(target=self.optimization_loop, args=(n_iters, is_important, True))
        self.optimization_thread.start()

    @torch.no_grad()
    def harmonize_test_exposure(self):
        """Harmonizes the exposure matrices of test keyframes by averaging the exposure of the previous and next keyframes."""
        for index, keyframe in enumerate(self.keyframes):
            if keyframe.is_test:
                idxm = index - 1 if index != 0 else 1
                idxp = (
                    index + 1
                    if index != len(self.keyframes) - 1
                    else len(self.keyframes) - 2
                )
                keyframe.exposure = (
                                            self.keyframes[idxm].exposure + self.keyframes[idxp].exposure
                                    ) / 2

    @torch.no_grad()
    def evaluate(self, with_LPIPS=False, all=False):
        # Make sure test keyframes have similar exposure matrices compared to their neighbors
        self.harmonize_test_exposure()
        torch.cuda.empty_cache()
        # Compute image quality metrics
        metrics = {"PSNR": 0.0, "SSIM": 0.0}
        if with_LPIPS:
            metrics["LPIPS"] = 0.0
        n_test_frames = 0
        start_index = 0 if all else self.active_anchor.keyframe_ids[0]
        for index, keyframe in enumerate(self.keyframes[start_index:]):
            if keyframe.is_test:
                gt_image = keyframe.image_pyr[0].to(self.device)
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                image = render_pkg["render"]
                mask = torch.ones_like(image[:1] > 0)
                mask = mask.expand_as(image)
                image = image * mask
                gt_image = gt_image * mask
                metrics["PSNR"] += psnr(image[mask], gt_image[mask])
                metrics["SSIM"] += fused_ssim(
                    image[None], gt_image[None], train=False
                ).item()
                if with_LPIPS and self.lpips is not None:
                    metrics["LPIPS"] += self.lpips(image[None], gt_image[None]).item()
                n_test_frames += 1

        if n_test_frames > 0:
            for metric in metrics:
                metrics[metric] /= n_test_frames
        else:
            metrics = {}
        metrics["n_test_frames"] = n_test_frames
        return metrics

    @torch.no_grad()
    def save_test_frames(self, out_dir):
        self.harmonize_test_exposure()
        os.makedirs(out_dir, exist_ok=True)
        diagnostics = []
        for i, keyframe in enumerate(self.keyframes):
            if keyframe.is_test:
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                image = torch.clamp(render_pkg["render"], 0, 1) * 255
                invdepth = render_pkg["invdepth"].squeeze(0).detach().cpu().numpy()  # [H,W]
                vmin = np.nanmin(invdepth)
                vmax = np.nanmax(invdepth)
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    invdepth_vis = np.zeros_like(invdepth, dtype=np.uint8)
                else:
                    invdepth_norm = (invdepth - vmin) / (vmax - vmin)
                    invdepth_vis = np.clip(invdepth_norm * 255.0, 0, 255).astype(np.uint8)
                image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                is_jpeg = os.path.splitext(keyframe.image_name)[-1].lower() in [
                    ".jpg",
                    ".jpeg",
                ]
                write_flag = [int(cv2.IMWRITE_JPEG_QUALITY), 100] if is_jpeg else []
                diagnostics.append((i, out_dir, keyframe.image_name))
                stem = os.path.splitext(os.path.basename(keyframe.image_name))[0]
                depth_gray_path = os.path.join(out_dir, f"{stem}_depth.png")
                # import pdb;pdb.set_trace()
                cv2.imwrite(depth_gray_path, invdepth_vis)
                
                cv2.imwrite(
                    os.path.join(out_dir, keyframe.image_name), image, write_flag
                )
                
        print(f"Saved {len(diagnostics)} test frames to {out_dir}")

    def render_from_id(
            self,
            keyframe_id,
            pyr_lvl=0,
            scaling_modifier=1,
            bg=torch.zeros(3),
    ):
        """
        Render the scene from a given keyframe id at a specified resolution level (pyr_lvl).
        Applies the exposure matrix of the keyframe to the rendered image.
        """
        bg = bg.to(self.device)
        keyframe = self.keyframes[keyframe_id]
        view_matrix = keyframe.get_Rt().transpose(0, 1).to(self.device)
        scale = 2 ** pyr_lvl
        width, height = self.width // scale, self.height // scale
        render_pkg = self.render(width, height, view_matrix, scaling_modifier, bg)
        render_pkg["render"] = (
                                       keyframe.exposure[:3, :3] @ render_pkg["render"].view(3, -1)
                               ) + keyframe.exposure[:3, 3, None]
        render_pkg["render"] = render_pkg["render"].clamp(0, 1).view(3, height, width)
        return render_pkg

    def render(
            self,
            width: int,
            height: int,
            view_matrix: torch.Tensor,
            scaling_modifier: float,
            bg: torch.Tensor = torch.zeros(3),
            top_view: bool = False,
            fov_x: Optional[float] = None,
            fov_y: Optional[float] = None,
    ):
        bg = bg.to(self.device)
        cam_centre = view_matrix.detach().inverse()[3, :3]
        # Use the scene's intrinsic parameters if not provided
        if fov_x is None and fov_y is None:
            tanfovx, tanfovy = self.tanfovx, self.tanfovy
            projection_matrix = self.projection_matrix
        # Use the provided FOV values
        elif fov_x is not None and fov_y is not None:
            tanfovx = math.tan(fov_x * 0.5)
            tanfovy = math.tan(fov_y * 0.5)
            projection_matrix = (
                getProjectionMatrix2(znear=0.01, zfar=100.0, cx=self.K[0, 2],
                                 cy=self.K[1, 2], fx=self.K[0, 0], fy=self.K[1, 1],
                                 W=self.width, H=self.height)
                .transpose(0, 1)
                .to(self.device)
            )
        else:
            raise ValueError("Both fov_x and fov_y should be provided or neither.")

        raster_settings = GaussianRasterizationSettings(
            height,
            width,
            tanfovx,
            tanfovy,
            bg,
            1 if top_view else scaling_modifier,
            projection_matrix,
            self.active_sh_degree,
            cam_centre,
            False,
            False,
        )
        rasterizer = GaussianRasterizer(raster_settings)
        with self.lock:
            # Load and blend anchors if in inference mode
            if self.inference_mode and not top_view:
                self.gaussian_params, self.anchor_weights = Anchor.blend(cam_centre,
                                                                         self.anchors,
                                                                         self.anchor_overlap,
                                                                         device=self.device)
            screenspace_points = torch.zeros_like(self.xyz, requires_grad=True)
            if self.xyz.shape[0] > 0:
                # Set constant scaling and opacity to visualize the Gaussians' positions in the top view
                if top_view:
                    scaling = torch.ones_like(self.scaling) * scaling_modifier
                    opacity = torch.ones_like(self.opacity)
                else:
                    scaling = self.scaling
                    opacity = self.opacity
                fl_x = width / (2 * tanfovx)
                fl_y = height / (2 * tanfovy)
                Ks = torch.tensor([
                    [fl_x, 0, width / 2.0],
                    [0, fl_y, height / 2.0],
                    [0, 0, 1],
                ],device="cuda",)[None]
                color, invdepth, mainGaussID, radii = rasterizer(
                    self.xyz,
                    screenspace_points,
                    opacity,
                    self.f_dc,
                    self.f_rest,
                    scaling,
                    self.rotation,
                    view_matrix,
                )
            else:
                # If no Gaussians are present, return empty tensors
                color = torch.zeros(3, height, width, device=self.device)
                invdepth = torch.zeros(1, height, width, device=self.device)
                mainGaussID = torch.zeros(
                    1, height, width, device=self.device, dtype=torch.int32
                )
                radii = torch.zeros(1, height, width, device=self.device)
                scaling = torch.zeros(1, height, width, device=self.device)
        return {
            "render": color,
            "invdepth": invdepth,
            "mainGaussID": mainGaussID,
            "radii": radii,
            "visibility_filter": radii > 0,
            "screenspace_points": screenspace_points,
            "scale": scaling,
        }

    def get_closest_by_cam(self, cam_centre, k=3):
        closest_anchors = []
        closest_anchors_ids = []
        approx_cam_centres = self.approx_cam_centres.clone()
        for l in range(min(k, len(self.anchors))):
            if approx_cam_centres.shape[0] == 0:
                break
            dists = torch.linalg.norm(approx_cam_centres - cam_centre[None], dim=-1)
            min_dist, min_id = torch.min(dists, dim=0)

            if min_dist < 1e9:
                for anchor_id, anchor in enumerate(self.anchors):
                    if min_id in anchor.keyframe_ids:
                        closest_anchors.append(anchor)
                        closest_anchors_ids.append(anchor_id)
                        approx_cam_centres[
                        anchor.keyframe_ids[0]: anchor.keyframe_ids[-1] + 1
                        ] = 1e9
                        break

        return closest_anchors, closest_anchors_ids

    @torch.no_grad()
    def get_prev_keyframes(self, n: int, update_3dpts: bool, desc_kpts: Optional[DescribedKeypoints] = None):
        """
        Get the n previous keyframes that are the closest to the last
        If desc_kpts is not None, we find the previous keyframes that have the most matches with desc_kpts. The search window is given by self.num_prev_keyframes_check
        """
        # Make sure the optimization thread is not running
        self.join_optimization_thread()

        # todo: 应该先向scene_model中增加关键帧再获取与其相关的关键帧，因为现在位姿比较准了，或者可以换成mast3r的回环检测
        # Look for the previous keyframes with the most matches with desc_kpts (if provided)
        if desc_kpts is not None and len(self.keyframes) > n:
            n_ckecks = min(self.num_prev_keyframes_check, len(self.keyframes))
            keyframes_indices_to_check = self.sorted_frame_indices[:n_ckecks]
            n_matches = torch.zeros(len(keyframes_indices_to_check), device=self.device)
            for i, index in enumerate(keyframes_indices_to_check):
                n_matches[i] = self.matcher.evaluate_match(self.keyframes[index].desc_kpts, desc_kpts)
            _, top_indices = torch.topk(n_matches, n)
            prev_keyframes_indices = keyframes_indices_to_check[top_indices.cpu()]
        else:
            prev_keyframes_indices = self.sorted_frame_indices[:n]
        prev_keyframes = [self.keyframes[i] for i in prev_keyframes_indices]

        # Re-run triangulation if necessary
        # 此处是更新原因有2
        # 1. 这些关键帧可能被之前的一些关键帧关联上了，既然关联上了就会有一些新的3D点可以可视化
        # 2. 这些关键帧可能有渲染的深度
        if update_3dpts:
            for keyframe in prev_keyframes:
                keyframe.update_3dpts(self.keyframes)
        return prev_keyframes

    def get_Rts(self):
        invalid_ids = torch.where(~self.valid_Rt_cache)[0]
        if len(invalid_ids) > 0:
            for keyframe_id in invalid_ids:
                self.cached_Rts[keyframe_id] = self.keyframes[keyframe_id].get_Rt()
            self.valid_Rt_cache[invalid_ids] = True
        return self.cached_Rts

    def get_gt_Rts(self, align):
        n_poses = min(self.gt_Rts_mask.shape[0], self.cached_Rts.shape[0])
        if align and n_poses > 0:
            Rts = self.get_Rts()[:n_poses][self.gt_Rts_mask[:n_poses]]
            return align_poses(self.gt_Rts[: len(Rts)], Rts)
        else:
            return self.gt_Rts

    def make_dummy_ext_tensor(self):
        return {
            "xyz": self.xyz[:0].detach(),
            "f_dc": self.f_dc[:0].detach(),
            "f_rest": self.f_rest[:0].detach(),
            "opacity": self.opacity[:0].detach(),
            "scaling": self.scaling[:0].detach(),
            "rotation": self.rotation[:0].detach(),
            "id": self.id[:0].detach(),
        }

    def reset(self, keyframe_id: int = -1):
        """Remove the Gaussians that are visible in the given keyframe."""
        valid_mask = self.opacity[:, 0] > 0.05
        render_pkg = self.render_from_id(keyframe_id)
        valid_mask[render_pkg["visibility_filter"]] = False
        self.optimizer.add_and_prune(self.make_dummy_ext_tensor(), valid_mask)

    @torch.no_grad()
    def add_new_gaussians(self, keyframe_id: int = -1):
        """Use the given keyframe to add new Gaussians to the scene model."""
        keyframe = self.keyframes[keyframe_id]

        # Skip if the keyframe is a test keyframe
        if keyframe.is_test:
            return

        ## Get the pixel-wise probability to add a Gaussian
        img = keyframe.image_pyr[0]
        img = F.avg_pool2d(img, 2)
        img = F.interpolate(img[None], (self.height, self.width), mode="bilinear", align_corners=True)[0]
        init_proba = get_lapla_norm(img, self.disc_kernel, device=self.device)  # eq. 1

        ## Compute the penalty based on the rendering from the new keyframe's point of view
        penalty = 0
        rendered_depth = None
        if self.xyz.shape[0] > 0:
            render_pkg = self.render_from_id(keyframe_id)
            render = render_pkg["render"]
            rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)
            penalty = get_lapla_norm(render, self.disc_kernel, device=self.device)

        ## Define which pixels should become Gaussians
        init_proba *= self.init_proba_scaler
        penalty *= self.init_proba_scaler
        # sample points
        sample_mask = torch.rand_like(init_proba) < init_proba - penalty  # eq. 3
        sample_mask = sample_mask.to(self.device)
        sampled_uv = self.uv[sample_mask]  # L
        sampled_depths = sample(
            keyframe.point_map[:, 2:],  # 1 1 H W
            sampled_uv[None, None, ...],  # 1 1 L 2
            keyframe.width,
            keyframe.height
        )[0, 0, 0]  # L

        sampled_conf = sample(
            keyframe.mono_depth_conf,  # 1 1 H W
            sampled_uv[None, None, ...],  # 1 1 L 2
            keyframe.width,
            keyframe.height
        )[0, 0, 0]  # L

        # todo:后处理对每个视角贡献都很低的高斯
        # 第一次过滤: 根据confidence过滤  todo:此处没有单位
        # quantile_depth_max = torch.quantile(keyframe.point_map[:, 2], 0.98)
        quantile_depth_min = min(1e-2, torch.quantile(keyframe.point_map[:, 2], 0.02).item())
        max_pixel_difference = 5
        min_conf = 1 / (max_pixel_difference - 2)
        # todo:warning!
        valid_mask = ((sampled_conf >= 0)
                      * (sampled_depths > quantile_depth_min))  # L
        sample_mask[sample_mask.clone()] = valid_mask
        sampled_uv = sampled_uv[valid_mask]  # L1 2
        sampled_depths = sampled_depths[valid_mask]  # L1 3
        sampled_conf = sampled_conf[valid_mask]

        # Remove Gaussians that are coarser than the newpoints
        if len(self.xyz) > 0:
            main_gaussians_map = render_pkg["mainGaussID"]
            accurate_sample_mask = sample_mask.clone()
            accurate_sample_mask[accurate_sample_mask.clone()] = True
            selected_main_gaussians = main_gaussians_map[:, accurate_sample_mask]  # L1
            ids, counts = torch.unique(
                selected_main_gaussians[selected_main_gaussians >= 0],
                return_counts=True,
            )
            valid_gs_mask = torch.ones_like(self.xyz[:, 0], dtype=torch.bool)
            valid_gs_mask[ids] = counts < 10
            with self.lock:
                self.optimizer.add_and_prune(
                    self.make_dummy_ext_tensor(), valid_gs_mask
                )
            render_pkg = self.render_from_id(keyframe_id)
            rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)

        # 第二次过滤
        # Check for occlusions
        if rendered_depth is not None:
            valid_mask = sampled_depths < rendered_depth[sample_mask]  # L1
            sample_mask[sample_mask.clone()] = valid_mask
            sampled_uv = sampled_uv[valid_mask]  # L2
            sampled_depths = sampled_depths[valid_mask]  # L2 3
            sampled_conf = sampled_conf[valid_mask]  # L2

        # todo: 内参矩阵目前假设fx=fy cx cy在中心
        sampled_points = depth2points(sampled_uv, sampled_depths.unsqueeze(-1), self.f, self.centre)

        # points_view = sampled_points / sampled_points[..., 2:]
        # points_view = (self.K[None, ...] @ points_view[..., None]).squeeze(-1)[..., :2]

        sampled_points = (sampled_points - keyframe.get_t()) @ keyframe.get_R()

        ## Initialize Colour
        f_dc = img[:, sample_mask]
        f_dc = RGB2SH(f_dc.permute(1, 0).unsqueeze(1))

        ## Initialize Scales
        sampled_init_proba = init_proba[sample_mask]
        # Expected distance to the nearest neighbour (eq. 4)
        scales = 1 / (torch.sqrt(sampled_init_proba))
        scales.clamp_(1, self.width / 10)
        # Scale by the distance to the camera centre
        scales.mul_(1 / self.f)
        scales *= torch.linalg.vector_norm(
            sampled_points - keyframe.approx_centre[None], dim=-1
        )
        scales = torch.log(scales.clamp(1e-6, 1e6)).unsqueeze(-1).repeat(1, 3)

        ## Initialize opacities
        opacities = torch.ones(f_dc.shape[0], 1, device=self.device)
        # Lower inital opacity depending for innacurate points
        opacities[: sampled_uv.shape[0]] *= 0.2 * sampled_conf[..., None]
        opacities = inverse_sigmoid(opacities)

        ## Initialize SH, rotations as identity
        f_rest = torch.zeros(
            f_dc.shape[0],
            (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
            3,
            device=self.device,
        )
        rots = torch.zeros(f_dc.shape[0], 4, device=self.device)
        rots[:, 0] = 1

        ## Get which Gaussians should be pruned
        if self.xyz.shape[0] > 0:
            # Only keep Gaussians with non neglectible opacity
            valid_gs_mask = self.opacity[:, 0] > 0.05

            # Discard huge Gaussians
            dist = torch.linalg.vector_norm(
                self.xyz - keyframe.approx_centre[None], dim=-1
            )
            screen_size = self.f * self.scaling.max(dim=-1)[0] / dist
            valid_gs_mask *= screen_size < 0.5 * self.width
        else:
            valid_gs_mask = torch.ones(0, device=self.device, dtype=torch.bool)

        ## Append the new Gaussians
        keyframe_id = len(self.keyframes) - 1 if keyframe_id == -1 else keyframe_id
        extension_tensors = {
            "xyz": sampled_points,
            "f_dc": f_dc,
            "f_rest": f_rest,
            "opacity": opacities,
            "scaling": scales,
            "rotation": rots,
            "id": torch.full((len(sampled_points), 1), keyframe_id, device="cuda", dtype=torch.long)
        }
        with self.lock:
            self.optimizer.add_and_prune(extension_tensors, valid_gs_mask)

    @torch.no_grad()
    def rigid_transform_gs(self, kf_id, old_Rt, new_Rt):
        mask = self.mask_by_id(kf_id).squeeze(-1)
        xyz_h = torch.cat([self.xyz[mask], torch.ones((len(self.xyz[mask]),1), device=self.xyz.device)], dim=-1)
        # xyz_h[mask] = (new_Rt @ (torch.linalg.inv(old_Rt) @ xyz_h[mask].T)).T
        # self.xyz = xyz_h[:, :3]
        self.xyz[mask] = (new_Rt @ (torch.linalg.inv(old_Rt) @ xyz_h.T)).T[:, :3]
        
    def init_intrinsics(self):
        self.FoVx = focal2fov(self.f, self.width)
        self.FoVy = focal2fov(self.f, self.height)
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)
        self.projection_matrix = (
            getProjectionMatrix2(znear=0.01, zfar=100.0, cx=self.K[0, 2],
                                 cy=self.K[1, 2], fx=self.K[0, 0], fy=self.K[1, 1],
                                 W=self.width, H=self.height)
            .transpose(0, 1)
            .to(self.device)
        )

    def add_keyframe(self, keyframe: Keyframe):
        """
            1. Add a keyframe to the scene
            2. Sort all the keyframe indices by the optical center distance from current camera in world camera
            3. add gt pose to viewer
            4. add keyframe to anchor
            5. move keyframe to cpu if necessary
        """

        # Make sure training is not running
        self.join_optimization_thread()

        ## Add the keyframe and update the indices (sorted by distance to last keyframe)
        self.keyframes.append(keyframe)
        if self.approx_cam_centres is None:
            self.approx_cam_centres = keyframe.approx_centre[None]
        else:
            self.approx_cam_centres = torch.cat(
                [self.approx_cam_centres, keyframe.approx_centre[None]], dim=0
            )
        dist_to_last = torch.linalg.vector_norm(
            self.approx_cam_centres - keyframe.approx_centre[None], dim=-1
        )
        self.sorted_frame_indices = torch.argsort(dist_to_last).cpu()

        ## Update cached Rts for the viewer
        self.cached_Rts = torch.cat(
            [self.cached_Rts, keyframe.get_Rt().unsqueeze(0)], dim=0
        )
        self.valid_Rt_cache = torch.cat(
            [self.valid_Rt_cache, torch.ones(1, device=self.device, dtype=torch.bool)], dim=0
        )
        gt_pose = keyframe.Rt_gt
        if gt_pose is not None:
            self.gt_Rts = torch.cat([self.gt_Rts, gt_pose.unsqueeze(0)], dim=0)
        self.gt_Rts_mask = torch.cat(
            [
                self.gt_Rts_mask,
                torch.Tensor([gt_pose is not None]).to(self.gt_Rts_mask),
            ],
            dim=0,
        )
        self.gt_f = keyframe.focal_gt if keyframe.focal_gt is not None else self.f

        if not self.inference_mode:
            ## Add keyframe to the active anchor
            self.active_anchor.add_keyframe(keyframe)

            ## Clear memory if there are many keyframes
            if (
                    self.n_active_keyframes - self.num_active_frames_cpu
                    > self.max_active_keyframes
            ):
                while True:
                    frame_id = np.random.randint(
                        self.first_active_frame, self.last_active_frame - 20
                    )
                    if self.keyframes[frame_id].device.type == "cuda":
                        self.keyframes[frame_id].to("cpu")
                        self.num_active_frames_cpu += 1
                        if self.num_active_frames_cpu % 10 == 0:
                            gc.collect()
                            torch.cuda.empty_cache()
                        break

    def enable_inference_mode(self):
        """Enable inference mode and sets the anchor position to the mean of the active keyframes."""
        self.inference_mode = True
        self.update_anchor()

    def update_anchor(self, n_left_frames: int = 0):
        """Update the anchor position and remove the last n_left_frames keyframes from the active anchor."""
        anchor_position = self.approx_cam_centres[
                          self.first_active_frame: self.last_active_frame - n_left_frames
                          ].mean(dim=0)
        self.active_anchor.position = anchor_position
        if n_left_frames > 0:
            self.active_anchor.keyframes = self.active_anchor.keyframes[:-n_left_frames]
            self.active_anchor.keyframe_ids = self.active_anchor.keyframe_ids[
                                              :-n_left_frames
                                              ]

    def place_anchor_if_needed(self):
        """Check if many Gaussians appear small on the screen.
        If so, place a new anchor. and merge the Gaussians."""
        small_prop_thresh = 0.4
        k = 3
        self.n_kept_frames = 20
        if (
                self.xyz.shape[0] > 0
                and self.first_active_frame < len(self.keyframes) - 2 * self.n_kept_frames
        ):
            with torch.no_grad():
                dist = torch.linalg.vector_norm(
                    self.xyz - self.approx_cam_centres[-1][None], dim=-1
                )
                screen_size = self.f * self.scaling.mean(dim=-1) / dist
                small_mask = screen_size < 1
                small_prop = small_mask.float().mean()

            if small_prop > small_prop_thresh:
                with torch.no_grad():
                    small_mask = screen_size < 1.5
                    # Update anchor positions using the camera poses used to optimize it
                    self.update_anchor(self.n_kept_frames)
                    self.num_active_frames_cpu = 0

                    ## Merge fine Gaussians for the current active set
                    # Select a subset and get their nearest neighbours for merging
                    small_gaussians = {
                        name: self.gaussian_params[name]["val"][small_mask]
                        for name in self.gaussian_params
                    }
                    xyz = small_gaussians["xyz"].contiguous()
                    _, nn_idx = distIndex2(xyz, k)
                    nn_idx = nn_idx.view(-1, k)
                    perm = torch.randperm(xyz.shape[0], device=xyz.device)
                    idx = perm[: (xyz.shape[0] // (k + 1))]
                    selected_nn_idx = torch.cat([idx[..., None], nn_idx[idx]], dim=-1)

                    # Compute merging weights based on contribution to the rendering
                    weights = self.gaussian_params["opacity"]["val"][
                                  selected_nn_idx, 0
                              ].sigmoid() * (screen_size[selected_nn_idx] ** 2)
                    weights = weights / weights.sum(dim=-1, keepdim=True)
                    weights.unsqueeze_(-1)

                    # Merge the Gaussians by averaging their parameters
                    merged_gaussians = {
                        "xyz": (self.gaussian_params["xyz"]['val'][selected_nn_idx, :] * weights).sum(dim=1),
                        "f_dc": (self.gaussian_params["f_dc"]['val'][selected_nn_idx, :] * weights.unsqueeze(-1)).sum(
                            dim=1),
                        "f_rest": (self.gaussian_params["f_rest"]['val'][selected_nn_idx, :] * weights.unsqueeze(
                            -1)).sum(dim=1),
                        "opacity": inverse_sigmoid(
                            self.gaussian_params["opacity"]['val'][selected_nn_idx, :].sigmoid() * weights).sum(dim=1),
                        "scaling": torch.log((torch.exp(
                            self.gaussian_params["scaling"]['val'][selected_nn_idx, :]) * weights * (k + 1)).sum(
                            dim=1)),
                        "rotation": (self.gaussian_params["rotation"]['val'][selected_nn_idx, :] * weights).sum(dim=1),
                        "id": torch.full((len(selected_nn_idx), 1), self.keyframes[-1].index, device="cuda", dtype=torch.long),
                    }

                    # Offload the previous Gaussians to the CPU
                    self.active_anchor.duplicate_param_dict()
                    self.active_anchor.to("cpu", with_keyframes=True)

                    ## Add the merged Gaussians to the set of Gaussians and reset the optimizer
                    with self.lock:
                        self.optimizer.add_and_prune(merged_gaussians, ~small_mask)

                    # Create a new active anchor with the merged Gaussians
                    self.active_anchor = Anchor(
                        self.gaussian_params,
                        self.approx_cam_centres[-1],
                        self.keyframes[-self.n_kept_frames:],
                    )
                    self.anchors.append(self.active_anchor)

                    # Visualization
                    self.anchor_weights = np.zeros(len(self.anchors))
                    self.anchor_weights[-1] = 1.0

                gc.collect()
                torch.cuda.empty_cache()

    def save(self, path: str, reconstruction_time: float = 0, n_frames: int = 0):
        # Get metrics
        metrics = {
            "num anchors": len(self.anchors),
            "num keyframes": len(self.keyframes),
        }
        if reconstruction_time > 0:
            metrics["time"] = reconstruction_time
            if n_frames > 0:
                metrics["FPS"] = n_frames / reconstruction_time
        metrics.update(self.evaluate( True, True))

        if path == "":
            print("No path provided, skipping save")
            return metrics

        # Save anchors
        pcd_path = os.path.join(path, "point_clouds")
        os.makedirs(pcd_path, exist_ok=True)
        for index, anchor in enumerate(self.anchors):
            anchor.save_ply(os.path.join(pcd_path), index)

        # Save metadata
        metadata = {
            "config": {
                "width": self.width,
                "height": self.height,
                "sh_degree": self.max_sh_degree,
                "f": self.f,
            },
            "anchors": [
                {
                    "position": anchor.position.cpu().numpy().tolist(),
                }
                for anchor in self.anchors
            ],
            "keyframes": [keyframe.to_json() for keyframe in self.keyframes],
        }
        metadata = {**metrics, **metadata}

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        # Save renders of test views
        self.save_test_frames(os.path.join(path, "test_images"))

        # Saving cameras with COLMAP format
        images = {}
        cameras = {}
        colmap_save_path = os.path.join(path, "colmap")
        os.makedirs(colmap_save_path, exist_ok=True)
        for index, keyframe in enumerate(self.keyframes):
            camera, image = keyframe.to_colmap(index)
            cameras[index] = camera
            images[index] = image
        write_model(cameras, images, {}, colmap_save_path, ext=".bin")
        
        # save combined points cloud with colmap
        all_pts = {'points': np.zeros((0, 3)), "colors": np.zeros((0, 3))}
        for index, anchor in enumerate(self.anchors):
            ply_path = os.path.join(pcd_path, f"anchor_{index}_xyz_rgb.ply")
            ply = PlyData.read(ply_path)
            vertex = ply['vertex']
            points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
            colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=-1)
            all_pts['points'] = np.concatenate([all_pts['points'], points], axis=0)
            all_pts['colors'] = np.concatenate([all_pts['colors'], colors], axis=0)

        attributes = np.concatenate((all_pts['points'], all_pts['colors']), axis=1)
        dtype_xyz_rgb = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        save_ply(attributes, dtype_xyz_rgb, os.path.join(colmap_save_path, "points3D.ply"))

        # save tum for evo metrics
        sorted_keyframes = sorted(
            [(Path(keyframe.image_name).stem, keyframe) for keyframe in self.keyframes],
            key=lambda x: x[0]  # 只按第一个元素(文件名)排序
        )
        # using save_poses_as_pyramid_ply
        gt_Rts = []
        onthfly_Rts = []
        with open(os.path.join(path, "onthefly.txt"), "w") as f1, open(os.path.join(path, "gt.txt"), "w") as f2:
            for name, keyframe in sorted_keyframes:
                Twc = torch.linalg.inv(keyframe.get_Rt())
                qw, qx, qy, qz = rotmat2qvec(Twc[:3, :3].cpu().detach().numpy()).tolist()
                twc = Twc[:3, 3]
                x, y, z = twc.cpu().detach().numpy().tolist()
                f1.write(f"{name} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
                onthfly_Rts.append(Twc.cpu().detach().numpy())

                if keyframe.Rt_gt is not None:
                    gt_Rt = torch.linalg.inv(keyframe.Rt_gt)  # T_cw -> T_wc
                    x, y, z = gt_Rt[:3, 3].cpu().detach().numpy().tolist()
                    qw, qx, qy, qz = rotmat2qvec(torch.linalg.inv(gt_Rt[:3, :3]).cpu().detach().numpy()).tolist()
                    f2.write(f"{name} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
                    gt_Rts.append(gt_Rt.cpu().detach().numpy())

        # 注意！ 输入的Rt_list: list/np.ndarray, shape=(N,4,4)
        # 修改你的代码
        onthefly_ply_path = os.path.join(path, "onthefly.ply")
        save_poses_as_pyramid_ply(onthfly_Rts, onthefly_ply_path, size=0.3, color='red')

        if len(gt_Rts) > 0:
            gt_ply_path = os.path.join(path, "gt.ply")
            save_poses_as_pyramid_ply(gt_Rts, gt_ply_path, size=0.3, color='green')

        return metrics

    def get_closest_keyframe(
            self, position: torch.Tensor, count: int = 1
    ) -> list[Keyframe]:
        dists = torch.linalg.vector_norm(
            self.approx_cam_centres - position[None], dim=-1
        )
        closest_ids = dists.argsort()[:count]
        return [self.keyframes[closest_id] for closest_id in closest_ids]

    def freeze_learning(self):
        self.gaussian_params["f_dc"]["lr"] = 0
        self.gaussian_params["f_rest"]["lr"] = 0
        self.gaussian_params["opacity"]["lr"] = 0
        self.gaussian_params["xyz"]["lr"] = 0
        self.gaussian_params["rotation"]["lr"] = 0
        self.gaussian_params["scaling"]["lr"] = 0
        self.lr_dict['xyz']['lr_init'] = 0
        for i in range(len(self.keyframes)):
            params = self.keyframes[i].optimizer.params
            for key in params:
                params[key]["lr"] = 0

    def finetune_epoch(self):
        """
        Go through all anchors and optimize them one by one.
        This is used for finetuning after the initial training.
        """
        self.anchor_weights = np.zeros(len(self.anchors))
        for anchor_id, anchor in enumerate(self.anchors):
            self.active_anchor = anchor
            # Load the anchor and make its parameters optimizable
            anchor.to(self.device, with_keyframes=True)
            self.gaussian_params = anchor.gaussian_params
            self.anchor_weights[anchor_id] = 1
            self.reset_optimizer()
            for key, param_group in self.optimizer.params.items():
                if key in self.optimizer.lr_dict:
                    num_points = self.gaussian_params["xyz"]["val"].shape[0]
                    new_lr = torch.full(
                        (num_points,),
                        self.optimizer.lr_dict[key]["lr_init"],
                        dtype=torch.float,
                        device=self.device
                    )
                    param_group["lr"] = new_lr
            # # Ensure other anchors are on cpu to save memory
            # if anchor_id >= 1:
            #     self.anchors[anchor_id-1].to("cpu", with_keyframes=True)

            # Optimize the anchor by going through its keyframes
            for _ in range(len(anchor.keyframes)):
                self.optimization_step(finetuning=True)

            # Update the anchor and store it on cpu
            anchor.gaussian_params = self.gaussian_params
            self.anchor_weights[anchor_id] = 0
