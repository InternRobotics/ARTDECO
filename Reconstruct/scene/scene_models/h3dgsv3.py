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
import torch.nn as nn
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import json
import math
import threading
import time
import warnings
import gsplat
import cv2
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
from typing import Optional
import shutil
from Reconstruct.utils import inverse_sigmoid, to_numpy
from Reconstruct.utils import save_ply
from plyfile import PlyData, PlyElement

import lpips
from torch_scatter import scatter_max
from fused_ssim import fused_ssim
from simple_knn._C import distIndex2
from Reconstruct.poses.feature_detector import DescribedKeypoints
from Reconstruct.scene.optimizers import SparseGaussianAdam
from Reconstruct.scene.keyframe import Keyframe
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
    save_poses_as_pyramid_ply, 
    sample, 
    getProjectionMatrix2,
    update_gaussians
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
            args: Arguments for the scene model. 
            matcher: Matcher for the scene model. Defaults to None (if inference_mode is True).
            inference_mode: Whether we load the scene for visualization. Defaults to False.
        """
        self.device = device
        self.width = width
        self.height = height
        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device=self.device)
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
            self.scaling_reg_factor = args.scaling_reg_factor
            self.rad_decay = args.rad_decay if args.rad_decay != -1 else 1.0
            self.debug_iter = 0

            self.lr_dict = {
                "xyz": {
                    "lr_init": args.position_lr_init,
                    "lr_decay": args.position_lr_decay,
                }
            }

            # Initialize Gaussian parameters
            self.gaussian_params = {
                "id": {
                    "val": torch.empty(0, 1, device=self.device, dtype=torch.long)
                },
                "cls_id": {
                    "val": torch.empty(0, 1, device=self.device, dtype=torch.long),
                },
                "d_max": {
                    "val": torch.empty(0, 1, device=self.device),
                },
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
                "local_feat": {
                    "val": torch.empty(0, args.local_feat_dim, device=self.device),
                    "lr": args.feat_lr,
                },
                "global_feat": {
                    "val": torch.empty(0, args.global_feat_dim, device=self.device),
                    "lr": args.feat_lr,
                }
            }

            self.mlp_cov = nn.Sequential(
                nn.Linear(args.global_feat_dim + args.local_feat_dim, args.global_feat_dim + args.local_feat_dim),
                nn.ReLU(True),
                nn.Linear(args.global_feat_dim + args.local_feat_dim, 7),
            ).cuda()

            self.mlp_params = {}

            for n, p in self.mlp_cov.named_parameters():
                name = "mlp_cov_" + n.replace(".", "_")
                self.mlp_params[name] = {"val": p, "lr": args.mlp_cov_lr_init}
                self.lr_dict[name] = {"lr_init": args.mlp_cov_lr_init, "lr_decay": args.mlp_cov_lr_decay}

            self.num_clusters = 0
            self.registered_hashes = torch.empty(0, dtype=torch.long, device=device)
            # Initialize optimizer
            self.reset_optimizer()

        self.keyframes: list[Keyframe] = []
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

        self.lods = [1, 2, 4, 8]
        self.uvs = dict()
        for lod in self.lods:
            self.uvs[lod] = torch.stack(torch.meshgrid(torch.arange(0, width // lod), torch.arange(0, height // lod), indexing="xy"), dim=-1).float().to(self.device)

    def update_voxel(self, new_xyz, xyz, cls_id, voxel_size=0.1):
        """
        Args:
            new_xyz: (M, 3), dtype=float
            xyz: (N, 3), dtype=float
            cls_id: (N, 1), dtype=torch.long
            voxel_size: float
        Returns:
            updated_orig_cls_id: (N, 1), dtype=torch.long
            updated_new_cls_id: (M, 1), dtype=torch.long
            new_voxel_count: int
        """
        device = new_xyz.device
        num_new = new_xyz.shape[0]
        num_orig = xyz.shape[0]

        # --- 1. 处理初始 xyz 为空的冷启动情况 ---
        if num_orig == 0:
            
            # 仅对新点进行体素化
            v_min = new_xyz.min(dim=0).values
            v_idx = torch.floor((new_xyz - v_min) / voxel_size).long()
            # 鲁棒的 stride 计算
            v_max = v_idx.max(dim=0).values + 1
            stride = torch.tensor([v_max[1] * v_max[2], v_max[2], 1], device=device)
            h_new = (v_idx * stride).sum(dim=1)

            u_hashes, u_inv = torch.unique(h_new, return_inverse=True)
            return u_inv.unsqueeze(-1), u_hashes.shape[0]

        # --- 2. 正常逻辑：准备 Hash ---
        cls_id_1d = cls_id.squeeze(-1)
        max_cls = cls_id_1d.max().item()
        
        # 合并点云以计算统一的坐标偏移和步长
        all_p = torch.cat([xyz, new_xyz], dim=0)
        min_c = all_p.min(dim=0).values
        
        # 计算体素索引
        v_idx_all = torch.floor((all_p - min_c) / voxel_size).long()
        v_max = v_idx_all.max(dim=0).values + 1
        stride = torch.tensor([v_max[1] * v_max[2], v_max[2], 1], device=device)
        
        # 分离出 hash
        h_all = (v_idx_all * stride).sum(dim=1)
        h_orig = h_all[:num_orig]
        h_new = h_all[num_orig:]

        # --- 3. 众数计算 (Majority Vote) ---
        # 找到原始体素及其对应的点索引
        unique_voxels, inv_idx = torch.unique(h_orig, return_inverse=True)
        
        # 建立组合 ID 用于频次统计: [Voxel_Index] -> [Class_ID]
        offset = max_cls + 1
        pair_id = inv_idx * offset + cls_id_1d
        
        # 统计 (体素, 类别) 组合出现的次数
        pair_unique_ids, pair_counts = torch.unique(pair_id, return_counts=True)
        v_indices_in_pair = pair_unique_ids // offset
        c_labels_in_pair = pair_unique_ids % offset

        # 使用 torch_scatter 提取每个体素出现次数最多的类别
        _, max_indices = scatter_max(pair_counts, v_indices_in_pair)
        voxel_mode_labels = c_labels_in_pair[max_indices] 

        # --- 4. 更新与映射 ---
        # 更新原始点云标签
        updated_orig_cls_id = voxel_mode_labels[inv_idx].unsqueeze(-1)

        # 检索 new_xyz 命中情况
        pos = torch.searchsorted(unique_voxels, h_new)
        pos_clamped = pos.clamp(max=unique_voxels.shape[0] - 1)
        mask = (unique_voxels[pos_clamped] == h_new)

        updated_new_cls_id = torch.zeros(num_new, dtype=torch.long, device=device)
        
        # 情况 A: 命中已有体素，赋予众数
        if mask.any():
            updated_new_cls_id[mask] = voxel_mode_labels[pos_clamped[mask]]
        
        # 情况 B: 落在新体素，分配递增新 ID
        new_voxel_count = 0
        if (~mask).any():
            unmatched_h = h_new[~mask]
            u_new_h, u_new_inv = torch.unique(unmatched_h, return_inverse=True)
            new_voxel_count = u_new_h.shape[0]
            # ID 紧接当前最大值
            updated_new_cls_id[~mask] = u_new_inv + max_cls + 1

        return updated_orig_cls_id, updated_new_cls_id.unsqueeze(-1), new_voxel_count

    def reset_optimizer(self):
        for key in self.gaussian_params:
            if key == 'id' or key == 'cls_id' or key == 'd_max' or key == 'mlp_cov':
                continue
            if not self.gaussian_params[key]["val"].requires_grad:
                self.gaussian_params[key]["val"].requires_grad = True
        for key in self.mlp_params:
            if not self.mlp_params[key]["val"].requires_grad:
                self.mlp_params[key]["val"].requires_grad = True
        train_params = {**self.gaussian_params, **self.mlp_params}
        self.optimizer = SparseGaussianAdam(
            train_params, (0.5, 0.99), lr_dict=self.lr_dict, device=self.device
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
        return self.gaussian_params["rotation"]["val"]


    @property
    def opacity(self):
        return torch.sigmoid(self.gaussian_params["opacity"]["val"])

    @property
    def id(self):
        return self.gaussian_params["id"]["val"]

    @property
    def cls_id(self):
        return self.gaussian_params["cls_id"]["val"]

    @property
    def d_max(self):
        return self.gaussian_params["d_max"]["val"]

    @property
    def local_feat(self):
        return self.gaussian_params["local_feat"]["val"]

    @property
    def global_feat(self):
        return self.gaussian_params["global_feat"]["val"]

    @property
    def n_active_gaussians(self):
        return self.xyz.shape[0]

    @property
    def first_active_frame(self):
        return self.keyframes[0].index

    @property
    def last_active_frame(self):
        return self.keyframes[-1].index

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
        gt_image = keyframe.image_pyr[lvl]
        mono_idepth = keyframe.get_mono_idepth(lvl)

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

        scaling_reg = scale.prod(dim=1).mean()
        loss = (
            self.lambda_dssim * ssim_loss
            + (1 - self.lambda_dssim) * l1_loss
            + keyframe.depth_loss_weight * depth_loss
            + self.scaling_reg_factor * scaling_reg
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
                    render_pkg["visibility_filter"], render_pkg["visibility_filter"].shape[0],
                    render_pkg["global_visibility_filter"], render_pkg["global_visibility_filter"].shape[0]
                )

            keyframe.latest_invdepth = render_pkg["invdepth"].detach()

        self.valid_Rt_cache[keyframe_id] = False
        self.last_trained_id = keyframe_id
        # self.clean()

    # def clean(self):
    #     if torch.isnan(self.local_feat).any():
    #         self.gaussian_params["local_feat"]["val"] = self.local_feat.nan_to_num(nan=0.0)
    #     if torch.isnan(self.global_feat).any():
    #         self.gaussian_params["global_feat"]["val"] = self.global_feat.nan_to_num(nan=0.0)


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
        metrics = {"PSNR": 0.0, "SSIM": 0.0, "Render": 0.0, "GS": 0.0}
        if with_LPIPS:
            metrics["LPIPS"] = 0.0
        n_test_frames = 0
        start_index = 0 if all else self.keyframes[0].index
        for index, keyframe in enumerate(self.keyframes[start_index:]):
            if keyframe.is_test:
                gt_image = keyframe.image_pyr[0].to(self.device)
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                image = render_pkg["render"]
                mask = torch.ones_like(image[:1] > 0)
                mask = mask.expand_as(image)
                image = image * mask
                gt_image = gt_image * mask
                metrics["Render"] += render_pkg["visibility_filter"].sum().item()
                metrics["GS"] += len(render_pkg["visibility_filter"])
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
                cv2.imwrite(depth_gray_path, invdepth_vis)
                
                cv2.imwrite(
                    os.path.join(out_dir, keyframe.image_name), image, write_flag
                )
                
        print(f"Saved {len(diagnostics)} test frames to {out_dir}")

    def render_from_id(
        self,
        keyframe_id,
        pyr_lvl=0,
        bg=torch.zeros(3),
    ):
        """
        Render the scene from a given keyframe id at a specified resolution level (pyr_lvl).
        Applies the exposure matrix of the keyframe to the rendered image.
        """
        bg = bg.to(self.device)
        keyframe = self.keyframes[keyframe_id]
        view_matrix = keyframe.get_Rt().to(self.device)
        scale = 2 ** pyr_lvl
        width, height = self.width // scale, self.height // scale
        render_pkg = self.render(width, height, view_matrix, bg)
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
        bg: torch.Tensor = torch.zeros(3),
    ):
        with self.lock:
            # Load and blend anchors if in inference mode
            xyz = self.xyz
            cam_centre = view_matrix.detach().inverse()[:3, 3].to(self.device)
            ob_view = xyz - cam_centre
            # dist
            ob_dist = ob_view.norm(dim=1, keepdim=True)
            # view
            ob_view = ob_view / ob_dist
            selection_mask = (ob_dist < 2 * self.d_max).squeeze(-1)
            alpha_mask = torch.logical_and(ob_dist > self.d_max, ob_dist < 2 * self.d_max).squeeze(-1)
            alpha_ratio = (2 * self.d_max - ob_dist) / self.d_max
            alpha_ratio[~alpha_mask] = 1.0

            xyz = xyz[selection_mask]
            opacity = (self.opacity * alpha_ratio)[selection_mask]
            f_dc = self.f_dc[selection_mask]
            f_rest = self.f_rest[selection_mask]
            scaling = self.scaling[selection_mask]
            rotation = self.rotation[selection_mask]
            f_dc = self.f_dc[selection_mask]
            f_rest = self.f_rest[selection_mask]
            feats = torch.concat([f_dc, f_rest], dim=1)

            fl_x = width / (2 * self.tanfovx)
            fl_y = height / (2 * self.tanfovy)
            Ks = torch.tensor([
                [fl_x, 0, width / 2.0],
                [0, fl_y, height / 2.0],
                [0, 0, 1],
            ],device="cuda",)[None]

            local_feat = self.local_feat[selection_mask]
            ids = self.cls_id[selection_mask].squeeze(-1).long()
            global_feat = self.global_feat[ids]
            cat_local_view = torch.cat([global_feat, local_feat], dim=1) # [N, c]
            scale_rot = self.mlp_cov(cat_local_view)
            scaling = scaling * torch.sigmoid(scale_rot[:,:3])
            rotation = F.normalize(rotation * scale_rot[:,3:])

            colors, alphas, meta = gsplat.rendering.rasterization(
                means=xyz,
                quats=rotation,
                scales=scaling,
                opacities=opacity.squeeze(-1),
                colors=feats,
                viewmats=view_matrix.unsqueeze(0),
                Ks=Ks,
                width=width,
                height=height,
                render_mode="RGB+D",
                rasterize_mode="classic",
                absgrad=False,
                packed=False,
                sh_degree=self.active_sh_degree,
                eps2d=self.args.low_pass_filter_eps,
            )  

            rendered_color = colors[..., 0:3].permute([0, 3, 1, 2])
            rendered_depth_unnormalized = colors[..., 3:4].permute([0, 3, 1, 2])
            rendered_alpha = alphas.permute([0, 3, 1, 2])
            rendered_color = rendered_color + (1.0 - rendered_alpha) * bg[None, :, None, None]
            invdepth = 1.0 / rendered_depth_unnormalized

            visible_mask = torch.zeros_like(selection_mask, dtype=torch.bool, device=self.device)
            visible_mask[selection_mask.clone()] = meta['radii'][0].max(dim=1).values > 0

            global_visible_mask = torch.zeros(len(self.global_feat), dtype=torch.bool, device=self.device)
            global_visible_mask[self.cls_id[visible_mask].squeeze(-1)] = True

        return {
            "render": rendered_color[0],
            "invdepth": invdepth[0],
            "visibility_filter": visible_mask,
            "global_visibility_filter": global_visible_mask,
            "scale": scaling
        }   

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
            "id": self.id[:0].detach(),
            "cls_id": self.cls_id[:0].detach(),
            "d_max": self.d_max[:0].detach(),
            "xyz": self.xyz[:0].detach(),
            "f_dc": self.f_dc[:0].detach(),
            "f_rest": self.f_rest[:0].detach(),
            "opacity": self.opacity[:0].detach(),
            "scaling": self.scaling[:0].detach(),
            "rotation": self.rotation[:0].detach(),
            "local_feat": self.local_feat[:0].detach(),
            "global_feat": self.global_feat[:0].detach(),
        }

    @torch.no_grad()
    def add_new_gaussians(self, keyframe_id: int = -1):
        """Use the given keyframe to add new Gaussians to the scene model."""
        keyframe = self.keyframes[keyframe_id]

        # Skip if the keyframe is a test keyframe
        if keyframe.is_test:
            return

        ## Get the pixel-wise probability to add a Gaussian
        org_img = keyframe.image_pyr[0]
        org_img = F.avg_pool2d(org_img, 2)
        extension_tensors = dict()
        for lod in self.lods:
            cur_h = self.height // lod
            cur_w = self.width // lod
            img = F.interpolate(org_img[None], (cur_h, cur_w), mode="bilinear", align_corners=True)[0]
            init_proba = get_lapla_norm(img, self.disc_kernel, device=self.device)  # eq. 1

            ## Compute the penalty based on the rendering from the new keyframe's point of view
            penalty = 0
            rendered_depth = None
            if self.xyz.shape[0] > 0:
                render_pkg = self.render_from_id(keyframe_id)
                render = F.interpolate(render_pkg["render"][None], (cur_h, cur_w), mode="bilinear", align_corners=True)[0]
                rendered_depth = 1 / F.interpolate(render_pkg["invdepth"][None], (cur_h, cur_w), mode="bilinear", align_corners=True)[0][0].clamp_min(1e-8)
                penalty = get_lapla_norm(render, self.disc_kernel, device=self.device)

            # Define which pixels should become Gaussians
            init_proba *= self.init_proba_scaler
            penalty *= self.init_proba_scaler

            # sample points
            sample_mask = torch.rand_like(init_proba) < (init_proba - penalty) * self.args.gs_add_ratio  # eq. 3
            sample_mask = sample_mask.to(self.device)
            sampled_uv = self.uvs[lod][sample_mask]  # L
            sampled_depths = sample(
                keyframe.point_map[:, 2:],  # 1 1 H W
                sampled_uv[None, None, ...],  # 1 1 L 2
                keyframe.width // lod,
                keyframe.height // lod
            )[0, 0, 0]  # L

            sampled_conf = sample(
                keyframe.mono_depth_conf,  # 1 1 H W
                sampled_uv[None, None, ...],  # 1 1 L 2
                keyframe.width // lod,
                keyframe.height // lod
            )[0, 0, 0]  # L

            quantile_depth_min = min(1e-2, torch.quantile(keyframe.point_map[:, 2], 0.02).item())
            max_pixel_difference = 5
            min_conf = 1 / (max_pixel_difference - 2)

            valid_mask = ((sampled_conf >= 0)
                        * (sampled_depths > quantile_depth_min))  # L
            sample_mask[sample_mask.clone()] = valid_mask
            sampled_uv = sampled_uv[valid_mask]  # L1 2
            sampled_depths = sampled_depths[valid_mask]  # L1 3
            sampled_conf = sampled_conf[valid_mask]

            # # Remove Gaussians that are coarser than the newpoints
            # if lod == 1:
            #     if len(self.xyz) > 0:
            #         main_gaussians_map = render_pkg["mainGaussID"]
            #         accurate_sample_mask = sample_mask.clone()
            #         accurate_sample_mask[accurate_sample_mask.clone()] = True
            #         selected_main_gaussians = main_gaussians_map[:, accurate_sample_mask]  # L1
            #         ids, counts = torch.unique(
            #             selected_main_gaussians[selected_main_gaussians >= 0],
            #             return_counts=True,
            #         )
            #         valid_gs_mask = torch.ones_like(self.xyz[:, 0], dtype=torch.bool)
            #         valid_gs_mask[ids] = counts < 10
            #         with self.lock:
            #             self.optimizer.add_and_prune(
            #                 self.make_dummy_ext_tensor(), valid_gs_mask
            #             )
            #         render_pkg = self.render_from_id(keyframe_id)
            #         rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)

            # todo: 内参矩阵目前假设fx=fy cx cy在中心
            f = self.f / lod
            centre = self.centre / lod
            sampled_points = depth2points(sampled_uv, sampled_depths.unsqueeze(-1), f, centre)
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
            scales = torch.log(lod * scales.clamp(1e-6, 1e6)).unsqueeze(-1).repeat(1, 3)

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
            local_feats = torch.zeros((f_dc.shape[0], self.args.local_feat_dim),device=self.device).float()
            
            if len(self.xyz) > 0:
                update_cls_ids, new_cls_ids, new_voxel_count = self.update_voxel(sampled_points, self.xyz, self.cls_id, self.args.voxel_size)
                self.gaussian_params["cls_id"]["val"] = update_cls_ids
            else:
                new_cls_ids, new_voxel_count = self.update_voxel(sampled_points, self.xyz, self.cls_id, self.args.voxel_size)
            global_feats = torch.zeros((new_voxel_count, self.args.global_feat_dim), device=self.device)
            rotation = torch.zeros((f_dc.shape[0], 4), device="cuda")
            rotation[:, 0] = 1
            d_maxs = (sampled_depths.unsqueeze(-1) * lod).to(self.device)

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
            extension_tensors[lod] = {
                "id": torch.full((len(sampled_points), 1), keyframe_id, device="cuda", dtype=torch.long),
                "cls_id": new_cls_ids,
                "d_max": d_maxs,
                "xyz": sampled_points,
                "f_dc": f_dc,
                "f_rest": f_rest,
                "opacity": opacities,
                "scaling": scales,
                "rotation": rotation,
                "local_feat": local_feats,
                "global_feat": global_feats
            }

        all_ext_tensors = {
            "id": torch.concat([extension_tensors[lod]["id"] for lod in self.lods], dim=0),
            "cls_id": torch.concat([extension_tensors[lod]["cls_id"] for lod in self.lods], dim=0),
            "d_max": torch.concat([extension_tensors[lod]["d_max"] for lod in self.lods], dim=0),
            "xyz": torch.concat([extension_tensors[lod]["xyz"] for lod in self.lods], dim=0),
            "f_dc": torch.concat([extension_tensors[lod]["f_dc"] for lod in self.lods], dim=0),
            "f_rest": torch.concat([extension_tensors[lod]["f_rest"] for lod in self.lods], dim=0),
            "opacity": torch.concat([extension_tensors[lod]["opacity"] for lod in self.lods], dim=0),
            "scaling": torch.concat([extension_tensors[lod]["scaling"] for lod in self.lods], dim=0),
            "rotation": torch.concat([extension_tensors[lod]["rotation"] for lod in self.lods], dim=0),
            "local_feat": torch.concat([extension_tensors[lod]["local_feat"] for lod in self.lods], dim=0),
            "global_feat": torch.concat([extension_tensors[lod]["global_feat"] for lod in self.lods], dim=0),
        }
        
        with self.lock:
            self.optimizer.add_and_prune(all_ext_tensors, valid_gs_mask)

        self.weed_out_gaussians()
    
    def weed_out_gaussians(self):
        visible_count = torch.zeros(self.xyz.shape[0], dtype=torch.int, device=self.device)
        for keyframe in self.keyframes:
            view_matrix = keyframe.get_Rt().transpose(0, 1).to(self.device)
            cam_centre = view_matrix.detach().inverse()[3, :3].to(self.device)
            ob_view = self.xyz - cam_centre
            ob_dist = ob_view.norm(dim=1, keepdim=True).to(self.device)
            selection_mask = (ob_dist < 2 * self.d_max).squeeze(-1).to(self.device)
            visible_count += selection_mask.int()
        visible_count = visible_count / len(self.keyframes)
        weed_mask = (visible_count > self.args.visible_threshold)
        self.optimizer.add_and_prune(self.make_dummy_ext_tensor(), weed_mask)

    @torch.no_grad()
    def rigid_transform_gs(self, old_c2ws, new_c2ws, cam_centres):
        old_c2ws = old_c2ws[self.id.squeeze(-1)]
        new_c2ws = new_c2ws[self.id.squeeze(-1)]

        new_xyz, new_rotation = update_gaussians(
            old_c2ws, new_c2ws, self.xyz, self.rotation
        )

        self.gaussian_params["xyz"]["val"] = new_xyz
        self.gaussian_params["rotation"]["val"] = new_rotation
        self.cam_centres = cam_centres

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
            # Clear memory if there are many keyframes
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
        for keyframe in self.keyframes:
            keyframe.to(self.device)
        self.num_active_frames_cpu = 0

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self.gaussian_params["f_dc"]["val"].shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(
            self.gaussian_params["f_rest"]["val"].shape[1]
            * self.gaussian_params["f_rest"]["val"].shape[2]
        ):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.gaussian_params["scaling"]["val"].shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.gaussian_params["rotation"]["val"].shape[1]):
            l.append("rotation_{}".format(i))
        return l
    
    def save_ply(self, directory: str):
        xyz = to_numpy(self.gaussian_params["xyz"]["val"])
        normals = np.zeros_like(xyz)
        f_dc = to_numpy(
            self.gaussian_params["f_dc"]["val"]
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
        )
        f_rest = to_numpy(
            self.gaussian_params["f_rest"]["val"]
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
        )
        opacities = to_numpy(self.gaussian_params["opacity"]["val"])    
        scaling = torch.exp(self.gaussian_params["scaling"]["val"])
        rotation = self.gaussian_params["rotation"]["val"]
        local_feat = self.gaussian_params["local_feat"]["val"]
        cls_id = self.gaussian_params["cls_id"]["val"].squeeze(-1)
        global_feat = self.gaussian_params["global_feat"]["val"][cls_id]
        cat_local_view = torch.cat([global_feat, local_feat], dim=1) # [N, c]
        scale_rot = self.mlp_cov(cat_local_view)
        scaling = to_numpy(torch.log(torch.sigmoid(scale_rot[:, :3]) * scaling))
        rotation = to_numpy(rotation * scale_rot[:, 3:])

        dtype_org = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scaling, rotation), axis=1
        )

        save_ply(attributes, dtype_org, os.path.join(directory, 'gs.ply'))

    def save_pcd(self, directory: str):
        def convert_rgb_to_spherical_harmonics(rgb: torch.Tensor) -> torch.Tensor:
            # Convert RGB to degree-0 spherical harmonics.
            coeff_degree0 = np.sqrt(1.0 / (4.0 * np.pi))
            return (rgb - 0.5) / coeff_degree0

        xyz = to_numpy(self.gaussian_params["xyz"]["val"])
        f_dc = to_numpy(
            self.gaussian_params["f_dc"]["val"]
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
        )
        f_dc_rgb = convert_rgb_to_spherical_harmonics(f_dc)  # shape (N, 3), values in [0,1]
        f_dc_rgb = np.clip(f_dc_rgb, 0, 1)
        f_dc_rgb_uint8 = (f_dc_rgb * 255).astype(np.uint8)
        attributes = np.concatenate((xyz, f_dc_rgb_uint8), axis=1)
        dtype_xyz_rgb = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        save_ply(attributes, dtype_xyz_rgb, os.path.join(directory, "xyz_rgb.ply"))

    def save(self, path: str, reconstruction_time: float = 0, n_frames: int = 0):
        # Get metrics
        metrics = {
            "num anchors": 1,
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
        self.save_ply(pcd_path)
        self.save_pcd(pcd_path)

        # Save metadata
        metadata = {
            "config": {
                "width": self.width,
                "height": self.height,
                "sh_degree": self.max_sh_degree,
                "f": self.f,
            },
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
        ply_path = os.path.join(pcd_path, f"xyz_rgb.ply")
        ply = PlyData.read(ply_path)
        vertex = ply['vertex']
        points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
        colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=-1)

        attributes = np.concatenate((points, colors), axis=1)
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

    def finetune_epoch(self):
        """
        Go through all anchors and optimize them one by one.
        This is used for finetuning after the initial training.
        """
        self.reset_optimizer()
        for key, param_group in self.optimizer.params.items():
            if key in self.optimizer.lr_dict:
                if key.startswith("mlp"):
                    param_group["lr"] = self.optimizer.lr_dict[key]["lr_init"]
                else:
                    num_points = self.gaussian_params[key]["val"].shape[0]
                    new_lr = torch.full(
                        (num_points,),
                        self.optimizer.lr_dict[key]["lr_init"],
                        dtype=torch.float,
                        device=self.device
                    )
                    param_group["lr"] = new_lr

        # Optimize the anchor by going through its keyframes
        for _ in range(len(self.keyframes)):
            self.optimization_step(finetuning=True)