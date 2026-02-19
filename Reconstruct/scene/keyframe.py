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

from __future__ import annotations
from argparse import Namespace
import torch
import torch.nn.functional as F
from Reconstruct.scene.optimizers import BaseAdam
from Reconstruct.utils import sample, sixD2mtx
from dataloaders.utils_IO import Camera, BaseImage, rotmat2qvec


class Keyframe:
    """
    A keyframe in the scene, containing the image, camera parameters, and other information used for optimization.
    """
    def __init__(
        self,
        image: torch.Tensor,
        image_name: str,
        is_test: bool,
        Rt: torch.Tensor,
        mapper_keyframe_idx: int,
        global_frame_id: int,
        last_keyframe_index: int,
        last_keyframe_frame_id: int,
        is_slam_keyframe: bool,
        f: torch.Tensor,
        args: Namespace,
        T_CkCf: torch.Tensor = None,
        prev_kf: Keyframe = None,
        inference_mode: bool = False,
        Rt_gt: torch.Tensor = None,
        focal_gt: torch.Tensor = None,
        point_map: torch.Tensor = None, # H W 3
        point_conf: torch.Tensor = None, # H W
        device_mapper = "cuda:0"
    ):
        self.device_mapper = device_mapper
        self.image_pyr = [image]
        self.image_name = image_name
        self.is_test = is_test
        self.width = image.shape[2]
        self.height = image.shape[1]
        self.index = mapper_keyframe_idx
        self.global_frame_id = global_frame_id
        self.last_keyframe_index = last_keyframe_index
        self.last_keyframe_frame_id = last_keyframe_frame_id
        self.is_slam_keyframe = is_slam_keyframe
        self.T_CkCf = T_CkCf
        self.latest_invdepth = None

        self.Rt_gt = Rt_gt
        self.focal_gt = focal_gt

        if not inference_mode:
            # mono_depth prediction
            self.point_map = point_map.permute(2, 0, 1)[None] # 1 3 H_slam W_slam
            depth_foundation = self.point_map[:, 2:, ...]
            idepth = torch.where(depth_foundation != 0, 1.0 / (depth_foundation + 1e-4), 1e4)
            self.mono_depth_conf = point_conf[None, None, ...].to(torch.float32)  # 1 1 H_slam W_slam
            self.idepth_pyr = [
                F.interpolate(
                    idepth,
                    (self.height, self.width),
                    mode="bilinear",
                    align_corners=True,
                )[0]
            ]
            self.idepth_conf_pyr = [
                F.interpolate(
                    self.mono_depth_conf,
                    (self.height, self.width),
                    mode="bilinear",
                    align_corners=True,
                )[0]
            ]
            # Create the multi-scale inverse depth pyramid
            for _ in range(args.pyr_levels - 1):
                self.idepth_pyr.append(F.avg_pool2d(self.idepth_pyr[-1], 2))
                self.idepth_conf_pyr.append(F.avg_pool2d(self.idepth_conf_pyr[-1], 2))

            self.centre = torch.tensor(
                [(self.width - 1) / 2, (self.height - 1) / 2]
            ).to(self.device_mapper)
            self.f = f
            self.depth_loss_weight = args.depth_loss_weight_init
            self.depth_loss_weight_decay = args.depth_loss_weight_decay
            # Build the multiscale pyramids
            for _ in range(args.pyr_levels - 1):
                self.image_pyr.append(F.avg_pool2d(self.image_pyr[-1], 2))
            self.pyr_lvl = args.pyr_levels - 1

        # Optimizable parameters
        self.rW2C = torch.nn.Parameter(Rt[:3, :2].clone().contiguous())
        self.tW2C = torch.nn.Parameter(Rt[:3, 3].clone().contiguous())
        exposure = (
            torch.eye(3, 4, device=self.device)
            if prev_kf is None
            else prev_kf.exposure.clone().detach()
        )
        self.exposure = torch.nn.Parameter(exposure)

        # Optimizer
        if not inference_mode: # Only create optimizer in training mode
            lr_poses = 0 if self.index == 0 else args.lr_poses
            # Test frame does not participate in optimizing the scene,
            # but we need its pose to be accurate to evaluate the reconstruction quality.
            if self.is_test:
                lr_poses = 1e-4
            params = {
                "rW2C": {"val": self.rW2C, "lr": lr_poses},
                "tW2C": {"val": self.tW2C, "lr": lr_poses},
            }
            if not self.is_test:
                params["exposure"] = {"val": self.exposure, "lr": args.lr_exposure}
            self.optimizer = BaseAdam(params, betas=(0.8, 0.99))
            self.num_steps = 0
        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    def to(self, device: str, only_train=False):
        if self.device.type == device:
            return
        for i in range(len(self.image_pyr)):
            self.image_pyr[i] = self.image_pyr[i].to(device)
            if self.idepth_pyr is not None:
                self.idepth_pyr[i] = self.idepth_pyr[i].to(device)
        if not only_train:
            if self.latest_invdepth is not None:
                self.latest_invdepth = self.latest_invdepth.to(device)

    @property
    def device(self):
        return self.image_pyr[0].device

    def get_R(self):
        return sixD2mtx(self.rW2C)

    def get_t(self):
        return self.tW2C

    def get_Rt(self):
        Rt = torch.eye(4, device=self.idepth_pyr[0].device)
        Rt[:3, :3] = self.get_R()
        Rt[:3, 3] = self.get_t()
        return Rt

    def set_Rt(self, Rt: torch.Tensor):
        self.rW2C.data.copy_(Rt[:3, :2])
        self.tW2C.data.copy_(Rt[:3, 3])
        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    def get_centre(self, approx=False):
        """
        Get the centre of the keyframe in the world coordinate
        """
        if approx:
            return self.approx_centre
        else:
            return -self.get_R().T @ self.get_t()


    def get_mono_idepth(self, lvl=0):
        """
            return 1 H W
        """
        if self.idepth_pyr[lvl].device.type != "cuda":
            self.idepth_pyr[lvl] = self.idepth_pyr[lvl].to(self.device_mapper)
        return self.idepth_pyr[lvl]


    @torch.no_grad()
    def sample_conf(self, uv):
        return sample(self.mono_depth_conf, uv.view(1, 1, -1, 2), self.width, self.height)[0, 0, 0]

    def zero_grad(self):
        self.optimizer.zero_grad()

    @torch.no_grad()
    def step(self):
        # Optimizer step
        self.optimizer.step()
        self.depth_loss_weight *= self.depth_loss_weight_decay
        self.num_steps += 1
        # decrement pyr_lvl
        # if self.num_steps % 5 == 0:
        #     if self.pyr_lvl > 0:
        #         self.image_pyr.pop()
        #         if self.idepth_pyr is not None:
        #             self.idepth_pyr.pop()
        #         self.pyr_lvl -= 1

    def to_json(self):
        info = {"is_test": self.is_test}
        info["name"] = self.image_name
        if self.Rt_gt is not None:
            info["gt_Rt"] = self.Rt_gt.cpu().numpy().tolist()

        return {
            "info": info,
            "Rt": self.get_Rt().detach().cpu().numpy().tolist(),
            "f": self.f.item(),
        }

    @classmethod
    def from_json(cls, config, index, height, width, device):
        if "gt_Rt" in config["info"]:
            config["info"]["Rt"] = torch.tensor(config["info"]["gt_Rt"]).cuda()
        keyframe = cls(
            image=None,
            image_name=config["info"]["name"],
            frame_idx=index,
            is_test=config["info"]["is_test"],
            desc_kpts=None,
            Rt=torch.tensor(config["Rt"]).cuda(),
            keyframe_idx=index,
            f=None,
            args=None,
            inference_mode=True,
        )
        keyframe.height = height
        keyframe.width = width
        keyframe.centre = torch.tensor(
            [(width - 1) / 2, (height - 1) / 2], device=device
        )
        return keyframe

    def to_colmap(self, id):
        """
        Convert the keyframe to a colmap camera and image.
        """
        # first param of params is focal length in pixels
        camera = Camera(
            id=id,
            model="SIMPLE_PINHOLE",
            width=self.width,
            height=self.height,
            params=[self.f.item(), self.centre[0].item(), self.centre[1].item()],
        )

        image = BaseImage(
            id=id,
            name=self.image_name,
            camera_id=id,
            qvec=-rotmat2qvec(self.get_R().cpu().detach().numpy()),
            tvec=self.get_t().flatten().cpu().detach().numpy(),
            xys=[],
            point3D_ids=[],
        )

        return camera, image
