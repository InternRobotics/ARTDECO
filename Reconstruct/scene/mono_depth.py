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
import cv2
import numpy as np
import torch
import os
import sys
import urllib.request
import torch.nn.functional as F

from Reconstruct.poses.feature_detector import DescribedKeypoints
from Reconstruct.utils import sample

sys.path.append("Reconstruct/submodules/Depth-Anything-V2")
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

size = 518
encoder = "vitl"


class MonoDepthInternal(torch.nn.Module):
    def __init__(self):
        super(MonoDepthInternal, self).__init__()
        model_path = f"models/depth_anything_v2_{encoder}.pth"
        if not os.path.exists(model_path):
            print(f"Downloading Depth-Anything-V2 model for {encoder}, may take a few minutes...")
            model_sizes = {
                "vits": "Small",
                "vitb": "Base",
                "vitl": "Large",
                "vitg": "Giant",
            }
            url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model_sizes[encoder]}/resolve/main/depth_anything_v2_{encoder}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(url, model_path)
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model = model.to("cuda").half().eval()
        self.sobel_x = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device="cuda", dtype=torch.half
            ).unsqueeze(0).unsqueeze(0)
        )
        self.sobel_y = (
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device="cuda", dtype=torch.half
            ).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, image: torch.Tensor):
        img = torch.nn.functional.interpolate(
            image[None].half(), (size, size), mode="bilinear", align_corners=True
        )
        depth = self.model(img)[None]
        t, s = get_t_s(depth)
        depth = (depth - t) / s

        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        edges = torch.cat((grad_x, grad_y), dim=0)

        edges_sq_norm = (edges**2).sum(0, keepdim=True)
        var = 0.2
        confidence = torch.exp(-edges_sq_norm / var)
        return depth.float(), confidence.float()


def get_t_s(d):
    t = d.median()
    s = (d - t).abs().median()
    return t, s


def align_samples(tri_idepth: torch.Tensor, mono_idepth: torch.Tensor):
    t_tri, s_tri = get_t_s(tri_idepth)
    t_mono, s_mono = get_t_s(mono_idepth)
    scale = s_tri / s_mono
    offset = t_tri - t_mono * scale
    return mono_idepth * scale + offset, scale, offset


def align_depth(mono_depth_map: torch.Tensor,
                desc_kpts: DescribedKeypoints,
                width: int, height: int,
                confidence: torch.Tensor = None):
    """Aligns the mono depth map with the triangulated depth from keypoints by finding the best scale and offset."""
    mono_idepth = sample(
        mono_depth_map,
        desc_kpts.kpts[desc_kpts.has_pt3d].view(1, 1, -1, 2),
        width,
        height,
    )[0, 0, 0]
    tri_idepth = 1 / desc_kpts.depth[desc_kpts.has_pt3d]

    # new add to filter out invalid depth
    mono_iconf = torch.ones_like(mono_idepth)
    if confidence is not None:
        mono_iconf = sample(
            confidence,
            desc_kpts.kpts[desc_kpts.has_pt3d].view(1, 1, -1, 2),
            width,
            height,
        )[0, 0, 0]
    mono_imask = mono_iconf >= 0.99


    tri_idepth = tri_idepth[mono_imask]
    mono_idepth = mono_idepth[mono_imask]

    mono_idepth_aligned, scale, offset = align_samples(tri_idepth, mono_idepth)
    err = (mono_idepth_aligned - tri_idepth).abs()
    valid = err < 5 * err.median()
    mono_idepth_aligned, scale, offset = align_samples(tri_idepth[valid], mono_idepth[valid])
    scale=1
    offset=0
    mono_depth_map_aligned = mono_depth_map * scale + offset

    return mono_depth_map_aligned

def align_depth_to_pointmap(
        mono_depth_map: torch.Tensor,
        pointmap: torch.Tensor,
        conf: torch.Tensor,
):
    """ Aligns a given monocular depth map to a subset of high confidence points from a pointmap. """
    high_conf_mask = conf > conf.mean()
    # pointmap as reference depth
    ref_idepth = 1.0 / pointmap[:, 2:3, ...][high_conf_mask].flatten()
    # mono inverse depth
    mono_idepth = 1.0 / mono_depth_map[high_conf_mask].flatten()
    mono_idepth_aligned, scale, offset = align_samples(ref_idepth, mono_idepth)
    # calculates abs diff as error
    err = (mono_idepth_aligned - ref_idepth).abs()
    # keep only samples with low error
    valid = err < 5 * err.median()
    # re-align with updated samples
    mono_idepth_aligned, scale, offset = align_samples(ref_idepth[valid], mono_idepth[valid])
    mono_depth_map_aligned = mono_depth_map * scale + offset
    return mono_depth_map_aligned

def save_depth_colormap(depth_map: torch.Tensor, path: str):
    # filepath: d:\Users\zhengzhewen\repos\on-the-fly-nvs\scene\mono_depth.py
    # Convert to CPU and numpy
    depth = depth_map.squeeze().detach().cpu().numpy()
    # Normalize to 0-255 (invert so closer is warmer)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_norm = 1.0 - depth_norm  # Invert: closer is warmer
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    # Apply colormap (e.g., JET)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    # Save as RGB
    cv2.imwrite(path, cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))

class MonoDepthEstimator:
    @torch.no_grad()
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        model = MonoDepthInternal()

        dummy = torch.zeros(3, height, width).cuda()
        self.model = torch.cuda.make_graphed_callables(model, [dummy])

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth, conf = self.model(image)
        return depth.clone(), conf.clone()
