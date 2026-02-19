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

# sys.path.append("Reconstruct/submodules/MoGe")
from moge.model.v2 import MoGeModel

class MonoGeometryInternal(torch.nn.Module):
    def __init__(self):
        super(MonoGeometryInternal, self).__init__()
        model_path = "models/cache/moge-2-vitl-normal"
        if not os.path.exists(model_path):
            print("Downloading MoGe-2 model, may take a few minutes...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
            # Save model locally
            # self.model.save_pretrained(model_path)
        else:
            self.model = MoGeModel.from_pretrained(model_path)
        
        self.model = self.model.to("cuda").eval()

    def forward(self, image: torch.Tensor):
        # Input image should be (3, H, W) with RGB values normalized to [0, 1]
        with torch.no_grad():
            output = self.model.infer(image)
            
            # Extract depth, points, normal maps and mask
            depth_map = output["depth"]  # (H, W)
            point_map = output["points"]  # (H, W, 3)
            normal_map = output["normal"]  # (H, W, 3)
            mask = output["mask"]  # (H, W)
            
            # Rearrange point and normal maps to match expected format (B, C, H, W)
            point_map = point_map.permute(2, 0, 1)[None]  # (1, 3, H, W)
            normal_map = normal_map.permute(2, 0, 1)[None]  # (1, 3, H, W)
            depth_map = depth_map[None, None]  # (1, 1, H, W)
            mask = mask[None, None]  # (1, 1, H, W)
            
            return depth_map, point_map, normal_map, mask

def save_normal_map(normal_map: torch.Tensor, path: str):
    # Convert to CPU and numpy
    normal = normal_map.squeeze().detach().cpu().numpy()
    # Scale from [-1, 1] to [0, 1]
    normal = (normal + 1) * 0.5
    # Convert to uint8
    normal_uint8 = (normal * 255).astype(np.uint8)
    # Save as RGB
    cv2.imwrite(path, cv2.cvtColor(normal_uint8, cv2.COLOR_RGB2BGR))

class MonoGeometryEstimator:
    @torch.no_grad()
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        model = MonoGeometryInternal()
        self.model = model
    
    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        depth, points, normal, mask = self.model(image)
        return 1.0 / depth.clone().float(), mask.clone().float()
