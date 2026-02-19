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

import math
import cupy
import torch

class GuidedMVS():
    @torch.no_grad()
    def __init__(self, args, num_depth_candidates=16):
        self.n_cams = args.num_prev_keyframes_miniba_incr
        self.num_depth_candidates = num_depth_candidates
        self.idepth_range = 2e-1

        # Read the CUDA source code and set the include directory to poses/
        with open('Reconstruct/poses/guided_mvs.cu', 'r') as f:
            cuda_source = f.read()
        cuda_source = cuda_source.replace("NUM_CAMS", str(self.n_cams))
        cuda_source = cuda_source.replace("NUM_DEPTH_CANDIDATES", str(num_depth_candidates))
        self.module = cupy.RawModule(
            code=cuda_source, 
            # options=('--std=c++14', '-Iposes'),
            options=("--std=c++14", "-I./Reconstruct/poses"),

        )
        self.uvToDepth = self.module.get_function("uvToDepth")

    @torch.no_grad()
    def __call__(self, uv, refKeyframe, keyframes: list):
        uv = uv.contiguous()
        # Get relative poses
        other2ref = [keyframe.get_Rt() @ torch.linalg.inv(refKeyframe.get_Rt()) for keyframe in keyframes] # Tkr
        other2ref = torch.stack(other2ref, dim=0)[..., :3, :4].contiguous()
        # Get feature maps for all neighbour keyframes
        refFeatMap = refKeyframe.feat_map.contiguous()
        featMaps = torch.stack([keyframe.feat_map.cuda().contiguous() for keyframe in keyframes], dim=0)
        intrinsics = torch.cat([refKeyframe.f, refKeyframe.centre], dim=0).contiguous()
        mono_idepth = refKeyframe.mono_idepth.contiguous()

        depth = -torch.ones_like(uv[..., 0]).contiguous()
        idist = -torch.ones_like(uv[..., 0]).contiguous()

        block_size = self.num_depth_candidates
        grid_size = math.ceil(uv.shape[0])
        self.uvToDepth(
            block=(block_size,),
            grid=(grid_size,),
            args=(
                uv.data_ptr(),
                refFeatMap.data_ptr(),
                featMaps.data_ptr(),
                other2ref.data_ptr(), 
                intrinsics.data_ptr(),
                mono_idepth.data_ptr(),
                depth.data_ptr(),
                idist.data_ptr(),
                cupy.float32(self.idepth_range),
                uv.shape[0],
                refFeatMap.shape[0],
                refFeatMap.shape[1],
                mono_idepth.shape[-2],
                mono_idepth.shape[-1],
                refKeyframe.image_pyr[0].shape[1],
                refKeyframe.image_pyr[0].shape[2],
            )
        )

        return depth, idist >= 0
