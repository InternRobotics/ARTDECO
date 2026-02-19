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

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from Reconstruct.poses.matcher import Matches

class DescribedKeypoints():
    """
    A class to store 2D keypoints, their descriptors, their matches to other images, and their estimated 3D positions.
    """
    @torch.no_grad()
    def __init__(self, kpts, feats):
        self.kpts = kpts   # N 2
        self.feats = feats # N C
        self.valid = feats.abs().sum(dim=-1) > 0
        self.nvalid = self.valid.sum()
        self.has_pt3d = torch.zeros(kpts.shape[0], dtype=torch.bool, device=kpts.device) # N
        self.pts_conf = torch.zeros(kpts.shape[0], dtype=torch.float, device=kpts.device) # N
        self.pts3d = torch.zeros(kpts.shape[0], 3, dtype=torch.float, device=kpts.device) # N 3 world coordinate
        self.depth = torch.zeros(kpts.shape[0], dtype=torch.float, device=kpts.device) # N
        # key: another frame id for match
        # value: match results
        self.matches = {}


    @torch.no_grad()
    def update_matches(self, other_id, matches, swap=False):
        if swap:
            matches = Matches(matches.kpts_other, matches.kpts, matches.idx_other, matches.idx)
        self.matches[other_id] = matches
 
    @torch.no_grad()
    def update_3D_pts(self, pts3D, depth, conf, idx):
        self.has_pt3d[idx] = True
        self.depth[idx] = depth
        self.pts3d[idx] = pts3D
        self.pts_conf[idx] = conf

    @torch.no_grad()
    def to(self, device):
        self.kpts = self.kpts.to(device)
        self.feats = self.feats.to(device)
        self.valid = self.valid.to(device)
        self.has_pt3d = self.has_pt3d.to(device)
        self.pts_conf = self.pts_conf.to(device)
        self.pts3d = self.pts3d.to(device)
        self.depth = self.depth.to(device)
        for key in self.matches:
            self.matches[key].kpts = self.matches[key].kpts.to(device)
            self.matches[key].kpts_other = self.matches[key].kpts_other.to(device)
            self.matches[key].idx = self.matches[key].idx.to(device)
            self.matches[key].idx_other = self.matches[key].idx_other.to(device)

# From https://github.com/verlab/accelerated_features
# XFeat: Accelerated Features for Lightweight Image Matching, Under Apache-2.0 license
class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """ 
    def __init__(self, mode = 'bicubic', align_corners = False): 
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False)
        return x.permute(0,2,3,1).squeeze(-2)
    
class Detector():
    @torch.no_grad()
    def __init__(self, top_k, width, height):
        cache_path = f"models/cache/xfeat_{width}_{height}_{top_k}.pt"
        dummy_img = torch.randn(1, 3, height, width).cuda().to(torch.half)
        if os.path.exists(cache_path):
            extractor = torch.jit.load(cache_path)
        else:
            print(f"Compiling feature extractor")
            extractor = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=top_k, source='local')
            extractor = extractor.cuda().eval().to(torch.half)

            ## Overriding the functions to run at fixed size
            
            # Adapted from https://github.com/verlab/accelerated_features to enable jit tracing
            # XFeat: Accelerated Features for Lightweight Image Matching, Under Apache-2.0 license
            def preprocess_tensor(x):
                H, W = x.shape[-2:]
                _H, _W = (H//32) * 32, (W//32) * 32
                rh, rw = H/_H, W/_W

                x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
                return x, rh, rw
            
            # Adapted from https://github.com/verlab/accelerated_features to enable jit tracing
            # XFeat: Accelerated Features for Lightweight Image Matching, Under Apache-2.0 license
            def NMS(x, threshold = 0.05, kernel_size = 5, nvalid = int(1.5 * top_k)):
                B, _, H, W = x.shape
                pad=kernel_size//2
                local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
                xTot = x * (x == local_max) * (x > threshold)
                xOut, pos1d = torch.topk(xTot.view(-1), nvalid)
                pos2d = torch.zeros((B, nvalid, 2), dtype=torch.long, device=x.device)
                pos2d[..., 0] = pos1d % W
                pos2d[..., 1] = pos1d // W
                return xOut, pos2d
            
            # Adapted from https://github.com/verlab/accelerated_features to enable jit tracing
            # XFeat: Accelerated Features for Lightweight Image Matching, Under Apache-2.0 license
            def detectAndCompute(x):
                top_k = extractor.top_k
                detection_threshold = extractor.detection_threshold
                x, rh1, rw1 = extractor.preprocess_tensor(x)

                B, _, _H1, _W1 = x.shape
                
                M1, K1, H1 = extractor.net(x)
                M1 = F.normalize(M1, dim=1)

                #Convert logits to heatmap and extract kpts
                K1h = extractor.get_kpts_heatmap(K1)
                xOut, mkpts = extractor.NMS(K1h, threshold=detection_threshold, kernel_size=5)

                #Compute reliability scores
                _nearest = InterpolateSparse2d('nearest')
                _bilinear = InterpolateSparse2d('bilinear')
                scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
                scores[torch.all(mkpts == 0, dim=-1)] = -1

                #Select top-k features
                # idxs = torch.argsort(-scores)
                idxs = torch.topk(scores, top_k)[1]
                mkpts_x  = torch.gather(mkpts[...,0], -1, idxs)
                mkpts_y  = torch.gather(mkpts[...,1], -1, idxs)
                xOut = torch.gather(xOut[None], -1, idxs)
                mkpts = torch.cat([mkpts_x[...,None], mkpts_y[...,None]], dim=-1)
                scores = torch.gather(scores, -1, idxs)
                scores *= xOut > 0

                #Interpolate descriptors at kpts positions
                feats = extractor.interpolator(M1, mkpts, H = _H1, W = _W1)

                #L2-Normalize
                feats = F.normalize(feats, dim=-1)

                #Correct kpt scale
                mkpts = mkpts.float() * torch.tensor([rw1,rh1], device=mkpts.device, dtype=torch.float).view(1, 1, -1)

                return mkpts[0], feats[0] * (scores[0][..., None] > 0)
            
            extractor.preprocess_tensor = preprocess_tensor
            extractor.NMS = NMS
            extractor.forward = detectAndCompute

            extractor = torch.jit.trace(extractor, [dummy_img])
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.jit.save(extractor, cache_path)
            extractor = torch.jit.load(cache_path)

        self.extractor = extractor

    @torch.no_grad()
    def __call__(self, image):
        return DescribedKeypoints(*(self.extractor(image[None].half())))
    