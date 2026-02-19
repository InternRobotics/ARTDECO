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
import cupy
import math
from enum import Enum
from Reconstruct.poses.mini_ba import MiniBA
from Reconstruct.utils import pts2px, sixD2mtx


class EstimatorType(Enum):
    FUNDAMENTAL_8PTS = 0
    P4P = 1


class RANSACEstimator:
    @torch.no_grad()
    def __init__(self, N: int, max_error: float, type: EstimatorType):
        """
        Initialize the RANSAC estimator.

        Args:
            N (int): Number of models to estimate.
            max_error (float): Maximum reprojection error for inliers.
            type (EstimatorType): Type of estimator to use.
        """
        self.N = N
        self.max_error = max_error
        self.type = type

        # Read the CUDA source code and set the include directory to poses/
        with open("./Reconstruct/poses/ransac.cu", "r") as f:
            cuda_source = f.read()
        self.module = cupy.RawModule(
            code=cuda_source,
            # options=("--std=c++14", "-Iposes"),
            options=("--std=c++14", "-I./Reconstruct/poses"),
            
        )

        # Set the functions and number of points required for each estimator
        if type == EstimatorType.FUNDAMENTAL_8PTS:
            self.model_estimator = self.module.get_function("batchFundMat8pts")
            self.inlier_mask_estimator = self.module.get_function("sampsonInliers")
            self.m = 8  # 8 pairs are required to estimate a fundamental matrix
            self.models = torch.zeros([N, 3, 3], device=torch.device("cuda"))
            self.valid_model_mask = torch.zeros(
                N, dtype=torch.bool, device=torch.device("cuda")
            )
        elif type == EstimatorType.P4P:
            self.m = 4  # 4 pairs are required per estimation
            self.model_estimator = MiniBA(
                self.N,
                1,
                0,
                self.m,
                optimize_focal=False,
                optimize_3Dpts=False,
                make_cuda_graph=True,
                outlier_mad_scale=0,
                iters=20,
            )
            self.models = torch.zeros([N, 3, 4], device=torch.device("cuda"))
        else:
            raise ValueError(f"Unknown EstimatorType {type}")

    def estimate(
        self,
        mkpts1: torch.Tensor,
        mkpts2: torch.Tensor,
        idxs: torch.Tensor,
        focal: torch.Tensor,
        centre: torch.Tensor,
        R6D_init: torch.Tensor,
        t_init: torch.Tensor,
    ):
        """
        Estimate N models from the given matches.
        """
        if self.type == EstimatorType.FUNDAMENTAL_8PTS:
            block_size = 64
            grid_size = math.ceil(idxs.shape[0] / block_size)
            self.model_estimator(
                block=(block_size,),
                grid=(grid_size,),
                args=(
                    mkpts1.data_ptr(),
                    mkpts2.data_ptr(),
                    idxs.data_ptr(),
                    self.models.data_ptr(),
                    self.valid_model_mask.data_ptr(),
                    idxs.shape[0],
                ),
            )
        elif self.type == EstimatorType.P4P:
            mkpts1_candidates = mkpts1[idxs].view(self.N, -1)
            mkpts2_candidates = mkpts2[idxs]
            Rs6D_init = R6D_init[None, None].repeat(self.N, 1, 1, 1)
            ts_init = t_init[None, None].repeat(self.N, 1, 1)
            Rs6D, ts, _, _, _, _, _ = self.model_estimator(
                Rs6D_init, ts_init, focal, mkpts2_candidates, centre, mkpts1_candidates
            )
            self.models[..., :3] = sixD2mtx(Rs6D).squeeze()
            self.models[..., 3] = ts.squeeze()

    def get_inlier_mask(self, mkpts1, mkpts2, focal, centre):
        if self.type == EstimatorType.FUNDAMENTAL_8PTS:
            inliers = torch.zeros(
                self.models.shape[0],
                mkpts1.shape[0],
                dtype=torch.bool,
                device=mkpts1.device,
            )
            block_size = 128
            grid_size = math.ceil((self.models.shape[0] * mkpts1.shape[0]) / block_size)
            self.inlier_mask_estimator(
                block=(block_size,),
                grid=(grid_size,),
                args=(
                    mkpts1.data_ptr(),
                    mkpts2.data_ptr(),
                    self.models.data_ptr(),
                    self.valid_model_mask.data_ptr(),
                    inliers.data_ptr(),
                    cupy.float32(self.max_error**2),
                    self.models.shape[0],
                    mkpts1.shape[0],
                ),
            )
        elif self.type == EstimatorType.P4P:
            mkpts2_cam = (
                torch.matmul(mkpts2, self.models[..., :3].transpose(-2, -1))
                + self.models[..., None, :, 3]
            )
            mkpts2_px = pts2px(mkpts2_cam, focal, centre)
            inliers = (
                torch.linalg.vector_norm(mkpts1[None] - mkpts2_px, dim=-1)
                < self.max_error
            )
        return inliers

    @torch.no_grad()
    def __call__(
        self,
        mkpts1,
        mkpts2,
        focal=None,
        centre=None,
        R6D_init=None,
        t_init=None,
        confs=None,
    ):
        """
        Run the RANSAC estimator to find the best model and inliers.
        args:
            mkpts1: (n, 2) 2D positions of the matched keypoints
            mkpts2: (n, 2) for FUNDAMENTAL_8PTS or (n, 3) for PnP
            focal: (1) for PnP, else None
            centre: (2) for PnP, else None
            R6D_init: (3, 2) initial 6D rotation for PnP, else None
            t_init: (3) initial translation for PnP, else None
            confs: (3) confidence to scale the weights of the inliers

        returns:
            best_model: (3, 4) or (3, 3) depending on the estimator type
            mask: (n) boolean mask of the inliers
        """

        if self.type == EstimatorType.P4P:
            assert focal is not None
            assert centre is not None

        # Select N x m random points
        random_scores = torch.rand(self.N, mkpts1.shape[0], device=mkpts1.device)
        _, idxs = torch.topk(random_scores, self.m, dim=1)

        # Run the batch estimator
        self.estimate(mkpts1, mkpts2, idxs, focal, centre, R6D_init, t_init)

        # Get the inlier mask
        inliers = self.get_inlier_mask(mkpts1, mkpts2, focal, centre)

        # Compute inlier mask and find the best model
        if confs is not None:
            inliers = inliers * confs[None]
        n_inliers = inliers.sum(dim=1)
        best_id = torch.argmax(n_inliers)
        best_model = self.models[best_id]
        mask = inliers[best_id] > 0

        return best_model, mask
