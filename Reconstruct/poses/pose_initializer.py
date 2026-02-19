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
import math

from Reconstruct.poses.feature_detector import DescribedKeypoints
from Reconstruct.poses.mini_ba import MiniBA
from Reconstruct.utils import fov2focal, depth2points, sixD2mtx, mtx2sixD, save_ply, save_poses_as_pyramid_ply
from Reconstruct.scene.keyframe import Keyframe
from Reconstruct.poses.ransac import RANSACEstimator, EstimatorType
import sys
import os

import open3d as o3d
import numpy as np


# using n view images with Rts to triangulation
# jcj add 07.07
def triangulate_nview(uv, f, centre, Rs, ts):
    # uv: (n, 2), Rs: (n, 3, 3), ts: (n, 3)
    n = uv.shape[0]
    A = []
    for i in range(n):
        # 构造投影矩阵
        K = torch.eye(3, device=uv.device)
        K[0, 0] = K[1, 1] = f
        K[0, 2] = centre[0]
        K[1, 2] = centre[1]
        P = K @ torch.cat([Rs[i], ts[i][:, None]], dim=1)  # (3, 4)
        x, y = uv[i]
        # 修正：P[0] - x * P[2], P[1] - y * P[2]
        A.append(P[0] - x * P[2])
        A.append(P[1] - y * P[2])
        
        
        
        
    A = torch.stack(A, dim=0)  # (2n, 4)
    # SVD solve
    _, _, V = torch.linalg.svd(A)
    X = V[-1]
    

    X = X / X[3]
    return X[:3]



class PoseInitializer():
    """Fast pose initializer using MiniBA and the previous frames."""
    def __init__(self, width, height, triangulator, matcher, max_pnp_error, args):
        self.width = width
        self.height = height
        self.triangulator = triangulator
        self.max_pnp_error = max_pnp_error
        self.matcher = matcher

        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device='cuda')
        self.num_pts_miniba_bootstrap = args.num_pts_miniba_bootstrap
        self.num_kpts = args.num_kpts

        self.num_pts_pnpransac = 2 * args.num_pts_miniba_incr
        self.num_pts_miniba_incr = args.num_pts_miniba_incr
        self.min_num_inliers = args.min_num_inliers
        self.enable_scaling = args.enable_scaling

        # Initialize the focal length
        if args.init_focal > 0:
            self.f_init = args.init_focal
        elif args.init_fov > 0:
            self.f_init = fov2focal(args.init_fov * math.pi / 180, width)
        else:
            self.f_init = 0.7 * width

        # Initialize MiniBA models
        self.miniba_bootstrap = MiniBA(
            1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  not args.fix_focal, True,
            make_cuda_graph=True, iters=args.iters_miniba_bootstrap)
        self.miniba_rebooting = MiniBA(
            1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  False, True,
            make_cuda_graph=True, iters=args.iters_miniba_bootstrap)
        self.miniBA_incr = MiniBA(
            1, 1, 0, args.num_pts_miniba_incr, optimize_focal=False, optimize_3Dpts=False,
            make_cuda_graph=True, iters=args.iters_miniba_incr)
        
        self.PnPRANSAC = RANSACEstimator(args.pnpransac_samples, self.max_pnp_error, EstimatorType.P4P)

    def build_problem(self,
                      desc_kpts_list: list[DescribedKeypoints],
                      npts: int,
                      n_cams: int,
                      n_primary_cam: int,
                      min_n_matches: int,
                      kfId_list: list[int],
    ):
        """Build the problem for mini ba by organizing the matches between the keypoints of the cameras."""
        npts_per_primary_cam = npts // n_primary_cam
        uvs = torch.zeros(npts, n_cams, 2, device='cuda') - 1
        xyz_indices = torch.zeros(npts, n_cams, dtype=torch.int64, device='cuda') - 1
        unused_kpts_mask = torch.ones((n_cams, desc_kpts_list[0].kpts.shape[0]), device='cuda', dtype=torch.bool)
        for k in range(n_primary_cam):
            idx_occurrences = torch.zeros(self.num_kpts, device="cuda", dtype=torch.int)
            for match in desc_kpts_list[k].matches.values():
                idx_occurrences[match.idx] += 1
            idx_occurrences *= unused_kpts_mask[k]
            if idx_occurrences.sum() == 0:
                print("No matches.")
                continue
            idx_occurrences = idx_occurrences > 0
            selected_indices = torch.multinomial(idx_occurrences.float(), npts_per_primary_cam, replacement=False)

            selected_mask = torch.zeros(self.num_kpts, device='cuda', dtype=torch.bool)
            selected_mask[selected_indices] = True
            aligned_ids = torch.arange(npts_per_primary_cam, device="cuda")
            all_aligned_ids = torch.zeros(self.num_kpts, device="cuda", dtype=aligned_ids.dtype)
            all_aligned_ids[selected_indices] = aligned_ids

            uvs_k = uvs[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam, :, :]
            xyz_indices_k = xyz_indices[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam]
            for l in range(n_cams):
                if l == k:
                    uvs_k[:, l, :] = desc_kpts_list[l].kpts[selected_indices]
                    xyz_indices_k[:, l] = selected_indices
                else:
                    lId = kfId_list[l]
                    if lId in desc_kpts_list[k].matches:
                        idxk = desc_kpts_list[k].matches[lId].idx
                        idxl = desc_kpts_list[k].matches[lId].idx_other

                        mask = selected_mask[idxk] 
                        idxk = idxk[mask]
                        idxl = idxl[mask]

                        set_idx = all_aligned_ids[idxk]
                        unused_kpts_mask[l, idxl] = False
                        uvs_k[set_idx, l, :] = desc_kpts_list[l].kpts[idxl]
                        xyz_indices_k[set_idx, l] = idxl

                        selected_indices_l = idxl.clone()
                        selected_mask_l = torch.zeros(self.num_kpts, device='cuda', dtype=torch.bool)
                        selected_mask_l[selected_indices_l] = True
                        all_aligned_ids_l = torch.zeros(self.num_kpts, device="cuda", dtype=aligned_ids.dtype)
                        all_aligned_ids_l[selected_indices_l] = set_idx.clone()

                        for m in range(l + 1, n_cams):
                            mId = kfId_list[m]
                            if mId in desc_kpts_list[l].matches:
                                idxl = desc_kpts_list[l].matches[mId].idx
                                idxm = desc_kpts_list[l].matches[mId].idx_other

                                mask = selected_mask_l[idxl] 
                                idxl = idxl[mask]
                                idxm = idxm[mask]

                                set_idx = all_aligned_ids_l[idxl]
                                set_mask = uvs_k[set_idx, m, 0] == -1
                                uvs_k[set_idx[set_mask], m, :] = desc_kpts_list[m].kpts[idxm[set_mask]]

        n_valid = (uvs >= 0).all(dim=-1).sum(dim=-1)
        mask = n_valid < min_n_matches
        uvs[mask, :, :] = -1
        xyz_indices[mask, :] = -1
        return uvs, xyz_indices

    @torch.no_grad()
    def initialize_bootstrap(self, desc_kpts_list: list[DescribedKeypoints], rebooting=False, init_Rts=None, debug_folder=None):
        """
        Estimate focal and initialize the poses of the frames corresponding to desc_kpts_list. 
        """
        n_cams = len(desc_kpts_list)
        npts = self.num_pts_miniba_bootstrap

        ## Exhaustive matching
        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                _ = self.matcher(desc_kpts_list[i], desc_kpts_list[j], remove_outliers=True, update_kpts_flag="inliers", kID=i, kID_other=j)
        
        ## Build the problem by organizing matches
        uvs, xyz_indices = self.build_problem(desc_kpts_list, npts, n_cams, n_cams, 2, list(range(n_cams)))

        ## Initialize for miniBA (poses at identity, 3D points with rand depth)
        f_init = (torch.tensor([self.f_init], device="cuda"))
        Rs6D_init = torch.eye(3, 2, device="cuda")[None].repeat(n_cams, 1, 1)
        ts_init = torch.zeros(n_cams, 3, device="cuda")

        

        xyz_init = torch.zeros(npts, 3, device="cuda")
        for k in range(n_cams):
            mask = (uvs[:, k, :] >= 0).all(dim=-1)
            xyz_init[mask] += depth2points(uvs[mask, k, :], 1, f_init, self.centre)
        xyz_init /= xyz_init[..., -1:].clamp_min(1)
        xyz_init[..., -1] = 1
        xyz_init *= 1 + torch.randn_like(xyz_init[:, :1]).abs()
        
        
        
        # if init_Rts given, initialize for miniBA (poses as given)
        if init_Rts is not None:

            init_Rts = torch.stack(init_Rts, dim=0).cuda()
            Rs = init_Rts[:, :3, :3]
            Rs6D_init = mtx2sixD(Rs)
            ts_init = init_Rts[:, :3, 3].clone()
            
            # debug
            # save 8 images init poses as pyramid ply
            if debug_folder is not None:
                init_Rts_inv = []
                
                for Rt in init_Rts:
                    Rt_inv = torch.linalg.inv(Rt).detach().cpu().numpy()
                    init_Rts_inv.append(Rt_inv)
                
                points_folder = os.path.join(debug_folder, "init_points")
                os.makedirs(points_folder, exist_ok=True)
                save_poses_as_pyramid_ply(init_Rts_inv, os.path.join(points_folder, "bootstrap_init_Rts.ply"), size = 0.3, color='blue')
 
            
            # import pdb; pdb.set_trace()
            # jcj add 07.07
            # using init_Rts poses to triangulation
            # provide better initial points for miniba_bootstrap
            xyz_init = torch.zeros(npts, 3, device="cuda")    # (2000, 3)
            
            # traverse all points
            for i in range(npts):
                # 找到该点在哪些相机被观测到
                obs_mask = (uvs[i, :, 0] >= 0) & (uvs[i, :, 1] >= 0)
                cam_ids = torch.where(obs_mask)[0]
                if len(cam_ids) < 2:
                    continue  # 至少需要两个观测才能三角化
                # 收集观测到该点的像素和相机位姿
                uv_obs = uvs[i, cam_ids, :]  # (n_obs, 2)
                Rs = []
                ts = []
                for k in cam_ids:
                    if init_Rts is not None:
                        Rt = init_Rts[k]
                    else:
                        Rt = torch.eye(4, device="cuda")
                        print("init_Rts is None")  
 
                    Rs.append(Rt[:3, :3])
                    ts.append(Rt[:3, 3])
                Rs = torch.stack(Rs, dim=0)  # (n_obs, 3, 3)
                ts = torch.stack(ts, dim=0)  # (n_obs, 3)
                # 三角化
                # 这里假设你有一个triangulate_nview函数，输入为像素、内参、Rs、ts，输出为三维点
                pt3d = triangulate_nview(uv_obs, f_init, self.centre, Rs, ts)  # (3,)
                # print("f_init = ", f_init)
                xyz_init[i] = pt3d            
                
            # debug
            if debug_folder is not None:
                points_folder = os.path.join(debug_folder, "init_points")
                os.makedirs(points_folder, exist_ok=True)
                save_ply(os.path.join(points_folder,"bootstrap_init_points.ply"), xyz_init)

        ## Run miniBA, estimating 3D points, camera focal and poses
        if rebooting:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_rebooting(Rs6D_init, ts_init, self.f, xyz_init, self.centre, uvs.view(-1))
        else:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_bootstrap(Rs6D_init, ts_init, f_init, xyz_init, self.centre, uvs.view(-1))
        final_residual = (r * mask).abs().sum()/mask.sum()
        
        # debug
        if debug_folder is not None:
            
            tmp_Rts = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
            tmp_Rts[:, :3, :3] = sixD2mtx(Rs6D)
            tmp_Rts[:, :3, 3] = ts
            
            
            points_folder = os.path.join(debug_folder, "init_points")
            os.makedirs(points_folder, exist_ok=True)        
            save_ply(os.path.join(points_folder,"bootstrap_opt_points.ply"), xyz)
            
            opt_Rts_inv = []
            
            for Rt_opt in tmp_Rts:
                Rt_opt_inv = torch.linalg.inv(Rt_opt).detach().cpu().numpy()
                opt_Rts_inv.append(Rt_opt_inv)
 
            save_poses_as_pyramid_ply(opt_Rts_inv, os.path.join(points_folder, "bootstrap_opt_Rts.ply"), size = 0.3, color='yellow')            

        self.f = f
        self.intrinsics = torch.cat([f, self.centre], dim=0) 

        ## Scale to 0.1 average translation
        print("enable_scaling = ", self.enable_scaling)
        if self.enable_scaling:
            rel_ts = ts[:-1] - ts[1:]
            scale = 0.1 / rel_ts.norm(dim=-1).mean()
            ts *= scale
            xyz = scale * xyz.clone()
        Rts = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        Rts[:, :3, :3] = sixD2mtx(Rs6D)
        Rts[:, :3, 3] = ts

        return Rts, f, final_residual

    @torch.no_grad()
    def initialize_incremental(self, keyframes: list[Keyframe], curr_desc_kpts: DescribedKeypoints,
                               index: int, is_test: bool, curr_img, init_Rt=None):
        """
        Initialize the pose of the frame given by curr_desc_kpts and index using the previously registered keyframes.
        """
        
        # Match the current frame with previous keyframes
        xyz = []
        uvs = []
        confs = []
        match_indices = []
        for keyframe in keyframes:
            matches = self.matcher(curr_desc_kpts, keyframe.desc_kpts, remove_outliers=True, update_kpts_flag="all", kID=index, kID_other=keyframe.index)
            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            xyz.append(keyframe.desc_kpts.pts3d[matches.idx_other[mask]]) # 在世界坐标系，keyframe在match后的所有有三维坐标的点
            uvs.append(matches.kpts[mask]) # 当前帧在match后所有具有3d点的keypoint
            confs.append(keyframe.desc_kpts.pts_conf[matches.idx_other[mask]])
            match_indices.append(matches.idx[mask])

        xyz = torch.cat(xyz, dim=0) # 3d
        uvs = torch.cat(uvs, dim=0) # 2d
        confs = torch.cat(confs, dim=0)
        match_indices = torch.cat(match_indices, dim=0) # 2d点在当前帧的index

        # Subsample the points if there are too many
        if len(xyz) > self.num_pts_pnpransac:
            selected_indices = torch.multinomial(confs, self.num_pts_miniba_incr, replacement=False)
            xyz = xyz[selected_indices]
            uvs = uvs[selected_indices]
            confs = confs[selected_indices]
            match_indices = match_indices[selected_indices]

        # Estimate an initial camera pose and inliers using PnP RANSAC
        Rs6D_init = keyframes[0].rW2C
        ts_init = keyframes[0].tW2C
        # If pose is given, initialize with it
        if init_Rt is not None:
            R = init_Rt[:3, :3]
            Rs6D_init = mtx2sixD(R)
            ts_init = init_Rt[:3, 3]
            
        # using PnPRANSAC to get GT poses
        Rt, inliers = self.PnPRANSAC(uvs, xyz, self.f, self.centre, Rs6D_init, ts_init, confs)

        xyz = xyz[inliers]
        uvs = uvs[inliers]
        confs = confs[inliers]
        match_indices = match_indices[inliers]

        # Subsample the points if there are too many
        if len(xyz) >= self.num_pts_miniba_incr:
            selected_indices = torch.topk(torch.rand_like(xyz[..., 0]), self.num_pts_miniba_incr, dim=0, largest=False)[1]
            xyz_ba = xyz[selected_indices]
            uvs_ba = uvs[selected_indices]
        # todo: 检测是否太少
        elif len(xyz) < self.num_pts_miniba_incr:
            xyz_ba = torch.cat([xyz, torch.zeros(self.num_pts_miniba_incr - len(xyz), 3, device="cuda")], dim=0)
            uvs_ba = torch.cat([uvs, -torch.ones(self.num_pts_miniba_incr - len(uvs), 2, device="cuda")], dim=0)

        # Run the incr ba
        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1))
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = sixD2mtx(Rs6D)[0]
        Rt[:3, 3] = ts[0]

        # Check if we have sufficiently many inliers
        if is_test or mask.sum() > self.min_num_inliers:
            # Return the pose of the current frame
            return Rt
        else:
            print("Too few inliers for pose initialization")
            
            
            # TODO 
            # if ba fails
            # using init Pose
            # or start reboot
            
            
            # Remove matches as we prevent the current frame from being registered
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
            return None