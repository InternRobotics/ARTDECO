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

from Reconstruct.poses.feature_detector import DescribedKeypoints
from Reconstruct.utils import depth2points, pts2px

def matches_to_points(uv, uv_matched, R, t, f, centre):
    """
        uv和uv_matched代表两个坐标系中的匹配像素点，根据这个匹配像素点三角化出uv对应的相机坐标系的点
        其中：
        xyz N,3 为uv对应的3D点
        disambiguation N,3 为深度为1和深度为10时将uv投影到另外一个坐标系的像素偏移
        error N,3 为将xyz投影到另外一个坐标系与gt像素的差
    """
    p2 = t[None]  # [1, 3]
    d1 = depth2points(uv, 1, f, centre)
    d2 = depth2points(uv_matched, 1, f, centre)
    d2 = torch.matmul(d2, R.T)  # Transform d2 by the rotation matrix

    # Normalize directions
    d1 = d1 / torch.linalg.vector_norm(d1, dim=-1, keepdim=True)
    d2 = d2 / torch.linalg.vector_norm(d2, dim=-1, keepdim=True)

    # Compute the normal vector and its secondary vector
    n = torch.cross(d1, d2, dim=-1)  # [N, 3]
    n2 = torch.cross(d2, n, dim=-1)  # [N, 3]

    # Compute distances
    dist = torch.matmul(n2, p2.T) / torch.bmm(n2.unsqueeze(1), d1.unsqueeze(-1)).squeeze(-1)

    # # Compute pose direction and angles
    angles = torch.acos(torch.sum(d1 * d2, dim=1))  # Angle between d1 and d2

    # Compute 3D points
    xyz = d1 * dist

    # Compute the views' disambiguation capability
    xyz1 = torch.matmul(d1 - p2, R)
    uv1 = pts2px(xyz1, f, centre)
    xyz2 = torch.matmul(d1 * 10 - p2, R)
    uv2 = pts2px(xyz2, f, centre)
    disambiguation = torch.linalg.vector_norm(uv1 - uv2, dim=-1)

    # Transform xyz to matched coordinates
    xyz_matched = torch.matmul(xyz - p2, R)
    expected_uv_matched = pts2px(xyz_matched, f, centre)

    # Compute reprojection error
    error = torch.linalg.vector_norm(expected_uv_matched - uv_matched, dim=-1)

    # Mark invalid points
    invalid = (xyz.isnan() | xyz.isinf()).any(dim=-1) | angles.isnan()
    angles = torch.where(invalid, 0, angles)

    # Return 3D points, disambiguation, and reprojection error
    return xyz, disambiguation, error

class TriangulatorInternal(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, uv, uvs_others, Rt, Rts_others, f, centre, max_error, min_dis):
        """
            当前帧uv依次与其他帧的对应uv进行三角化，如果当前帧某一个像素点与多个其他帧像素点有三角化，
            则取视差最大的三角化点作为最终点,返回的是世界坐标系的点
        """
        n_pts = uv.shape[0]
        kpts3d = torch.zeros(n_pts, 3, device="cuda")
        best_disambiguation = torch.zeros(n_pts, device="cuda")  # 记录当前帧某个像素三角化时的视差
        Rts_others_inv = torch.linalg.inv_ex(Rts_others)[0]

        for cam_idx in range(uvs_others.shape[0]):
            uv_other = uvs_others[cam_idx]
            Rt_other_inv = Rts_others_inv[cam_idx]
            rel_Rt = Rt @ Rt_other_inv # Rcp
            kpts3dTmp, disTmp, error = matches_to_points(uv, uv_other, rel_Rt[:3, :3], rel_Rt[:3, 3], f, centre)
            validMask = (kpts3dTmp[:, 2] > 1e-6) * (disTmp > best_disambiguation) * (error < max_error)
            validMask *= uv_other.min(dim=-1).values > 0

            kpts3d = torch.where(validMask.unsqueeze(-1), kpts3dTmp, kpts3d)
            best_disambiguation = torch.where(validMask, disTmp, best_disambiguation)

        depth = kpts3d[:, 2].clone()
        kpts3d = (kpts3d - Rt[None, :3, 3]) @ Rt[:3, :3]
        return kpts3d, depth, best_disambiguation, best_disambiguation > min_dis
    
class Triangulator():
    @torch.no_grad()
    def __init__(self, n_pts, n_cams, max_error):
        self.n_cams = n_cams
        self.model = TriangulatorInternal().eval().cuda()
        uv = torch.rand(n_pts, 2, device="cuda")
        uvs_others = torch.rand(n_cams, n_pts, 2, device="cuda")
        Rt = torch.eye(4, device="cuda")
        Rts_others = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        f = torch.rand(1, device="cuda")
        centre = torch.rand(2, device="cuda")
        self.max_error = torch.tensor(max_error, device="cuda")
        self.min_dis = torch.tensor(max_error * 30, device="cuda")
        
        # self.model = torch.cuda.make_graphed_callables(
        #     self.model, (uv, uvs_others, Rt, Rts_others, f, centre, self.max_error, self.min_dis))

    def __call__(self, uv, uvs_others, Rt, Rts_others, f, centre):
        return self.model(uv, uvs_others, Rt, Rts_others, f, centre, self.max_error, self.min_dis)


    def prepare_matches(self, desc_kpts: DescribedKeypoints):
        """
        Des:
            已知：当前帧与之前的一些关键帧有点的匹配关系
            获取与当前帧的topk个匹配数量最多的帧
            如果与当前帧匹配的帧数量不满足self.n_cams，则有几个算几个
        return:
            uv L,2 当前帧关键点坐标
            uvs_others K,L,2 与当前帧的关键点匹配的其他帧的坐标
            chosen_kfs_ids K 与当前帧匹配的其他帧的index
        """
        uv = desc_kpts.kpts
        uvs_others = -torch.ones(self.n_cams, uv.shape[0], 2, device="cuda")
        # 找出所有与当前帧匹配的关键帧
        n_matches = torch.tensor([matches.idx.shape[0] for matches in desc_kpts.matches.values()])
        kf_indices = torch.tensor(list(desc_kpts.matches.keys()))
        # 找出topk个关键帧
        chosen_ids = torch.topk(n_matches, min(self.n_cams, n_matches.shape[0])).indices
        chosen_kfs_ids = kf_indices[chosen_ids].tolist()
        # 记录下topk个关键帧的匹配坐标
        for i, index in enumerate(chosen_kfs_ids):
            matches = desc_kpts.matches[index]
            uvs_others[i, matches.idx, :] = matches.kpts_other
        return uv, uvs_others, chosen_kfs_ids
