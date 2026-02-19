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

import numpy as np
import torch
import math
import time
import cv2
import torch.nn.functional as F
from pathlib import Path
import open3d as o3d
from torch import nn
from plyfile import PlyData, PlyElement

import kornia
import scipy
from PIL import Image
from e3nn.o3 import matrix_to_angles, wigner_D

def update_gaussians(
    old_c2ws: torch.Tensor,       # Shape: (N, 4, 4)
    new_c2ws: torch.Tensor,       # Shape: (N, 4, 4)
    gaussians_positions: torch.Tensor,    # Shape: (N, 3)
    gaussians_quaternions: torch.Tensor,  # Shape: (N, 4), 假设为 wxyz
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    使用 Kornia 加速 Gaussian 位姿更新，全程保留在 GPU。
    """
    # 1. 计算增量变换
    # delta_T = new * old^-1
    # 对于位姿矩阵，inverse 也可以用 [R^T, -R^T @ t] 快速计算，
    # 但 torch.inverse 在 GPU 上对 4x4 矩阵已经很快了。
    delta_T = torch.bmm(new_c2ws, torch.inverse(old_c2ws))

    R = delta_T[:, :3, :3]  # (N, 3, 3)
    t = delta_T[:, :3, 3]   # (N, 3)

    # 2. 将原始四元数转换为旋转矩阵
    # kornia 默认支持 wxyz (scalar_first)
    rot_mats_orig = kornia.geometry.conversions.quaternion_to_rotation_matrix(gaussians_quaternions)

    # 3. 计算新的位置
    new_positions = torch.einsum('bij,bj->bi', R, gaussians_positions) + t

    # 4. 计算新的旋转并转回四元数
    # New_R = Delta_R * Old_R
    new_rot_mats = torch.bmm(R, rot_mats_orig)
    
    # 转回四元数 (wxyz)
    new_quaternions = kornia.geometry.conversions.rotation_matrix_to_quaternion(new_rot_mats)

    return new_positions, new_quaternions

def transform_gaussians(R, t, gaussians_positions, gaussians_quaternions):
    device = gaussians_positions.device
    
    # 1. 变换位置 (Vectorized)
    # p' = p @ R.T + t
    new_positions = torch.matmul(gaussians_positions, R.t()) + t

    # 将高斯四元数转为矩阵 (N, 3, 3)
    R_original = kornia.geometry.conversions.quaternion_to_rotation_matrix(gaussians_quaternions)
    
    # 批量矩阵乘法 (N, 3, 3) = (1, 3, 3) @ (N, 3, 3)
    R_new = torch.matmul(R.unsqueeze(0), R_original)
    
    # 转回四元数 (N, 4) wxyz
    new_quaternions = kornia.geometry.conversions.rotation_matrix_to_quaternion(R_new)
    
    return new_positions, new_quaternions

def parse_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def get_image_names(in_folder, image_extensions=[".jpg", ".png", ".jpeg"]):
    return [p.name for p in Path(in_folder).glob('*') if p.suffix.lower() in image_extensions]

def psnr(img1, img2):
    return 10 * torch.log10(1 / F.mse_loss(img1, img2)).item()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def get_lapla_norm(img, kernel, device="cuda:0"):
    laplacian_kernel = (
        torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=torch.float32
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    laplacian_kernel = laplacian_kernel.repeat(1, img.shape[0], 1, 1)
    laplacian = F.conv2d(img[None], laplacian_kernel, padding="same")
    laplacian_norm = torch.linalg.vector_norm(laplacian, ord=1, dim=1, keepdim=True)
    laplacian_norm[..., :, 0] = 0
    laplacian_norm[..., :, -1] = 0
    laplacian_norm[..., 0, :] = 0
    laplacian_norm[..., -1, :] = 0
    return F.conv2d(laplacian_norm, kernel, padding="same")[0, 0].clamp(0, 1)



def increment_runtime(runtime, start_time):
    # torch.cuda.synchronize()
    runtime[0] += time.time() - start_time
    runtime[1] += 1

# COLOR

C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
    left = ((2 * cx - W) / W - 1.0) * W / 2.0
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
    left = znear / fx * left
    right = znear / fx * right
    top = znear / fy * top
    bottom = znear / fy * bottom
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P

## Camera/triangulation/projection functions
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def depth2points(uv, depth, f, centre):
    xyz = torch.cat([(uv[..., :2] - centre) / f, torch.ones_like(uv[..., 0:1])], dim=-1)
    return depth * xyz

def pointmap2points(uv, pointmap):
    u_idx, v_idx = uv[:, 0].long(), uv[:, 1].long()
    new_pts = pointmap[v_idx, u_idx].reshape(-1, 3) # (H*W, 3)
    return new_pts

def reproject(uv, depth, f, centre, relR, relt):
    xyz = depth2points(uv, depth, f, centre)
    xyz = xyz @ relR.T + relt
    return pts2px(xyz, f, centre)


def make_torch_sampler(uv, width, height):
    """
    Converts OpenCV UV coordinates to a sampler for torch's grid_sample.
    To be used with align_corners=True
    """
    sampler = uv.clone()  # + 0.5
    sampler[..., 0] = sampler[..., 0] * (2.0 / (width - 1)) - 1.0
    sampler[..., 1] = sampler[..., 1] * (2.0 / (height - 1)) - 1.0
    return sampler


def sample(map, uv, width, height):
    sampler = make_torch_sampler(uv, width, height)
    return F.grid_sample(map, sampler, mode="bilinear", align_corners=True)


def pts2px(xyz, f, centre):
    return f * xyz[..., :2] / xyz[..., 2:3] + centre


def sixD2mtx(r):
    b1 = r[..., 0]
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)
    b2 = r[..., 1] - torch.sum(b1 * r[..., 1], dim=-1, keepdim=True) * b1
    b2 = b2 / torch.norm(b2, dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def mtx2sixD(R):
    return R[..., :2].clone()


## Visualization functions
def display_matches(mkpts1, mkpts2, img1, img2, scale=1, match_step=1, indices=None):
    image1 = img1.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
    image2 = img2.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
    if indices is not None:
        mkpts1 = mkpts1[indices]
        mkpts2 = mkpts2[indices]
    matched_mkptsi_np = mkpts1[::match_step].cpu().float().numpy()
    matched_mkptsj_np = mkpts2[::match_step].cpu().float().numpy()
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in matched_mkptsi_np]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in matched_mkptsj_np]
    mask_np = (
        ((mkpts1 != -1).all(dim=-1) * (mkpts2 != -1).all(dim=-1))[::match_step]
        .cpu()
        .numpy()
    )
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask_np)) if mask_np[i]]
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)
    if scale != 1:
        img_matches = cv2.resize(img_matches, (0, 0), fx=scale, fy=scale)
    cv2.imshow("matches_img", img_matches[..., ::-1])
    cv2.waitKey()


@torch.no_grad()
def draw_poses(image, view_matrix, view_fovx, scale, cam_width, cam_height, Rts, cam_f, color):
    """
    Overlay the camera frustums on the np image

    Args:
       image (np.ndarray): The image to draw on
       view_matrix (torch.Tensor): The point of view to render from
       view_fov (float): The field of view to render with
       scale (float): The scale of the drawn poses
       cam_width (int): The width of the image to draw the frustums
       cam_height (int): The height of the image to draw the frustums
       Rts (torch.Tensor): The camera poses to draw (camera to world)
       cam_f (float): The focal length of the poses to draw
    Returns:
       image (np.ndarray): The image with the frustums drawn on
    """
    if len(Rts) > 0:
        # Rendering options
        width, height = image.shape[1], image.shape[0]
        f = fov2focal(view_fovx, width)
        centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device='cuda')

        # Camera intrinsics to draw
        cam_centre = torch.tensor([(cam_width - 1) / 2, (cam_height - 1) / 2], device='cuda')
        # Make a 3D frustum using intrinsics
        origin = torch.tensor([0, 0, 0], device='cuda')
        corners2d = torch.tensor([[0, 0], [cam_width, 0], [cam_width, cam_height], [0, cam_height]], device='cuda')
        corners3d = depth2points(corners2d, scale, cam_f, cam_centre)
        # Duplicate and transform frustums for each pose
        cams_verts = torch.cat([origin.unsqueeze(0), corners3d], dim=0)
        n_cams = Rts.shape[0]
        cams_verts = torch.bmm((cams_verts - Rts[:n_cams, None, :3, 3]), Rts[:n_cams, :3, :3])
        cams_verts_view = (cams_verts @ view_matrix[:3, :3] + view_matrix[3:4, :3])
        cams_verts_2d = pts2px(cams_verts_view, f, centre).view(n_cams, -1, 2)
        # Out of view check
        valid_cams = (cams_verts_view[..., 2] > 0).all(dim=-1)
        cams_verts_2d = cams_verts_2d[valid_cams]

        # Draw frustums on the image
        draw_order = torch.tensor([1, 2, 0, 3, 4, 0, 1, 4, 3, 2], device="cuda")
        cams_verts_2d = cams_verts_2d[..., draw_order, :]
        image = cv2.polylines(
            image,
            cams_verts_2d.detach().cpu().numpy().astype(int),
            isClosed=False,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return image


@torch.no_grad()
def draw_anchors(image, view_matrix, view_fovx, scale, anchors, anchor_weights=[]):
    coords = [
        [ 1,  1,  1],
        [ 1,  1, -1],
        [ 1, -1,  1],
        [ 1, -1, -1],
        [-1,  1,  1],
        [-1,  1, -1],
        [-1, -1,  1],
        [-1, -1, -1],
    ]
    draw_order = [0,4,6,2,0,1,5,7,3,1,5,4,6,7,3,2,0]
    centred_cube_verts = scale * torch.tensor([coords[i] for i in draw_order], device='cuda')

    # Rendering options
    width, height = image.shape[1], image.shape[0]
    f = fov2focal(view_fovx, width)
    centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device='cuda')

    if len(anchors) != len(anchor_weights):
        anchor_weights = np.zeros(len(anchors))

    for anchor_weight, anchor in zip(anchor_weights, anchors):
        cube_verts = centred_cube_verts + anchor.position
        cube_vert_view = cube_verts @ view_matrix[:3, :3] + view_matrix[3:4, :3]
        if cube_vert_view[..., 2].min() > 0:
            cube_verts_2d = pts2px(cube_vert_view, f, centre)
            verts_2d = cube_verts_2d.cpu().numpy().astype(int)[None]
            cv2.polylines(image, verts_2d, isClosed=False, color=(anchor_weight * 255, 0, (1-anchor_weight)*255), thickness=2, lineType=cv2.LINE_AA)
    return image


def get_transform_mean_up_fwd(input, target, w_scale):
    """
    Get the transform that aligns input poses to target mean position, up and forward vectors.
    This appears more stable than Procrustes analysis.

    The input and target are both [N,4,4] transforms from world to camera.
    We want to:
      - match the mean position (camera center) of 'input' to that of 'target'
      - align the average "up" direction from 'input' to the average "up" direction of 'target'
      - align the average "forward" direction from 'input' to the average "forward" direction of 'target'

    """
    inv_input = torch.linalg.inv(input)
    inv_target = torch.linalg.inv(target)
    center_input = inv_input[:, :3, 3]
    center_target = inv_target[:, :3, 3]

    # Compute average up and forward vectors in world coords
    up_input_avg = inv_input[:, :3, 1].mean(dim=0)
    up_target_avg = inv_target[:, :3, 1].mean(dim=0)
    fwd_input_avg = inv_input[:, :3, 2].mean(dim=0)
    fwd_target_avg = inv_target[:, :3, 2].mean(dim=0)

    # Normalize these average directions to get unit vectors
    up_input_avg = up_input_avg / up_input_avg.norm()
    up_target_avg = up_target_avg / up_target_avg.norm()
    fwd_input_avg = fwd_input_avg / fwd_input_avg.norm()
    fwd_target_avg = fwd_target_avg / fwd_target_avg.norm()

    # Input basis
    right_input = torch.cross(up_input_avg, fwd_input_avg)
    right_input = right_input / right_input.norm()

    R_in = torch.stack([right_input, up_input_avg, fwd_input_avg], dim=1)

    # Target basis
    right_target = torch.cross(up_target_avg, fwd_target_avg)
    right_target = right_target / right_target.norm()

    R_tgt = torch.stack([right_target, up_target_avg, fwd_target_avg], dim=1)

    # This rotation aligns the input basis to target basis
    R = R_tgt @ R_in.transpose(0, 1)

    # This scale aligns the input center to target center
    center_input_mean = center_input.mean(dim=0)
    center_target_mean = center_target.mean(dim=0)
    if w_scale:
        s_input = ((center_input - center_input_mean)**2).sum(dim=-1).mean().sqrt()
        s_target = ((center_target - center_target_mean)**2).sum(dim=-1).mean().sqrt()
        s = s_target / s_input
    else:
        s = 1.0

    # This translation aligns the input center to target center
    t = center_target_mean - R @ center_input_mean * s

    return R, t, s


def align_mean_up_fwd(input, target, w_scale=False):
    """
    Align input poses to target mean position, up and forward vectors.

    Returns:
      A set of [N,4,4] transforms, which are the aligned poses of 'input'.
    """

    R, t, s = get_transform_mean_up_fwd(input, target, w_scale)
    inv_input = torch.linalg.inv(input)
    inv_input[:, :3, :3] = R @ inv_input[:, :3, :3]
    inv_input[:, :3, 3] = (R @ inv_input[:, :3, 3:4]).squeeze(-1) * s + t[None]

    return torch.linalg.inv(inv_input)

## Pose alignment and evaluation functions
def align_poses(input, target, w_scale=True):
    """Align input poses to target using Procrustes analysis on camera centers"""
    center_input = torch.linalg.inv(input)[:, :3, 3]
    center_target = torch.linalg.inv(target)[:, :3, 3]
    t0, t1, s0, s1, R = procrustes_analysis(center_target, center_input, w_scale)
    center_aligned = (center_input - t1) / s1 @ R.t() * s0 + t0
    R_aligned = input[:, :3, :3] @ R.t()
    t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
    aligned = torch.eye(4, device=input.device).repeat(input.shape[0], 1, 1)
    aligned[:, :3, :3] = R_aligned[:, :3, :3]
    aligned[:, :3, 3] = t_aligned
    return aligned


## From https://github.com/chenhsuanlin/bundle-adjusting-NeRF
# BARF: Bundle-Adjusting Neural Radiance Fields
# Copyright (c) 2021 Chen-Hsuan Lin
# Under the MIT License.
# Modified to interface with our pose format 
def rotation_distance(R1, R2, eps=1e-9):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1 @ R2.transpose(-2, -1)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = (
        ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()
    )  # numerical stability near -1/+1
    return angle


def procrustes_analysis(X0, X1, w_scale=True):  # [N,3]
    # translation
    t0 = X0.mean(dim=0, keepdim=True)
    t1 = X1.mean(dim=0, keepdim=True)
    X0c = X0 - t0
    X1c = X1 - t1
    # scale
    if w_scale:
        s0 = (X0c**2).sum(dim=-1).mean().sqrt()
        s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    else:
        s0, s1 = 1, 1
    X0cs = X0c / s0
    X1cs = X1c / s1
    # rotation (use double for SVD, float loses precision)
    U, S, V = (X0cs.t() @ X1cs).double().svd(some=True)
    R = (U @ V.t()).float()
    if R.det() < 0:
        R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    return t0[0], t1[0], s0, s1, R


def save_ply(attributes, dtype, path: str):
    elements = np.empty(attributes.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)

    
    
# jcj add 07.08
def save_debug_images(image, image_name):
    """
    save the image as image_name
    """
    if torch.is_tensor(image):
        image_np = image.cpu().numpy()
        # CHW --> HWC
        if image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
        # normalize to 0-255 
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
    else:
        image_np = image
        
    # rgb --> bgr
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


    cv2.imwrite(image_name, image_np)
    
    
# jcj add 07.08
def save_debug_matches(img1, img2, matches, image_name):
    if torch.is_tensor(img1):
        img1_np = img1.cpu().numpy()
        if img1_np.shape[0] == 3:
            img1_np = np.transpose(img1_np, (1, 2, 0))
        if img1_np.max() <= 1.0:
            img1_np = (img1_np * 255).astype(np.uint8)
        else:
            img1_np = img1_np.astype(np.uint8)
    else:
        img1_np = img1
        
    img1_np = cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)
        
    if torch.is_tensor(img2):
        img2_np = img2.cpu().numpy()
        if img2_np.shape[0] == 3:
            img2_np = np.transpose(img2_np, (1, 2, 0))
        if img2_np.max() <= 1.0:
            img2_np = (img2_np * 255).astype(np.uint8)
        else:
            img2_np = img2_np.astype(np.uint8)
    else:
        img2_np = img2
    
    img2_np = cv2.cvtColor(img2_np, cv2.COLOR_RGB2BGR)
    
    # stich
    h1, w1 = img1_np.shape[:2]
    h2, w2 = img2_np.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    combined_img = np.zeros((h, w, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1_np
    combined_img[:h2, w1:w1+w2] = img2_np
    
 
    kpts1 = matches.kpts.cpu().numpy()
    kpts2 = matches.kpts_other.cpu().numpy()
    
    for k in range(min(len(kpts1), 20000)):   
        pt1 = (int(kpts1[k, 0]), int(kpts1[k, 1]))
        pt2 = (int(kpts2[k, 0] + w1), int(kpts2[k, 1]))
        cv2.line(combined_img, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(combined_img, pt1, 3, (255, 0, 0), -1)
        cv2.circle(combined_img, pt2, 3, (0, 0, 255), -1)
 
    cv2.imwrite(image_name, combined_img)
    
    
    
# jcj add 07.09
def save_poses_as_pyramid_ply(Rt_list, out_ply, axis='z', size=0.4, color='red'):
    """
    save the poses as pyramid ply
    Rt_list: list of [4,4] np.array or np.ndarray
    out_ply: str, the path to save the ply file
    axis: str, the axis to save the pyramid
    size: float, the size of the pyramid
    color: str, the color of the pyramid
    """
    color_map = {
        'red': [1.0, 0.0, 0.0],
        'green': [0.0, 1.0, 0.0],
        'blue': [0.0, 0.0, 1.0],
        'yellow': [1.0, 1.0, 0.0],
        'purple': [1.0, 0.0, 1.0],
        'cyan': [0.0, 1.0, 1.0],
        'white': [1.0, 1.0, 1.0],
        'black': [0.0, 0.0, 0.0],
        'gray': [0.5, 0.5, 0.5],
        'brown': [0.5, 0.25, 0.0],
    }
    if isinstance(color, str):
        color = color_map.get(color.lower(), [1.0, 0.0, 0.0])

    Rt_arr = np.array(Rt_list)
    if Rt_arr.ndim == 2:
        Rt_arr = Rt_arr[None, ...]
    N = Rt_arr.shape[0]
    apex = np.array([[0, 0, 0]])
    if axis == 'z':
        base = np.array([
            [ size,  size, size],
            [-size,  size, size],
            [-size, -size, size],
            [ size, -size, size],
        ])
    elif axis == 'y':
        base = np.array([
            [ size, size,  size],
            [-size, size,  size],
            [-size, size, -size],
            [ size, size, -size],
        ])
    elif axis == 'x':
        base = np.array([
            [size,  size,  size],
            [size, -size,  size],
            [size, -size, -size],
            [size,  size, -size],
        ])
    else:
        raise ValueError('axis只能为x/y/z')
    pyramid_local = np.vstack([apex, base])  # (5,3)
    all_points = []
    triangles = []
    for i in range(N):
        Rt = Rt_arr[i]
        points_homo = np.hstack([pyramid_local, np.ones((5,1))])
        points_world = (Rt @ points_homo.T).T[:, :3]
        idx_offset = len(all_points)
        all_points.extend(points_world.tolist())

        for j in range(4):
            triangles.append([idx_offset, idx_offset+1+(j%4), idx_offset+1+((j+1)%4)])

        triangles.append([idx_offset+1, idx_offset+2, idx_offset+3])
        triangles.append([idx_offset+1, idx_offset+3, idx_offset+4])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(all_points))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (len(all_points), 1)))
    o3d.io.write_triangle_mesh(str(out_ply), mesh)


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

def make_empty(*shape):
    return nn.Parameter(torch.empty(*shape, device="cuda"), requires_grad=True)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def visualize_points_on_image(uv_coords, H, W, point_size=3, color=(255, 255, 255)):
    """
    在黑色图像上绘制uv坐标点
    Args:
        uv_coords: L*2 tensor，UV坐标 (u, v)
        H: 图像高度
        W: 图像宽度
        point_size: 点的大小
        color: 点的颜色 (B, G, R)
    Returns:
        img: 绘制后的图像
    """
    # 创建黑色图像
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # 转换为numpy
    if isinstance(uv_coords, torch.Tensor):
        uv_coords = uv_coords.cpu().numpy()

    # 遍历所有点
    for i in range(len(uv_coords)):
        u, v = uv_coords[i]

        # 检查是否为nan
        if np.isnan(u) or np.isnan(v):
            continue

        x, y = int(u), int(v)

        # 检查边界
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(img, (x, y), point_size, color, -1)

    return img

import torch
from torchvision.utils import save_image

def project_points_to_image(
    points: torch.Tensor,      # (N,3) world or camera coords
    K: torch.Tensor,           # (3,3) intrinsics
    H: int,
    W: int,
    Rt: torch.Tensor = None,   # (3,4) or (4,4) world→camera; identity if None
    img: torch.Tensor = None,
    save_path: str = "projection.png",
) -> torch.Tensor:
    """
    Projects 3D points into a single‐channel H×W image and saves it.

    Returns the image tensor of shape (1, H, W).
    """
    device, dtype = points.device, points.dtype

    # 1) bring into camera coords
    if Rt is not None:
        if Rt.shape == (4, 4):
            ones = torch.ones(points.shape[0], 1, device=device, dtype=dtype)
            pts_h = torch.cat([points, ones], dim=1)          # N×4
            pts_cam = (Rt @ pts_h.t()).t()[:, :3]            # N×3
        elif Rt.shape == (3, 4):
            R, t = Rt[:, :3], Rt[:, 3]
            pts_cam = (R @ points.t()).t() + t               # N×3
        else:
            raise ValueError("Rt must be (3,4) or (4,4)")
    else:
        pts_cam = points

    # 2) keep only points in front of camera
    mask = pts_cam[:, 2] > 0
    pts = pts_cam[mask]

    # 3) project
    proj = (K @ pts.t()).t()                              # M×3
    uv   = (proj[:, :2] / proj[:, 2:3]).round().long()    # M×2

    # 4) rasterize into 1×H×W
    if img is not None:
        img = torch.zeros(1, H, W, device=device, dtype=dtype)
    valid = (
        (uv[:, 0] >= 0) & (uv[:, 0] < W) &
        (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )
    uv = uv[valid]
    img[0, uv[:, 1], uv[:, 0]] = 1.0

    return img

def radial_decay_kernel(H, W, sigma=0.5):
    # Create meshgrid of normalized coordinates in [-1, 1]
    y = torch.linspace(-1, 1, steps=H)
    x = torch.linspace(-1, 1, steps=W)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    # Compute distance from center for each pixel
    r2 = xx**2 + yy**2
    # Gaussian weighting: max at center, decays to edge
    weights = torch.exp(-r2 / (2 * sigma**2))
    return weights