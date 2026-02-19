import json
import pathlib

import einops
import pypose as pp
import numpy as np
import torch

from evo.core.metrics import PoseRelation, APE, RPE
from evo.core.trajectory import PoseTrajectory3D


from VSLAM.SharedKeyframes import SharedKeyframes
from VSLAM.mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement

from VSLAM.utils_mast3r import inverse_normalize


def as_SE3(X):

    if X.data.shape[-1] == 7:
        return X
    t, q, s = einops.rearrange(X.data.detach().cpu(), "... c -> (...) c").split(
        [3, 4, 1], -1
    )
    T_WC = pp.SE3(torch.cat([t, q], dim=-1))
    pp.quat2unit(T_WC)
    return T_WC

def evaluate_trajectory(save_path, name, Twc_est, Twc_gt):
    """
    评估轨迹，使用SE(3) Umeyama对齐

    Args:
        Twc_est: 估计位姿 numpy数组 (N, 8) [t, tx,ty,tz,qx,qy,qz,qw] - Twc
        Twc_gt: 位姿 numpy数组 (N, 8) [t, tx,ty,tz,qx,qy,qz,qw] - Twc

    Returns:
        dict: 包含APE、RPE统计结果和有效位姿数量
    """
    from evo.core.metrics import PoseRelation, APE, RPE
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core import sync
    import copy
    import numpy as np

    # 过滤nan值
    valid_est_mask = ~np.any(np.isnan(Twc_est), axis=1)
    valid_gt_mask = ~np.any(np.isnan(Twc_gt), axis=1)

    Twc_est_clean = Twc_est[valid_est_mask]
    Twc_gt_clean = Twc_gt[valid_gt_mask]

    if Twc_gt_clean.shape[0] <= 0:
        return
    # 创建轨迹对象
    traj_est = PoseTrajectory3D(
        positions_xyz=Twc_est_clean[:, 1:4],
        orientations_quat_wxyz=np.roll(Twc_est_clean[:, 4:8], 1, axis=1),  # 转为[qw,qx,qy,qz]
        timestamps=Twc_est_clean[:, 0]
    )

    traj_ref = PoseTrajectory3D(
        positions_xyz=Twc_gt_clean[:, 1:4],
        orientations_quat_wxyz=np.roll(Twc_gt_clean[:, 4:8], 1, axis=1),  # 转为[qw,qx,qy,qz]
        timestamps=Twc_gt_clean[:, 0]
    )

    # 自动时间戳同步
    traj_ref_sync, traj_est_sync = sync.associate_trajectories(traj_ref, traj_est)

    # SE(3) Umeyama对齐
    traj_est_aligned = copy.deepcopy(traj_est_sync)
    traj_est_aligned.align(traj_ref_sync, correct_scale=True)

    # 计算APE
    ape_metric = APE(PoseRelation.translation_part)
    ape_metric.process_data((traj_ref_sync, traj_est_aligned))
    ape_stats = ape_metric.get_all_statistics()

    # 计算RPE
    rpe_metric = RPE(PoseRelation.translation_part, delta=1.0)
    rpe_metric.process_data((traj_ref_sync, traj_est_aligned))
    rpe_stats = rpe_metric.get_all_statistics()

    results =  {
        'ape': {
            'rmse': ape_stats['rmse'],
            'mean': ape_stats['mean'],
            'std': ape_stats['std']
        },
        'rpe': {
            'rmse': rpe_stats['rmse'],
            'mean': rpe_stats['mean'],
            'std': rpe_stats['std']
        },
        'num_poses': len(traj_ref_sync.timestamps)
    }

    with open(save_path.joinpath(name), 'w') as f:
        json.dump(results, f, indent=4)

    return results


def save_traj(logdir, keyframes: SharedKeyframes, frmaes_info):
    T_WCs_all = []
    timestamps_all = []

    T_WCs_keyframes = []
    timestamps_keyframes = []
    # 第一个循环
    for i in range(len(frmaes_info)):
        time = frmaes_info[i][1]
        keyframe_index = frmaes_info[i][2]
        ref_Tckcf = frmaes_info[i][3]
        keyframe = keyframes[keyframe_index]

        T_WCs_all.append(keyframe.T_WC.cpu().mul(ref_Tckcf.cpu()))
        timestamps_all.append(time)

    # 第二个循环
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        time = keyframe.frame_time
        T_WC = keyframe.T_WC.cpu()

        T_WCs_all.append(T_WC)
        T_WCs_keyframes.append(T_WC)
        timestamps_all.append(time)
        timestamps_keyframes.append(time)

    # 对所有帧按时间戳排序
    sorted_indices = np.argsort(timestamps_all)
    sorted_T_WCs = [T_WCs_all[i] for i in sorted_indices]
    sorted_timestamps = [timestamps_all[i] for i in sorted_indices]

    # 对关键帧按照时间戳排序
    sorted_indices_keyframe = np.argsort(timestamps_keyframes)
    sorted_T_WCs_keyframe = [T_WCs_keyframes[i] for i in sorted_indices_keyframe]
    sorted_timestamps_keyframe = [timestamps_keyframes[i] for i in sorted_indices_keyframe]


    # 保存日志
    tum_all = save_tum(logdir, "frames.txt", sorted_T_WCs, sorted_timestamps)
    tum_keyframe = save_tum(logdir, "keyframe.txt", sorted_T_WCs_keyframe, sorted_timestamps_keyframe)
    return tum_all, tum_keyframe


def save_tum(logdir, name, sorted_T_WCs, sorted_timestamps):
    # 保存日志
    tum_array = []
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)

    with open(logdir.joinpath(name), "w") as f:
        for i in range(len(sorted_T_WCs)):
            t = sorted_timestamps[i]
            T_WC = as_SE3(sorted_T_WCs[i])
            x, y, z, qx, qy, qz, qw = T_WC.cpu().data.numpy().reshape(-1)
            tum_array.append([t, x, y, z, qx, qy, qz, qw])
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")

    tum_array = np.array(tum_array)
    return tum_array

def save_keyframe(savedir, keyframes, c_conf_threshold, device="cuda:0", use_calib = True):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    confidences = []

    pointcloudsPerKeyframe = []
    colorsPerKeyframe = []
    confidencePerKeyframe = []

    pose = np.zeros((len(keyframes), 9))
    for i in range(len(keyframes)):
        keyframe = keyframes[i].to(device)
        if use_calib:
            img_shape = torch.tensor(keyframe.img.shape[1:], device=device)
            X_canon = constrain_points_to_ray(img_shape, keyframe.X_canon[None], keyframe.K)
            keyframe.X_canon = X_canon.squeeze(0).to(torch.float32)
        pW = keyframe.T_WC.Act(keyframe.X_canon.to(device)).cpu().numpy().reshape(-1, 3)
        color_img = inverse_normalize(keyframe.img.cpu())
        color = (color_img.numpy() * 255).astype(np.uint8).reshape(-1, 3)
        confidence = keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)

        # original
        pointcloudsPerKeyframe.append(pW)
        colorsPerKeyframe.append(color)
        confidencePerKeyframe.append(confidence)

        # valid
        valid = confidence> c_conf_threshold
        pointclouds.append(pW[valid])
        colors.append(color[valid])
        confidences.append(confidence[valid])

        # pose
        pose[i, 0] = keyframe.frame_time
        pose[i, 1:] = keyframe.T_WC.cpu().data[0]

    pointcloudsPerKeyframe = np.stack(pointcloudsPerKeyframe, axis=0)
    colorsPerKeyframe = np.stack(colorsPerKeyframe, axis=0)
    confidencePerKeyframe = np.stack(confidencePerKeyframe, axis=0)
    np.save(savedir / "keyframe_point_world_per.npy", pointcloudsPerKeyframe)
    np.save(savedir / "keyframe_color_per.npy", colorsPerKeyframe)
    np.save(savedir / "keyframe_confidence_per.npy", confidencePerKeyframe)
    np.save(savedir / "keyframe_pose_per.npy", pose)

    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)
    save_ply(savedir / "keyframe_points_all.ply", pointclouds, colors)


def save_frame_wise_points(savedir, frames, c_conf_threshold, use_calib = True):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    confidences = []

    pointcloudsPerframe = []
    colorsPerframe = []
    confidencePerframe = []

    pose = np.zeros((len(frames), 9))
    for i in range(len(frames)):
        frame = frames[i]
        if use_calib:
            X_canon = constrain_points_to_ray(
                frame.img_shape.flatten()[:2], frame.X_canon[None], frame.K
            )
            frame.X_canon = X_canon.squeeze(0)


        pW = frame.T_WC.Act(frame.X_canon.float()).cpu().numpy().reshape(-1, 3)
        color_img = inverse_normalize(frame.img.cpu())
        color = (color_img.numpy() * 255).astype(np.uint8).reshape(-1, 3)
        confidence = frame.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)

        # original
        pointcloudsPerframe.append(pW)
        colorsPerframe.append(color)
        confidencePerframe.append(confidence)

        # valid
        valid = confidence > c_conf_threshold
        pointclouds.append(pW[valid])
        colors.append(color[valid])
        confidences.append(confidence[valid])

        # pose
        pose[i, 0] = frame.frame_time
        pose[i, 1:] = frame.T_WC.cpu().data[0]

    pointcloudsPerframe = np.stack(pointcloudsPerframe, axis=0)
    colorsPerframe = np.stack(colorsPerframe, axis=0)
    confidencePerframe = np.stack(confidencePerframe, axis=0)
    np.save(savedir / "frame_point_world_per.npy", pointcloudsPerframe)
    np.save(savedir / "frame_color_per.npy", colorsPerframe)
    np.save(savedir / "frame_confidence_per.npy", confidencePerframe)
    np.save(savedir / "frame_pose_per.npy", pose)


def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)
