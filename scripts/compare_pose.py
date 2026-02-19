import pypose as pp
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from pathlib import Path
slam_frame_path = Path('/home/pjlab/Documents/SLAM/Artdeco-V1/results/test_t123123/slam/frame_poses')
map_frame_path =  Path('/home/pjlab/Documents/SLAM/Artdeco-V1/results/test_t123123/onthefly.txt')
slam_frame_dict = {}
map_frame_dict = {}
with open(slam_frame_path, 'r') as f1, open(map_frame_path, 'r') as f2:
    slam_lines = f1.readlines()
    map_lines = f2.readlines()
    for line in slam_lines:

        timestamp, x, y, z, qx, qy, qz, qw = map(float, line.split())
        if qw < 0:
            qw = -qw
            qx, qy, qz = -qx, -qy, -qz
        slam_frame_dict[timestamp] = (
            timestamp,
            (x, y, z),
            (qx, qy, qz, qw)
        )
    for line in map_lines:
        timestamp, x, y, z, qx, qy, qz, qw = map(float, line.split())
        if qw < 0:
            qw = -qw
            qx, qy, qz = -qx, -qy, -qz
        map_frame_dict[timestamp] = (
            timestamp,
            (x, y, z),
            (qx, qy, qz, qw)
        )
translations = []
angles = []
times = []
for k, v in map_frame_dict.items():
    assert k in slam_frame_dict, f"Key {k} not found in slam lines"
    timestamp, (x, y, z), (qx, qy, qz, qw) = map_frame_dict[k]
    Twc_map = pp.SE3(torch.tensor([x, y, z, qx, qy, qz, qw]))
    timestamp, (x, y, z), (qx, qy, qz, qw) = slam_frame_dict[k]
    Twc_slam = pp.SE3(torch.tensor([x, y, z, qx, qy, qz, qw]))
    T_rel = Twc_map.mul(Twc_slam.Inv()).matrix().numpy()
    # 提取相对平移
    rel_translation = T_rel[:3, 3]
    translation_magnitude = np.linalg.norm(rel_translation)
    translations.append(translation_magnitude)
    # 提取相对旋转角度
    rel_rotation_matrix = T_rel[:3, :3]
    rotation = R.from_matrix(rel_rotation_matrix)
    # 转换为角度 (度)
    angle_radians = rotation.magnitude()
    angle_degrees = np.degrees(angle_radians)
    angles.append(angle_degrees)
    times.append(timestamp)

max_index = np.argmax(translations)
print(times[max_index])
print(translations[max_index])
print(angles[max_index])
print(np.max(translations), np.max(angles))
print(translations, angles)