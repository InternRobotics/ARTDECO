import torch
import numpy as np
import open3d as o3d
from pathlib import Path

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

# Usage:
# Assuming you have an ImageDataset instance called 'dataset'
base_dir = Path('D:/Users/zhengzhewen/datasets/l1/3f/bedroom/bedroom-20250611-2_full/usb0/slam')
pointmaps_path = base_dir / 'frame_point_per.npy'  # Replace with your actual path
tum_pose_path = base_dir / '0_frames.txt'  # Replace with your actual path
pointmaps = np.load(pointmaps_path) # .reshape(-1, 512, 384, 3)

def invert_pose(qvec, tvec):
    q_conjugate = np.array([qvec[0], -qvec[1], -qvec[2], -qvec[3]])
    R = qvec2rotmat([qvec[0], qvec[1], qvec[2], qvec[3]])
    tvec_cw = -R.T @ tvec
    return q_conjugate, tvec_cw

Rts = []
with open(tum_pose_path, 'r') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.strip().split()
        ts, x, y, z, qx, qy, qz, qw = map(float, parts[:9])
        qvec, tvec = np.array([qw, qx, qy, qz]), np.array([x, y, z])
        # qvec, tvec = invert_pose([qw, qx, qy, qz], [x, y, z])
        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = qvec2rotmat(qvec)
        Rt[:3, 3] = tvec
        name = f"{ts:.7f}.jpg"
        Rts.append(Rt)

Rts = np.stack(Rts, axis=0).astype(np.float32)  # [N, 4, 4]
all_points = []
num_frames = min(len(pointmaps), len(Rts))
# for i in range(num_frames):
pts = pointmaps
N, M, _ = pts.shape
ones = np.ones((N, M, 1), dtype=pts.dtype)
pts_h = np.concatenate([pts, ones], axis=-1)  # [N, M, 4]
np.save('frame_point_per_c.npy', np.einsum('nij,nmj->nmi', np.linalg.inv(Rts), pts_h)[:, :, :3])

pts_world = pts_h[...,:3]  # [N, M, 3]
pts_world = pts_world.reshape(-1, 3)

# o3d
# all_points = pts_world.reshape(-1, 3)  # Flatten to [N*M, 3]
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pts_world.reshape(-1, 3))
# o3d.visualization.draw_geometries([pcd])
# ---
left, right = 0, 1  # initial window
def get_window_points():
    return pts_world[left*M:right*M]

def update_pcd(pcd):
    points = get_window_points()
    pcd.points = o3d.utility.Vector3dVector(points)
    print(f"Showing frames [{left}, {right}) with {points.shape[0]} points.")

pcd = o3d.geometry.PointCloud()
update_pcd(pcd)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.add_geometry(pcd)

def refresh():
    update_pcd(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

def on_key_up(vis):
    global right
    if right < N:
        right += 1
        refresh()
    return False

def on_key_down(vis):
    global right
    if right > left + 1:
        right -= 1
        refresh()
    return False

def on_key_left(vis):
    global left, right
    if left > 0:
        left -= 1
        right -= 1
        refresh()
    return False

def on_key_right(vis):
    global left, right
    if right < N:
        left += 1
        right += 1
        refresh()
    return False

vis.register_key_callback(264, on_key_down)
vis.register_key_callback(265, on_key_up)
vis.register_key_callback(262, on_key_right)  # right arrow
vis.register_key_callback(263, on_key_left)   # left arrow

vis.run()
vis.destroy_window()