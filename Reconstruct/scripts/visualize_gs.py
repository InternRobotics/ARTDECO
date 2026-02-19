import numpy as np
import open3d as o3d
from pathlib import Path
from torch.nn import functional as F
import torch
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import cv2

w, h = 1296, 972  # width, height
cx, cy = 648.0, 486.0  # principal point
f = 1517.6684216796323 * w / 2592

bdir = Path('D:/Users/zhengzhewen/datasets/l1/3f/bedroom/bedroom-20250611-2_full/usb0/slam')
pts = np.load(bdir / 'pts.npy', allow_pickle=True).reshape(-1, 3)[None]

N = len(pts)
left, right = 0, 1  # initial window
def get_window_points():
    return np.concatenate(pts[left:right], axis=0)

def update_pcd(pcd):
    points = get_window_points().astype(float)
    valid = np.linalg.norm(points, axis=1) < 100
    points = points[valid]
    print(points.max(), points.min())
    pcd.points = o3d.utility.Vector3dVector(points)
    print(f"Showing frames [{left}, {right}) with {points.shape[0]} points.")

pcd = o3d.geometry.PointCloud()
update_pcd(pcd)
o3d.io.write_point_cloud(bdir / 'pts.ply', pcd)

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