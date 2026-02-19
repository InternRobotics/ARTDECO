import sys
import numpy as np
import open3d as o3d

def tum_pose_to_points(tum_txt):
    points = []
    with open(tum_txt, 'r') as f:
        for line in f:
            if line.strip() == '' or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            # TUM格式: timestamp tx ty tz qx qy qz qw
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            points.append([x, y, z])
    return np.array(points)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('用法: python tum_pose_to_ply.py poses.txt output.ply')
        sys.exit(1)
    tum_txt = sys.argv[1]
    ply_file = sys.argv[2]
    points = tum_pose_to_points(tum_txt)
    if len(points) == 0:
        print('未找到任何点')
        sys.exit(1)
    # 创建open3d点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 设置所有点为红色
    colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(ply_file, pcd)
    print(f'已保存 {len(points)} 个点到 {ply_file}')        