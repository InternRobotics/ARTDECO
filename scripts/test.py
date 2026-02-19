import lietorch
import numpy as np
import pypose as pp
import torch
from scipy.spatial.transform import Rotation
import torch.nn.functional as F
pose = torch.load("../pose.pt").cpu()
# pose[:, 3:7] = F.normalize(pose[:, 3:7])

points = torch.load("../points.pt").cpu()


def pose7d_to_matrix(pose_7d, s):
    translation = pose_7d[:3]
    quaternion = pose_7d[3:]  # [qx, qy, qz, qw]

    # 转换为旋转矩阵
    R = Rotation.from_quat(quaternion).as_matrix()

    # 构建4x4变换矩阵
    T = np.eye(4)
    T[:3, :3] = R * s
    T[:3, 3] = translation
    return T


# T_test = pp.from_matrix([[ 9.9518e-01, -6.2058e-04, -4.7471e-03, -5.9260e-02],
#          [ 6.0061e-04,  9.9518e-01, -4.1873e-03, -1.0458e-02],
#          [ 4.7497e-03,  4.1844e-03,  9.9517e-01,  1.2579e-01],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], ltype=pp.Sim3_type)


T_lie = lietorch.Sim3(pose)
T_pp = pp.quat2unit(pp.Sim3(pose))
# T_true = pose7d_to_matrix(pose[0, :7], pose[0, 7].item())

point_lie = T_lie.act(points)
point_pp = T_pp.Act(points)
res = point_pp.cpu() - point_lie.cpu()
print(res.max(), res.min())


mat_lie = T_lie.matrix()
mat_pp = T_pp.matrix()
point_lie = (mat_lie[:, :3, :3] @ points[..., None] + mat_lie[:, :3, 3:4]).squeeze(-1)
point_pp = (mat_pp[:, :3, :3] @ points[..., None] + mat_pp[:, :3, 3:4]).squeeze(-1)
res = point_pp.cpu() - point_lie.cpu()
print(res.max(), res.min())



pose = torch.tensor([[0,0,0,0,0,0,1.,1]])
T_lie = lietorch.Sim3(pose)
T_pp = pp.Sim3(pose)
mat_lie = T_lie.matrix()
mat_pp = T_pp.matrix()
pass