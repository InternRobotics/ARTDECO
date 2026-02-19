import torch
import pypose as pp

# 反对称矩阵
def skew_sym(x):
    b = x.shape[:-1]
    x, y, z = x.unbind(dim=-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).view(*b, 3, 3)

# 计算点的模长
def point_to_dist(X):
    d = torch.linalg.norm(X, dim=-1, keepdim=True)
    return d

# 返回归一化之后的X，最后一位附加了其模长d，如有必要也返回归一化的X和d对输入点云X的jacobian
def point_to_ray_dist(X, jacobian=False):
    b = X.shape[:-1] # number of points

    d = point_to_dist(X) # 计算每个点代表向量的模长
    d_inv = 1.0 / d
    r = d_inv * X #
    rd = torch.cat((r, d), dim=-1)  # Dim 4
    if not jacobian:
        return rd
    else:
        d_inv_2 = d_inv**2
        I = torch.eye(3, device=X.device, dtype=X.dtype).repeat(*b, 1, 1)
        dr_dX = d_inv.unsqueeze(-1) * (
            I - d_inv_2.unsqueeze(-1) * (X.unsqueeze(-1) @ X.unsqueeze(-2))
        )
        dd_dX = r.unsqueeze(-2)
        drd_dX = torch.cat((dr_dX, dd_dX), dim=-2)
        return rd, drd_dX

# 根据内参重新计算相机坐标系点云
# 2 B*(H*W)*3  3*3
def constrain_points_to_ray(img_size, Xs, K):
    uv = get_pixel_coords(Xs.shape[0], img_size, device=Xs.device, dtype=Xs.dtype).view(
        *Xs.shape[:-1], 2
    )
    Xs = backproject(uv, Xs[..., 2:3], K)
    return Xs


# 相似变换Pc到Pw，如有必要返回Pw对位相似变换的Jacobian 3 * 7
def act_Sim3(X: pp.Sim3, pC: torch.Tensor, jacobian=False):
    pW = X.Act(pC)
    if not jacobian:
        return pW
    dpC_dt = torch.eye(3, device=pW.device).repeat(*pW.shape[:-1], 1, 1)
    dpC_dR = -skew_sym(pW)
    dpc_ds = pW.reshape(*pW.shape[:-1], -1, 1)
    return pW, torch.cat([dpC_dt, dpC_dR, dpc_ds], dim=-1)  # view(-1, mdim)

# 求得内参
def decompose_K(K):
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    return fx, fy, cx, cy

# 在知道内参矩阵的情况下，将P映射到图像上去
# 返回P对应的像素坐标和对数深度pz， pz对P的jacobian， 在边界内且深度合理的像素
def project_calib(P, K, img_size, jacobian=False, border=0, z_eps=0.0, dXf_Ck_d_f=None):
    # 投影到像素平面
    b = P.shape[:-1]
    # print(K.shape)
    # print(K.view(1, 1, 3, 3))
    K_rep = K.repeat(*b, 1, 1)

    p = (K_rep @ P[..., None]).squeeze(-1)
    p = p / p[..., 2:3]
    p = p[..., :2]

    u, v = p.split([1, 1], dim=-1)
    x, y, z = P.split([1, 1, 1], dim=-1)

    # 获得在边界内且深度合理的像素
    # Check if pixel falls in image
    valid_u = (u > border) & (u < img_size[1] - 1 - border)
    valid_v = (v > border) & (v < img_size[0] - 1 - border)
    # Check if in front of camera
    valid_z = z > z_eps
    # Get total valid
    valid = valid_u & valid_v & valid_z

    # Depth transformation
    logz = torch.log(z)
    invalid_z = torch.logical_not(valid_z)
    logz[invalid_z] = 0.0  # Need to avoid nans

    # Output
    pz = torch.cat((p, logz), dim=-1)

    if not jacobian:
        return pz, valid
    else:
        fx, fy, cx, cy = decompose_K(K)
        z_inv = 1.0 / z[..., 0]
        dpz_dP = torch.zeros(*b + (3, 4), device=P.device, dtype=P.dtype)
        dpz_dP[..., 0, 0] = fx
        dpz_dP[..., 1, 1] = fy
        dpz_dP[..., 0, 2] = -fx * x[..., 0] * z_inv
        dpz_dP[..., 1, 2] = -fy * y[..., 0] * z_inv
        dpz_dP *= z_inv[..., None, None]
        dpz_dP[..., 2, 2] = z_inv  # Only z itself in bottom row

        dpz_dP[..., 0, 3] = x[..., 0] * z_inv + K[0,0] * (dXf_Ck_d_f[:, 0, 0] * z[..., 0] - dXf_Ck_d_f[:, 2, 0] * x[..., 0]) / (z_inv ** 2)
        dpz_dP[..., 1, 3] = y[..., 0] * z_inv + K[1,1] * (dXf_Ck_d_f[:, 1, 0] * z[..., 0] - dXf_Ck_d_f[:, 2, 0] * y[..., 0]) / (z_inv ** 2)
        dpz_dP[..., 2, 3] = z_inv * dXf_Ck_d_f[:, 2, 0]
        return pz, dpz_dP, valid

# 将所有像素坐标p根据其深度z和内参矩阵K重新投影到相机坐标系
def backproject(p, z, K):
    tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
    tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
    dP_dz = torch.empty(p.shape[:-1] + (3, 1), device=z.device, dtype=K.dtype)
    dP_dz[..., 0, 0] = tmp1
    dP_dz[..., 1, 0] = tmp2
    dP_dz[..., 2, 0] = 1.0
    P = torch.squeeze(z[..., None, :] * dP_dz, dim=-1)
    return P

# 获取图像像素坐标网格，即b h w 2
def get_pixel_coords(b, img_size, device, dtype):
    h, w = img_size
    u, v = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
    uv = torch.stack((u, v), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    uv = uv.to(device=device, dtype=dtype)
    return uv
