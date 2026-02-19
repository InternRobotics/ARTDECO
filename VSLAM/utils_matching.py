import torch
import torch.nn.functional as F
import mast3r_slam_backends


# 根据角度和特征计算X21中每个点与X11中的像素的对应关系，存储在idx_1_to_2_init
def match_pi3(config ,X11, X21, idx_1_to_2_init=None):
    cfg = config["matching"]
    b, h, w = X21.shape[:3]
    device = X11.device

    # 计算梯度，预处理
    # B H W C(coor, xgrad, ygrad)        b h w c              b (h*w) 2(u, v)
    rays_with_grad_img, pts3d_norm, p_init = prep_for_iter_proj(X11, X21, idx_1_to_2_init)

    # 匹配
    # todo: LM算法中梯度计算没有除以归一化项
    # p1 更新后的2中所有点到1中像素的匹配关系
    # valid_proj2 p1中的点的收敛情况
    with torch.cuda.device(device):
        p1, valid_proj2 = mast3r_slam_backends.iter_proj(
            rays_with_grad_img,
            pts3d_norm,
            p_init,
            cfg["max_iter"],
            cfg["lambda_init"],
            cfg["convergence_thresh"],
        )
    p1 = p1.long() # todo:整数匹配？？？
    # 对上述的角度匹配的点进行二次距离的过滤，距离过大直接删除
    # Check for occlusion based on distances
    batch_inds = torch.arange(b, device=device)[:, None].repeat(1, h * w)
    dists2 = torch.linalg.norm(
        X11[batch_inds, p1[..., 1], p1[..., 0], :].reshape(b, h, w, 3) - X21, dim=-1
    )
    valid_dists2 = (dists2 < cfg["dist_thresh"]).view(b, -1)
    valid_proj2 = valid_proj2 & valid_dists2


    # todo：并没有去处一对多的情况  clamp(v_new, 1, h-2)对应情况不同
    # todo：valid_proj2仅仅参与了角度和距离的更新，没有参与最后基于特征的更新
    # Convert to linear index
    idx_1_to_2 = pixel_to_lin(p1, w)
    # unique_values = torch.unique(idx_1_to_2)
    # # 判断是否有重复值
    # has_duplicates = len(unique_values) != len(idx_1_to_2)
    # print("是否有重复值:", has_duplicates)
    # print("21",p1)
    return idx_1_to_2, valid_proj2


# 计算相机坐标系中归一化的射线对像素坐标uv的梯度
def img_gradient(img):
    device = img.device
    dtype = img.dtype
    b, c, h, w = img.shape

    gx_kernel = (1.0 / 32.0) * torch.tensor(
        [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    gx_kernel = gx_kernel.repeat(c, 1, 1, 1)

    gy_kernel = (1.0 / 32.0) * torch.tensor(
        [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    gy_kernel = gy_kernel.repeat(c, 1, 1, 1)

    gx = F.conv2d(
        F.pad(img, (1, 1, 1, 1), mode="reflect"),
        gx_kernel,
        groups=img.shape[1],
    )

    gy = F.conv2d(
        F.pad(img, (1, 1, 1, 1), mode="reflect"),
        gy_kernel,
        groups=img.shape[1],
    )

    return gx, gy

# 根据角度和特征计算匹配，返回2对1的匹配关系（从0-hw每一个像素对应的1中的uv关系）以及每个匹配关系的有效数
def match(config, X11, X21, D11, D21, idx_1_to_2_init=None):
    idx_1_to_2, valid_match2 = match_iterative_proj(config, X11, X21, D11, D21, idx_1_to_2_init)
    return idx_1_to_2, valid_match2

# 像素的uv转为序列数
def pixel_to_lin(p1, w):
    idx_1_to_2 = p1[..., 0] + (w * p1[..., 1])
    return idx_1_to_2

# 序列数转为像素的uv
def lin_to_pixel(idx_1_to_2, w):
    u = idx_1_to_2 % w
    v = idx_1_to_2 // w
    p = torch.stack((u, v), dim=-1)
    return p

# 返回
# X11每个位置的梯度（归一化坐标xyz队像素坐标uv的梯度） rays_with_grad_img
# X21 对应的归一化坐标 pts3d_norm
# 以及初始化的X21的每个点和X11的uv的对应关系 p_init
def prep_for_iter_proj(X11, X21, idx_1_to_2_init):
    b, h, w, _ = X11.shape
    device = X11.device

    # Ray image
    rays_img = F.normalize(X11, dim=-1)
    rays_img = rays_img.permute(0, 3, 1, 2)  # (b,c,h,w)
    gx_img, gy_img = img_gradient(rays_img) # (b,c,h,w)
    rays_with_grad_img = torch.cat((rays_img, gx_img, gy_img), dim=1)
    rays_with_grad_img = rays_with_grad_img.permute(
        0, 2, 3, 1
    ).contiguous()  # (b,h,w,c)

    # 3D points to project
    X21_vec = X21.view(b, -1, 3)
    pts3d_norm = F.normalize(X21_vec, dim=-1)

    # Initial guesses of projections
    if idx_1_to_2_init is None:
        # Reset to identity mapping
        idx_1_to_2_init = torch.arange(h * w, device=device)[None, :].repeat(b, 1)
    p_init = lin_to_pixel(idx_1_to_2_init, w)
    p_init = p_init.float()

    return rays_with_grad_img, pts3d_norm, p_init

# 根据角度和特征计算X21中每个点与X11中的像素的对应关系，存储在idx_1_to_2_init
def match_iterative_proj(config ,X11, X21, D11, D21, idx_1_to_2_init=None):
    cfg = config["matching"]
    b, h, w = X21.shape[:3]
    device = X11.device

    # 计算梯度，预处理
    # B H W C(coor, xgrad, ygrad)        b h w c              b (h*w) 2(u, v)
    rays_with_grad_img, pts3d_norm, p_init = prep_for_iter_proj(
        X11, X21, idx_1_to_2_init
    )

    # 匹配
    # todo: LM算法中梯度计算没有除以归一化项
    # p1 更新后的2中所有点到1中像素的匹配关系
    # valid_proj2 p1中的点的收敛情况
    with torch.cuda.device(device):
        p1, valid_proj2 = mast3r_slam_backends.iter_proj(
            rays_with_grad_img,
            pts3d_norm,
            p_init,
            cfg["max_iter"],
            cfg["lambda_init"],
            cfg["convergence_thresh"],
        )
    p1 = p1.long() # todo:整数匹配？？？
    # 对上述的角度匹配的点进行二次距离的过滤，距离过大直接删除
    # Check for occlusion based on distances
    batch_inds = torch.arange(b, device=device)[:, None].repeat(1, h * w)
    dists2 = torch.linalg.norm(
        X11[batch_inds, p1[..., 1], p1[..., 0], :].reshape(b, h, w, 3) - X21, dim=-1
    )
    valid_dists2 = (dists2 < cfg["dist_thresh"]).view(b, -1)
    valid_proj2 = valid_proj2 & valid_dists2

    # todo:这个cuda实现中，for循环写的太过多余，直接两层循环遍历即可，两个参数改成一个就行
    if cfg["radius"] > 0:
        with torch.cuda.device(device):
            (p1,) = mast3r_slam_backends.refine_matches(
                D11.half(),
                D21.view(b, h * w, -1).half(),
                p1,
                cfg["radius"],
                cfg["dilation_max"],
            )

    # todo：并没有去处一对多的情况  clamp(v_new, 1, h-2)对应情况不同
    # todo：valid_proj2仅仅参与了角度和距离的更新，没有参与最后基于特征的更新
    # Convert to linear index
    idx_1_to_2 = pixel_to_lin(p1, w)
    # unique_values = torch.unique(idx_1_to_2)
    # # 判断是否有重复值
    # has_duplicates = len(unique_values) != len(idx_1_to_2)
    # print("是否有重复值:", has_duplicates)
    # print("21",p1)
    return idx_1_to_2, valid_proj2.unsqueeze(-1)
