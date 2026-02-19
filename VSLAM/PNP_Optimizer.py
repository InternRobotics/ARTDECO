import time

import torch
import pypose as pp

class CholeskySolver(torch.autograd.Function):
    """
        Cholesky solver function
            H torch.Tensor [B,N,N]
            b torch.Tensor [B,N,M]
        return
            xs torch.Tensor [B,N,M]
    """
    @staticmethod
    def forward(ctx, H, b):



        # don't crash training if cholesky decomp fails
        L, info = torch.linalg.cholesky_ex(H)

        if torch.any(info):
            ctx.failed = True
            return torch.zeros_like(b)

        xs = torch.cholesky_solve(b, L)
        ctx.save_for_backward(L, xs)
        ctx.failed = False

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        L, xs = ctx.saved_tensors

        dz = torch.cholesky_solve(grad_x, L)
        dH = -torch.matmul(xs, dz.transpose(-1,-2)).transpose(-1,-2)

        return dH, dz

class BlockDiagonalInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block_diagonal_matrix, size):
        """
        Forward pass: Compute the inverse of a block-diagonal matrix.

        Args:
            block_diagonal_matrix (Tensor): A tensor of block-diagonal matrices with shape (B, N, N),
                                             where B is the number of batches and N is the dimension of each block.
            size (int): The size of each block.

        Returns:
            Tensor: The inverse of the block-diagonal matrix.
        """
        device = block_diagonal_matrix.device

        # Save the input tensor and block size
        ctx.size = size

        B, N, _ = block_diagonal_matrix.shape
        block_count = N // size

        batch_indices = torch.arange(B)[..., None, None, None].expand(-1, block_count, size, size).to(device)
        x_indices = torch.arange(block_count)[None, :, None, None].expand(B, -1, size, size).to(device) * size + \
                    torch.arange(size)[None, None, :, None].expand(B, block_count, -1, size).to(device)
        y_indices = torch.arange(block_count)[None, :, None, None].expand(B, -1, size, size).to(device) * size + \
                    torch.arange(size)[None, None, None, :].expand(B, block_count, size, -1).to(device)
        ctx.batch_indices = batch_indices
        ctx.x_indices = x_indices
        ctx.y_indices = y_indices

        # 通过索引一次性提取所有块 (B, block_count, size, size)
        blocks = block_diagonal_matrix[batch_indices, x_indices, y_indices]

        # 计算所有块的逆矩阵
        blocks = blocks.reshape(B * block_count, size, size)
        inv_blocks, info = torch.linalg.inv_ex(blocks)
        if torch.any(info):
            ctx.failed = True
            return torch.zeros_like(block_diagonal_matrix)
        inv_blocks = inv_blocks.reshape(B, block_count, size, size)

        # 重构原始形状的逆矩阵
        inv_matrix = torch.zeros_like(block_diagonal_matrix).to(device)
        inv_matrix[batch_indices, x_indices, y_indices] = inv_blocks
        ctx.save_for_backward(inv_matrix)

        return inv_matrix

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute the gradient of the input block-diagonal matrix.

        Args:
            grad_output (Tensor): Gradient of the loss w.r.t. the output of the forward pass.

        Returns:
            Tensor: The gradient of the input block-diagonal matrix.
        """
        if ctx.failed:
            return None, None
        inv_matrix, = ctx.saved_tensors
        B, N, _ = inv_matrix.shape
        size = ctx.size
        block_count = N // size

        batch_indices = ctx.batch_indices
        x_indices = ctx.x_indices
        y_indices = ctx.y_indices

        grad_input = torch.zeros_like(inv_matrix)

        # 通过索引一次性提取所有块 (B, block_count, size, size)
        inv_blocks = inv_matrix[batch_indices, x_indices, y_indices]
        grad_blocks = grad_output[batch_indices, x_indices, y_indices]

        # 计算梯度
        grad_input[batch_indices, x_indices, y_indices] = \
            -torch.matmul(torch.matmul( inv_blocks.transpose(-2, -1), grad_blocks), inv_blocks.transpose(-2, -1))

        return grad_input, None  # None for the `size` gradient as it's not trainable


def compute_huber_weights(residuals, threshold=1.0, epsilon=1e-6):
    """
    Huber核函数计算权重

    参数:
    - residuals: [B, M, C] 每条边的C个通道的残差 (pixels_es - pixels_jj)
    - threshold: Huber阈值

    返回:
    - weights: [B, M] 权重
    """
    # 计算残差L2范数
    residual_norms = torch.norm(residuals, dim=-1)  # [B, M]

    # Huber权重: weight = 1 if |r| <= threshold, else threshold/|r|
    weights = torch.where(
        residual_norms <= threshold,
        torch.ones_like(residual_norms).to(residuals.device),
        threshold / (residual_norms + epsilon)  # 避免除零
    )

    return weights


def residual_PW2Pix(B, M, Tcjw, points_world, pixels_jj, K):
    """
        B: int - batch size
        M: int - edges number
        Tcjw: [B, M] - transformation matrix
        points_world: [B, M, 3] - world points
        pixels_jj: [B, M, 2] - target pixels
        K: [3, 3] - intrinsics
    """

    Points_j = Tcjw.Act(points_world)
    pixels_es = Points_j / Points_j[..., 2:3]
    K_batch = K[None, None, ...].expand(B, M, 3, 3)
    pixels_es = (K_batch @ pixels_es.unsqueeze(-1)).squeeze(-1)[..., :2]

    residuals = pixels_es - pixels_jj

    return residuals, Points_j, pixels_es




def build_optimization_matrix(J_x, J_y,
                              B, M,
                              dimx, dimy,
                              numx, numy,
                              indexx, indexy,
                              device="cuda:0",
                              damping=1e-4):
    Matrix = torch.zeros(B, dimx * numx * dimy * numy, device=device)
    Matrix_xy = torch.matmul(J_x.transpose(-2, -1), J_y)  # B M (dimx) (dimy)

    indexx_ex = indexx[:, :, None, None]  # [B, M, 1, 1]
    indexy_ex = indexy[:, :, None, None]  # [B, M, 1, 1]
    row_idx_xy = (indexx_ex * dimx).expand(B, M, dimx, dimy) + torch.arange(dimx, device=device).view(1, 1, dimx, 1)
    col_idx_xy = (indexy_ex * dimy).expand(B, M, dimx, dimy) + torch.arange(dimy, device=device).view(1, 1, 1, dimy)
    xy_idx_all = row_idx_xy * numy * dimy + col_idx_xy

    Matrix.scatter_add_(1, xy_idx_all.reshape(B, -1), Matrix_xy.reshape(B, -1))
    Matrix = Matrix.reshape(B, dimx * numx, dimy * numy)  # B 3L 3L

    # todo:  此处检测是通过也不一定需要增加阻尼，但目前没什么问题
    if dimx == dimy and numx == numy:
        damping_matrix = damping * torch.eye(dimx * numx,
                                             device=device,
                                             dtype=torch.float64)[None].repeat(B, 1, 1)
        Matrix += damping_matrix

    return Matrix


def build_optimization_matrix_right(J_x, res, B ,M, dimx, numx, indexx, device="cuda:0"):
    right = torch.zeros(B, dimx * numx, device=device)  # B 6N
    right_xx = torch.matmul(J_x.transpose(-2, -1), res)  # B M 6 1

    indexx_ex = indexx[:, :, None, None]  # [B, M, 1, 1]
    row_idx_xx = (indexx_ex * dimx).expand(B, M, dimx, 1) + torch.arange(dimx, device=device).view(1, 1, dimx, 1)

    right.scatter_add_(1, row_idx_xx.reshape(B, -1), right_xx.reshape(B, -1))
    right = right[..., None]
    return -right



# 要求points中每个点至少有一条有效边
# 要求Tcws中每个位置至少有一条有效边
# ii 0-N
# jj 0-N
# kk 0-L
def opt_single_pnp(Tcws,
                   jj: torch.Tensor,
                   kk: torch.Tensor,
                   points: torch.Tensor,
                   pixels_jj: torch.Tensor,
                   valid_mask: torch.Tensor,
                   K: torch.Tensor,
                   w: int,
                   h: int,
                   fix_pose = 1,
                   damping=1e-4,
                   huber_thres = 2.,
                   optimize_points = True,
                   optimize_xy = False,
                   device = "cuda:0"):
    """
    执行一次Bundle Adjustment迭代

    参数:
        Tcws: lietorch.SE3 [B, N] - 初始相机位姿 (Tcw: 相机到世界)
        jj: [B, M] - 观测到3D点的相机索引
        kk: [B, M] - 观测到3D点的索引
        points: [B, L, 3] - 世界坐标系的3D点
        pixels_jj: [B, M, 2] - 每个边在jj坐标系下的目标像素
        valid_mask: [B, M] - 每个边的有效性
        damping: float - 阻尼系数
    返回:
        Tcws_updated: lietorch.SE3 [B, N, 7] - 更新后的位姿
        residuals: [B, M, 3] - 残差
    """
    B, N, _ = Tcws.shape
    B, M = jj.shape
    B, L, _ = points.shape

    # ============================================================================
    # 第1步：计算残差和weights
    # ============================================================================
    batch_indices = torch.arange(B, device=Tcws.device).unsqueeze(1).expand(B, M)
    # 获取目标相机位姿 (Tcw)
    Tcjw = Tcws[batch_indices, jj]  # [B, M, 7] - 目标相机位姿
    # 选择每条边对应的世界点
    points_selected = points[batch_indices, kk]  # [B, M, 3]

    residuals_init, Points_j, _ = residual_PW2Pix(B, M, Tcjw, points_selected, pixels_jj, K) # [B, M, 2]
    weights = compute_huber_weights(residuals_init, threshold=huber_thres)
    # apply masks and weights
    residuals = weights[..., None] * residuals_init
    residuals = valid_mask[..., None] * residuals

    # ============================================================================
    # 第2步：计算雅可比矩阵
    # ============================================================================
    X, Y, Z = Points_j.unbind(dim=-1)  # [B, M]
    o = torch.zeros_like(X)
    W = torch.ones_like(X)

    J_pixj_Pj = torch.stack([K[0, 0] / Z, o, -X * K[0, 0] / (Z ** 2),
                             o, K[1, 1] / Z, -Y * K[1, 1] / (Z ** 2)], dim=-1).view(B, M, 2, 3)
    J_Pj_Tcjw = torch.stack([
                W, o, o, o, Z, -Y,
                o, W, o, -Z, o, X,
                o, o, W, Y, -X, o,
            ], dim=-1).view(B, M, 3, 6)

    J_r_Tcjw = J_pixj_Pj @ J_Pj_Tcjw # B M 2 6

    # apply masks and weights
    J_r_Tcjw = J_r_Tcjw * weights[..., None, None] * valid_mask[..., None, None]

    if optimize_points:
        J_Pj_PW = Tcjw.matrix()[:, :, :3, :3]
        J_r_PW = J_pixj_Pj @ J_Pj_PW# B M 2 3
        J_r_PW = J_r_PW * weights[..., None, None] * valid_mask[..., None, None]

    # ============================================================================
    # 第3步：构建线性系统
    # ============================================================================

    # Build B
    B_mat = build_optimization_matrix(J_r_Tcjw, J_r_Tcjw, B, M, 6, 6, N, N, jj, jj,
                                  device=device, damping=damping)
    # build v
    v = build_optimization_matrix_right(J_r_Tcjw, residuals[..., None], B ,M, 6, N, jj, device=device)

    if optimize_points:
        # build C
        C = build_optimization_matrix(J_r_PW, J_r_PW, B, M, 3, 3,L, L, kk, kk,
                                      device=device, damping=damping)
        # build E
        E = build_optimization_matrix(J_r_Tcjw, J_r_PW, B, M, 6, 3, N, L, jj, kk,
                                      device=device, damping=damping)
        # build w
        w = build_optimization_matrix_right(J_r_PW, residuals[..., None], B, M, 3, L, kk, device=device)

    # ============================================================================
    # 第4步：求解线性系统
    # ============================================================================
    B_mat = B_mat[:, 6*fix_pose:, 6*fix_pose:]
    v     =     v[:, 6*fix_pose:, :]

    if optimize_points:
        E = E[:, 6 * fix_pose:, :]

        # inverse C
        C_inv = BlockDiagonalInverse.apply(C, 3)
        C_inv.nan_to_num_()

        # solve for poses
        S = B_mat - E @ C_inv @ E.transpose(-1, -2)
        b = v - E @ C_inv @ w

        delta_Tcws = CholeskySolver.apply(S, b)

        # solve for points
        F = w - E.transpose(-1, -2) @ delta_Tcws
        delta_points = C_inv @ F
    else:
        delta_Tcws = CholeskySolver.apply(B_mat, v)

    # ============================================================================
    # 第5步：更新位姿并且评估残差
    # ============================================================================

    # 将更新向量重新排列为 [B, N, 6]
    delta_poses = delta_Tcws.reshape(B, N - fix_pose, 6)

    # 应用SE3更新：
    Tcws_updated = pp.se3(delta_poses).Exp().mul(Tcws[:, fix_pose:])
    Tcws_updated = pp.quat2unit(Tcws_updated)
    Tcws_updated.data = torch.concatenate([Tcws.data[:, :fix_pose, ...], Tcws_updated.data], dim=1)

    # 应用points更新
    if optimize_points:
        if not optimize_xy:
            points_updated = torch.concatenate([points[..., :2],
                                                points[..., 2:] + delta_points.reshape(B, L, 3)[..., 2:]], dim=-1)
        else:
            points_updated = points + delta_points.reshape(B, L, 3)


    batch_indices = torch.arange(B, device=Tcws.device).unsqueeze(1).expand(B, M)
    # 获取目标相机位姿 (Tcw)
    Tcjw = Tcws_updated[batch_indices, jj]  # [B, M, 7] - 目标相机位姿
    # 选择每条边对应的世界点
    if optimize_points:
        points_selected = points_updated[batch_indices, kk]  # [B, M, 3]

    residuals_after, _, __ = residual_PW2Pix(B, M, Tcjw, points_selected, pixels_jj, K)  # [B, M, 2]
    residuals_init = valid_mask[..., None] * residuals_init
    residuals_after = valid_mask[..., None] * residuals_after

    if optimize_points:
        return Tcws_updated, points_updated, residuals_init, residuals_after
    else:
        return Tcws_updated, points, residuals_init, residuals_after

def opt_pnp(Tcws_init,
            jj: torch.Tensor,
            kk: torch.Tensor,
            points: torch.Tensor,
            pixels_jj: torch.Tensor,
            valid_mask: torch.Tensor,
            K, w, h,
            damping=1e-4,
            huber_thres=0.1,
            fix_pose = 0,
            iter=20,
            verbose=False,
            optimize_points=False,
            optimize_xy=False,
            device="cuda:0"):

    # setting damping factor
    Tcws_current = Tcws_init
    points_current = points
    t1 = time.time()
    for i in range(iter):
        (Tcws_updated, points_updated,
         residuals_init, residuals_after) = opt_single_pnp(Tcws_current, jj, kk,
                                                           points_current, pixels_jj, valid_mask,
                                                           K, w, h,
                                                           fix_pose=fix_pose,
                                                           damping=damping,
                                                           huber_thres=huber_thres,
                                                           optimize_points=optimize_points,
                                                           optimize_xy=optimize_xy,
                                                           device=device, )

        initLoss = torch.abs(residuals_init).mean()
        afterLoss = torch.abs(residuals_after).mean()

        if afterLoss < initLoss:
            damping *= 0.5
            Tcws_current = Tcws_updated
            points_current = points_updated
        else:
            damping *= 2

        if verbose:
            print(f"迭代次数{i:5d}, "
                  f"残差: {initLoss.item()}->{afterLoss.item():.8f}")

    return Tcws_current, residuals_after