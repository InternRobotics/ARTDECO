import torch
import torch.nn.functional as F

@torch.no_grad()
def local_diag_cov_from_X1(
    X1: torch.Tensor,     # (H*W, 3)  3D点云(任意坐标系)，展平顺序为行优先
    H: int, W: int,       # 图像尺寸
    win: int = 5,         # 邻域窗口(奇数)，如 3/5/7
    valid: torch.Tensor | None = None,  # (H*W,) 可选有效掩码；None则自动: finite & z>0
    var_floor: float = 1e-12            # 方差下限，避免数值为负/零
) -> torch.Tensor:
    """
    返回: Sigma (H*W, 3, 3) —— 仅包含对角项的局部协方差矩阵
          diag = [Var(X), Var(Y), Var(Z)]，其余为 0
    """
    device = X1.device
    assert X1.shape[0] == H*W and X1.shape[1] == 3, "X1应为(H*W,3)"
    pad = win // 2

    # (H,W,3)
    Xv = X1.view(H, W, 3)

    # 有效掩码
    if valid is None:
        valid_hw1 = (torch.isfinite(Xv).all(-1) & (Xv[..., 2] > 0)).float().unsqueeze(-1)
    else:
        assert valid.shape[0] == H*W
        valid_hw1 = valid.view(H, W, 1).float()

    # 盒滤波（用 avg_pool2d 实现）
    def boxfilter(img_hwC):
        x = img_hwC.permute(2,0,1).unsqueeze(0)             # (1,C,H,W)
        x = F.pad(x, (pad,pad,pad,pad), mode='reflect')
        y = F.avg_pool2d(x, kernel_size=win, stride=1)      # (1,C,H,W)
        return y.squeeze(0).permute(1,2,0)                  # (H,W,C)

    denom = boxfilter(valid_hw1).clamp_min(1e-9)
    mean  = boxfilter(Xv * valid_hw1) / denom               # E[X]
    ex2   = boxfilter((Xv * Xv) * valid_hw1) / denom        # E[X^2]
    var   = (ex2 - mean * mean).clamp_min(var_floor)        # Var = E[X^2]-E[X]^2

    # 组装对角协方差 (H*W,3,3)
    N = H*W
    Sigma = torch.zeros(N, 3, 3, device=device)
    var_flat = var.view(N, 3)
    Sigma[:, 0, 0] = var_flat[:, 0]   # Var(X)
    Sigma[:, 1, 1] = var_flat[:, 1]   # Var(Y)
    Sigma[:, 2, 2] = var_flat[:, 2]   # Var(Z)

    return Sigma

import numpy as np
import cv2

def visualize_covariance_max(Sigma, H, W,
                             cmap: str = "inferno",
                             win_name: str = "Max Covariance (per-pixel)",
                             low_percent: float = 20.0,
                             high_percent: float = 80.0):
    """
    可视化每像素最大协方差，并用 OpenCV 显示。

    Args:
      Sigma: (H*W, 3, 3) 协方差（对角为 [VarX, VarY, VarZ]；也可非对角，但只取 diag）
             (numpy 或 torch.Tensor 都可)
      H, W : 图像尺寸
      cmap : 颜色映射，可选: "inferno"(默认), "magma", "plasma", "viridis", "turbo", "jet"
      win_name: 窗口名
      low_percent, high_percent: 用分位数做归一化的下上界（抑制异常值）

    效果:
      - INFERNO: 协方差小=暗紫/黑，协方差大=亮黄/白
      - JET:     协方差小=蓝，中=绿，大=红（不推荐）
    """
    # 兼容 torch
    try:
        import torch
        if isinstance(Sigma, torch.Tensor):
            Sigma = Sigma.detach().cpu().numpy()
    except ImportError:
        pass

    assert Sigma.shape == (H*W, 3, 3), f"Sigma shape should be {(H*W,3,3)}, got {Sigma.shape}"

    # 取对角三通道的最大值作为该像素的协方差代表
    diag = np.stack([Sigma[:,0,0], Sigma[:,1,1], Sigma[:,2,2]], axis=-1)  # (N,3)
    smax = np.max(diag, axis=1).reshape(H, W)  # (H,W)
    smax = np.linalg.det(Sigma).reshape(H, W)

    # 稳健归一化（按分位数拉伸到 0..1）
    finite_mask = np.isfinite(smax)
    if np.any(finite_mask):
        lo = np.percentile(smax[finite_mask], low_percent)
        hi = np.percentile(smax[finite_mask], high_percent)
        if hi <= lo: hi = lo + 1e-12
        norm = np.clip((smax - lo) / (hi - lo), 0, 1)
    else:
        norm = np.zeros((H, W), dtype=np.float32)

    # 转 0..255 灰度
    gray = (norm * 255.0).astype(np.uint8)

    # 选择 colormap
    cmap_dict = {
        "inferno": cv2.COLORMAP_INFERNO,
        "magma":   cv2.COLORMAP_MAGMA,
        "plasma":  cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo":   cv2.COLORMAP_TURBO,
        "jet":     cv2.COLORMAP_JET,
    }
    cv_cmap = cmap_dict.get(cmap.lower(), cv2.COLORMAP_INFERNO)

    # 伪彩色（OpenCV 返回 BGR）
    color_bgr = cv2.applyColorMap(gray, cv_cmap)

    # 显示
    cv2.imshow(win_name, color_bgr)
    cv2.waitKey(-1)
    cv2.destroyWindow(win_name)

    return color_bgr  # 若你想另存：cv2.imwrite("cov_max.png", color_bgr)