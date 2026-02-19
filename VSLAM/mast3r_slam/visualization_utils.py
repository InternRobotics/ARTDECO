import functools

import cv2
import imgui
import matplotlib
import torch
import numpy as np
from scipy.spatial import cKDTree

from VSLAM.thirdparty.in3d.in3d.geometry import LineGeometry


@functools.cache
def get_colormap(colormap):
    colormap = matplotlib.colormaps[colormap]
    return colormap(np.linspace(0, 1, 256))[:, :3]


def depth2rgb(depth, min=None, max=None, colormap="turbo", add_alpha=False, alpha=1.0):
    # depth: HxW
    dmin = np.nanmin(depth) if min is None else min
    dmax = np.nanmax(depth) if max is None else max
    d = (depth - dmin) / np.maximum((dmax - dmin), 1e-8)
    d = np.clip(d * 255, 0, 255).astype(np.int32)
    img = get_colormap(colormap)[d].astype(np.float32)
    if add_alpha:
        img = np.concatenate([img, alpha * np.ones_like(img[..., :1])], axis=-1)
    return np.ascontiguousarray(img)


class Frustums(LineGeometry):
    def __init__(self, program):
        super().__init__()
        self.program = program
        self.lines = []
        self.colors = []
        self.frustum = self.make_frustum(1, 1)

    def make_frustum(self, h, w):
        self.aspect_ratio = float(w / h)
        origin = [0.0, 0.0, 0.0]
        topleft = [-self.aspect_ratio, -1.0, 1.0]
        topright = [self.aspect_ratio, -1.0, 1.0]
        bottomleft = [-self.aspect_ratio, 1.0, 1.0]
        bottomright = [self.aspect_ratio, 1.0, 1.0]
        self.frustum = np.array(
            [
                origin,
                topleft,
                origin,
                topright,
                origin,
                bottomleft,
                origin,
                bottomright,
                topleft,
                topright,
                topright,
                bottomright,
                bottomright,
                bottomleft,
                bottomleft,
                topleft,
            ],
            dtype=np.float32,
        )

    def add(self, T_WC, thickness=3, scale=1, color=None):
        frustum = T_WC.Act(torch.from_numpy(self.frustum * scale)).numpy()
        thickness = np.ones_like(frustum[..., :1]) * thickness
        frustum = np.concatenate([frustum, thickness], axis=-1).reshape(-1, 4)
        color = [1.0, 1.0, 1.0, 1.0] if color is None else color
        colors = np.tile(color, (frustum.shape[0], 1)).astype(np.float32)
        self.lines.append(frustum)
        self.colors.append(colors)

    def render(self, camera, mode=None):
        if len(self.lines) == 0:
            return
        self.lines = np.concatenate(self.lines, axis=0)
        self.colors = np.concatenate(self.colors, axis=0)
        self.clear()
        super().render(camera, mode=mode)
        self.lines = []
        self.colors = []


class Lines(LineGeometry):
    def __init__(self, program):
        super().__init__()
        self.program = program
        self.lines = []
        self.colors = []

    def add(self, start, end, thickness=1, color=None):
        start = start.reshape(-1, 3).astype(np.float32)
        end = end.reshape(-1, 3).astype(np.float32)

        thickness = np.ones_like(start[..., :1]) * thickness
        start_xyzw = np.concatenate([start, thickness], axis=-1)
        end_xyzw = np.concatenate([end, thickness], axis=-1)
        line = np.concatenate([start_xyzw, end_xyzw], axis=1).reshape(-1, 4)
        if isinstance(color, np.ndarray):  # TODO Bit hacky!
            colors = color.reshape(-1, 4).astype(np.float32)
        else:
            color = [1.0, 1.0, 1.0, 1.0] if color is None else color
            colors = np.tile(color, (line.shape[0], 1)).astype(np.float32)

        # make sure that the dimensions match!
        assert line.shape[0] == colors.shape[0]
        assert line.shape[1] == 4 and colors.shape[1] == 4

        self.lines.append(line)
        self.colors.append(colors)

    def render(self, camera, mode=None):
        if len(self.lines) == 0:
            return
        self.lines = np.concatenate(self.lines, axis=0)
        self.colors = np.concatenate(self.colors, axis=0)
        self.clear()
        super().render(camera, mode=mode)
        self.lines = []
        self.colors = []


def image_with_text(img, size, text, same_line=False):
    # check if the img is too small to render
    if size[0] < 16:
        return
    text_cursor_pos = imgui.get_cursor_pos()
    imgui.image(img.texture.glo, *size)
    if same_line:
        imgui.same_line()
    next_cursor_pos = imgui.get_cursor_pos()
    imgui.set_cursor_pos(text_cursor_pos)
    imgui.text(text)
    imgui.set_cursor_pos(next_cursor_pos)


def save_pointcloud_ply(points, colors, filename):
    """
    保存点云为PLY格式

    Args:
        points: (N, 3) tensor，点云坐标
        colors: (N, 3) tensor，RGB颜色值(0-1之间)
        filename: 输出文件名
    """
    # 转换为numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()

    # 将颜色从0-1转换为0-255
    colors = (colors * 255).astype(np.uint8)

    N = points.shape[0]

    # 写入PLY文件
    with open(filename, 'w') as f:
        # PLY头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # 点云数据
        for i in range(N):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

def non_maximum_suppression(x, y, r1, r2):
    # 将坐标(x, y)堆叠成一个n x 2的数组
    points = np.vstack((x, y)).T

    # 使用cKDTree加速最近邻查找
    tree = cKDTree(points)

    # 创建一个布尔型的mask，初始化为True，表示所有点都被选中
    mask = np.ones(len(x), dtype=bool)

    # 对每个点，查找半径r1到r2内的点
    for i in range(len(points)):
        if mask[i]:  # 如果当前点已经被选中
            # 查询半径r2内的邻居点
            indices_r2 = tree.query_ball_point(points[i], r2)

            # 查询半径r1内的邻居点
            indices_r1 = tree.query_ball_point(points[i], r1)

            # 需要抑制的点是在半径r1和r2之间的点
            suppress_indices = [idx for idx in indices_r2 if idx not in indices_r1]

            # 将这些点标记为不被选中
            mask[suppress_indices] = False
            # 重新选中当前点（因为它没有被抑制）
            mask[i] = True

    return mask

def visualize_matches_corr(img_i, img_j, index_i, flow_i_on_j, flow_i_on_j_gt, valid_mask,
                         pt_radius=1, line_thickness=1, filter_i=[0, 10], filter_j=[0, 10], vis_line=True):

    # 张量转 numpy、0-255-uint8、RGB→BGR
    def to_bgr(img):
        img = (img.clamp(0, 1) * 255).byte().cpu().numpy()
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. 拼接图像
    img_i = to_bgr(img_i)
    img_j = to_bgr(img_j)
    H, W = img_i.shape[:2]
    canvas = cv2.hconcat([img_i, img_j])

    # 2. 计算在两个图像中的估计点和对应点的行、列坐标
    # 有效 j-像素索引 k
    k_all = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    if k_all.numel() == 0:
        return canvas

    # 计算行、列坐标
    tOriginal = index_i[k_all]
    i_rows = tOriginal[: ,1]
    i_cols = tOriginal[: ,0]

    tEstimate = flow_i_on_j[k_all]
    j_rows = tEstimate[:, 1]
    j_cols = tEstimate[:, 0] + W

    tgt = flow_i_on_j_gt[k_all]
    j_rows_gt = tgt[: ,1]
    j_cols_gt = tgt[: ,0] + W

    # 3. 非极大值抑制半径内的点
    mask_i = np.ones(i_rows.shape[0], dtype=bool)
    mask_j = np.ones(i_rows.shape[0], dtype=bool)
    if filter_i[1] - filter_i[0] > 0:
        valid_mask = non_maximum_suppression(i_rows, i_cols, filter_i[0], filter_i[1])
        mask_i = valid_mask
    if filter_j[1] - filter_j[0] > 0:
        valid_mask = non_maximum_suppression(j_rows, j_cols, filter_j[0], filter_j[1])
        mask_j = valid_mask
    overall_mask = mask_i & mask_j

    j_rows = j_rows[overall_mask]
    j_cols = j_cols[overall_mask]
    i_rows = i_rows[overall_mask]
    i_cols = i_cols[overall_mask]
    j_rows_gt = j_rows_gt[overall_mask]
    j_cols_gt = j_cols_gt[overall_mask]

    # 在 canvas 上画线与端点
    for r_i, c_i, r_j, c_j, r_j_gt, c_j_gt, in zip(i_rows, i_cols, j_rows, j_cols, j_rows_gt, j_cols_gt):
        pt_i = (int(c_i), int(r_i))
        pt_j = (int(c_j), int(r_j))
        pt_j_gt = (int(c_j_gt ), int(r_j_gt))  # j 的 x 要整体右移 W
        if vis_line:
            cv2.line(canvas, pt_i, pt_j, (0, 255, 255), line_thickness, cv2.LINE_AA)  # j-i 黄线
        cv2.circle(canvas, pt_i, pt_radius, (255, 0, 0), -1, cv2.LINE_AA)  # j: 蓝点
        cv2.circle(canvas, pt_j, pt_radius, (0, 0, 255), -1, cv2.LINE_AA)  # i: 红点

        if (r_j_gt < 0) or (r_j_gt > H-1) or (c_j_gt < W) or (c_j_gt > 2*W-1):
            cv2.rectangle(canvas, (int(c_j)-3, int(r_j)-3), (int(c_j)+3, int(r_j)+3), (0, 0, 255), 2)
        else:
            cv2.line(canvas, pt_j, pt_j_gt, (255, 0, 0), line_thickness, cv2.LINE_AA)  # i-i_gt 蓝线
            cv2.circle(canvas, pt_j_gt, pt_radius, (0, 255, 0), -1, cv2.LINE_AA)  # i_gt: 绿点

    return canvas