import glm
import numpy as np
import imgui

from in3d.logger import setup_logger
from in3d.pose_utils import exp_angle_axis, translation_matrix

logger = setup_logger()


def unproject(win_x, win_y, win_z, view_mat, proj_mat, viewport):
    vx, vy, vx_w, vy_h = viewport
    # normalize x, y, z
    x = (2 * win_x - vx) / vx_w - 1
    y = (2 * win_y - vy) / vy_h - 1
    z = 2 * win_z - 1
    inv_proj_view = np.linalg.inv(proj_mat @ view_mat)
    p = np.array([x, y, z, 1], dtype="f4")
    p = inv_proj_view @ p
    return p[:3] / p[-1]


def projection_matrix(w, h, vfov, cx, cy, znear, zfar):
    # glm::mat4 P = glm::perspective(glm::radians(FoV_), window_->ratio(),
    #    min_disp_range_, max_disp_range_);
    aspect = w / h
    P = glm.perspective(glm.radians(vfov / 2), aspect, znear, zfar)
    # to numpy
    return np.array(P, dtype=np.float32)

    # vfov = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(hfov / 2)) * h / w))
    # fx = fov2focal(hfov, w)
    # fy = fov2focal(vfov, h)

    # L = -cx * znear / fx
    # R = (w - cx) * znear / fx
    # T = cy * znear / fy
    # B = -(h - cy) * znear / fy

    # a = 2 * znear / (R - L)
    # b = 2 * znear / (T - B)
    # c = (R + L) / (R - L)
    # d = (T + B) / (T - B)
    # e = -(zfar + znear) / (zfar - znear)
    # f = -(2 * zfar * znear) / (zfar - znear)
    # return np.array(
    #     [
    #         [a, 0, c, 0],
    #         [0, b, d, 0],
    #         [0, 0, e, f],
    #         [0, 0, -1, 0],
    #     ],
    #     dtype=np.float32,
    # )


def lookat(eye, obj, up):
    f = (obj - eye).astype(np.float32)
    f /= np.linalg.norm(f)
    up = up.astype(np.float32)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    u /= np.linalg.norm(u)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = np.array([s, u, -f])
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def fov2focal(fov, size):
    return size / (2 * np.tan(np.deg2rad(fov / 2)))


def focal2fov(focal, size):
    return 2 * np.rad2deg(np.arctan(size / (2 * focal)))


def create_camera_vertices(near, far, fov, aspect_ratio, color=None):
    if color is None:
        color = np.array([1, 1, 1, 1], dtype=np.float32)
    # use opencv coordinate
    h_near = 2 * np.tan(np.radians(fov / 2)) * near
    w_near = h_near * aspect_ratio
    h_far = 2 * np.tan(np.radians(fov / 2)) * far
    w_far = h_far * aspect_ratio

    # Near plane vertices
    near_top_left = np.array([-w_near / 2, h_near / 2, near, 1.0])
    near_top_right = np.array([w_near / 2, h_near / 2, near, 1.0])
    near_bottom_left = np.array([-w_near / 2, -h_near / 2, near, 1.0])
    near_bottom_right = np.array([w_near / 2, -h_near / 2, near, 1.0])

    # Far plane vertices
    far_top_left = np.array([-w_far / 2, h_far / 2, far, 1.0])
    far_top_right = np.array([w_far / 2, h_far / 2, far, 1.0])
    far_bottom_left = np.array([-w_far / 2, -h_far / 2, far, 1.0])
    far_bottom_right = np.array([w_far / 2, -h_far / 2, far, 1.0])
    # fmt:off
    vertices = np.array([
        near_top_left, near_top_right, near_bottom_left, near_bottom_right,
        far_top_left, far_top_right, far_bottom_left, far_bottom_right
    ])
    lines = [
        vertices[0], vertices[1], vertices[1], vertices[3], vertices[3], vertices[2], vertices[2], vertices[0],  # Near plane  
        vertices[4], vertices[5], vertices[5], vertices[7], vertices[7], vertices[6], vertices[6], vertices[4],  # Far plane
        vertices[0], vertices[4], vertices[1], vertices[5], vertices[2], vertices[6], vertices[3], vertices[7],  # Connecting lines
    ]
    axis = [np.array([0, 0, 0, 1]), np.array([near / 2, 0, 0, 1]), np.array([0, 0, 0, 1]), np.array([0, near / 2, 0, 1]), np.array([0, 0, 0, 1]), np.array([0, 0, near / 2, 1])]
    axis_color = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
    # fmt:on

    lines = np.array(lines, dtype=np.float32).reshape(-1, 4)
    colors = np.tile(color, (len(lines), 1))
    axis = np.array(axis, dtype=np.float32).reshape(-1, 4)
    lines = np.concatenate([lines, axis], axis=0)
    colors = np.concatenate([colors, axis_color], axis=0)
    return lines, colors


class ProjectionMatrix:
    def __init__(self, w, h, vfov, cx, cy, znear, zfar):
        self.w = w
        self.h = h
        self.hfov = vfov
        self.cx = cx
        self.cy = cy
        self.znear = znear
        self.zfar = zfar
        self.cv2gl = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self.matrix = projection_matrix(w, h, vfov, cx, cy, znear, zfar)

    def update(
        self, w=None, h=None, hfov=None, cx=None, cy=None, znear=None, zfar=None
    ):
        self.w = w or self.w
        self.h = h or self.h
        self.hfov = hfov or self.hfov
        self.cx = cx or self.cx
        self.cy = cy or self.cy
        self.znear = znear or self.znear
        self.zfar = zfar or self.zfar

        self.matrix = projection_matrix(
            self.w, self.h, self.hfov, self.cx, self.cy, self.znear, self.zfar
        )

    def gl_matrix(self):
        return self.matrix.T.copy()


class Camera:
    def __init__(self, proj_mat, T_CW):
        self.T_CW = T_CW
        self.proj_mat = proj_mat

        self.mouse_left_down, self.mouse_right_down = False, False
        self.drag_start_x, self.drag_start_y, self.drag_start_z = 0.0, 0.0, 0.8
        self.last_x, self.last_y, self.last_z = 0.0, 0.0, 0.8
        self.drag_start_T_CW = self.T_CW.copy()

    def resize(self, width, height):
        self.proj_mat.update(w=width, h=height, cx=width // 2, cy=height // 2)

    def gl_matrix(self):
        return self.T_CW.T.copy()

    @staticmethod
    def is_valid_depth(z):
        return z > 0.0 and z < 0.999

    def update(self, wnd):
        pix_ratio = wnd.pixel_ratio
        viewport = wnd.ctx.screen.viewport

        x, y = imgui.get_io().mouse_pos
        # set origin to bottom-left
        x, y = int(x), int(wnd.size[1] - y)
        left, right = (
            imgui.get_io().mouse_down[0],
            imgui.get_io().mouse_down[1],
        )
        scroll = imgui.get_io().mouse_wheel
        w = 31
        z = np.frombuffer(
            wnd.ctx.screen.read(
                attachment=-1,
                alignment=4,
                components=1,
                dtype="f4",
                viewport=(x * pix_ratio - w // 2, y * pix_ratio - w // 2, w, w),
            ),
            dtype="f4",
        )
        z = np.median(z)

        if not imgui.get_io().want_capture_mouse:
            if abs(scroll) > 0.1:
                self.zoom(scroll, x, y, viewport, pix_ratio)
            # only trigger after the initial left/right down so we can set the correct last/drag-start points.
            if self.mouse_left_down:
                self.pan(x, y, viewport, pix_ratio)
            if self.mouse_right_down:
                self.orbit(x, y, viewport, pix_ratio)

        drag_start_xy = left and not self.mouse_left_down
        drag_start_xy |= right and not self.mouse_right_down
        drag_start_xy |= not left and not right

        if drag_start_xy:
            self.drag_start_x, self.drag_start_y = x, y
            self.drag_start_T_CW = self.T_CW.copy()
        if drag_start_xy and (left or right) and self.is_valid_depth(z):
            self.drag_start_z = z

        if left or right:
            self.last_x, self.last_y = x, y
            self.last_z = z if self.is_valid_depth(z) else self.last_z

        self.mouse_left_down = left
        self.mouse_right_down = right

    def mouse_drag_delta(self, x, y):
        return x - self.drag_start_x, y - self.drag_start_y

    def mouse_delta(self, x, y, pix_ratio=1.0):
        return (x - self.last_x) * pix_ratio, (y - self.last_y) * pix_ratio

    def zoom(self, scroll, x, y, viewport, pix_ratio):
        view_mat = self.T_CW
        proj_mat = self.proj_mat.matrix
        inv_view_mat = np.linalg.inv(view_mat)

        pt_W = unproject(
            x * pix_ratio, y * pix_ratio, 1.0, view_mat, proj_mat, viewport
        )
        eye = inv_view_mat[:3, 3]
        ray = pt_W - eye
        ray /= np.linalg.norm(ray)

        inv_view_mat[:3, 3] += ray * scroll * 0.3
        self.T_CW = np.linalg.inv(inv_view_mat)

    def pan(self, x, y, viewport, pix_ratio):
        view_mat = self.T_CW
        proj_mat = self.proj_mat.matrix
        inv_view_mat = np.linalg.inv(view_mat)

        wnd_to = np.array([x * pix_ratio, y * pix_ratio, self.last_z])
        wnd_from = np.array(
            [
                self.last_x * pix_ratio,
                self.last_y * pix_ratio,
                self.last_z,
            ]
        )
        inv_view_mat = np.linalg.inv(view_mat)

        pt_from_W = unproject(*wnd_from, view_mat, proj_mat, viewport)
        pt_to_W = unproject(*wnd_to, view_mat, proj_mat, viewport)
        v_W = pt_to_W - pt_from_W
        inv_view_mat[:3, 3] -= v_W
        self.T_CW = np.linalg.inv(inv_view_mat)

    def orbit(self, x, y, viewport, pix_ratio):
        proj_mat = self.proj_mat.matrix
        view_mat = self.T_CW

        dx, dy = self.mouse_delta(x, y, pix_ratio)
        if np.abs(dx) > 0 or np.abs(dy) > 0:
            pivot = unproject(
                self.drag_start_x * pix_ratio,
                self.drag_start_y * pix_ratio,
                self.drag_start_z,
                self.drag_start_T_CW,
                proj_mat,
                viewport,
            )
            pivot = view_mat[:3, :3] @ pivot + view_mat[:3, 3]
            axis = np.array([-dy, dx, 0], dtype=np.float32)
            axis = axis / np.linalg.norm(axis)
            angle = np.linalg.norm(np.array([dx / viewport[2], dy / viewport[3]])) * 5.0
            # make into 4x4
            R = np.eye(4, dtype=np.float32)
            R[:3, :3] = exp_angle_axis(angle, axis)
            R = translation_matrix(pivot) @ R @ translation_matrix(-pivot)
            self.T_CW = R @ view_mat


class CameraHandler:
    def __init__(self, T_CW):
        self.T_CW = T_CW
        self.drag_start_x, self.drag_start_y, self.drag_start_z = 0.0, 0.0, 0.8
        self.last_x, self.last_y, self.last_z = 0.0, 0.0, 0.8
        self.drag_start_T_CW = self.T_CW.copy()
        self.is_following = False
        self.follow_T_CW = np.eye(4, dtype=np.float32)
        self.cv2gl = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self.zoom_factor = 0.15

    @staticmethod
    def is_valid_depth(z, has_focus):
        return z > 0 and z < 1 and has_focus

    def update(self, io_state, proj_mat):
        x, y, z, scroll, left, right, clicked, has_focus, size = io_state
        if clicked:
            self.drag_start_T_CW = self.T_CW.copy()
            self.drag_start_x, self.drag_start_y = x, y
            self.drag_start_z = (
                z if self.is_valid_depth(z, has_focus) else self.drag_start_z
            )
            self.last_x, self.last_y = x, y
            self.last_z = z if self.is_valid_depth(z, has_focus) else self.last_z
        if abs(scroll) > 0.0:
            self.last_z = z if self.is_valid_depth(z, has_focus) else self.last_z
            self.zoom(scroll, x, y, size, proj_mat)
        if left:
            self.pan(x, y, size, proj_mat)
        if right:
            dx, dy = self.mouse_delta(x, y)
            self.orbit(dx, dy, size, proj_mat)

        if left or right:
            self.last_x, self.last_y = x, y
            self.last_z = z if self.is_valid_depth(z, has_focus) else self.last_z

    def mouse_delta(self, x, y):
        return (x - self.last_x), (y - self.last_y)

    def zoom(self, scroll, x, y, viewport, proj_mat):
        view_mat = self.T_CW
        inv_view_mat = np.linalg.inv(view_mat)
        pt_W = unproject(x, y, self.last_z, view_mat, proj_mat, viewport)
        eye = inv_view_mat[:3, 3]
        ray = pt_W - eye
        ray_dist = np.linalg.norm(ray)
        ray = ray / ray_dist
        zoom_factor = self.zoom_factor * ray_dist
        # clamp zoom factor
        zoom_factor = np.clip(zoom_factor, 0.01, 10)

        inv_view_mat[:3, 3] += ray * scroll * zoom_factor
        self.T_CW = np.linalg.inv(inv_view_mat)

    def pan(self, x, y, viewport, proj_mat):
        view_mat = self.T_CW
        inv_view_mat = np.linalg.inv(view_mat)

        wnd_to = np.array([x, y, self.last_z])
        wnd_from = np.array([self.last_x, self.last_y, self.last_z])
        inv_view_mat = np.linalg.inv(view_mat)

        pt_from_W = unproject(*wnd_from, view_mat, proj_mat, viewport)
        pt_to_W = unproject(*wnd_to, view_mat, proj_mat, viewport)
        v_W = pt_to_W - pt_from_W
        inv_view_mat[:3, 3] -= v_W
        self.T_CW = np.linalg.inv(inv_view_mat)

    def orbit(self, dx, dy, viewport, proj_mat):
        view_mat = self.T_CW

        if np.abs(dx) > 0 or np.abs(dy) > 0:
            pivot = unproject(
                self.drag_start_x,
                self.drag_start_y,
                self.drag_start_z,
                self.drag_start_T_CW,
                proj_mat,
                viewport,
            )
            pivot = view_mat[:3, :3] @ pivot + view_mat[:3, 3]
            axis = np.array([-dy, dx, 0], dtype=np.float32)
            axis = axis / np.linalg.norm(axis)
            angle = np.linalg.norm(np.array([dx / viewport[2], dy / viewport[3]])) * 5.0
            # make into 4x4
            R = np.eye(4, dtype=np.float32)
            R[:3, :3] = exp_angle_axis(angle, axis)
            R = translation_matrix(pivot) @ R @ translation_matrix(-pivot)
            self.T_CW = R @ view_mat

    def follow_cam(self, T_CW):
        if not self.is_following:
            self.T_CW = np.eye(4, dtype=np.float32)
        self.is_following = True
        self.follow_T_CW = T_CW.reshape(4, 4)

    def unfollow_cam(self):
        if self.is_following:
            self.T_CW = self.T_CW @ self.follow_T_CW
        self.is_following = False


class Camera:
    def __init__(self, proj_mat, T_CW=None, handler=None):
        self.proj_mat = proj_mat
        if handler is None:
            handler = CameraHandler(
                np.eye(4, dtype=np.float32) if T_CW is None else T_CW
            )
        self.handler = handler

    def resize(self, width, height):
        self.proj_mat.update(w=width, h=height, cx=width // 2, cy=height // 2)

    def gl_matrix(self):
        return self.T_CW.T.copy()

    @staticmethod
    def is_valid_depth(z, has_focus):
        print(z)
        return z > 1e-3 and z < 1 and has_focus

    def update(self, io_state):
        # x, y, z, scroll, left, right, clicked, has_focus, size = io_state
        self.handler.update(io_state, self.proj_mat.matrix)

    @property
    def viewport_size(self):
        return self.proj_mat.w, self.proj_mat.h

    @property
    def T_CW(self):
        if self.handler.is_following:
            return self.handler.T_CW @ self.handler.follow_T_CW
        return self.handler.T_CW

    @T_CW.setter
    def T_CW(self, T_CW):
        self.handler.T_CW = T_CW

    def follow_cam(self, T_CW):
        self.handler.follow_cam(self.handler.cv2gl @ T_CW)

    def unfollow_cam(self):
        self.handler.unfollow_cam()
