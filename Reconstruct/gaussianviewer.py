#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import numpy as np
from argparse import ArgumentParser, Namespace
from imgui_bundle import imgui_ctx, imgui
from enum import IntEnum, auto
import time

from graphdecoviewer import Viewer
from graphdecoviewer.types import ViewerMode
from graphdecoviewer.widgets.image import TorchImage
from graphdecoviewer.widgets.radio import RadioPicker
from graphdecoviewer.widgets.cameras.fps import FPSCamera
from graphdecoviewer.widgets.ellipsoid_viewer import EllipsoidViewer

class Dummy(object):
    pass

class SnapMode(IntEnum):
    free = auto()
    keyframe = auto()
    last = auto()

class GaussianViewer(Viewer):
    atttrs_to_sync = [
            "render_mode_id", "draw_poses", "draw_gt_poses", "pose_sizes",
            "draw_anchors", "anchor_sizes", "scaling_factor", "bg_color",
            "show_top_view", "keyframe_id", "altitude_control", "altitude_smoothing",
            "snap_to_closest", "next_keyframe", "prev_keyframe", "reset_intrinsics_flag",
        ]

    def __init__(self, mode: ViewerMode, device="cuda:0"):
        super().__init__(mode)
        self.window_title = "Gaussian Viewer"
        self.throttling = False
        self.device = device

    def import_server_modules(self):
        global torch
        import torch

        global cv2
        import cv2

        global SceneModel
        from Reconstruct.scene.scene_models.h3dgs import SceneModel

        global draw_poses, draw_anchors
        from Reconstruct.utils import draw_poses, draw_anchors # TODO: move this to viewer?

    @classmethod
    def from_scene(cls, scene_dir: str, mode: ViewerMode, args: Namespace):
        viewer = cls(mode)
        viewer.scene_model = SceneModel.from_scene(scene_dir, args)
        return viewer
    
    @classmethod
    def from_scene_model(cls, scene_model: 'SceneModel', mode: ViewerMode, device="cuda:0"):
        viewer = cls(mode, device=device)
        viewer.scene_model = scene_model
        return viewer

    def create_widgets(self):
        if self.mode is not ViewerMode.CLIENT:
            width = self.scene_model.width
            height = self.scene_model.height
            fov_y = np.rad2deg(self.scene_model.FoVy)
            self.num_keyframes = len(self.scene_model.keyframes)
        else:
            width, height, fov_y = 2, 2, 1
            self.num_keyframes = 0
        self.point_view_camera = FPSCamera(self.mode, width, height, fov_y, 0.01, 100)
        self.top_view_camera = FPSCamera(self.mode, 480, 480, 60, 0.01, 100,
            to_world=np.array([[-1, 0, 0, 0],
                               [0, 0.7071, 0.7071, -3],
                               [0, -0.7071, 0.7071, -2],
                               [0, 0, 0, 1]])
        )
        self.cameras = {"top_view": self.top_view_camera, "point_view": self.point_view_camera}
        self.point_view = TorchImage(self.mode)
        self.top_view = TorchImage(self.mode)
        self.views = {"top_view": self.top_view, "point_view": self.point_view}
        self.ellipsoid_viewer = EllipsoidViewer(self.mode)

        # Render modes
        self.render_modes = ["Splats", "Depth", "Ellipsoids"]
        self.render_mode_id = 0

        # Render settings
        views = ["top_view", "point_view"]
        self.draw_poses = {view: False for view in views}
        self.draw_gt_poses = {view: False for view in views}
        self.pose_sizes = {view: 0.1 for view in views}
        self.draw_anchors = {view: False for view in views}
        self.anchor_sizes = {view: 0.1 for view in views}
        self.scaling_factor = {"top_view": 0.002, "point_view": 1}
        self.reset_intrinsics_flag = {view: False for view in views}
        self.bg_color = [0.0, 0.0, 0.0]
        self.show_top_view = False
        self.max_fps = 20
        self.last_show_gui_time = time.time()

        # Camera settings
        self.keyframe_id = 0
        self.reset_pose = False
        self.altitude_control = False
        self.altitude_smoothing = 0.9
        self.snap_to_closest = False
        self.snap_mode = RadioPicker(ViewerMode.LOCAL, SnapMode.free)
        self.next_keyframe = False
        self.prev_keyframe = False
        self.updated_pose = None
    
    def render_mode(self):
        return self.render_modes[self.render_mode_id]

    def reset_intrinsics(self, view):
        camera = self.cameras[view]
        camera.res_x = self.scene_model.width // 2 if view == "top_view" else self.scene_model.width
        camera.fov_x = self.scene_model.FoVx
        camera.res_y = self.scene_model.height // 2 if view == "top_view" else self.scene_model.height
        camera.fov_y = self.scene_model.FoVy

    def onconnect(self, websocket):
        if self.mode == ViewerMode.SERVER:
            websocket.send(json.dumps({
                "num_keyframes": self.num_keyframes,
                "width": self.point_view_camera.res_x,
                "height": self.point_view_camera.res_y,
                "fov_y": self.point_view_camera.fov_y,
                "ellipsoid_enabled": self.ellipsoid_viewer.enabled,
            }), text=True)
        if self.mode == ViewerMode.CLIENT:
            data = json.loads(websocket.recv())
            self.num_keyframes = data["num_keyframes"]
            self.point_view_camera.res_x = data["width"]
            self.point_view_camera.res_y = data["height"]
            self.point_view_camera.fov_y = data["fov_y"]
            self.point_view_camera.compute_fov_x()
            self.ellipsoid_viewer.enabled = data["ellipsoid_enabled"]
    
    def step(self):
        # Get camera matrix
        self.updated_pose = None
        if self.snap_mode.value in [SnapMode.keyframe, SnapMode.last]:
            self.num_keyframes = len(self.scene_model.keyframes)
            if self.next_keyframe:
                self.keyframe_id = min(self.num_keyframes - 1, self.keyframe_id + 1)
            if self.prev_keyframe:
                self.keyframe_id = max(0, self.keyframe_id - 1)
            keyframe_id = self.keyframe_id if self.snap_mode.value == SnapMode.keyframe else -1
            point_viewmatrix = self.scene_model.keyframes[keyframe_id].get_Rt()
            self.updated_pose = torch.linalg.inv(point_viewmatrix.detach()).cpu().numpy()
        else:
            if self.altitude_control:
                camera_position = torch.tensor(self.point_view_camera.origin, dtype=torch.float32).to(self.device)
                n_closest = 4
                closest_keyframes = self.scene_model.get_closest_keyframe(camera_position, n_closest)
                mean_closest_altitude = (torch.stack([kf.approx_centre for kf in closest_keyframes]).sum(axis=0) / n_closest)[1]

                if abs(mean_closest_altitude - self.point_view_camera.origin[1]) > 1e-4:
                    dist = mean_closest_altitude - self.point_view_camera.origin[1]
                    to_world = self.point_view_camera.to_world.copy()
                    to_world[1, 3] += (1.0 - self.altitude_smoothing) * self.point_view_camera.speed * dist
                    self.updated_pose = to_world
            if self.snap_to_closest:
                camera_position = torch.tensor(self.point_view_camera.origin, dtype=torch.float32).to(self.device)
                closest_keyframe = self.scene_model.get_closest_keyframe(camera_position)[0]
                keyframe_pose = torch.linalg.inv(closest_keyframe.get_Rt()).detach().cpu().numpy()
                self.updated_pose = keyframe_pose

        # Render scene
        for view in ["point_view", "top_view"]:
            camera = self.cameras[view]
            if self.reset_intrinsics_flag[view]:
                self.reset_intrinsics(view)

            # Draw Gaussians
            if (view == "point_view" and self.render_mode() in ["Splats", "Depth"]) or (view == "top_view" and self.show_top_view):
                width = camera.res_x
                height = camera.res_y
                viewmatrix = torch.tensor(camera.to_camera, dtype=torch.float32).to(self.device).transpose(0, 1)
                with torch.cuda.device(self.device):
                    render_pkg = self.scene_model.render(width, height, viewmatrix, self.scaling_factor[view], torch.tensor(self.bg_color,
                                                                                                                        device=self.device), view=="top_view", camera.fov_x, camera.fov_y)
                if self.render_mode() == "Splats":
                    image = render_pkg["render"].clamp(0, 1.0).mul(255).permute(1, 2, 0).byte()
                elif self.render_mode() == "Depth":
                    image = render_pkg["invdepth"][0].mul(100).clamp(0, 255).byte().cpu().numpy()
                    image = cv2.cvtColor(cv2.applyColorMap(image, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
                    image = torch.tensor(image).to(self.device)

                # Draw overlays
                if self.draw_anchors[view] or self.draw_poses[view] or self.draw_gt_poses[view] or view == "top_view":
                    image = image.contiguous().cpu().numpy()
                    common_opt = (image, viewmatrix, camera.fov_x, self.pose_sizes[view], self.scene_model.width, self.scene_model.height)
                    if self.draw_gt_poses[view]:
                        image = draw_poses(*common_opt, self.scene_model.get_gt_Rts(True), self.scene_model.gt_f, (255, 0, 0))
                    if self.draw_poses[view]:
                        Rts = self.scene_model.get_Rts()
                        image = draw_poses(*common_opt, Rts, self.scene_model.f, (255, 255, 255))
                    if self.draw_anchors[view]:
                        image = draw_anchors(image, viewmatrix, camera.fov_x, self.anchor_sizes[view],
                                             self.scene_model.anchors, self.scene_model.anchor_weights)
                    if view == "top_view":
                        point_viewmatrix = torch.tensor(self.point_view_camera.to_camera, dtype=torch.float32).to(self.device)[None]
                        image = draw_poses(*common_opt, point_viewmatrix, self.scene_model.f, (0, 255, 255))
                    image = torch.tensor(image).to(self.device)

                # Update the buffer
                self.views[view].step(image)

            # Draw ellipsoids
            elif self.render_mode() == "Ellipsoids" and view == "point_view":
                self.ellipsoid_viewer.step(camera)

        # Upload to OpenGL only after first anchor has been loaded (will be done in first render call)
        if self.ellipsoid_viewer.num_gaussians != self.scene_model.n_active_gaussians:
            self.ellipsoid_viewer.upload(
                self.scene_model.xyz.detach().cpu().numpy(),
                self.scene_model.rotation.detach().cpu().numpy(),
                self.scene_model.scaling.detach().cpu().numpy(),
                self.scene_model.opacity.detach().cpu().numpy(),
                self.scene_model.f_dc.detach().cpu().numpy()
            )


    def show_gui(self):
        if self.updated_pose is not None:
            self.point_view_camera.update_pose(self.updated_pose)
            self.updated_pose = None

        with imgui_ctx.begin(f"Point View Settings"):
            render_modes = self.render_modes.copy()
            if not self.ellipsoid_viewer.enabled:
                render_modes.remove("Ellipsoids")

            _, self.render_mode_id = imgui.list_box("Render Mode", self.render_mode_id, render_modes)

            imgui.separator_text("Render Settings")
            if self.render_mode() in ["Splats", "Depth"]:
                _, self.scaling_factor["point_view"] = imgui.slider_float("Scaling Factor", self.scaling_factor["point_view"], 1e-2, 1)
                _, self.draw_poses["point_view"] = imgui.checkbox("Draw Poses", self.draw_poses["point_view"])
                _, self.draw_gt_poses["point_view"] = imgui.checkbox("Draw GT Poses", self.draw_gt_poses["point_view"])
                if self.draw_poses["point_view"] or self.draw_gt_poses["point_view"]:
                    _, self.pose_sizes["point_view"] = imgui.drag_float("Pose Sizes", self.pose_sizes["point_view"], 0.01, 0, 1e8, "%.2f")
                _, self.draw_anchors["point_view"] = imgui.checkbox("Draw Anchors", self.draw_anchors["point_view"])
                if self.draw_anchors["point_view"]:
                    _, self.anchor_sizes["point_view"] = imgui.drag_float("Anchor Sizes", self.anchor_sizes["point_view"], 0.01, 0, 1e8, "%.2f")
            if self.render_mode() == "Ellipsoids":
                _, self.ellipsoid_viewer.scaling_modifier = imgui.drag_float("Scaling Factor", self.ellipsoid_viewer.scaling_modifier, v_min=0, v_max=10, v_speed=0.01)
                
                _, self.ellipsoid_viewer.render_floaters = imgui.checkbox("Render Floaters", self.ellipsoid_viewer.render_floaters)
                _, self.ellipsoid_viewer.limit = imgui.drag_float("Alpha Threshold", self.ellipsoid_viewer.limit, v_min=0, v_max=1, v_speed=0.01)
            _, self.throttling = imgui.checkbox("Throttling", self.throttling)
            if self.throttling:
                _, self.max_fps = imgui.slider_int("Max FPS", self.max_fps, 2, 60)
            _, self.bg_color = imgui.color_edit3("Background Color", self.bg_color)

            imgui.separator_text("Camera Settings")
            self.snap_mode.show_gui()
            if self.snap_mode.value == SnapMode.free:
                self.reset_pose = imgui.button("Reset Pose")
                imgui.same_line()
                self.snap_to_closest = imgui.button("Snap to Closest")
                self.snap_to_closest |= imgui.is_key_pressed(imgui.Key.p)
            if self.snap_mode.value == SnapMode.keyframe:
                imgui.text("Keyframe ID")
                self.prev_keyframe = imgui.button("-")
                imgui.same_line()
                _, self.keyframe_id = imgui.slider_int("##", self.keyframe_id, 0, max(0, self.num_keyframes - 1))
                imgui.same_line()
                self.next_keyframe = imgui.button("+")
            imgui.separator()
            self.point_view_camera.show_gui()
            _, self.altitude_control = imgui.checkbox("Altitude Control", self.altitude_control)
            if self.altitude_control:
                _, self.altitude_smoothing = imgui.slider_float("smoothing", self.altitude_smoothing, 0.9, 0.9999)
            self.reset_intrinsics_flag["point_view"] = imgui.button("Reset Intrinsics")

        with imgui_ctx.begin("Point View"):
            if self.render_mode() in ["Splats", "Depth"]:
                self.point_view.show_gui()
            else:
                self.ellipsoid_viewer.show_gui()

            if imgui.is_item_hovered():
                self.point_view_camera.process_mouse_input()
            
            if imgui.is_item_focused() or imgui.is_item_hovered():
                self.point_view_camera.process_keyboard_input()

        if self.show_top_view:
            with imgui_ctx.begin("Top View"):
                self.top_view.show_gui()

                if imgui.is_item_hovered():
                    self.top_view_camera.process_mouse_input()
                
                if imgui.is_item_focused() or imgui.is_item_hovered():
                    self.top_view_camera.process_keyboard_input()
            
        with imgui_ctx.begin("Top View Settings"):
            _, self.show_top_view = imgui.checkbox("Show Top View", self.show_top_view)
            if self.show_top_view:
                imgui.separator_text("Render Settings")
                _, self.scaling_factor["top_view"] = imgui.slider_float("Scaling Factor", self.scaling_factor["top_view"], 1e-3, 1e-2)
                _, self.draw_poses["top_view"] = imgui.checkbox("Draw Poses", self.draw_poses["top_view"])
                _, self.draw_gt_poses["top_view"] = imgui.checkbox("Draw GT Poses", self.draw_gt_poses["top_view"])
                if self.draw_poses["top_view"] or self.draw_gt_poses["top_view"]:
                    _, self.pose_sizes["top_view"] = imgui.drag_float("Pose Sizes", self.pose_sizes["top_view"], 0.01, 0, 1e8, "%.2f")
                _, self.draw_anchors["top_view"] = imgui.checkbox("Draw Anchors", self.draw_anchors["top_view"])
                if self.draw_anchors["top_view"]:
                    _, self.anchor_sizes["top_view"] = imgui.drag_float("Anchor Sizes", self.anchor_sizes["top_view"], 0.01, 0, 1e8, "%.2f")

                imgui.separator_text("Camera Settings")
                self.top_view_camera.show_gui()
                self.reset_intrinsics_flag["top_view"] = imgui.button("Reset Intrinsics")

        if self.reset_pose:
            self.point_view_camera.update_pose(np.eye(4))
        
        # Throttling
        if self.throttling:
            elapsed = time.time() - self.last_show_gui_time
            if elapsed < 1 / self.max_fps:
                time.sleep(1 / self.max_fps - elapsed)

        self.last_show_gui_time = time.time()

    def server_send(self):
        to_send = {
            "num_keyframes": self.num_keyframes,
            "keyframe_id": self.keyframe_id,
            "res_x": {view: camera.res_x for view, camera in self.cameras.items()},
            "res_y": {view: camera.res_y for view, camera in self.cameras.items()},
            "fov_x": {view: camera.fov_x for view, camera in self.cameras.items()},
            "fov_y": {view: camera.fov_y for view, camera in self.cameras.items()},
        }
        if self.updated_pose is not None:
            to_send["updated_pose"] = self.updated_pose.tolist()
        return None, to_send
    
    def client_recv(self, _, text):
        self.num_keyframes = text["num_keyframes"]
        self.keyframe_id = text["keyframe_id"]
        if "updated_pose" in text:
            self.updated_pose = np.array(text["updated_pose"])
        for view in self.cameras:
            self.cameras[view].res_x = text["res_x"][view]
            self.cameras[view].res_y = text["res_y"][view]
            self.cameras[view].fov_x = text["fov_x"][view]
            self.cameras[view].fov_y = text["fov_y"][view]
        
    def client_send(self):
        attrs = {key: getattr(self, key) for key in GaussianViewer.atttrs_to_sync}
        return None, attrs
    
    def server_recv(self, _, text):
        for attr in GaussianViewer.atttrs_to_sync:
            if attr in text:
                setattr(self, attr, text[attr])
    
if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="mode", dest="mode", required=True)
    local = subparsers.add_parser("local")
    local.add_argument("scene_dir")
    local.add_argument('--anchor_overlap', type=float, default=0.3)
    client = subparsers.add_parser("client")
    client.add_argument("--ip", default="localhost")
    client.add_argument("--port", type=int, default=6009)
    server = subparsers.add_parser("server")
    server.add_argument("scene_dir")
    server.add_argument('--anchor_overlap', type=float, default=0.3)
    server.add_argument("--ip", default="localhost")
    server.add_argument("--port", type=int, default=6009)
    args = parser.parse_args()

    match args.mode:
        case "local":
            mode = ViewerMode.LOCAL
        case "client":
            mode = ViewerMode.CLIENT
        case "server":
            mode = ViewerMode.SERVER

    if mode is ViewerMode.CLIENT:
        viewer = GaussianViewer(mode)
    else:
        viewer = GaussianViewer.from_scene(args.scene_dir, mode, args)

    if args.mode in ["client", "server"]:
        viewer.run(args.ip, args.port)
    else:
        viewer.run()
