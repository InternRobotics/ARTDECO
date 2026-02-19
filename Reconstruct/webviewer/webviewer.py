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

import cv2
import json
import time
import torch
from Reconstruct.utils import fov2focal, focal2fov
from threading import Thread
from Reconstruct.webviewer.scene_models import SceneModel
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve, ServerConnection


class WebViewer:
    def __init__(self, scene_model: SceneModel, ip="0.0.0.0", port=6009):
        self.ip = ip
        self.port = port
        self.num_clients = 0
        self.scene_model = scene_model
        self.state = "stop"
        self.trainer_state = "disconnected"
    
    def run(self):
        with serve(self.main, self.ip, self.port, max_size=None, compression=None) as server:
            server_thread = Thread(target=server.serve_forever)
            server_thread.start()
            while True:
                try:
                    time.sleep(1)  # Keep the main thread alive
                except KeyboardInterrupt:
                    break
                pass

    def main(self, websocket: ServerConnection):
        if self.num_clients >= 1:
            print("Only one client supported at a time.")
            while self.num_clients >= 1:
                time.sleep(1)

        print("Client connected.")
        self.num_clients += 1
        self.state = "stop"
        while True:
            try:
                try:
                    cam_centers = self.scene_model.approx_cam_centres
                    cam_centers[:, 1] *= -1  # Flip Y
                    cam_centers[:, 2] *= -1  # Flip Z
                    max_pos = cam_centers.max(0)[0].cpu().numpy().tolist()
                    min_pos = cam_centers.min(0)[0].cpu().numpy().tolist()
                    mean_pose = self.scene_model.keyframes[len(self.scene_model.keyframes) // 2].get_Rt().detach()
                    mean_pose = torch.linalg.inv(mean_pose)
                    mean_pose[:3, 1] *= -1  # Flip Y
                    mean_pose[:3, 2] *= -1  # Flip Z
                    mean_pose = mean_pose.cpu().numpy().flatten().tolist()
                except (AttributeError,TypeError):
                    max_pos = [0.0, 0.0, 0.0]
                    min_pos = [0.0, 0.0, 0.0]
                    mean_pose = [1.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 1.0, 0.0,
                                 0.0, 0.0, 0.0, 1.0]
                websocket.send(json.dumps({
                    "trainer_state": self.trainer_state,
                    "max_pos": max_pos,
                    "min_pos": min_pos,
                    "mean_pose": mean_pose
                }))
                
                # Receive state from client
                data = json.loads(websocket.recv())
                self.state = data["state"]
                res_x = data["res_x"] // 2
                res_y = data["res_y"] // 2
                fov_y = self.scene_model.FoVy
                focal = fov2focal(fov_y, res_y)
                fov_x = focal2fov(focal, res_x)
                
                if data["snapToLast"]:
                    if len(self.scene_model.keyframes) == 0:
                        pose = torch.eye(4).cuda()
                    else:
                        pose = self.scene_model.keyframes[-1].get_Rt()
                else:
                    pose = torch.tensor(data["pose"], dtype=torch.float32).cuda().reshape(4, 4) # CM,C2W
                    pose = pose.transpose(0, 1) # RM,C2W
                    pose[:3,1] *= -1 # Flip Y
                    pose[:3,2] *= -1 # Flip Z
                    pose = torch.linalg.inv(pose) # RM,W2C
                pose = pose.transpose(0, 1) # CM,W2C

                # Render image and send it to client
                render_pkg = self.scene_model.render(res_x, res_y, pose, 1, fov_x=fov_x, fov_y=fov_y)
                image = render_pkg["render"]
                image = image.clamp(0, 1.0).mul(255).permute(1, 2, 0).byte().detach().cpu().numpy()

                _, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 50])
                websocket.send(buffer.tobytes())
            except ConnectionClosed:
                print("Client disconnected.")
                self.num_clients -= 1
                break
