import pathlib
import time

import numpy as np
import pypose as pp
from tqdm import tqdm
import torch
import json
from VSLAM.CameraTracker import CameraTracker
import VSLAM.mast3r_slam.evaluate as eval
from VSLAM.ImageFrame import Mode, ImageFrame
from VSLAM.utils_mast3r import load_mast3r

class Frontend:
    @torch.no_grad()
    def __init__(self, args, config,dataset, keyframes, states, device="cuda:0"):
        self.args = args
        self.config = config
        self.use_calib = config["use_calib"]
        self.device = device

        # args: camera intrinsics for slam
        self.dataset = dataset
        self.H_slam, self.W_slam = dataset.H_slam, dataset.W_slam

        # Shared variables
        self.keyframes = keyframes  # shared
        self.states = states  # shared
        self.model = load_mast3r(device=self.device)  # shared
        self.model.share_memory()

        # initialize tracker and backend
        self.min_displacement = max(self.args.min_displacement * self.W_slam, 30)
        self.tracker = CameraTracker(args, self.config, self.min_displacement, args.thres_keyframe,
                                     self.model, self.keyframes,
                                     self.H_slam, self.W_slam, dataset.K_slam.clone().to(device),
                                     self.device)

        # class domain variable
        self.frames_info = []
        self.frames_Twc_gt = []


    def init_pose_per_frame(self, current_idx, Twc_gt=None):
        if current_idx == 0:
            T_WC = pp.identity_Sim3(1, device=self.device)
        else:
            T_WC = pp.Sim3(self.states.T_WC).to(self.device)
        if Twc_gt is not None:
            T_WC.data[:, :7] = Twc_gt.data[:, :7]
        T_WC = pp.quat2unit(T_WC)
        return T_WC

    @torch.no_grad()
    def run(self):
        frameID = 0
        bar = tqdm(total=len(self.dataset), desc=f"Frontend: ", position=0, leave=True)
        while True:
            # 1. read data
            try:
                original_image, info = self.dataset.getnext()
            except Exception as e:
                break
            is_test = info["is_test"]
            timestamp = info["timestamp"]
            img_slam = self.dataset.transform.to_slam(original_image, device=self.device)

            Twc_gt = None
            if "Twc_gt" in info:
                x, y, z, qx, qy, qz, qw = info["Twc_gt"]
                self.frames_Twc_gt.append([timestamp, x, y, z, qx, qy, qz, qw])
                if self.args.use_gt_pose:
                    Twc_gt = pp.Sim3(torch.tensor([[x, y, z, qx, qy, qz, qw, 1]]))
                    Twc_gt = pp.quat2unit(Twc_gt).to(self.device)
            T_WC = self.init_pose_per_frame(frameID, Twc_gt)
            frame = ImageFrame(frameID, 0, float(timestamp), img_slam, T_WC, K=self.tracker.K_slam)

            # 2. Track
            with torch.cuda.device(self.device):
                lost, is_keyframe, is_keyframe_map = self.tracker.track(frame)
            if self.args.use_same_set_of_keyframes:
                is_keyframe = is_keyframe or is_keyframe_map

            # 3. Add keyframe
            keyframe_style = -1
            if lost:
                # 0 Lost
                self.states.lost_number.value += 1
                keyframe_style = 0
            elif is_keyframe:
                # 1 是keyframe且是keyframe_map
                self.keyframes.append(frame)
                keyframe_style = 1
            elif is_keyframe_map or is_test or self.args.use_all_frames:
                # 2 是keyframe_map
                keyframe = self.keyframes.last_keyframe().to(self.device)
                self.store_relative_pose(frame, len(self.keyframes) - 1, keyframe.T_WC.Inv().mul(frame.T_WC))
                keyframe_style = 2
            else:
                keyframe = self.keyframes.last_keyframe().to(self.device)
                self.store_relative_pose(frame, len(self.keyframes) - 1, keyframe.T_WC.Inv().mul(frame.T_WC))

            # 4. send to backend
            if keyframe_style != -1:
                keyframe_dict = {
                    "keyframe_style": keyframe_style,
                    "is_important": is_keyframe_map or is_test,
                    "is_test": is_test,
                    "keyframe_id": len(self.keyframes) - 1,
                    "frame_id": frame.frame_id,
                    "T_WC": frame.T_WC.data.cpu(),
                    "focal": self.tracker.K_slam.cpu()[0,0].item()
                }
                self.states.msg2Backend(keyframe_dict)
                self.states.queue_backend_execute()
            self.states.set_frame(frame)

            # 5. 等待后端对当前帧优化完成
            if self.args.sync_hard:
                while True:
                    with self.states.lock:
                        if self.states.backend_execute.value == 0:
                            break
                    time.sleep(0.001)


            frameID += 1
            bar.update(1)

        self.states.set_mode(Mode.OPTIMIZING)
        while self.states.get_mode() != Mode.TERMINATED:
            time.sleep(0.1)
        self.sav_results(self.args.model_path)

    def store_relative_pose(self, frame, index_keyframe, Tckc):
        self.frames_info.append([frame.frame_id, frame.frame_time, index_keyframe, Tckc])

    def sav_results(self, path):
        print("Saving the poses and point clouds...")
        save_dir = pathlib.Path(path).joinpath("slam")
        save_dir.mkdir(exist_ok=True, parents=True)

        lost_percentage = self.states.lost_number.value / len(self.dataset)
        with open(save_dir / f"lost_percentage.txt", 'w') as f:
            f.write(str(lost_percentage))

        # save config
        configFile = save_dir / f"config.json"
        with open(configFile, 'w') as f:
            json.dump(self.config, f, indent=4)

        # save pose
        Twc_est, Twc_est_keyframe = eval.save_traj(save_dir, self.keyframes, self.frames_info)
        if len(self.frames_Twc_gt) > 0:
            self.frames_Twc_gt = np.array(self.frames_Twc_gt)
            eval.evaluate_trajectory(save_dir, "evaluate_frames.json", Twc_est, self.frames_Twc_gt)
            eval.evaluate_trajectory(save_dir, "evaluate_keyframes.json", Twc_est_keyframe, self.frames_Twc_gt)

            with open(save_dir.joinpath("gt_pose.txt"), "w") as f:
                for i in range(len(self.frames_Twc_gt)):
                    t, x, y, z, qx, qy, qz, qw = self.frames_Twc_gt[i]
                    f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")

        # save point cloud
        if self.args.save_point_could:
            eval.save_keyframe(save_dir, self.keyframes,1.5, device=self.device, use_calib=self.use_calib)
        # eval.save_frame_wise_points(
        #     save_dir,
        #     self.frames,
        #     self.last_msg.C_conf_threshold, use_calib=self.use_calib)
        print("Saved!")