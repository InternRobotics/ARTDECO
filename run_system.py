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

import os
from queue import Empty
import shutil
import pypose as pp
import torch.nn.functional as F
from Reconstruct.scene.keyframe import Keyframe
from VSLAM.Backend import Backend
from VSLAM.ImageFrame import Mode
from VSLAM.SharedKeyframes import SharedKeyframes
from VSLAM.SharedStates import SharedStates
from Reconstruct.gaussianviewer import GaussianViewer
from VSLAM.mast3r_slam.visualization import run_visualization
from VSLAM.utils_mp import new_queue
from dataloaders.utils_load import load_dataset
from VSLAM.utils_config import load_config
from VSLAM.Frontend import Frontend
from plyfile import PlyElement, PlyData

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import time

import numpy as np
import torch
from tqdm import tqdm
from socketserver import TCPServer
from http.server import SimpleHTTPRequestHandler
from Reconstruct.gaussianviewer import GaussianViewer
from Reconstruct.webviewer.webviewer import WebViewer
from dataloaders.args import get_args
from threading import Thread
from graphdecoviewer.types import ViewerMode
import torch.multiprocessing as mp
import os

import scipy
from PIL import Image
from e3nn.o3 import matrix_to_angles, wigner_D

def _inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
    return torch.log(tensor / (1.0 - tensor))

def convert_rgb_to_spherical_harmonics(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to degree-0 spherical harmonics.

    Reference:
        https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    """
    coeff_degree0 = np.sqrt(1.0 / (4.0 * np.pi))
    return (rgb - 0.5) / coeff_degree0

def sixD2mtx(r):
    b1 = r[..., 0]
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)
    b2 = r[..., 1] - torch.sum(b1 * r[..., 1], dim=-1, keepdim=True) * b1
    b2 = b2 / torch.norm(b2, dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True

    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    args = get_args()
    config = load_config(args.config)
    device_frontend = args.device_frontend
    device_backend = args.device_backend
    device_mapper = args.device_mapper
    device_shared = args.device_shared

    dataset = load_dataset(args)
    H_map, W_map = dataset.H_map, dataset.W_map
    H_slam, W_slam = dataset.H_slam, dataset.W_slam
    K_slam, K_map = dataset.K_slam, dataset.K_map
    lc_count = {'big': 0, 'small': 0}

    # 2. Shared variables
    manager = mp.Manager()
    main2viz = new_queue(manager, True)  # shared
    viz2main = new_queue(manager, True)  # shared
    keyframes = SharedKeyframes(config, manager, H_slam, W_slam, K_slam, device=device_shared)  # shared
    states = SharedStates(manager, H_slam, W_slam, device=device_shared)  # shared

    # 3. Initialize slam system
    frontend = Frontend(args, config, dataset, keyframes, states, device=device_frontend)
    backend = Backend(args, config, dataset,
                      H_slam, W_slam, K_slam,
                      states, keyframes,
                      model=frontend.model if device_frontend == device_backend else None,
                      device=device_backend)
    backend = mp.Process(target=backend.run)
    backend.start()

    print(f"Using dataset: {args.source_path}")
    frontend = mp.Process(target=frontend.run)
    frontend.start()

    # 4. Initialize the viewer
    modules = __import__('Reconstruct.scene.scene_models.'+args.base_model, fromlist=[''])
    scene_model = getattr(modules, "SceneModel")(W_map, H_map, K_map.to(device_backend), args, device=device_mapper)
    if args.viewer_mode in ["server", "local"]:
        viewer_mode = ViewerMode.SERVER if args.viewer_mode == "server" else ViewerMode.LOCAL
        viewer = GaussianViewer.from_scene_model(scene_model, viewer_mode, device = device_mapper)
        viewer_thd = Thread(target=viewer.run, args=(args.ip, args.port), daemon=True)
        viewer_thd.start()
        viewer.throttling = True # Enable throttling when training
    elif args.viewer_mode == "web":
        ip = "0.0.0.0"
        server = TCPServer((ip, 8000), SimpleHTTPRequestHandler)
        server_thd = Thread(target=server.serve_forever, daemon=True)
        server_thd.start()
        print(f"Visit http://{ip}:8000/webviewer to for the viewer")
        viewer = WebViewer(scene_model, args.ip, args.port)
        viewer_thd = Thread(target=viewer.run, daemon=True)
        viewer_thd.start()
    else:
        viewer = None

    # 5. Track and Reconstruct
    # Dict of runtimes for each step
    runtimes = ["Load", "BAB", "tri", "BAI", "Add", "Init", "Opt", "anc"]
    runtimes = {key: [0, 0] for key in runtimes}
    metrics = {}
    pbar = tqdm(total=0, desc=f"GS Mapper: ", position=1, leave=False)
    reconstruction_start_time = time.time()
    mapper_index = 0
    important_idx = 0
    related_frames = {}
    while True:
        mode = states.get_mode()
        try:
            keyframe_map_dict = states.msgFromBackend()
        except Empty:
            time.sleep(0.001)
            if mode == Mode.TERMINATED:
                break
            continue
        
        # construct keyframe
        frame_id = keyframe_map_dict["frame_id"]
        last_keyframe_index = keyframe_map_dict["last_keyframe_index"]
        last_keyframe_frame_id = keyframe_map_dict.get("last_keyframe_frame_id", None)
        if last_keyframe_index not in related_frames:
            related_frames[last_keyframe_index] = []
        related_frames[last_keyframe_index].append(mapper_index)
        
        T_CkC = keyframe_map_dict["T_CkC"]
        T_CkC = T_CkC.to(device_mapper) if T_CkC is not None else None
        original_img, info = dataset[frame_id]
        T_WC = pp.Sim3(keyframe_map_dict["T_WC"]).to(device_mapper)
        densePoint = keyframe_map_dict["densePoint"].to(device_mapper)
        is_slam_keyframe = keyframe_map_dict["is_slam_keyframe"]
        is_important = keyframe_map_dict["is_important"]
        is_test = keyframe_map_dict["is_test"]
        point_map = densePoint[...,:3].to(device_mapper) #  H W 3
        point_map_conf = densePoint[...,3].to(device_mapper) # H W
        Twc = pp.SE3(T_WC.data[:, :7])
        Twc = pp.quat2unit(Twc)
        Tcw = Twc.Inv().matrix()[0]

        image_for_mapper = dataset.transform.to_map(original_img, device=device_mapper)

        keyframe_map = Keyframe(image_for_mapper,
                                info["name"],
                                is_test,
                                Tcw,
                                mapper_index,
                                frame_id,
                                last_keyframe_index,
                                last_keyframe_frame_id,
                                keyframe_map_dict["is_slam_keyframe"],
                                torch.tensor([K_map[0, 0].item()]).to(device_mapper),
                                args,
                                T_CkCf=T_CkC,
                                point_map=point_map.to(device_mapper),
                                point_conf=point_map_conf.to(device_mapper),  # 1 1 H W
                                device_mapper=device_mapper
                                )

        if is_slam_keyframe and frame_id > 0:
            old_c2ws = torch.zeros(len(scene_model.keyframes), 4, 4).to(device_mapper)
            new_c2ws = torch.zeros(len(scene_model.keyframes), 4, 4).to(device_mapper)
            cam_centres = torch.zeros(len(scene_model.keyframes), 3).to(device_mapper)
            for index in related_frames.keys():
                for mapper_frame_id in related_frames.get(index, []):
                    if mapper_frame_id == len(scene_model.keyframes):
                        continue
                    frame = scene_model.keyframes[mapper_frame_id]
                    
                    if frame.is_slam_keyframe:
                        frame_slam = keyframes[frame.last_keyframe_index]
                        T_WCk = pp.SE3(frame_slam.T_WC.data[:, :7])
                        T_WCk = pp.quat2unit(T_WCk).to(device_mapper)
                        T_WCf = T_WCk
                    else:
                        frame_slam = keyframes[frame.last_keyframe_index]
                        T_WCk = frame_slam.T_WC.to(device_mapper)
                        T_WCf = T_WCk.mul(frame.T_CkCf)
                        T_WCf = pp.SE3(pp.quat2unit(T_WCf).data[:, :7])
                    new_Rt = T_WCf.Inv().matrix()[0]
                    old_Rt = frame.get_Rt()
                    
                    frame.set_Rt(new_Rt.to(old_Rt.device))
                    view_matrix = frame.get_Rt().transpose(0, 1).to(device_mapper)
                    centre = view_matrix.detach().inverse()[3, :3].to(device_mapper)
                    old_c2ws[mapper_frame_id] = torch.linalg.inv(old_Rt).to(device_mapper)
                    new_c2ws[mapper_frame_id] = torch.linalg.inv(new_Rt).to(device_mapper)
                    cam_centres[mapper_frame_id] = centre
            
            old_c2ws = old_c2ws.squeeze(1)
            new_c2ws = new_c2ws.squeeze(1)
            cam_centres = cam_centres.squeeze(1)
            scene_model.rigid_transform_gs(old_c2ws, new_c2ws, cam_centres)

        with torch.cuda.device(device_mapper):
            scene_model.add_keyframe(keyframe_map)
            if is_important:
                scene_model.add_new_gaussians()
            num_iterations = args.num_key_iterations if is_important else args.num_common_iterations
            scene_model.optimization_loop(num_iterations, is_important)
            
            ## Intermediate evaluation
            if (
                len(scene_model.keyframes) % args.test_frequency == 0
                and args.test_frequency > 0
                and (args.test_hold > 0)
            ):
                metrics = scene_model.evaluate()

            ## Save intermediate model
            if (
                mapper_index % args.save_every == 0
                and args.save_every > 0
            ):
                scene_model.save(
                    os.path.join(args.model_path, "progress", f"{mapper_index:05d}")
                )

            ## Display optimization progress and metrics
            bar_postfix = []
            for key, value in metrics.items():
                bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
            if args.display_runtimes:
                for key, value in runtimes.items():
                    if value[1] > 0:
                        bar_postfix += [
                            f"\033[35m{key}:{1000 * value[0] / value[1]:.1f}\033[0m"
                        ]
            bar_postfix += [
                # f"\033[36mFocal:{focal:.1f}",
                f"\033[36mKeyframes:{len(scene_model.keyframes)}\033[0m",
                f"\033[36mGaussians:{scene_model.n_active_gaussians}\033[0m",
            ]
            pbar.set_postfix_str(",".join(bar_postfix), refresh=False)
        mapper_index += 1
        pbar.n = frame_id
        pbar.refresh()

    if len(scene_model.keyframes) <= 0:
        time.sleep(5)
        exit(0)
    reconstruction_time = time.time() - reconstruction_start_time

    # Set to inference mode so that the model can be rendered properly
    scene_model.enable_inference_mode()

    # Save the model and metrics
    with torch.cuda.device(device_mapper):
        print("Saving the reconstruction to:", args.model_path)
        metrics = scene_model.save(args.model_path, reconstruction_time, len(dataset))
        # Fine tuning after initial reconstruction
        if len(args.save_at_finetune_epoch) > 0 or len(args.save_at_finetune_iteration) > 0:
            if len(args.save_at_finetune_epoch) > 0:
                finetune_epochs = max(args.save_at_finetune_epoch)
            else:
                finetune_epochs = int(max(args.save_at_finetune_iteration) / mapper_index)
                args.save_at_finetune_epoch = [finetune_epochs]
            torch.cuda.empty_cache()
            scene_model.inference_mode = False
            pbar = tqdm(range(0, finetune_epochs), desc="Fine tuning")
            for epoch in pbar:
                # Run one epoch of fine-tuning
                epoch_start_time = time.time()
                scene_model.finetune_epoch()
                epoch_time = time.time() - epoch_start_time
                reconstruction_time += epoch_time
                # Save the model and metrics
                if epoch + 1 in args.save_at_finetune_epoch:
                    torch.cuda.empty_cache()
                    scene_model.inference_mode = True
                    metrics = scene_model.save(
                        os.path.join(args.model_path, str(epoch + 1)), reconstruction_time
                    )
                    bar_postfix = []
                    for key, value in metrics.items():
                        bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
                    pbar.set_postfix_str(",".join(bar_postfix))
                    scene_model.inference_mode = False
                    torch.cuda.empty_cache()

            # Set to inference mode so that the model can be rendered properly
            scene_model.inference_mode = True

        if args.save_to_data_for_gsplat:
            colmap_dir = os.path.join(args.model_path, "colmap")
            target_dir = os.path.join(args.source_path, "artdeco_colmap")
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            print(colmap_dir, target_dir)
            shutil.copytree(colmap_dir, target_dir)
        print(
            ", ".join(
                f"{metric}: {value:.3f}"
                if isinstance(value, float)
                else f"{metric}: {value}"
                for metric, value in metrics.items()
            )
        )
    if args.viewer_mode != "none":
        if args.viewer_mode == "web":
            while True:
                time.sleep(1)
        else:
            viewer.throttling = False  # Disable throttling when done training
            # Loop to keep the viewer alive
            while viewer.running:
                time.sleep(1)
    frontend.join()
    backend.join()
