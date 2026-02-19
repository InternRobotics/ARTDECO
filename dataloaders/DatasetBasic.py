"""
BaseDataset interface for unified data loading (images, poses, pointmaps, etc.)
All dataset types (disk, stream, MASt3R-SLAM, etc.) should implement this interface.
"""
import os
from argparse import Namespace
from pathlib import Path

import cv2
import torch
import logging
import numpy as np
from typing import Tuple, Dict, Any

import yaml

from dataloaders.CameraModel import PinholeCamera
from dataloaders.utils_IO import read_model, qvec2rotmat
from geocalib import GeoCalib

class BaseDataset:
    """
        Base dataset interface for unified data loading.
        对于其中为None的部分需要在子类中定义。
    """
    def __init__(self, args: Namespace):
        """
            Must specific the following variables:

            Required:
                image_dir           [os.Path]            the folder path that contains images
                image_name_list     [list: string]       the list of image full names
                timestamp           [list: float]        the timestamp of images

            Optional:
                Twc_gt              [numpy.array]        poses of each camera: tx ty tz qx qy qz qw
        """

        assert len(self.image_name_list) == len(self.timestamp), \
            "Mismatch between number of times and number of images"
        if self.Twc_gt is not None:
            assert len(self.Twc_gt) == len(self.image_name_list), \
            "Mismatch between number of poses and number of images"


        if hasattr(args, "image_sampling") and args.image_sampling > 1:
            self.image_name_list = self.image_name_list[::args.image_sampling]
            self.timestamp = self.timestamp[::args.image_sampling]
            if self.Twc_gt is not None:
                self.Twc_gt = self.Twc_gt[::args.image_sampling]
        self.start_at = args.start_at
        self.end_at = len(self.image_name_list) - args.end_at
        self.image_name_list = self.image_name_list[self.start_at:self.end_at]
        self.timestamp = self.timestamp[self.start_at:self.end_at]
        if self.Twc_gt is not None:
            self.Twc_gt = self.Twc_gt[self.start_at:self.end_at]

        # dataset downsample
        if hasattr(args, "seq_length") and args.seq_length > 0:
            self.image_name_list = self.image_name_list[:args.seq_length]
            self.timestamp = self.timestamp[:args.seq_length]
            if self.Twc_gt is not None:
                self.Twc_gt = self.Twc_gt[:args.seq_length]

        self.image_paths = [
            os.path.join(self.image_dir, image_name)
            for image_name in self.image_name_list
        ]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        print(self.image_paths)


        # decide test images
        self.infos = {
            name: {
                "is_test": (args.test_hold > 0) and (i % args.test_hold == 0) and (i != 0),
                "name": name,
                "timestamp": self.timestamp[i],
            }
            for i, name in enumerate(self.image_name_list)
        }

        # args: camera intrinsics for slam
        intrinsics = {}
        if args.calib is None:
            if getattr(args, 'use_colmap_calib', False):
                H, W, fx, fy, cx, cy = self.estimate_calib_from_colmap(args)
            else:
                H, W, fx, fy, cx, cy = self.estimate_calib()
            intrinsics["width"] = W
            intrinsics["height"] = H
            # 格式1：8个参数 [fx, fy, cx, cy, k1, k2, p1, p2]
            # 格式2：9个参数 [fx, fy, cx, cy, k1, k2, p1, p2, k3]
            # 格式3：12个参数 [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]
            intrinsics["calibration"] = [fx, fy, cx, cy]
        else:
            with open(args.calib, "r") as f:
                intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
            if hasattr(self, "frame_intrinsic"):
                intrinsics["calibration"] = [self.frame_intrinsic[0, 0, 0],
                                             self.frame_intrinsic[0, 1, 1],
                                             self.frame_intrinsic[0, 0, 2],
                                             self.frame_intrinsic[0, 1, 2]]
        self.downsampling = args.downsampling
        self.load_calib(intrinsics, args.max_size_slam, args.optimize_focal)

        # class variable
        self.current_index = 0


    def estimate_calib(self):
        results = []
        H, W = None, None
        model = GeoCalib().cuda()
        for i in range(4):
            image_path = self.image_paths[i]
            image = self._load_image(image_path, cv2.IMREAD_UNCHANGED)
            H, W = image.shape[:2]
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
            results.append(model.calibrate(image.cuda())["camera"].K)
        results = torch.concatenate(results, dim=0).mean(dim=0)
        fx, fy, cx, cy = results[0, 0], results[1, 1], results[0, 2], results[1, 2]

        return H, W, fx.item(), fy.item(), cx.item(), cy.item()

    def estimate_calib_from_colmap(self, args: Namespace):
        first_n = getattr(args, 'colmap_first_n', 800)
        stride = getattr(args, 'colmap_stride', 4)

        # Select image subset
        total = len(self.image_paths)
        target = max(1, first_n // stride)
        list_count = min(total, first_n)
        selected_paths = [self.image_paths[i] for i in range(0, list_count, stride)]

        if len(selected_paths) < target:
            if list_count >= target:
                indices = np.linspace(0, list_count - 1, num=target, dtype=int)
                selected_paths = [self.image_paths[i] for i in indices]
            else:
                selected_paths = list(self.image_paths)

        if len(selected_paths) < 16:
            raise ValueError(f"Not enough images for COLMAP: {len(selected_paths)}")

        print(f"[Estimate Intrinsics] Estimating intrinsics from {len(selected_paths)} images using COLMAP (total={total}, first_n={first_n}, stride={stride})")

        first_img = cv2.imread(selected_paths[0])
        H, W = first_img.shape[:2]

        import shutil

        work_dir = os.path.join(self.image_dir, "tmp")
        subset_dir = os.path.join(work_dir, "images")
        sparse_dir = os.path.join(work_dir, "sparse")
        sparse_txt_dir = os.path.join(work_dir, "sparse_txt")
        db_path = os.path.join(work_dir, "database.db")

        # Clean up existing files
        for f in [db_path, db_path + "-shm", db_path + "-wal"]:
            if os.path.exists(f):
                os.remove(f)
        for d in [subset_dir, sparse_dir, sparse_txt_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)

        os.makedirs(subset_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)

        for i, src_path in enumerate(selected_paths):
            ext = os.path.splitext(src_path)[1]
            dst_path = os.path.join(subset_dir, f"{i:06d}{ext}")
            os.symlink(os.path.abspath(src_path), dst_path)

        import subprocess
        DEVNULL = subprocess.DEVNULL

        print("[Estimate Intrinsics] Step 1/5: Feature extraction...")
        ret = subprocess.run(
            f"colmap feature_extractor --database_path {db_path} --image_path {subset_dir} "
            f"--ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE "
            f"--SiftExtraction.use_gpu 1",
            shell=True, stdout=DEVNULL, stderr=DEVNULL
        ).returncode
        if ret != 0:
            raise RuntimeError("COLMAP feature_extractor failed")

        print("[Estimate Intrinsics] Step 2/5: Sequential matching...")
        ret = subprocess.run(
            f"colmap sequential_matcher --database_path {db_path} "
            f"--SiftMatching.use_gpu 1 --SiftMatching.max_num_matches 32768 "
            f"--SequentialMatching.overlap 30",
            shell=True, stdout=DEVNULL, stderr=DEVNULL
        ).returncode
        if ret != 0:
            print("[Estimate Intrinsics] GPU matcher failed, retrying with CPU...")
            ret = subprocess.run(
                f"colmap sequential_matcher --database_path {db_path} "
                f"--SiftMatching.use_gpu 0 --SiftMatching.max_num_matches 32768 "
                f"--SequentialMatching.overlap 30",
                shell=True, stdout=DEVNULL, stderr=DEVNULL
            ).returncode
            if ret != 0:
                raise RuntimeError("COLMAP sequential_matcher failed")

        print("[Estimate Intrinsics] Step 3/5: Mapping...")
        ret = subprocess.run(
            f"colmap mapper --database_path {db_path} --image_path {subset_dir} "
            f"--output_path {sparse_dir}",
            shell=True, stdout=DEVNULL, stderr=DEVNULL
        ).returncode
        if ret != 0:
            raise RuntimeError("COLMAP mapper failed")

        # Find the largest model by images.bin size
        model_dirs = [d for d in os.listdir(sparse_dir) if os.path.isdir(os.path.join(sparse_dir, d))]
        if not model_dirs:
            raise RuntimeError("COLMAP did not generate any model")

        best_model = None
        best_size = 0
        for d in model_dirs:
            images_bin = os.path.join(sparse_dir, d, "images.bin")
            if os.path.exists(images_bin):
                size = os.path.getsize(images_bin)
                if size > best_size:
                    best_size = size
                    best_model = os.path.join(sparse_dir, d)

        if best_model is None:
            raise RuntimeError("No valid COLMAP model found")

        print("[Estimate Intrinsics] Step 4/5: Bundle adjustment...")
        subprocess.run(
            f"colmap bundle_adjuster --input_path {best_model} --output_path {best_model}",
            shell=True, stdout=DEVNULL, stderr=DEVNULL
        )

        print("[Estimate Intrinsics] Step 5/5: Exporting model...")
        os.makedirs(sparse_txt_dir, exist_ok=True)
        ret = subprocess.run(
            f"colmap model_converter --input_path {best_model} "
            f"--output_path {sparse_txt_dir} --output_type TXT",
            shell=True, stdout=DEVNULL, stderr=DEVNULL
        ).returncode
        if ret != 0:
            raise RuntimeError("COLMAP model_converter failed")

        # Parse cameras.txt
        cameras_txt = os.path.join(sparse_txt_dir, "cameras.txt")
        with open(cameras_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                model = parts[1]
                width, height = int(parts[2]), int(parts[3])

                if model == "SIMPLE_PINHOLE":
                    fx = fy = float(parts[4])
                    cx, cy = float(parts[5]), float(parts[6])
                elif model == "PINHOLE":
                    fx, fy = float(parts[4]), float(parts[5])
                    cx, cy = float(parts[6]), float(parts[7])
                else:
                    raise RuntimeError(f"Unsupported camera model: {model}")

                print(f"[Estimate Intrinsics] Done. Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                return height, width, fx, fy, cx, cy

        raise RuntimeError("No camera found in cameras.txt")

    def load_calib(self, intrinsics, max_size_slam=512, optimize_focal=False):

        self.transform = PinholeCamera(max_size_slam,
                                    self.downsampling,
                                    intrinsics["width"],
                                    intrinsics["height"],
                                    intrinsics["calibration"],
                                    optimize_focal=optimize_focal)
        self.H, self.W = intrinsics["height"], intrinsics["width"]
        self.H_slam, self.W_slam = self.transform.H_slam, self.transform.W_slam
        self.H_map, self.W_map = self.transform.H_map, self.transform.W_map
        self.K_slam = self.transform.K_slam
        self.K_map = self.transform.K_map


    def __len__(self) -> int:
        """Returns the number of images/frames (or a large number for streams)."""
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self._load_image(image_path, cv2.IMREAD_UNCHANGED)
        H_c, W_c = image.shape[:2]
        assert (H_c == self.H) and (W_c == self.W), \
            f"Image {index}: Size not match between the actual and the size specified in the file"
        info = self.infos[os.path.basename(image_path)]
        if self.Twc_gt is not None:
            info["Twc_gt"] = self.Twc_gt[index]
        return image, info

    def _load_image(self, image_path, mode=cv2.IMREAD_COLOR):
        image = cv2.imread(image_path, mode)
        if image is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
        return image

    def getnext(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns the next image and associated info (e.g., pose, pointmap, etc.).
        Returns:
            image (torch.Tensor): The image tensor (C, H, W)
            info (dict): Dictionary with keys like 'pose', 'pointmap', 'is_test', etc.
        """
        frame, info = self.__getitem__(self.current_index)
        self.current_index += 1
        return frame, info

    def get_image_size(self) -> Tuple[int, int]:
        """Returns (height, width) of images."""
        return self.H_map, self.W_map

