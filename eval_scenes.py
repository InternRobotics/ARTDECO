import subprocess
import argparse
import sys
import os
from pathlib import Path
import shutil

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
# Setup Class
@dataclass
class Setup:
    name: str
    base_args: List[str] = field(default_factory=list)
    # a Python‐format string; if None, we won’t add any --gt_poses_from
    reference_template: Optional[str] = None
    # If every setup always needs calibration + extra_params, you can
    # bake those in here, or keep them global:
    apply_calibration: bool = True
    apply_extra: bool = True
    def get_args(self, camera_id: str) -> List[str]:
        args = list(self.base_args)
        if self.reference_template is not None:
            ref = self.reference_template.format(camera_id=camera_id)
            args += ["--gt_poses_from", ref]
        return args
    
# Presets
calibration = ["--fix_focal", "--init_focal", "758.8342108398161"]
extra_params = ["--no-viz"]
setups = {
    "slam-tracker_init": Setup(
        name="slam-tracker_init",
        base_args=["--use_slam_pose", "--init_poses_with", "MASt3R-SLAM_Rt"],
        reference_template="slam/{camera_id}/0_frames.txt"
    ),
    "slam-tracker_overwrite": Setup(
        name="slam-tracker_overwrite",
        base_args=["--use_slam_pose", "--overwrite_poses_with", "MASt3R-SLAM_Rt"],
        reference_template="slam/{camera_id}/0_frames.txt"
    ),
    "slam-tracker_overwrite-bs_init-inc": Setup(
        name="slam-tracker_overwrite-bs_init-inc",
        base_args=["--use_slam_pose", "--init_incremental_poses_with", "MASt3R-SLAM_Rt",
        "--overwrite_bootstrap_poses_with",  "MASt3R-SLAM_Rt"],
        reference_template="slam/{camera_id}/0_frames.txt"
    ),
    "slam-tum_init_pose": Setup(
        name="slam-tum_init_pose",
        base_args=["--init_poses_with", "Rt"],
        reference_template="slam/{camera_id}/0_frames.txt"
    ),
    "slam-tum_overwrite_pose": Setup(
        name="slam-tum_overwrite_pose",
        base_args=["--overwrite_poses_with", "Rt","--use_lidar_dataset", "--save_lidar_ply",
                   "--align_matrix","0701",],
        reference_template="slam/{camera_id}/0_frames.txt"
    ),
    "LiDAR-tum_init_pose": Setup(
        name="LiDAR-tum_init_pose",
        base_args=["--init_poses_with", "Rt"],
        reference_template="lidar/pose/{camera_id}_lidar_interpolate_image_poses.txt"
    ),
    "LiDAR-tum_overwrite": Setup(
        name="LiDAR-tum_overwrite",
        base_args=["--overwrite_poses_with", "Rt"],
        reference_template="lidar/pose/{camera_id}_lidar_interpolate_image_poses.txt"
    ),
    # SfM variants—swap out “colmap” or “hloc” to change backend
    "SfM-colmap": Setup(
        name="SfM-colmap",
        base_args=["--init_poses_with", "COLMAP"],
        reference_template="colmap/sparse/0"
    ),
    "SfM-hloc": Setup(
        name="SfM-hloc",
        base_args=["--init_poses_with", "Hloc"],
        reference_template="hloc/sparse/0"
    ),
    "onthefly": Setup(
        name="onthefly",
        base_args=["--enable_scaling"],
        apply_calibration=False,
        apply_extra=False
    ),
}
# 3) In your main(), replace all the inline logic with something like:
def build_cmd(python, scene, camera_id, setup: Setup, args):
    base = [
        python, "run_system.py",
        "-s", str(scene),
        "--images_dir", str(Path("images") / camera_id),
        "--config", args.config,
        "--calib", args.calib,
        "--downsampling", str(args.downsampling),
        "--test_hold", str(args.test_hold),
        "-m", str(args.save_dir),
    ]
    if setup.apply_calibration:
        base += calibration
    if setup.apply_extra:
        base += extra_params
    base += setup.get_args(camera_id)
    return base


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run run_system.py over multiple scene directories"
    )
    parser.add_argument(
        'scenes', nargs='+',
        help="One or more scene directories to process"
    )
    parser.add_argument(
        '--config', default='config/base.yaml',
        help="Path to the config YAML"
    )
    parser.add_argument(
        '--calib', default='config/camera_intrinsics/intrinsics_self_captured.yaml',
        help="Path to the calibration YAML"
    )
    parser.add_argument(
        '--downsampling', type=float, default=2.0,
        help="Downsampling factor"
    )
    parser.add_argument(
        '--test_hold', type=int, default=8,
        # '--test_hold', type=int, default=-1,         # jcj add 07.08 do not test
        help="Number of held-out test views"
    )
    
    # result_saving path
    parser.add_argument(
        '--save_dir', type=str, default=None,
        help="Path to the result directory"
    )    
    # setting
    parser.add_argument(
        '--eval_setup', type=str
    )
    parser.add_argument(
        '--always_redo', action='store_true'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    exp_name = args.eval_setup
    for scene_camera in args.scenes:
        scene_camera = Path(scene_camera)
        camera_id = scene_camera.stem
        scene = scene_camera.parent.parent
        
        # jcj add 07.09
        # override: saving result to args.result_folder
        if args.save_dir is None:
            args.save_dir = Path('results') / scene.relative_to(scene.parts[0]) / camera_id / exp_name       
        else:
            args.save_dir = Path(args.save_dir) / scene.relative_to(scene.parts[0]) / camera_id / exp_name
    
        if args.save_dir.exists():
            if not args.always_redo:
                skip = input(f"{args.save_dir} exists, skip? y/N").lower() == 'y'
                if skip:
                    continue
            shutil.rmtree(args.save_dir)
        setup = setups[args.eval_setup]
        cmd = build_cmd(sys.executable, scene, camera_id, setup, args)
        print(f"\n\033[34m→ Running on scene: {scene}\033[0m")
        print(f"\033[32mUsing pose: {setup.name}\033[0m")
        print("\033[34mEntire command:", " ".join(cmd), "\033[0m")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!][Train] Error on scene {scene}: {e}", file=sys.stderr)
        
        # rendering
        cmd = [
            sys.executable, 'render_traj.py',
            str(args.save_dir)
        ]
        print(f"\n→ Rendering on scene: {scene}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!][Render] Error on scene {scene}: {e}", file=sys.stderr)
        
        # create videos
        cmd = [
            sys.executable, 'scripts/img2vid.py',
            str(args.save_dir), 
            '-o', str(args.save_dir)
        ]
        print(f"\n→ Creating video on scene: {scene}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!][Video] Error on scene {scene}: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()