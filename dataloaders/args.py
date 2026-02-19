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

import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Options for data loading and training")

    ## Data and Images
    parser.add_argument('-s', '--source_path', type=str, required=True,
                        help="Path to the data folder (should have sparse/0/ if using COLMAP or evaluating poses)")
    parser.add_argument('-i', '--images_dir', type=str, default="images",
                        help="source_path/images_dir is the path to the images (with extensions jpg, png or jpeg).")
    parser.add_argument('--downsampling', type=float, default=1.0, help="Downsampling ratio for input images")
    parser.add_argument('--max_size_slam', type=int, default=512,
                        help="Maximum size of slam image process.")
    parser.add_argument('--start_at', type=int, default=0,
                        help="Number of frames to skip from the start of dataset.")
    parser.add_argument('--end_at', type=int, default=0,
                        help="Number of frames to skip from the end of dataset.")
    parser.add_argument('--seq_length', type=int, default=0,
                        help="Number of frames to skip from the end of dataset.")
    parser.add_argument('--image_sampling', type=int, default=0,
                        help='sample every N images')
    parser.add_argument('--save_lidar_ply', action='store_true', default=False,
                        help='Save lidar and pointmap as ply files(test function)')
    parser.add_argument('-d', '--dataset_name', type=str, default="selfCaptured",
                        choices=["selfCaptured"],)
    parser.add_argument('--save_to_data_for_gsplat', action="store_true")
    parser.add_argument('--rigid_transform_gaussians', action="store_true")
    parser.add_argument('--base_model', type=str, default="sharp")

    ## Learning Rates
    parser.add_argument('--lr_poses', type=float, default=1e-4, help="Pose learning rate")
    parser.add_argument('--lr_exposure', type=float, default=5e-4, 
                        help="Exposure compensation learning rate")
    parser.add_argument('--lr_depth_scale_offset', type=float, default=1e-4)
    parser.add_argument('--position_lr_init', type=float, default=0.00005,
                        help="Initial position learning rate")
    parser.add_argument('--position_lr_decay', type=float, default=1-2e-5, 
                        help="multiplicative decay factor for position learning rate")
    parser.add_argument('--mlp_cov_lr_init', type=float, default=0.004,
                        help="Initial feature learning rate")
    parser.add_argument('--mlp_cov_lr_decay', type=float, default=1-2e-5, 
                        help="multiplicative decay factor for feature learning rate")
    parser.add_argument('--feat_lr', type=float, default=0.004, help="feat learning rate")
    parser.add_argument('--feature_lr', type=float, default=0.005, help="Feature learning rate")
    parser.add_argument('--opacity_lr', type=float, default=0.1, help="Opacity learning rate")
    parser.add_argument('--scaling_lr', type=float, default=0.01, help="Scaling learning rate")
    parser.add_argument('--rotation_lr', type=float, default=0.002, help="Rotation learning rate")
    parser.add_argument('--low_pass_filter_eps', type=float, default=0.01, help="Epsilon for low-pass filter")

    ## Training schedule and losses
    parser.add_argument('--lambda_dssim', type=float, default=0.2, help="Weight for DSSIM loss")
    parser.add_argument('--num_key_iterations', type=int, default=30, 
                        help="Number of training iterations per key frame")
    parser.add_argument('--num_common_iterations', type=int, default=0, 
                        help="Number of training iterations per common frame")
    parser.add_argument('--depth_loss_weight_init', type=float, default=1e-2)
    parser.add_argument('--depth_loss_weight_decay', type=float, default=0.9, 
                        help="Weight decay for depth loss, multiply depth loss weight by this factor every iterations")
    parser.add_argument('--save_at_finetune_epoch', type=int, nargs='+', default=[], 
                        help="Enable finetuning after the initial on-the-fly reconstruction and save the scene at the end of the specified epochs when fine-tuning.")
    parser.add_argument('--save_at_finetune_iteration', type=int, nargs='+', default=[], 
                        help="Enable finetuning after the initial on-the-fly reconstruction and save the scene at the end of the specified iterations when fine-tuning.")
    parser.add_argument('--use_last_frame_proba', type=float, default=0.2, 
                        help="Probability of using the last registered frame for each training iteration")

    ## Pose initialization options
    # Matching
    parser.add_argument('--num_kpts', type=int, default=int(4096*1.5),
                        help="Number of keypoints to extract from each image")
    parser.add_argument('--match_max_error', type=float, default=2e-3,
                        help="Maximum reprojection error for matching keypoints, proportion of the image width. This is used to filter outliers and discard points at triangulation.")
    parser.add_argument('--fundmat_samples', type=int, default=2000,
                        help="Maximum number of set of matches used to estimate the fundamental matrix for outlier removal")
    parser.add_argument('--min_num_inliers', type=int, default=100,
                        help="The keyframe will be added only if the number of inliers is greater than this value")

    # voxel size
    parser.add_argument('--scaling_reg_factor', type=float, default=0)
    parser.add_argument('--voxel_size', type=float, default=0.1)
    parser.add_argument('--visible_threshold', type=float, default=0.01)
    parser.add_argument('--gs_add_ratio', type=float, default=0.3)
    parser.add_argument('--rad_decay', type=float, default=5 ** 0.5)
    parser.add_argument('--use_loop_closure', action='store_true')
    parser.add_argument('--use_all_frames', action="store_true")

    parser.add_argument('--num_keyframes_miniba_bootstrap', type=int, default=8,
                        help="Number of first keyframes accumulated for pose and focal estimation before optimization")
    parser.add_argument('--num_pts_miniba_bootstrap', type=int, default=2000,
                        help="Number of keypoints considered for initial mini bundle adjustment")
    parser.add_argument('--iters_miniba_bootstrap', type=int, default=200)
    parser.add_argument('--enable_reboot', action='store_true')
    parser.add_argument('--enable_scaling', action='store_true')

    # Focal estimation
    parser.add_argument('--fix_focal', action='store_true', 
                        help="If set, will use init_focal or init_fov without reoptimizing focal")
    parser.add_argument('--init_focal', type=float, default=-1.0, 
                        help="Initial focal length in pixels. If not set, will use init_fov or be set as 0.7*width of the image if init_fov is also not set")
    parser.add_argument('--init_fov', type=float, default=-1.0, 
                        help="Initial horizontal FoV in degrees. Used only if init_focal is not set")
    # Incremental pose optimization
    parser.add_argument('--num_prev_keyframes_miniba_incr', type=int, default=6,
                        help="Number of previous keyframes for incremental pose initialization")
    parser.add_argument('--num_prev_keyframes_check', type=int, default=20,
                        help="Number of previous keyframes to check for matches with new keyframe")
    parser.add_argument('--pnpransac_samples', type=int, default=2000,
                        help="Maximum number of set of 2D-3D matches used to estimate the initial pose and outlier removal")
    parser.add_argument('--num_pts_miniba_incr', type=int, default=2000,
                        help="Number of keypoints considered for initial mini bundle adjustment")
    parser.add_argument('--iters_miniba_incr', type=int, default=20)


    ## Gaussian initialization options
    parser.add_argument('--checkpoint_path', type=str, default="./models/sharp_2572gikvuh.pt", 
                        help="Path to checkpoint for SHARP pretrained model")
    parser.add_argument('--sh_degree', default=3)
    parser.add_argument('--local_feat_dim', type=int, default=32)
    parser.add_argument('--global_feat_dim', type=int, default=32)
    parser.add_argument('--pyr_levels', type=int, default=2,
                        help="Number of pyramid levels. Each level l will downsample the image 2^l times in width and height")
    parser.add_argument('--init_proba_scaler', type=float, default=2,
                        help="Scale the laplacian-based probability of using a pixel to make a new Gaussian primitive. Set to 0 to only use triangulated points.")

    ## Keyframe management
    parser.add_argument('--max_active_keyframes', type=int, default=400,
                        help="Maximum number of keyframes to keep in GPU memory. Will start offloading keyframes to CPU if this number is exceeded.")

    ## Evaluation
    parser.add_argument('--test_hold', type=int, default=-1, 
                        help="Holdout for test set, will exclude every test_hold image from the Gaussian optimization and use them for testing. The test frames will still be used for training the pose. If set to -1, no keyframes will be excluded from training.")
    parser.add_argument('--test_frequency', type=int, default=-1, 
                        help="Test and get metrics every test_frequency keyframes")
    parser.add_argument('--display_runtimes', action='store_true', 
                        help="Display runtimes for each step in the tqdm bar")

    ## Checkpoint options
    parser.add_argument('-m', '--model_path', default="", 
                        help="Directory to store the renders from test view and checkpoints after training. If not set, will be set to results/xxxxxx.")
    parser.add_argument('--save_every', default=-1, type=int, 
                        help="Frequency of exporting renders w.r.t input frames.")
    parser.add_argument("--save_point_could", action="store_true")

    #################################### system ########################################
    parser.add_argument("--device_frontend", default="cuda:0")
    parser.add_argument("--device_backend", default="cuda:0")
    parser.add_argument("--device_mapper", default="cuda:0")
    parser.add_argument("--device_shared", default="cpu")

    ## Debug options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug file saving and additional logging statements.')
    ## Viewer
    parser.add_argument('--viewer_mode', choices=['local', 'server', 'web', 'none'], default='none')
    parser.add_argument('--ip', type=str, default="0.0.0.0", 
                        help="IP address of the viewer client, if using server viewer_mode")
    parser.add_argument('--port', type=int, default=6009,
                        help="Port of the viewer client, if using server viewer_mode")

    # VSLAM
    parser.add_argument("--optimize_focal", action="store_true")
    parser.add_argument("--point_fusion_frontend", action="store_true")
    parser.add_argument("--covariance_filter", action="store_true")
    parser.add_argument("--accurate_loop_closure", action="store_true")
    parser.add_argument('--num_GBA', type=int, default=1)
    parser.add_argument("--use_gt_pose", action="store_true")
    parser.add_argument('--min_displacement', type=float, default=0.03,
                        help="Minimum median keypoint displacement for a new keyframe to be added. ")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--calib", default=None)
    parser.add_argument("--use_colmap_calib", action="store_true",
                        help="Use COLMAP to estimate camera calibration")
    parser.add_argument("--colmap_first_n", type=int, default=400,
                        help="Number of first images to use for COLMAP calibration")
    parser.add_argument("--colmap_stride", type=int, default=4,
                        help="Sampling stride for COLMAP calibration")
    parser.add_argument("--sync_hard", action="store_false")
    parser.add_argument('--thres_keyframe', type=float, default=0.8,
                        help="Size of the overlapping regions when blending between anchors")
    parser.add_argument("--use_same_set_of_keyframes", action="store_true")
    args = parser.parse_args()

    ## Set the output directory if not specified
    if args.model_path == "":
        i = 0
        while os.path.exists(f"results/{i:06d}"):
            i += 1
        args.model_path = f"results/{i:06d}"

    return args
