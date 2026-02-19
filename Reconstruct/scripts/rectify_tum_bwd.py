import argparse
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append('.')

from scripts.rectify_tum import cam_params_dict, get_K_in_K_out
from utils import get_image_names

if __name__ == "__main__":
    """
    Rectify from what is used from training (centred principal point) 
    to something without black borders for visualization
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="../fast-end2end-nvs-comparisons/results")
    args = parser.parse_args()

    for scene, cam_params in cam_params_dict.items():
        in_scene_dir = os.path.join(args.base_dir, scene)
        out_scene_dir = os.path.join(args.base_dir, "derectified", scene)

        methods = os.listdir(in_scene_dir)
        for method_id, method in enumerate(methods):
            in_folder = os.path.join(in_scene_dir, method)
            out_folder = os.path.join(out_scene_dir, method)

            os.makedirs(out_folder, exist_ok=True)

            image_names = get_image_names(in_folder)
            h, w = cv2.imread(f"{in_folder}/{image_names[0]}").shape[:2]
            if h ==336 and method_id == 0:
                print("Mannually adjusting scale in intrinsics")
                cam_params[0] *= 448 / 640
                cam_params[1] *= 336 / 480
                cam_params[2] *= 448 / 640
                cam_params[3] *= 336 / 480

            # Get the matrix used for optimization
            K_in, K_train = get_K_in_K_out(cam_params, h, w)
            # Get a matrix that fills the full image
            K_out = cv2.getOptimalNewCameraMatrix(
                K_in, np.array(cam_params[4:]), (w, h), 0, (w, h), True
            )[0]

            rectify_map = cv2.initUndistortRectifyMap(
                K_train, np.array(cam_params[4:]), None, K_out, (w, h), cv2.CV_32FC2
            )[0]

            def process_image(image_name):
                image = cv2.imread(f"{in_folder}/{image_name}")
                dst = cv2.remap(
                    image,
                    rectify_map,
                    None,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                pad = 4
                dst = dst[pad:-pad, pad: -pad]
                cv2.imwrite(f"{out_folder}/{image_name}", dst)
                cv2.imwrite(f"{out_folder}/{os.path.splitext(image_name)[0]}.jpg", dst)

            with ThreadPoolExecutor() as executor:
                executor.map(process_image, image_names)
