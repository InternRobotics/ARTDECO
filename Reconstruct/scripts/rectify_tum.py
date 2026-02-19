import argparse
import cv2 
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import sys

sys.path.append('.')
from utils import get_image_names

# OpenCV format: fx, fy, cx, cy, k1, k2, p1, p2, k3
cam_params_dict = {
    "TUM/rgbd_dataset_freiburg1_desk": [517.306408, 516.469215, 318.643040, 255.313989, 0.262383, -0.953104, -0.005358, 0.002628, 1.163314],
    "TUM/rgbd_dataset_freiburg2_xyz": [520.908620, 521.007327, 325.141442, 249.701764, 0.231222, -0.784899, -0.003257, -0.000105, 0.917205],
    "TUM/rgbd_dataset_freiburg3_long_office_household": [535.4, 539.2, 320.1, 247.6, 0.0, 0.0, 0.0, 0.0],
}

def get_K_in_K_out(cam_params, h, w):
    K_in = np.array([[cam_params[0], 0, cam_params[2]], [0, cam_params[1], cam_params[3]], [0, 0, 1]])
    K_out = cv2.getOptimalNewCameraMatrix(K_in, np.array(cam_params[4:]), (w, h), 1, (w, h), True)[0]
    K_out[0, 0] = K_out[1, 1] = (K_out[0, 0] + K_out[1, 1]) / 2
    return K_in, K_out

def rectify_and_mask(image, rectify_map, initial_mask, threshold=250, zero_invalid=True, add_alpha=True):
    dst = cv2.remap(image, rectify_map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.remap(initial_mask, rectify_map, None, cv2.INTER_LINEAR)
    mask[mask <= threshold] = 0
    mask[mask != 0] = 255
    if zero_invalid:
        dst[mask == 0] = 0
    if add_alpha:
        dst = np.concatenate([dst, mask[..., None]], axis=-1)
    return dst, mask

if __name__ == '__main__':
    """
    Rectifies images so that they have no distortion, centred principal point and square pixels.
    Will read from rgb/ and put fully rectify images is images/ with masks in masks/.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="../data")
    args = parser.parse_args()


    for scene, cam_params in cam_params_dict.items():
        scene_folder = os.path.join(args.base_dir, scene) 

        in_folder = f"{scene_folder}/rgb"
        out_folder = f"{scene_folder}/images"
        mask_folder = f"{scene_folder}/masks"
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)

        image_names = get_image_names(in_folder)
        h, w = cv2.imread(f"{in_folder}/{image_names[0]}").shape[:2]
        K_in, K_out = get_K_in_K_out(cam_params, h, w)

        rectify_map = cv2.initUndistortRectifyMap(
            K_in, np.array(cam_params[4:]), None, K_out, (w, h), cv2.CV_32FC2)[0]
        initial_mask = np.ones((h, w), dtype=np.uint8) * 255

        def process_image(image_name):
            image = cv2.imread(f"{in_folder}/{image_name}")

            dst, mask = rectify_and_mask(image, rectify_map, initial_mask, threshold=0)
            cv2.imwrite(f"{out_folder}/{image_name}", dst)
            cv2.imwrite(f"{mask_folder}/{os.path.splitext(image_name)[0]}.png", mask)

        with ThreadPoolExecutor() as executor:
            executor.map(process_image, image_names)
