import argparse
import cv2 
import os
import concurrent.futures

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_folder', default="../data/Meta/university2")
    parser.add_argument('--downsampling', default=1.5)
    args = parser.parse_args()

    in_folder = f"{args.scene_folder}/images"
    out_folder = f"{args.scene_folder}/images_{args.downsampling}"
    os.makedirs(out_folder, exist_ok=True)

    image_names = [file for file in os.listdir(in_folder) if file.endswith('.png') or file.endswith('.jpg')]
    
    def process_image(image_name):
        image = cv2.imread(f"{in_folder}/{image_name}")
        dst = cv2.resize(image, (0, 0), fx=1/args.downsampling, fy=1/args.downsampling, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{out_folder}/{image_name}", dst, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_image, image_names)
