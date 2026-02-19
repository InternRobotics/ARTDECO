import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('image_dir', type=str, help='Directory containing timestamped images to pad.')
args = parser.parse_args()
image_fps = Path(args.image_dir).rglob("*.jpg")
for image_fp in image_fps:
    if not image_fp.is_file():
        continue
    print(f"Renaming {image_fp} to {float(image_fp.stem):.7f}.jpg")
    image_fp.rename(f"{float(image_fp.stem):.7f}.jpg")