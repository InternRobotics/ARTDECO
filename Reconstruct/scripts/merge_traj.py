import os
import sys
import shutil
from pathlib import Path

# Usage: python merge_traj.py out_dir dir1 dir2 dir3 ...
# Example: python merge_traj.py merged 0 4 2 1

def main():
    if len(sys.argv) < 3:
        print("Usage: python merge_traj.py out_dir dir1 dir2 ...")
        sys.exit(1)
    out_dir = Path(sys.argv[1])
    dir_names = sys.argv[2:]
    dirs = [Path(d) for d in dir_names]

    # Get sorted image lists for each directory
    img_lists = []
    for d in dirs:
        imgs = sorted([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        img_lists.append(imgs)

    # Find the max number of images in any directory
    max_len = max(len(imgs) for imgs in img_lists)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_idx = 0
    order = list(range(len(dirs)))
    reverse = False
    for i in range(max_len):
        if reverse:
            order = order[::-1]
        for idx in order:
            if i < len(img_lists[idx]):
                src = img_lists[idx][i]
                dst = out_dir / f"{out_idx:06d}{src.suffix.lower()}"
                shutil.copy(src, dst)
                out_idx += 1
        reverse = not reverse

if __name__ == "__main__":
    main()