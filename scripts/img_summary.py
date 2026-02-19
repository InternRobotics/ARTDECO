#!/usr/bin/env python3
import sys, os
from pathlib import Path

try:
    from PIL import Image  # Pillow or Pillow-SIMD
except Exception:
    Image = None  # size will be blank if Pillow isn't available

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def count_and_first(root: Path):
    count, first = 0, None
    for dp, _, files in os.walk(root):
        for f in files:
            if Path(f).suffix.lower() in EXTS:
                count += 1
                if first is None:
                    first = Path(dp) / f
    return count, first

def img_size_str(path: Path) -> str:
    if not Image or path is None:
        return ""
    try:
        with Image.open(path) as im:
            return f"{im.width}x{im.height}"
    except Exception:
        return ""

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {Path(sys.argv[0]).name} DIR [DIR ...]", file=sys.stderr)
        sys.exit(1)

    print("name,count,image_size")
    max_count, min_count = None, None
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.is_dir():
            print(f"{p.name},0,")
            continue
        count, first = count_and_first(p)
        size = img_size_str(first)
        if min_count is None:
            min_count = count
        if max_count is None:
            max_count = count
        min_count = min(min_count, count)
        max_count = max(max_count, count)
        print(f"{p.name},{count},{size}")  # use {p} for full path instead of basename
        print(min_count, max_count)
if __name__ == "__main__":
    main()
