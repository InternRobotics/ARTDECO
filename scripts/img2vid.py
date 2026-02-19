#!/usr/bin/env python3
"""
make_videos.py: Recursively find image-containing directories and create a video per directory
using imageio-ffmpeg with optional GPU acceleration and multithreading.

Usage:
    make_videos.py [-o OUTPUT_DIR] [--fps N] [--workers W] [--hwaccel {none,nvenc}] DIR [DIR ...]
"""
import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import imageio
import numpy as np

EXTS = ('jpg', 'jpeg', 'png')

def pad_to_block(img: np.ndarray, block_size: int = 16) -> np.ndarray:
    """
    Pad the bottom and right edges of `img` so its height and width
    are multiples of `block_size`, filling with black (zeros).
    """
    h, w = img.shape[:2]
    new_h = ((h + block_size - 1) // block_size) * block_size
    new_w = ((w + block_size - 1) // block_size) * block_size
    pad_h = new_h - h
    pad_w = new_w - w

    # construct padding widths: ((top, bottom), (left, right), (0,0) for channels)
    if img.ndim == 3:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    else:
        pad_width = ((0, pad_h), (0, pad_w))

    return np.pad(img, pad_width, mode='constant', constant_values=0)

def find_image_dirs(roots, exts=EXTS):
    """
    Return sorted unique directories under `roots` containing files with given extensions.
    """
    dirs = {
        p.parent
        for root in map(Path, roots)
        for ext in exts
        for p in root.rglob(f'*.{ext}')
    }
    return sorted(dirs)

def make_video(src: Path, dst: Path, fps: int, hwaccel: str):
    """
    Read and sort all images in `src`, then write them to `dst` with imageio.
    """
    # 1) gather & sort
    files = []
    for ext in EXTS:
        files.extend(src.glob(f'*.{ext}'))
        files.extend(src.glob(f'*.{ext.upper()}'))
    files = sorted(files, key=lambda p: p.name.lower())
    if not files:
        print(f"[WARN] no images in {src}")
        return

    # 2) pick codec
    if hwaccel == 'nvenc':
        codec = 'h264_nvenc'
    else:
        codec = 'libx264'

    # 3) write video
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(dst),
        fps=fps,
        codec=codec,
    )
    for img_path in files:
        img = imageio.v3.imread(img_path)
        img = pad_to_block(img)
        writer.append_data(img)
    writer.close()
    print(f"Saved: {dst}")

def safe_name(src: Path) -> str:
    """
    Generate a filesystem-safe name from a Path by replacing separators with '_'.
    """
    name = src.as_posix().lstrip('./')
    return name.replace('/', '_').replace('\\', '_') or src.name

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('dirs', nargs='+', help='Root directories to scan')
    parser.add_argument('-o', '--output', default='.', help='Output directory')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='Number of parallel jobs')
    parser.add_argument('--hwaccel', default='none',
                        choices=['none', 'nvenc'],
                        help='Enable GPU encoder (nvenc)')
    args = parser.parse_args()
    cwd = Path.cwd().resolve()

    dirs = find_image_dirs(args.dirs)
    if not dirs:
        print("No image directories found.")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = []
        for src in dirs:
            # make a unique name for each src
            rel = src.relative_to(cwd) if src.is_relative_to(cwd) else src
            # name = safe_name(rel)
            name = rel
            print(name)
            if args.output:
                out_dir = Path(args.output).expanduser().resolve()
            else:
                out_dir = src.parent.resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / f"{name}.mp4"
            print(f"Processing: {src} → {dst}")
            futures.append(
                exe.submit(make_video, src, dst, args.fps, args.hwaccel)
            )
        for fut in as_completed(futures):
            fut.result()

if __name__ == '__main__':
    main()