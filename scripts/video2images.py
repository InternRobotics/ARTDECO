import os
import subprocess
from pathlib import Path


def video_to_frames(video_dir):
    """将视频文件转换为图片序列"""
    video_dir = Path(video_dir)

    if not video_dir.exists():
        print(f"错误: 目录 {video_dir} 不存在")
        return

    # 遍历所有mp4文件
    for video_file in video_dir.glob("*.MP4"):
        # 获取文件名（不含扩展名）
        filename = video_file.stem

        # 创建输出目录
        output_dir = video_dir / filename
        output_dir.mkdir(exist_ok=True)

        print(f"转换: {video_file} -> {output_dir}/")

        # 使用ffmpeg转换为图片序列
        cmd = [
            "ffmpeg", "-i", str(video_file),
            "-q:v", "2",
            str(output_dir / "%06d.png")
        ]

        subprocess.run(cmd, check=True)
        print(f"完成: {filename}")

    print("所有转换完成")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("用法: python script.py <视频目录路径>")
        sys.exit(1)

    video_to_frames(sys.argv[1])
