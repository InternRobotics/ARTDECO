import os
import shutil
from pathlib import Path
from PIL import Image


def process_images(parent_dir):
    """处理图片：排序、数量降采样20倍、尺寸降采样4倍"""
    parent_dir = Path(parent_dir)

    if not parent_dir.exists():
        print(f"错误: 目录 {parent_dir} 不存在")
        return

    # 遍历所有子文件夹
    for subdir in parent_dir.iterdir():
        if subdir.is_dir():
            print(f"处理子文件夹: {subdir.name}")

            # 获取所有图片文件并排序
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
            images = [f for f in subdir.iterdir()
                      if f.is_file() and f.suffix.lower() in image_extensions]

            # 按文件名排序
            images.sort(key=lambda x: x.name)

            if not images:
                print(f"  跳过: 没有找到图片文件")
                continue

            print(f"  找到 {len(images)} 张图片")

            # 每20张取1张进行处理
            kept_images = []
            for i in range(0, len(images), 20):
                img_path = images[i]
                print(f"  处理: {img_path.name}")

                try:
                    # 打开图片
                    with Image.open(img_path) as img:
                        # 计算新尺寸（降采样4倍，即缩小到25%）
                        new_width = img.width // 4
                        new_height = img.height // 4

                        # 缩放图片
                        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                        # 覆盖原图
                        resized_img.save(img_path)
                        kept_images.append(img_path)

                except Exception as e:
                    print(f"    错误处理 {img_path.name}: {e}")

            # 删除未被选中的图片
            for img_path in images:
                if img_path not in kept_images:
                    img_path.unlink()

            print(f"  完成: 保留并缩小了 {len(kept_images)} 张图片")

    print("所有处理完成")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("用法: python script.py <包含子文件夹的目录路径>")
        sys.exit(1)

    process_images(sys.argv[1])
