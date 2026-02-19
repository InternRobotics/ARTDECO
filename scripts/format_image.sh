#!/bin/bash

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <图片目录路径>"
    exit 1
fi

DIR="$1"

# 检查目录是否存在
if [ ! -d "$DIR" ]; then
    echo "错误: 目录 $DIR 不存在"
    exit 1
fi

# 进入目录
cd "$DIR"

# 遍历匹配的图片文件并重命名
for file in output_*.png; do
    if [ -f "$file" ]; then
        # 提取数字部分
        number=$(echo "$file" | sed 's/output_\([0-9]*\)\.png/\1/')
        # 重命名
        mv "$file" "${number}.png"
        echo "重命名: $file -> ${number}.png"
    fi
done

echo "完成"
