#!/bin/bash
work_dir=$(pwd)
proj_dir=/cpfs04/shared/IDC_yumulin_group/Artdeco-V1
base_dir=$proj_dir/self_captured_baselines
result_dir=$proj_dir/Results
img_dir=images
run() { printf '%q ' "$@"; echo; "$@"; }

for data_dir in $base_dir/08_05_canon-r8*; do
    calib_file="$work_dir/config/camera_intrinsics/${data_dir##*_}.yaml"
    for src_dir in $data_dir/MVI_0644*; do
        [[ ! -d "$src_dir/images" ]] && { echo "No images/ in $src_dir"; continue; }
        dst_dir=$result_dir/"${src_dir#$proj_dir/}"/artdeco-rigidtransform
        [[ -e "$dst_dir" && -f "$dst_dir/metadata.json" ]] && { echo "$dst_dir contains results"; continue; }
        scene_dir="${src_dir#$proj_dir/*/}"
        echo $calib_file
        CUDA_VISIBLE_DEVICES=1 run python run_system.py \
            -s "$src_dir" \
            -i "$img_dir" \
            -m "$dst_dir" \
            --config config/base.yaml \
            --calib $calib_file \
            -d selfCaptured \
            --device_frontend cuda:0 \
            --device_backend cuda:0 \
            --device_mapper cuda:0 \
            --device_shared cpu \
            --save_at_finetune_epoch 120 \
            --num_iterations 30 \
            --downsampling 2.0 \
            --test_hold 8 \
            --lr_poses 0.0 \
            --use_loop_closure \
            --rigid_transform_gaussians
        python scripts/img2vid.py $dst_dir/test_images $dst_dir/120/test_images
    done
done
