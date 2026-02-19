#!/bin/bash
work_dir=$(pwd)
proj_dir=/cpfs04/shared/IDC_yumulin_group/Artdeco-V1
base_dir=$proj_dir/dataset/raw
base_data_dir=$base_dir/self_captured_baselines
data_dirs=$base_data_dir/08_05_canon-r8

result_dir=$proj_dir/Results
img_dir=images
run() { printf '%q ' "$@"; echo; "$@"; }

for data_dir in $data_dirs; do
    calib_file="$work_dir/config/camera_intrinsics/${data_dir##*_}.yaml"
    for src_dir in $data_dir/MVI_0636 $data_dir/MVI_0640 $data_dir/MVI_0644 $data_dir/MVI_0650; do
        [[ ! -d "$src_dir/$img_dir" ]] && { echo "No images/ in $src_dir"; continue; }
        dst_dir=$result_dir/"${src_dir#$base_dir/}"/artdeco-norigidtransform
        [[ -e "$dst_dir" && -f "$dst_dir/metadata.json" ]] && { echo "$dst_dir contains results"; continue; }
        scene_dir="${src_dir#$proj_dir/*/}"
        echo $calib_file
        CUDA_VISIBLE_DEVICES=0 run taskset -c 0-9 python run_system.py \
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
            --save_at_finetune_epoch 218 \
            --num_iterations 30 \
            --downsampling 2.0 \
            --test_hold 8 \
            --lr_poses 0.0 \
            --use_loop_closure \
            --save_point_could \
            --save_to_data_for_gsplat
            # --rigid_transform_gaussians 
        python scripts/img2vid.py $dst_dir/test_images $dst_dir/218/test_images
        break 2
    done
done
