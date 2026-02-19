input_dir=data/pingpong
scene_name=$(basename "$input_dir")

image_dir=images
time=$(date "+%Y-%m-%d_%H:%M:%S")
output_dir="results/${scene_name}/${time}"

python run_system.py \
  --calib config/camera_intrinsics/usbcam.yaml \
  -s $input_dir \
  -i $image_dir \
  -m $output_dir \
  -d selfCaptured \
  --config config/base.yaml \
  --device_frontend cuda:0 \
  --device_backend cuda:0 \
  --device_mapper cuda:0 \
  --device_shared cpu \
  --downsampling 2.0 \
  --test_hold 8 \
  --use_all_frames \
  --base_model h3dgsv3 \
  --num_key_iterations 20 \
  --num_common_iterations 10 \
  --save_at_finetune_iteration 10000 \
  --local_feat_dim 16 \
  --global_feat_dim 16 \
  --visible_threshold 0 \
  --gs_add_ratio 1.0 \
  --covariance_filter \
  --point_fusion_frontend \
  --accurate_loop_closure