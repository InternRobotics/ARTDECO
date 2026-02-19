import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import json

# 提取数字部分进行排序
def extract_number(filename):
    return int(filename.split("DSC")[-1].replace('DSC', '').replace('.JPG', '').replace('.jpg', '').replace('dsc', ''))

 
def calculate_relative_pose_metrics(main_path):
    """
    计算每个子文件夹中相邻帧间的相对位姿指标
    
    Args:
        main_path: 包含多个子文件夹的主路径
    
    Returns:
        dict: 每个子文件夹的指标统计
    """
    results = {}
    
    # 遍历所有子文件夹
    nums_folder = 0
    for subfolder in os.listdir(main_path):
        print("processing folder:", nums_folder, subfolder)
        subfolder_path = os.path.join(main_path, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
            
        metadata_path = os.path.join(subfolder_path, "scene_metadata.npz")
        
        if not os.path.exists(metadata_path):
            continue
            
        # 加载数据
        image_datas = np.load(metadata_path)
        poses = image_datas["trajectories"]  # N x 4 x 4
        image_names = image_datas["images"]  # N str
        
        # 按图片名称排序
        sorted_indices = np.argsort([extract_number(x) for x in image_names])
        poses = poses[sorted_indices]
        
        # 计算相邻帧间的相对位姿
        angles = []
        translations = []
        
        end_at = 50

        for i in range(len(poses) - 1 - end_at):
            # 当前帧和下一帧的位姿 (Twc)
            Twc_curr = poses[i]
            Twc_next = poses[i + 1]
            
            # 检查是否有nan
            if np.isnan(Twc_curr).any() or np.isnan(Twc_next).any():
                continue
                
            # 计算相对位姿: T_rel = inv(Twc_curr) @ Twc_next
            T_rel = np.linalg.inv(Twc_curr) @ Twc_next
            
            # 提取相对平移
            rel_translation = T_rel[:3, 3]
            translation_magnitude = np.linalg.norm(rel_translation)
            translations.append(translation_magnitude)
            
            # 提取相对旋转角度
            rel_rotation_matrix = T_rel[:3, :3]
            rotation = R.from_matrix(rel_rotation_matrix)
            
            # 转换为角度 (度)
            angle_radians = rotation.magnitude()
            angle_degrees = np.degrees(angle_radians)
            angles.append(angle_degrees)
        
        # 计算统计指标
        if len(angles) > 0 and len(translations) > 0:
            results[subfolder] = {
                'mean_angle': np.mean(angles),
                'max_angle': np.max(angles),
                'mean_translation': np.mean(translations),
                'max_translation': np.max(translations)
            }
        else:
            results[subfolder] = {
                'mean_angle': 0.0,
                'max_angle': 0.0,
                'mean_translation': 0.0,
                'max_translation': 0.0
            }
        nums_folder += 1
        print(np.max(angles), np.mean(angles), np.max(translations), np.mean(translations))
    return results
 
# 存储为JSON格式
def save_metrics_json(metrics, output_path):
    """保存指标为JSON格式"""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
 
# 存储为pickle格式  
def save_metrics_pickle(metrics, output_path):
    """保存指标为pickle格式"""
    with open(output_path, 'wb') as f:
        pickle.dump(metrics, f)
 
# 存储为numpy格式
def save_metrics_npz(metrics, output_path):
    """保存指标为npz格式"""
    np.savez(output_path, **metrics)
 
# 完整使用示例
if __name__ == "__main__":
    main_path = "/nas/shared/pjlab_lingjun_landmarks/yumulin_group/nerf_data/scannetpp_v2/scannetpp_processed/"
    
    # 计算指标
    metrics = calculate_relative_pose_metrics(main_path)
    
    # 存储到不同格式
    save_metrics_json(metrics, "pose_metrics.json")
    
    print("指标已保存完成")
    print(f"处理了 {len(metrics)} 个子文件夹")