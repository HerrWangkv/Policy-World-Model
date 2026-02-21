import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_cases(json_path, data_root, mode='worst', num_samples=3):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 过滤出行驶中的样本 (速度 > 1m/s)
    valid_samples = [s for s in data if s.get('speed', 0) > 1.0 and s.get('mask', [[0]])[5][0] > 0]
    
    if not valid_samples:
        print("没有找到有效行驶样本。")
        return

    # 根据 mode 选择样本
    if mode == 'worst':
        # 按 L2 误差从大到小排序，挑出最糟糕的样本
        valid_samples.sort(key=lambda x: x.get('l2_3s_sample', 0), reverse=True)
        selected = valid_samples[:num_samples]
    elif mode == 'random':
        # 随机挑选样本
        selected = random.sample(valid_samples, min(num_samples, len(valid_samples)))
    else:
        print(f"未知模式 '{mode}'，请使用 'worst' 或 'random'。")
        return

    cmd_map = {0: 'Left', 1: 'Right', 2: 'Forward', -1: 'Unknown'}

    # 创建画布：左边画图片，右边画轨迹
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    if num_samples == 1: axes = [axes]

    for i, sample in enumerate(selected):
        ax_img = axes[i][0]
        ax_traj = axes[i][1]

        # 1. 动态拼接图片名
        scene_name = sample.get('scene_name', 'unknown')
        frame_idx = sample.get('frame_idx', 0)
        img_name = f"{scene_name}_{int(frame_idx):04d}.jpg"
        
        # 尝试从 samples 或 sweeps 文件夹读取图片
        img_path = os.path.join(data_root, 'samples', 'CAM_FRONT', img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_root, 'sweeps', 'CAM_FRONT', img_name)
            
        # 2. 画左侧的真实图片
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax_img.imshow(img)
            ax_img.set_title(f"Camera View: {img_name}", fontsize=10)
        else:
            ax_img.text(0.5, 0.5, 'Image File Not Found', ha='center', va='center')
            ax_img.set_title(f"Missing: {img_name}", fontsize=10)
        ax_img.axis('off')

        # 3. 画右侧的轨迹对比图
        plan = np.array(sample['plan'])
        gt = np.array(sample['gt'])
        
        # 坐标系: X向前方，Y向左方
        ax_traj.plot(gt[:, 1], gt[:, 0], 'bo-', label='Ground Truth', linewidth=2)
        ax_traj.plot(plan[:, 1], plan[:, 0], 'rx--', label='Prediction', linewidth=2)
        ax_traj.plot(0, 0, 'ks', label='Ego Vehicle', markersize=8)

        cmd = cmd_map.get(sample.get('command', -1), 'Unknown')
        speed = sample.get('speed', 0.0)
        l2 = sample.get('l2_3s_sample', 0.0)

        ax_traj.set_title(f"Command: {cmd} | Speed: {speed:.1f} m/s | L2 Error: {l2:.2f}m", fontsize=11)
        ax_traj.set_xlabel('Lateral (Y) [m]')
        ax_traj.set_ylabel('Longitudinal (X) [m]')
        ax_traj.legend(loc='upper left')
        ax_traj.grid(True)
        ax_traj.axis('equal')
        ax_traj.invert_xaxis()
    plt.tight_layout()
    save_filename = f'debug_{mode}_cases.png'
    plt.savefig(save_filename, dpi=150)
    print(f"已成功生成包含图像的轨迹对比图: {save_filename}")

if __name__ == "__main__":
    DATA_ROOT = "my_carla_dataset_pwm" 
    json_file = 'resume_checkpoint/sft_nuscenes/validation_only/traj_pred_0.json'
    
    # 你可以在这里切换 'random' 或 'worst'
    visualize_cases(json_file, DATA_ROOT, mode='random', num_samples=3)