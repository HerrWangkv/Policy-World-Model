import os
import json
import argparse
import math
import gzip
from tqdm import tqdm

CMD_MAP = {
    1: 0,
    2: 1,
    3: 2,
    4: 2
}

def convert_dataset(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 遍历所有场景 (例如 town01_route0_...)
    scenes = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    
    for scene_name in tqdm(scenes, desc="Processing Scenes"):
        scene_dir = os.path.join(args.input_dir, scene_name, "measurements")
        if not os.path.exists(scene_dir):
            continue

        # 获取所有 measurements 的 json.gz 文件并排序
        scene_data = sorted([os.path.join(scene_dir, f) for f in os.listdir(scene_dir) if f.endswith('.json.gz')])
        
        output_scene_dir = os.path.join(args.output_dir, scene_name, "measurements")
        os.makedirs(output_scene_dir, exist_ok=True)

        for current_idx, file_path in enumerate(scene_data):
            with gzip.open(file_path, 'rt') as f:
                data = json.load(f)

            # 修正1：键名改为 'pos_global'，同时翻转 Y 轴以匹配右手系
            ego_pos_raw = data.get('pos_global', [0.0, 0.0, 0.0])
            ego_x, ego_y = ego_pos_raw[0], -ego_pos_raw[1]
            
            # 修正2：你的数据中 theta 已经是弧度(3.14)，不需要 math.radians！
            # 并且将顺时针取反变为标准的逆时针
            theta = -data.get('theta', 0.0)

            plan = []
            plan_mask = []

            # 预测未来3秒，每0.5秒一个点，共6个点
            for i in range(1, 7):
                future_idx = current_idx + int(i * 0.5 * 10) # 假设10Hz
                
                if future_idx < len(scene_data):
                    with gzip.open(scene_data[future_idx], 'rt') as f:
                        future_data = json.load(f)
                    
                    # 同样使用 pos_global 并翻转 Y
                    future_pos_raw = future_data.get('pos_global', [0.0, 0.0, 0.0])
                    future_x, future_y = future_pos_raw[0], -future_pos_raw[1]
                    future_yaw = future_data.get('theta', 0.0)
                    
                    # 计算相对位移
                    dx = future_x - ego_x
                    dy = future_y - ego_y
                    
                    # 修正3：标准右手系旋转矩阵
                    local_x = dx * math.cos(theta) + dy * math.sin(theta)
                    local_y = -dx * math.sin(theta) + dy * math.cos(theta)
                    
                    plan.append([local_x, local_y, future_yaw - data.get('theta', 0.0)])
                    plan_mask.append([1.0, 1.0])
                else:
                    # 如果超出边界，填充最后一点并设掩码为0
                    plan.append(plan[-1] if plan else [0.0, 0.0, 0.0])
                    plan_mask.append([0.0, 0.0])

            # 构建符合评估脚本的格式
            processed_data = {
                "plan_command": CMD_MAP.get(data.get('command', 3), 2),
                "ego_status": [
                    data.get('speed', 0.0),
                    data.get('steer', 0.0),
                    data.get('throttle', 0.0),
                    data.get('brake', 0.0),
                    data.get('gear', 0)
                ],
                "plan": plan,
                "plan_mask": plan_mask
            }

            # 保存处理后的文件
            file_name = os.path.basename(file_path)
            save_file_path = os.path.join(output_scene_dir, file_name)
            
            with gzip.open(save_file_path, 'wt') as f:
                json.dump(processed_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CARLA measurements for model evaluation.")
    parser.add_argument("--input_dir", type=str, default="/mrtstorage/users/kwang/my_sunny_dataset", help="Path to raw CARLA dataset.")
    parser.add_argument("--output_dir", type=str, default="./my_carla_dataset_pwm", help="Path to save processed dataset.")
    args = parser.parse_args()
    
    convert_dataset(args)
    print("Conversion complete!")