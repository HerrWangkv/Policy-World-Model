import gzip
import json
import glob
import os
import numpy as np
from pathlib import Path

def verify_direction_logic(data_root):
    json_files = glob.glob(os.path.join(data_root, "**/measurements/*.json.gz"), recursive=True)
    
    print(f"{'文件名':<15} | {'原始CMD':<6} | {'映射后':<6} | {'Y位移(3s)':<10} | {'实际转向'}")
    print("-" * 75)

    found_turning = 0
    for f_path in json_files:
        with gzip.open(f_path, 'rt') as f:
            data = json.load(f)
        
        # 获取原始命令和映射后的命令
        raw_cmd = data.get('command', -1)
        mapped_cmd = data.get('plan_command', -1)
        plan = np.array(data['plan'])
        
        # 检查 3s 处的 Y 轴位移 (Index 5)
        # NuScenes 右手系：Y > 0 是左转，Y < 0 是右转
        y_offset = plan[5][1] 
        
        # 只看有明显转向的帧 (位移 > 1.0米)
        if abs(y_offset) > 1.0:
            actual_dir = "左转 (Left)" if y_offset > 0 else "右转 (Right)"
            print(f"{Path(f_path).name:<15} | {raw_cmd:<8} | {mapped_cmd:<8} | {y_offset:>9.2f} | {actual_dir}")
            found_turning += 1
        
        if found_turning >= 10: # 找到10个样本就停止
            break

if __name__ == "__main__":
    # 指向你处理后的数据目录
    DATA_PATH = "./my_carla_dataset_pwm"
    verify_direction_logic(DATA_PATH)