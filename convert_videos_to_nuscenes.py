import cv2
import os
import argparse
from pathlib import Path
from nuscenes.nuscenes import NuScenes

def process_video_to_nuscenes(input_dir, output_root, nusc, sensor_name="CAM_FRONT"):
    input_dir = Path(input_dir)
    output_root = Path(output_root)
    
    video_files = list(input_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos to process.")

    for video_path in video_files:
        scene_name = video_path.stem  # 获取 "scene-0784"
        
        # 1. 在 nuScenes 中查找到对应的 scene token
        scene_tokens = nusc.field2token('scene', 'name', scene_name)
        if not scene_tokens:
            print(f"Warning: Scene {scene_name} not found in nuScenes metadata. Skipping.")
            continue
        scene_token = scene_tokens[0]
        scene_rec = nusc.get('scene', scene_token)
        
        # 2. 获取该 scene 的第一帧和最后一帧的 sample 记录
        first_sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        last_sample_rec = nusc.get('sample', scene_rec['last_sample_token'])
        
        # 3. 获取特定传感器（CAM_FRONT）对应的第一个和最后一个 sample_data token
        first_sd_token = first_sample_rec['data'][sensor_name]
        last_sd_token = last_sample_rec['data'][sensor_name]
        
        # 4. 遍历 sample_data 链表，按顺序收集所有原图的 filename
        sd_filenames = []
        current_sd_token = first_sd_token
        while True:
            sd_rec = nusc.get('sample_data', current_sd_token)
            
            # sd_rec['filename'] 的格式自带目录，例如: 'samples/CAM_FRONT/n015-2018...jpg' 
            # 或者是 'sweeps/CAM_FRONT/n015-2018...jpg'
            sd_filenames.append(sd_rec['filename'])
            
            if current_sd_token == last_sd_token:
                break
            
            current_sd_token = sd_rec['next']
            if not current_sd_token:
                break # 链表异常断裂时的安全退出
                
        # 5. 打开视频并获取总帧数
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[{scene_name}] Video frames: {frame_count}, nuScenes frames: {len(sd_filenames)}")
        
        # 6. 核心断言：生成的视频帧数必须与 nuScenes 真值帧数完全一致
        assert frame_count == len(sd_filenames), (
            f"Frame count mismatch for {scene_name}! "
            f"Video has {frame_count}, but nuScenes defines {len(sd_filenames)}."
        )
        
        # 7. 按顺序读取视频并保存到原定路径
        for filename in sd_filenames:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Unexpected end of video stream in {scene_name}")
                
            # 拼接最终输出路径
            out_path = output_root / filename
            
            # 自动创建所需的 samples/CAM_FRONT 或 sweeps/CAM_FRONT 文件夹
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存图像
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
        cap.release()
        print(f"Successfully aligned and exported: {scene_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert generated videos back to nuScenes dataset structure")
    parser.add_argument("--input_dir", type=str, default="/mrtstorage/users/kwang/ppd/flux_30_wan_30/", help="Path to your re-rendered .mp4s")
    parser.add_argument("--output_root", type=str, default="ppd/flux_30_wan_30/", help="nuScenes root directory for saving images")
    parser.add_argument("--nusc_version", type=str, default="v1.0-trainval", help="nuScenes version (e.g., v1.0-trainval, v1.0-mini)")
    parser.add_argument("--nusc_dataroot", type=str, default="nuscenes", help="Path to the ORIGINAL nuScenes dataset (needed to read metadata JSONs)")
    args = parser.parse_args()

    # 初始化 nuScenes (需要读取原版数据集的 JSON 获取链表信息)
    print("Loading nuScenes metadata...")
    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_dataroot, verbose=False)
    
    process_video_to_nuscenes(args.input_dir, args.output_root, nusc)