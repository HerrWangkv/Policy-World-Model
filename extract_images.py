import cv2
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_frames_nuscenes_style(video_folder, output_root, sample_step=5):
    vid_dir = Path(video_folder)
    out_samples = Path(output_root) / "samples" / "CAM_FRONT"
    out_sweeps = Path(output_root) / "sweeps" / "CAM_FRONT"
    
    out_samples.mkdir(parents=True, exist_ok=True)
    out_sweeps.mkdir(parents=True, exist_ok=True)
    
    for vid_path in tqdm(list(vid_dir.glob("*.mp4")), desc="Extracting Videos"):
        scene_name = vid_path.stem
        cap = cv2.VideoCapture(str(vid_path))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            
            # Resize to match NuScenes standard if needed
            frame = cv2.resize(frame, (1600, 900))
            filename = f"{scene_name}_{frame_idx:04d}.jpg"
            
            # 10Hz / 5 = 2Hz (Keyframes go to samples)
            if frame_idx % sample_step == 0:
                cv2.imwrite(str(out_samples / filename), frame)
            # Remaining 8Hz go to sweeps
            else:
                cv2.imwrite(str(out_sweeps / filename), frame)
                
            frame_idx += 1
            
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos in NuScenes style.")
    parser.add_argument("video_dir", type=str, help="Directory containing input videos", default="/mrtstorage/users/kwang/my_sunny_videos/rgb")
    parser.add_argument("--output_dir", type=str, help="Directory to save extracted frames", default="./my_carla_dataset_pwm")
    parser.add_argument("--sample_step", type=int, default=5, help="Step size for keyframe sampling (default: 5)")
    args = parser.parse_args()
    print(f"Extracting from {args.video_dir}")
    VIDEO_DIR = args.video_dir
    OUTPUT_DIR = args.output_dir

    extract_frames_nuscenes_style(VIDEO_DIR, OUTPUT_DIR, sample_step=args.sample_step)