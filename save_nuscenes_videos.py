import cv2
import argparse
import subprocess
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

def process_nuscenes_to_video(nusc, output_dir, fps=12, sensor_name="CAM_FRONT"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    val_scenes = create_splits_scenes()['val']
    
    for scene in nusc.scene:
        scene_name = scene['name']
        if scene_name not in val_scenes:
            continue
            
        first_sample_rec = nusc.get('sample', scene['first_sample_token'])
        last_sample_rec = nusc.get('sample', scene['last_sample_token'])
        
        first_sd_token = first_sample_rec['data'][sensor_name]
        last_sd_token = last_sample_rec['data'][sensor_name]
        
        current_sd_token = first_sd_token
        image_paths = []
        
        while True:
            sd_rec = nusc.get('sample_data', current_sd_token)
            img_path = nusc.get_sample_data_path(current_sd_token)
            image_paths.append(img_path)
            
            if current_sd_token == last_sd_token:
                break
                
            current_sd_token = sd_rec['next']
            if not current_sd_token:
                break
                
        if not image_paths:
            print(f"No images found for {scene_name}. Skipping.")
            continue
            
        first_img = cv2.imread(image_paths[0])
        height, width, _ = first_img.shape
        
        video_path = output_dir / f"{scene_name}.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24", 
            "-r", str(fps),      
            "-i", "-",          
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(video_path)
        ]
        
        # Start FFmpeg process
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            # Write raw bytes to stdin
            process.stdin.write(img.tobytes())
            
        # Close stdin and wait for FFmpeg to finish
        process.stdin.close()
        process.wait()
        
        print(f"[{scene_name}] Saved video with {len(image_paths)} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert nuScenes images to videos using FFmpeg")
    parser.add_argument("--output_dir", type=str, default="/mrtstorage/users/kwang/nuscenes_videos", help="Directory to save the .mp4 files")
    parser.add_argument("--nusc_version", type=str, default="v1.0-trainval", help="nuScenes version")
    parser.add_argument("--nusc_dataroot", type=str, default="nuscenes", help="Path to the ORIGINAL nuScenes dataset")
    parser.add_argument("--fps", type=int, default=12, help="Video framerate")
    args = parser.parse_args()

    print("Loading nuScenes metadata...")
    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_dataroot, verbose=False)
    
    process_nuscenes_to_video(nusc, args.output_dir, args.fps)