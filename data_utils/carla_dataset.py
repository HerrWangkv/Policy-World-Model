import os
import glob
import json
import gzip
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union

# Define the collator class that the eval script expects
class DataCollatorForSupervisedNuScenes(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        idx = [instance["idx"] for instance in instances if "idx" in instance]
        seq_idx = [instance["seq_idx"] for instance in instances if "seq_idx" in instance] # <--- 新增这行
        scene_name = [instance["scene_name"] for instance in instances if "scene_name" in instance]
        batch = dict(
            idx = idx,
            seq_idx = seq_idx,
            scene_name = scene_name,
        )

        # Batch all tensor fields required by the model
        for i in ['prev_img_context', 'prev_img_dynamic', 'next_img_context', 'next_img_dynamic','planning_a', 'real_len','ego_status','H_cmd','plan_mask']:
            if i in instances[0]:
                state = [instance[i] for instance in instances]
                new_images = []
                for image in state:
                    if isinstance(image, list):
                        for item in image:
                            new_images.append(item)
                    else:
                        new_images.append(image)
                
                images = new_images
                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch[i] = torch.stack(images)
                else:
                    batch[i] = images
        return batch

class DatasetNuScenes(Dataset):
    def __init__(self, config, split='val', **kwargs):
        super().__init__()
        self.data_root = config.dataset.json_root 
        
        self.Con_resolution_h, self.Con_resolution_w = config.dataset.ctd.c_resolution
        self.resolution_h, self.resolution_w = config.dataset.ctd.d_resolution
        self.prev_frames = config.dataset.ctd.prev_frames  # 12
        self.next_frames = config.dataset.ctd.next_frames  # 12
        
        # Initialize the collator
        self.collate_fn = DataCollatorForSupervisedNuScenes()
        self.frames_index = self._build_index()

    def _build_index(self):
        index = []
        scene_dirs = [d for d in glob.glob(os.path.join(self.data_root, '*')) 
                     if os.path.isdir(d) and os.path.basename(d) not in ['samples', 'sweeps']]
        
        for scene_dir in scene_dirs:
            scene_name = os.path.basename(scene_dir)
            meas_dir = os.path.join(scene_dir, "measurements")
            if not os.path.exists(meas_dir): continue
                
            json_files = sorted(glob.glob(os.path.join(meas_dir, "*.json.gz")))
            
            # --- 修改点 1: 剔除最后 30 帧 (3.0s)，防止截断轨迹干扰评估 ---
            valid_limit = len(json_files) - 30 
            if valid_limit <= 0: continue # 场景太短则跳过

            for i in range(valid_limit):
                index.append({
                    'scene_name': scene_name,
                    'seq_idx': i,
                    'json_path': json_files[i]
                })
        return index

    def __len__(self):
        return len(self.frames_index)

    def _find_image(self, scene_name, idx):
        filename = f"{scene_name}_{idx:04d}.jpg"
        for folder in ["samples", "sweeps"]:
            path = os.path.join(self.data_root, folder, "CAM_FRONT", filename)
            if os.path.exists(path):
                return path
        return None

    def _upsample_10_to_12(self, frames):
        if not frames: return []
        k = len(frames)
        target_len = 12
        indices = [int(i * k / target_len) for i in range(target_len)]
        return [frames[i] for i in indices]

    def _get_images(self, scene_name, current_idx):
        prev_img_12 = []
        for i in range(11, -1, -1):
            idx = max(0, current_idx - i) # 场景开始时重复第 0 帧
            img_path = self._find_image(scene_name, idx)
            if img_path:
                prev_img_12.append(Image.open(img_path).convert('RGB'))
                
        # 2. 获取未来序列 (Next)：从当前帧开始往后的 10 帧
        # 索引范围：[current_idx, ..., current_idx+9]
        next_img_12 = []
        for i in range(0, 12): # 从 0 开始，确保第一帧是当前帧
            idx = current_idx + i 
            img_path = self._find_image(scene_name, idx)
            
            # 如果越界（场景结束），则通过回溯寻找最后一张存在的图片
            # 这比简单的 idx - 1 更稳健
            temp_idx = idx
            while not img_path and temp_idx > 0:
                temp_idx -= 1
                img_path = self._find_image(scene_name, temp_idx)
            
            if img_path:
                next_img_12.append(Image.open(img_path).convert('RGB'))

        # 返回插值后的 12 帧序列
        return prev_img_12, next_img_12
        # # 1. 获取历史序列 (Prev)：包含当前帧在内的过去 10 帧
        # # 索引范围：[current_idx-9, ..., current_idx]
        # prev_img_10 = []
        # for i in range(9, -1, -1):
        #     idx = max(0, current_idx - i) # 场景开始时重复第 0 帧
        #     img_path = self._find_image(scene_name, idx)
        #     if img_path:
        #         prev_img_10.append(Image.open(img_path).convert('RGB'))
                
        # # 2. 获取未来序列 (Next)：从当前帧开始往后的 10 帧
        # # 索引范围：[current_idx, ..., current_idx+9]
        # next_img_10 = []
        # for i in range(0, 10): # 从 0 开始，确保第一帧是当前帧
        #     idx = current_idx + i 
        #     img_path = self._find_image(scene_name, idx)
            
        #     # 如果越界（场景结束），则通过回溯寻找最后一张存在的图片
        #     # 这比简单的 idx - 1 更稳健
        #     temp_idx = idx
        #     while not img_path and temp_idx > 0:
        #         temp_idx -= 1
        #         img_path = self._find_image(scene_name, temp_idx)
            
        #     if img_path:
        #         next_img_10.append(Image.open(img_path).convert('RGB'))

        # # 返回插值后的 12 帧序列
        # return self._upsample_10_to_12(prev_img_10), self._upsample_10_to_12(next_img_10)

    def data_augmentation(self, images):
        if not images: return None, None
        tensor = transforms.ToTensor()
        new_images_context, new_images = [], []
        
        for i, img in enumerate(images):
            if i in [0, 1]:
                img_c = transforms.Resize((self.Con_resolution_h, self.Con_resolution_w))(img)
                img_c = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(tensor(img_c))
                new_images_context.append(img_c)

            img_d = transforms.Resize((self.resolution_h, self.resolution_w))(img)
            img_d = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(tensor(img_d))
            new_images.append(img_d)
            
        return torch.stack(new_images_context, 0), torch.stack(new_images, 0)

    def pad_frame(self, prev_img_context, prev_img_dynamic, next_img_context, next_img_dynamic):
        t_p, c_p, h_p, w_p = prev_img_dynamic.shape if prev_img_dynamic is not None else (0, 3, self.resolution_h, self.resolution_w)
        t_n, c_n, h_n, w_n = next_img_dynamic.shape if next_img_dynamic is not None else (0, 3, self.resolution_h, self.resolution_w)
        
        if t_p < self.prev_frames:
            pad = torch.ones((self.prev_frames - t_p, c_p, h_p, w_p), dtype=torch.float32) * -100
            prev_img_dynamic = torch.cat([pad, prev_img_dynamic], dim=0) if prev_img_dynamic is not None else pad
        if prev_img_context is not None and prev_img_context.shape[0] == 1:
            prev_img_context = torch.cat([prev_img_context, prev_img_context], dim=0)
            
        if t_n < self.next_frames:
            pad = torch.ones((self.next_frames - t_n, c_n, h_n, w_n), dtype=torch.float32) * -100
            next_img_dynamic = torch.cat([next_img_dynamic, pad], dim=0) if next_img_dynamic is not None else pad
        if next_img_context is not None and next_img_context.shape[0] == 1:
            next_img_context = torch.cat([next_img_context, next_img_context], dim=0)
            
        return prev_img_context, prev_img_dynamic, next_img_context, next_img_dynamic, t_p, t_n

    def __getitem__(self, idx):
        info = self.frames_index[idx]
        with gzip.open(info['json_path'], 'rt') as f:
            data = json.load(f)
            
        prev_img, next_img = self._get_images(info['scene_name'], info['seq_idx'])
        prev_c, prev_d = self.data_augmentation(prev_img)
        next_c, next_d = self.data_augmentation(next_img)
        
        prev_c, prev_d, next_c, next_d, t_p, t_n = self.pad_frame(prev_c, prev_d, next_c, next_d)

        return {
            'prev_img_context': prev_c,
            'prev_img_dynamic': prev_d,
            'next_img_context': next_c,
            'next_img_dynamic': next_d,
            'idx': idx,
            'seq_idx': info['seq_idx'],
            'scene_name': info['scene_name'],
            'real_len': torch.tensor([t_p, t_n]),
            'planning_a': torch.tensor(data['plan'], dtype=torch.float32),
            'plan_mask': torch.tensor(data['plan_mask'], dtype=torch.float32),
            'ego_status': torch.tensor(data['ego_status'], dtype=torch.float32),
            'H_cmd': torch.tensor(data['plan_command'], dtype=torch.int64)
        }