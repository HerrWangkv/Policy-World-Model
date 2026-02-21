
import copy
from dataclasses import dataclass, field
import json
from torch import Tensor
from datasets import load_dataset, load_from_disk
import numpy as np
from data_utils.dataset_config import _get_rawvideo_dec,process_coco_image
from data_utils.dataset_config import OPENDV_LOCAL, OPENDV_MINI, OPENDV_FULL, NUSCENES_FRONT, NUSCENES_BACK, NUSCENES_FRONT_LEFT, NUSCENES_FRONT_RIGHT, NUSCENES_BACK_LEFT, NUSCENES_BACK_RIGHT
import os
from training.utils import get_config, flatten_omega_conf, image_transform
from PIL import Image
from llava.llava import conversation as conversation_lib
from nuscenes.utils.splits import create_splits_scenes
import math
from PIL import UnidentifiedImageError
local_rank = None
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union
import os
from navsim.common.dataclasses import SensorConfig, Scene, Trajectory, NAVSIM_INTERVAL_LENGTH
from hydra.utils import instantiate
from pathlib import Path
import lzma
import pickle
import os
import torch
import logging
from dataclasses import dataclass, field, asdict
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
DataConfig = {
    "OPENDV_LOCAL": [OPENDV_LOCAL],
    "OPENDV_MINI": [OPENDV_MINI],
    "OPENDV_FULL": [OPENDV_FULL],
}
prompt = {
    "generate_scene": "Draw image in front of a vehicle.",
    "desc": "Summarize what the image shows.",
    "action": "What should be the next move?",
    "plan": "Give the next 6 plans.",
    "image": "Draw the image of the next frame."
}
DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."

def preprocess_multimodal(sources):
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = sentence['value'].strip()

    return sources

def preprocess_v0(
        sources,
        tokenizer,
        return_system = False,
        max_len=None,
):
    has_image = False
    conv = conversation_lib.default_conversation.copy()
    roles = {"USER": conv.roles[0], "ASSISTANT": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversation_str = str(conv.get_prompt()).strip()
        conversations.append(conversation_str)
    if max_len is not None:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=max_len,
            truncation=True,
        ).input_ids
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "                   # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):        # loop for instances in a batch
        # total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(conv.sep2)  # in phi-2, pad_token_id == eos_token_id
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        pad_len = sum(target==tokenizer.pad_token_id)
        rounds = conversation.split(conv.sep2)              # handle multi-round conversation regarding one image
        cur_len = pad_len                                         # no bos token in phi, so set the initial len to 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids) + 1  # +1 for <|endoftext|>
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX#usr->-100


    input_ids_system = tokenizer(
        [SYSTEM_PROMPT for _ in range(len(conversations))],
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    if return_system == True:

        return dict(
            input_ids=input_ids,
            labels=targets,
            input_ids_system=input_ids_system
        )
    else:
        return dict(input_ids=input_ids,
            labels=targets,)
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    # tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_text = [instance["input_text"] for instance in instances]
        batch = dict(
            input_text=input_text,
        )
        for i in ['image_clip', 'image_vq', 'action']:
            if i in instances[0]:
                state_action = [instance[i] for instance in instances]

                new_images = []
                for image in state_action:
                    if type(image) is list:
                        for i in image:
                            new_images.append(i)
                    else:
                        new_images.append(image)
                images = new_images

                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch[i] = torch.stack(images)
                else:
                    batch[i] = images
                #image_vq(batch,T,C,H,W)
        return batch

class DatasetNuScenes(Dataset):#camera ready dataset
    def __init__(self,
                 config,
                 split,
                 version='v1.0-trainval',
                 aug_enable=False,
                 scene_split=False,
                 scene_name=None,
                 aug= {
                        'brightness': [0.9, 1.1],
                        'contrast': [0.9, 1.1],
                        'saturation': [0.9, 1.1],
                        'hue': [-0.05, 0.05],
                        'random_resized_crop_scale': (0.9, 1.0),
                        'random_resized_crop_ratio': (0.5, 0.6),

                        },
                 ):
        super(DatasetNuScenes, self).__init__()
        assert config.dataset.ctd.nuscenes_data_path is not None, "Either nusc or nuscenes_data_path"
        # with open(os.path.join(config.dataset.ctd.anno_path, f'nuscenes2d_ego_temporal_infos_{split}.pkl'), 'rb') as f:
        #     self.nus_ori_annos = pickle.load(f)['infos']
        # self.omini_anno_root = config.dataset.ctd.anno_path  # [[conv[qas]]]
        self.scenes = create_splits_scenes()
        self.image_file = config.dataset.ctd.image_file
        with open(os.path.join(config.dataset.ctd.image_file, f'CAM_FRONT_{split}_imgs_path.json'), 'r')as f:
            self.image_path = json.load(f)
        # self.image_root = config.dataset.ctd.image_root
        if split == 'train':
            with open(os.path.join(config.dataset.ctd.image_file, 'ominidrive', f'plan_{split}_filter_w_ego_w_cmd_1s_to_19s.json'), 'r') as f:
                self.omini_annos = json.load(f)
            self.scenes = self.scenes['train']
        elif split == 'val':
            if scene_split:
                with open(os.path.join(config.dataset.ctd.image_file, 'ominidrive', 'val_saparate_scene/val_1s_20s.json'), 'r') as f:
                    omini_annos = json.load(f)
                assert scene_name is not None, "require scene name"
                self.omini_annos = omini_annos[scene_name]
            else:
                # with open(os.path.join(config.dataset.ctd.omini_path, f'plan_{split}_filter_w_ego_w_cmd_1s_to_20s.json'), 'r') as f:
                with open(os.path.join(config.dataset.ctd.image_file, 'ominidrive', f'plan_{split}_filter_w_ego_w_cmd_1s_to_19s.json'), 'r') as f:
                    self.omini_annos = json.load(f)
            self.scenes = self.scenes['val']
        else:
            raise NotImplementedError
        self.camera_views= [k for k, v in config.dataset.ctd.views.items() if v is not '']
        self.num_images = config.dataset.ctd.segment_length#过去2s，那么就是2x8=16帧 frames condition
        self.condition_frames = config.dataset.ctd.condition_length #2 for one second
        # self.data = self._prepare_text_vqa()
        # random.shuffle(self.data)#字典，image file
        self.vq_processor = image_transform
        self.Con_resolution_h, self.Con_resolution_w = config.dataset.ctd.c_resolution
        self.resolution_h, self.resolution_w = config.dataset.ctd.d_resolution
        self.prev_frames = config.dataset.ctd.prev_frames  # 12 #1s
        self.next_frames = config.dataset.ctd.next_frames
        self.aug_enable = aug_enable
        self.aug = aug
        self.split = split
        self.collate_fn = DataCollatorForSupervisedNuScenes()
        self.fps = config.dataset.ctd.next_frames #12
    def _check_frame_images(self, frame_token):
        sample = self.nusc.get('sample', frame_token)
        for _ in range(self.num_images-1):
            if sample['next'] == '':
                return False
            sample = self.nusc.get('sample', sample['next'])
        return True
    def data_augmentation(self, images):

        con_len_count = 0
        new_images_context = []
        new_images = []
        tensor = transforms.ToTensor()
        for image_0 in images:
            if con_len_count in [0, 1]:
                image = transforms.Resize((self.Con_resolution_h, self.Con_resolution_w), interpolation=transforms.InterpolationMode.BICUBIC)(image_0)
                image = transforms.CenterCrop((self.Con_resolution_h, self.Con_resolution_w))(image)
                image = tensor(image) #/255
                image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
                new_images_context.append(image)

            image = transforms.Resize((self.resolution_h, self.resolution_w), interpolation=transforms.InterpolationMode.BICUBIC)(image_0)
            image = transforms.CenterCrop((self.resolution_h, self.resolution_w))(image)
            image = tensor(image) #/255
            image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
            new_images.append(image)
            con_len_count += 1
        return torch.stack(new_images_context,0) if new_images_context is not [] else None, torch.stack(new_images,0) if new_images_context is not [] else None
    def get_jittor_params(self,
            brightness: Optional[List[float]],
            contrast: Optional[List[float]],
            saturation: Optional[List[float]],
            hue: Optional[List[float]],
    ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h
    def get_crop_params(self, img: Tensor, scale: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop with fixed aspect ratio.

        Args:
            img (Tensor): Input image (C, H, W).
            scale (list): Range of scale of the origin size cropped.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop, keeping the original aspect ratio.
        """
        _, height, width = F.get_dimensions(img)
        area = height * width

        original_ratio = width / height

        for _ in range(3):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            h = int(round(math.sqrt(target_area / original_ratio)))
            w = int(round(h * original_ratio))
            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w
        if width > height:
            w = int(round(height * original_ratio))
            h = height
        else:
            h = int(round(width / original_ratio))
            w = width

        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
    def get_segment(self, num_frames, start_index, end_index, select_num, short_flag=False, mode=None):
        start_index = start_index
        end_index = end_index
        if start_index >= num_frames or end_index >= num_frames:
            print(f'start_index: {start_index}, start_index: {end_index},total_frames: {num_frames}')
            raise ValueError("start_index must be less than num_frames")
        indices = np.arange(start_index, end_index)
        if mode == 'dynamic':
            weights = indices - start_index + 1
        else:
            weights = np.ones_like(indices)
        # select_num index
        if short_flag:
            selected_indices = np.random.choice(indices, size=select_num, replace=True, p=weights/weights.sum())
        else:
            selected_indices = np.random.choice(indices, size=select_num, replace=False, p=weights/weights.sum())

        return selected_indices.tolist()
    def _load_image(self, img_path):
        # Implement your image loading logic here
        # For example, using PIL:

        img = Image.open(img_path)
        # print(f"Loading image from: {img_path}")
        # img = img.resize((224, 224))  # Resize if necessary
        # img = torch.tensor(np.array(img)).permute(2, 0, 1)  # Convert to tensor and rearrange dimensions
        return img

    def __len__(self):
        return len(self.omini_annos)

    def process_desc(self,desc):
        banned = ['rear', 'Behind', 'behind', 'right-rear', 'left-rear']
        sentences = desc.split('. ')
        filtered = []
        for s in sentences:
            f = True
            for b in banned:
                if b in s:
                    f = False
                    break
            if f:
                filtered.append(s)
        new_desc = ''
        for s in filtered:
            new_desc += s
            new_desc += '. '
        new_desc = new_desc[:-2]
        return new_desc

    def _get_images(self, frame_token):
        images = {}
        images_prev_root, images_next_root = self.image_path[frame_token]['prev'], self.image_path[frame_token]['next']
        len_prev_max = self.prev_frames
        len_next_max = self.next_frames
        #prev
        prev_img = []
        next_img = []
        if len(images_prev_root) >= len_prev_max:
            for i in range(len_prev_max):
                prev_img.append(self._load_image(os.path.join(self.image_file,images_prev_root[i-len_prev_max])))
        else:
            for i in range(len(images_prev_root)):
                prev_img.append(self._load_image(os.path.join(self.image_file, images_prev_root[i - len(images_prev_root)])))
        if len(images_next_root) >= len_next_max:
            for i in range(len_next_max):
                next_img.append(self._load_image(os.path.join(self.image_file, images_next_root[i])))
        else:
            for i in range(len(images_next_root)):
                next_img.append(self._load_image(os.path.join(self.image_file, images_next_root[i])))
        return prev_img, next_img
    def command_to_text(self,command):

        if command == 2: #'FORWARD'
            text_command = 'FORWARD'
        elif command == 0: #'LEFT'
            text_command = 'LEFT'
        elif command == 1:  # 'RIGHT'
            text_command = 'RIGHT'
        else:
            raise NotImplementedError
        return text_command
    def pad_frame(self,
                  prev_img_context,
                  prev_img_dynamic,
                  next_img_context,
                  next_img_dynamic):
        t_p,c_p,h_p,w_p = prev_img_dynamic.shape
        t_n,c_n,h_n,w_n = next_img_dynamic.shape
        # if t_p < self.fps or t_n<self.fps:
        #     print("debug in here")
        if prev_img_dynamic.shape[0] == 1:
            assert prev_img_context.shape[0]==1
            prev_img_context = torch.cat([prev_img_context, prev_img_context], dim=0)#assert prev_img_context.shape[0]==2
        prev_img_dynamic = torch.cat([torch.ones((self.prev_frames-t_p,c_p,h_p,w_p), dtype=torch.float32)*-100, prev_img_dynamic], dim=0)#pad using -100

        if next_img_dynamic.shape[0] == 1:
            assert next_img_context.shape[0] == 1
            next_img_context = torch.cat([next_img_context, next_img_context], dim=0)
        next_img_dynamic = torch.cat([next_img_dynamic, torch.ones((self.next_frames-t_n,c_n,h_n,w_n), dtype=torch.float32)*-100], dim=0)#pad using -100

        return prev_img_context, prev_img_dynamic, next_img_context, next_img_dynamic, t_p, t_n
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = {}
        omini_anno = self.omini_annos[idx][0]
        token = omini_anno[-1]['token']
        planning_a = omini_anno[-2]['plan'] #trajectory
        plan_mask = omini_anno[-2]['plan_mask']
        scene_name = omini_anno[-2]['scene_name']
        prev_img, next_img = self._get_images(token)
        images_prev_miss, images_next_miss = self.prev_frames-len(prev_img), self.next_frames-len(next_img)
        assert images_prev_miss>=0 and images_next_miss>=0

        prev_img_context, prev_img_dynamic = self.data_augmentation(prev_img)
        next_img_context, next_img_dynamic = self.data_augmentation(next_img)
        #using -100 as pad value
        prev_img_context, prev_img_dynamic, next_img_context, next_img_dynamic, t_p,t_n = self.pad_frame(prev_img_context, prev_img_dynamic, next_img_context, next_img_dynamic)
        assert [0,0] not in omini_anno[-2]['plan_mask'], print('plan_mask', omini_anno[-2]['plan_mask'])
        sample['prev_img_context'] = prev_img_context
        sample['prev_img_dynamic'] = prev_img_dynamic
        sample['next_img_context'] = next_img_context
        sample['next_img_dynamic'] = next_img_dynamic
        sample['idx'] = omini_anno[-2]['id']
        sample['planning_a'] = torch.tensor(planning_a, dtype=torch.float32)[0]
        sample['plan_mask'] = torch.tensor(plan_mask, dtype=torch.float32)[0]
        sample['real_len'] = torch.tensor([t_p, t_n], dtype=torch.int64)
        sample['ego_status'] = torch.tensor(omini_anno[-2]['ego_status'], dtype=torch.float32)
        sample['H_cmd'] = torch.tensor(omini_anno[-2]['plan_command'], dtype=torch.int64)
        sample['scene_name'] = scene_name
        # assert self.nus_ori_annos[omini_anno[-2]['id']]['token']==token
        return sample

@dataclass
class DataCollatorForSupervisedNuScenes(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        idx = [instance["idx"] for instance in instances if "idx" in instance]
        scene_name = [instance["scene_name"] for instance in instances if "scene_name" in instance]
        batch = dict(
            idx = idx,
            scene_name = scene_name,
        )

        for i in ['prev_img_context', 'prev_img_dynamic', 'next_img_context', 'next_img_dynamic','planning_a', 'real_len','ego_status','H_cmd','plan_mask']:
            if i in instances[0]:
                state = [instance[i] for instance in instances]

                new_images = []
                for image in state:
                    if isinstance(image, list):
                        for i in image:
                            new_images.append(i)
                    else:
                        new_images.append(image)
                images = new_images

                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch[i] = torch.stack(images)
                else:
                    batch[i] = images

        return batch

if '__main__' == __name__:
    pass