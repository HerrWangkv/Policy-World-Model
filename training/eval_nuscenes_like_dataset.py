# coding=utf-8
# Copyright 2024 HuggingFace, NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import warnings


warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
from pathlib import Path
from typing import Union
import numpy as np
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from data_utils.carla_dataset import DatasetNuScenes
from models import Showo
from models.modeling_showo import get_vq_model_class
from models.prompting_utils import UniversalPrompting, create_attention_mask_for_nusc
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from training.utils import flatten_omega_conf, AverageMeter

SYSTEM_PROMPT_LEN = 28
try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls) and 'mm_projector' not in name.split('.'):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1]) #unique linear layer name

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    if 'embed_tokens' in lora_module_names:
        lora_module_names.remove('embed_tokens')
    return list(lora_module_names)
def batch_forward(batch_size, input, forward, context_length=None, special_token=None, verbose=False):
    if context_length is None and special_token is None:
        return torch.cat([forward(input[i: i + batch_size], ) for i in trange(0, input.shape[0], batch_size, disable=not verbose)], dim=0)
    else:
        return torch.cat(
            [forward(input[i: i + batch_size], context_length=context_length, special_token=special_token ) for i in trange(0, input.shape[0], batch_size, disable=not verbose)],
            dim=0)

def img_token2pixel(image_tokens_ori, uni_prompting, vq_model, gen_image_token_ids=None):
    if gen_image_token_ids is not None:
        img_token = dict(context=image_tokens_ori["context"], dynamic=gen_image_token_ids)
    else:
        img_token = image_tokens_ori
    img_pixel, _ = vq_model.detokenize(img_token,
                                      offset_tokenzier=len(uni_prompting.text_tokenizer),
                                      sptids_dict=uni_prompting.sptids_dict,
                                      )  # (T-1,C,H,W)
    img_pixel = torch.clamp((img_pixel + 1.0) / 2.0, min=0.0, max=1.0)
    return img_pixel
def video_concate(o_images, r_images, p_images, context_length=None):
    len_o = len(o_images)
    len_r = len(r_images)
    len_p = len(p_images)

    max_len = max(len_o, len_r, len_p)
    t2d_v = []
    for i in range(max_len):
        i_o = o_images[i % len_o]
        i_r = r_images[i % len_r]
        i_p = p_images[i % len_p]
        t2d_v.append(np.concatenate((i_o, i_r, i_p), axis=-2))
    return t2d_v
def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config_path = 'configs/sft_nuscenes_like/carla.yaml' #path/to/nuscenes.config
    config = OmegaConf.load(config_path)
    # config = get_config()
    # Enable TF32 on Ampere GPU
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    log_dir = os.path.join(config.experiment.output_dir, "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=log_dir,
        split_batches=True,
    )

    if accelerator.is_local_main_process:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{time_str}_train.log") if accelerator.is_local_main_process else None

    if accelerator.is_local_main_process:
        # Formatter created
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            "%m/%d/%Y %H:%M:%S"
        )


        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

        # StreamHandler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logging.getLogger().addHandler(stream_handler)

    os.environ["WANDB_MODE"] = "offline"  # debug
    total_batch_size_per_gpu = config.training.batch_size_train_nus #must have context frame tokens
    total_batch_size = config.training.batch_size_train_nus* config.training.gradient_accumulation_steps


    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = total_batch_size_per_gpu

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.output_dir,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.eval_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    # unified prompting for show-o
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sod|>", "<|eod|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2d|>", "<|act|>", "<|lvg|>"# special token
                                       ),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    print('special tokens : \n', uni_prompting.sptids_dict)
    # Initialize model
    if config.model.showo.load_from_showo:
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(accelerator.device)
        if config.model.showo.vocab_size != model.vocab_size:
            model.showo.resize_token_embeddings(config.model.showo.vocab_size)
            model.config.codebook_size = config.model.showo.codebook_size
            model.config.vocab_size = config.model.showo.vocab_size
            model.vocab_size = config.model.showo.vocab_size
            model.output_size = config.model.showo.vocab_size
            model.config.mask_token_id = model.config.vocab_size - 1
            model.mask_token_id = model.config.vocab_size - 1
    else:
        model = Showo(**config.model.showo).to(accelerator.device)
    #codebook expanding
    if config.model.showo.dynamic_size:
        dynamic_size = config.model.showo.dynamic_size
        model.resize_dynamic_size(dynamic_size, 'sft', config)
        from nuscenes_kit.Trj_eval import Planning_Evaluator_mask
        trj_evaluator_val = Planning_Evaluator_mask(config.dataset.ctd.nuscenes_data_path, config.experiment.eval.trj_anno_path)
        vq_name = get_vq_model_class(config.model.vq_model.type)
        vq_model = vq_name(config_exps=config,
                           num_vq_embeddings=config.model.vq_model.num_vq_embeddings,
                           num_dyn_embeddings=config.model.vq_model.num_dyn_embeddings)

        if config.model.vq_model.get("pretrained_model_path", None):
            from safetensors.torch import load_file
            state_dict = load_file(config.model.vq_model.pretrained_model_path)  # ['model']
            vq_model.load_state_dict(state_dict, strict=True)
        vq_model.eval()
        vq_model.requires_grad_(False)

    if config.model.showo.resume_from_pretrain:
        logger.info('load video pretrain from {}'.format(config.model.showo.resume_from_pretrain))
        # weights = torch.load(config.model.showo.resume_from_pretrain)
        model.load_state_dict(torch.load(config.model.showo.resume_from_pretrain), strict=False)
    else:
        logger.info('no video pretrain from {}'.format(config.model.showo.resume_from_pretrain))
    ##################################
    #   Optimizer and LR scheduler   #
    ##################################
    optimizer_config = config.optimizer.params
    if config.training.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32*2,
            target_modules=find_all_linear_names(model),
            lora_dropout=0.01,
            bias= "none",
            task_type="CAUSAL_LM",
        )
        print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )
    logger.info("Creating dataloaders and lr_scheduler")
    dataset_config = config.dataset.params
    ####################
    #     Dataloader   #
    ####################

    if config.dataset.dataset_use == "sft_nuscenes":
        total_batch_size_without_accum = config.training.batch_size_train_nus * accelerator.num_processes
        total_batch_size = (total_batch_size_without_accum * config.training.gradient_accumulation_steps)
        dataset_nus_val = DatasetNuScenes(config=config, split='val', aug_enable=False)
        print('process index : ',
          accelerator.process_index, ', total_gpus:', accelerator.num_processes,
          "Length of dataset_val:", len(dataset_nus_val), "samples")

        if accelerator.num_processes > 1:
            sampler_nusc_val = DistributedSampler(dataset_nus_val,
                                         num_replicas=accelerator.num_processes,
                                         rank=accelerator.process_index,
                                         shuffle=True,
                                         seed=config.training.seed
                                         )
            shuffle_val = False
        else:
            sampler_nusc_val = None
            shuffle_val = False
        val_dataloader_nusc = DataLoader(dataset_nus_val, batch_size=config.training.batch_size_val_nus,
                                          sampler=sampler_nusc_val, collate_fn=dataset_nus_val.collate_fn,
                                          shuffle=shuffle_val, num_workers=dataset_config.num_workers)
        
    else:
        raise ValueError(f"Unsupported dataset")

    ####################
    #     Load ckpt    #
    ####################
    global_step = 0

    if config.experiment.eval_from_checkpoint and config.experiment.eval_only:
        path = config.experiment.eval.eval_dir
        logger.info(f"only evaluation from ckpt:{path}")
        if path is not None:

            accelerator.print(f"evaluate from checkpoint {path}")
            # state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
            model.load_state_dict(torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu"), strict=True)
            # del state_dict

    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    vq_model = vq_model.to(accelerator.device)
    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.showo.model.embed_tokens.weight.dtype


    ####################
    #     Training     #
    ####################

    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    @torch.no_grad()
    def prepare_inputs_and_labels(
            prev_context: Union[torch.FloatTensor, torch.LongTensor],
            prev_dynamic: Union[torch.FloatTensor, torch.LongTensor],
            next_context: Union[torch.FloatTensor, torch.LongTensor],
            next_dynamic: Union[torch.FloatTensor, torch.LongTensor],
            planning: Union[torch.FloatTensor, torch.LongTensor],
            input_caption=None,
            condition_len=None,
            real_len=None,
            mode=None,
            ego_status=None,
            H_cmd=None,
            is_train: bool = True,

    ):
        input_ids_prev, labels_prev = vq_model.tokenize(prev_dynamic,
                                              context_pixel_values=prev_context,
                                              context_length=condition_len,
                                              special_token=uni_prompting.sptids_dict,
                                              return_label=False)# (batch*T,3,H,W)
        input_ids_next, labels_next = vq_model.tokenize(next_dynamic,
                                              context_pixel_values=next_context,
                                              context_length=condition_len,
                                              special_token=uni_prompting.sptids_dict)
        labels_prev = {key_label: torch.ones_like(labels_prev[key_label])*-100 for key_label in labels_prev}

        vocab_offset = len(uni_prompting.text_tokenizer)# add offset
        for k, v in input_ids_prev.items():
            mask = (v > 0) & (v < (vocab_offset - len(uni_prompting.sptids_dict)))
            input_ids_prev[k][mask] += vocab_offset
            if k in labels_prev:
                mask_label = (labels_prev[k] > 0) & (labels_prev[k] < (vocab_offset - len(uni_prompting.sptids_dict)))
                labels_prev[k][mask_label] += vocab_offset

        for k, v in input_ids_next.items():
            mask = (v > 0) & (v < (vocab_offset - len(uni_prompting.sptids_dict)))
            input_ids_next[k][mask] += vocab_offset
            if k in labels_next:
                mask_label = (labels_next[k] > 0) & (labels_next[k] < (vocab_offset - len(uni_prompting.sptids_dict)))
                labels_next[k][mask_label] += vocab_offset

        action_num = planning.shape[1]
        if input_caption is not None:
            texts =dict(input_caption=input_caption)
        else:
            texts = dict(input_caption=[""]*real_len.shape[0])
        final_input_ids, labels = uni_prompting((texts, input_ids_prev, input_ids_next, labels_prev, labels_next, real_len, action_num, ego_status, H_cmd), mode)
        return final_input_ids, labels, [input_ids_prev, input_ids_next]

    num_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    print("Num of Parameters require gradients: {}M".format(num_params / 1e6))
    if config.experiment.eval_only:
        eval_logs = evaluate(model,
                             config,
                             mask_dtype,
                             accelerator,
                             global_step,
                             uni_prompting,
                             val_dataloader_nusc,
                             trj_evaluator_val,
                             prepare_inputs_and_labels)
        return eval_logs


@torch.no_grad
def evaluate(model,
             config,
             mask_dtype,
             accelerator,
             global_step,
             uni_prompting,
             eval_dataloader,
             trj_evaluator_val,
             prepare_inputs_and_labels):

    model.eval()
    future_seconds = 3
    l2, cnt = np.zeros(2 * future_seconds), 0
    colls = [0., 0., 0.]
    action_pair = []
    # mse_values, psnr_values, ssim_values, lpips_values, fvds, desc_a_pair, action_pair = [], [], [], [], [], [], []
    eval_iters = min(len(eval_dataloader), config.experiment.eval.max_eval_iters)
    bar = tqdm(range(eval_iters), desc="validation", disable=not accelerator.is_local_main_process)
    logger.info("validation ...")
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    output_root = config.experiment.output_dir
    os.makedirs(output_root, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            if i == config.experiment.eval.max_eval_iters:
                break
            prev_context = batch['prev_img_context'].to(accelerator.device)
            prev_dynamic = batch['prev_img_dynamic'].to(accelerator.device)
            next_context = batch['next_img_context'].to(accelerator.device)
            next_dynamic = batch['next_img_dynamic'].to(accelerator.device)
            planning_gt = batch['planning_a'].to(accelerator.device)
            plan_mask_gt = batch['plan_mask'].to(accelerator.device)
            real_len = batch['real_len'].to(accelerator.device)
            ego_status = batch['ego_status'].to(accelerator.device) if config.experiment.add_ego else None
            H_cmd = batch['H_cmd'].to(accelerator.device) if config.experiment.add_cmd else None

            ego_status = batch['ego_status'].to(accelerator.device, non_blocking=True) if config.experiment.add_ego else None
            H_cmd = batch['H_cmd'].to(accelerator.device, non_blocking=True) if config.experiment.add_cmd else None

            prev_context = prev_context.to(accelerator.device, non_blocking=True)
            prev_dynamic = prev_dynamic.to(accelerator.device, non_blocking=True)
            next_context = next_context.to(accelerator.device, non_blocking=True)
            next_dynamic = next_dynamic.to(accelerator.device, non_blocking=True)
            planning_gt = planning_gt.to(accelerator.device, non_blocking=True)
            plan_mask_gt = plan_mask_gt.to(accelerator.device, non_blocking=True)
            real_len = batch['real_len'].to(accelerator.device)
            ego_status = batch['ego_status'].to(accelerator.device) if config.experiment.add_ego else None
            H_cmd = batch['H_cmd'].to(accelerator.device) if config.experiment.add_cmd else None

            context_length = config.dataset.ctd.context_length
            input_ids, labels, image_tokens_ori = prepare_inputs_and_labels(prev_context=prev_context,
                                                                            prev_dynamic=prev_dynamic,
                                                                            next_context=next_context,
                                                                            next_dynamic=next_dynamic, #PLACEHOLDER, torch.zeros_like(next_dynamic),#
                                                                            planning=torch.zeros_like(planning_gt), #PLACEHOLDER
                                                                            # input_caption=desc_a,
                                                                            condition_len=context_length,
                                                                            real_len=real_len,
                                                                            ego_status=ego_status,
                                                                            H_cmd=H_cmd,
                                                                            mode='nusc_add_front')  # pad:50295, end token

            attention_mask = create_attention_mask_for_nusc(input_ids,  # (B,1,L,L)
                                                            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                            sod_id=int(uni_prompting.sptids_dict['<|sod|>']),
                                                            eod_id=int(uni_prompting.sptids_dict['<|eod|>']),
                                                            rm_pad_in_image=True,
                                                            return_inverse_mask=True)
            attention_mask = attention_mask.to(mask_dtype)
            sod_input_idx = torch.where(input_ids == uni_prompting.sptids_dict['<|sod|>'].to(input_ids.device))[1].unique()
            eod_input_idx = torch.where(input_ids == uni_prompting.sptids_dict['<|eod|>'].to(input_ids.device))[1].unique()
            init_next_frame_tokens = input_ids[:, sod_input_idx[[prev_dynamic.shape[1]-1]]:eod_input_idx[[prev_dynamic.shape[1]-1]]+1]
            action_len = planning_gt.shape[1]
            #add cmd token
            input_embed = model.showo.model.embed_tokens(input_ids[:, :sod_input_idx[[prev_dynamic.shape[1]-1]]])
            pad_info = 0
            mmu_index = torch.where(input_ids == uni_prompting.sptids_dict['<|sot|>'].to(input_ids.device))[1].unique()
            if ego_status is not None:
                ego_token = model.ego_forward(ego_status.to(input_embed.dtype))
                input_embed[:, mmu_index[0]-1, :] = ego_token
                pad_info = 1
            if H_cmd is not None:
                cmd_queries = model.cmd_queries(H_cmd).to(input_embed.dtype)
                input_embed[:, mmu_index[0]-1-pad_info, :] = cmd_queries
            with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):  # uni_prompting.sptids_dict as input
                gen_qa_token_ids, gen_image_token_ids, gen_trj, eot_index = accelerator.unwrap_model(model).nusc_gen(input_embed=input_embed,
                                            attention_mask=attention_mask[:, :, :sod_input_idx[[prev_dynamic.shape[1]-1]], :sod_input_idx[[prev_dynamic.shape[1]-1]]],
                                            init_next_frame_embed=model.showo.model.embed_tokens(init_next_frame_tokens),
                                            config=config,
                                            labels=labels,
                                            action_len=action_len,
                                            uni_prompting=uni_prompting
                                            )

            #####################################
            # ---------video metrics------------#
            #####################################

            if config.experiment.eval.use_fvd:
                raise NotImplementedError("FVD computation is currently disabled due to the need for a large batch size. Please enable it when you have sufficient resources.")

            # #video metrics:mse,psnr,ssim,lpips
            if config.experiment.eval.use_frame_metrics:
                raise NotImplementedError("Frame-level metric computation is currently disabled due to the need for a large batch size. Please enable it when you have sufficient resources.")

            #####################################
            # ---------text  metrics------------#
            #####################################
            #text metrics:pass

            if config.experiment.eval.use_text_metrics:
                raise NotImplementedError("Text metric computation is currently disabled. Please enable it when you have sufficient resources.")

            #####################################
            # ---------action metrics-----------#
            #####################################
            # action metrics
            if config.experiment.eval.use_trj_metrics:
                # 假设 batch 是当前 dataloader 返回的原始数据包
                # 需要确保这些字段在调用时是可用的
                for b in range(planning_gt.shape[0]):
                    raw_seq_idx = batch['seq_idx'][b]
                    frame_seq_idx = raw_seq_idx.item() if hasattr(raw_seq_idx, 'item') else raw_seq_idx
                    action_pair.append({
                        'plan': gen_trj[b].detach().cpu(),
                        'gt': planning_gt[b].detach().cpu(),
                        'mask': plan_mask_gt[b].detach().cpu(),
                        # --- 新增 Debug 字段 ---
                        'command': batch['H_cmd'][b].item() if 'H_cmd' in batch else -1,
                        'speed': batch['ego_status'][b][0].item() if 'ego_status' in batch else 0.0,
                        'yaw_rate': batch['ego_status'][b][-1].item() if 'ego_status' in batch else 0.0,
                        'scene_name': batch['scene_name'][b] if 'scene_name' in batch else "unknown",
                        'frame_idx': frame_seq_idx,
                    })

            bar.update(1)

        eval_logs = {}
        if accelerator.is_main_process:
            if config.experiment.eval.use_trj_metrics and len(action_pair) > 0:
                l2_1s, l2_2s, l2_3s = 0.0, 0.0, 0.0
                cnt_1s, cnt_2s, cnt_3s = 0, 0, 0
                
                # 用于统计不同指令下的平均 L2
                cmd_stats = {} # {cmd_id: [total_l2_3s, count]}

                for pair in action_pair:
                    pred, gt, mask = pair['plan'], pair['gt'], pair['mask']
                    cmd = pair['command']
                    
                    # 记录 3s 处的单帧误差
                    current_l2_3s = 0.0
                    
                    if mask[1, 0] > 0:
                        l2_1s += torch.norm(pred[1] - gt[1, :2]).item()
                        cnt_1s += 1
                    if mask[3, 0] > 0:
                        l2_2s += torch.norm(pred[3] - gt[3, :2]).item()
                        cnt_2s += 1
                    if mask[5, 0] > 0:
                        err_3s = torch.norm(pred[5] - gt[5, :2]).item()
                        l2_3s += err_3s
                        cnt_3s += 1
                        current_l2_3s = err_3s

                    # 按指令统计 L2 (帮助判断是否映射反了)
                    if cmd not in cmd_stats: cmd_stats[cmd] = [0.0, 0]
                    cmd_stats[cmd][0] += current_l2_3s
                    cmd_stats[cmd][1] += 1
                
                eval_logs = {
                    "L2_1s": l2_1s / max(1, cnt_1s),
                    "L2_2s": l2_2s / max(1, cnt_2s),
                    "L2_3s": l2_3s / max(1, cnt_3s),
                    "L2_avg": (l2_1s + l2_2s + l2_3s) / (max(1, cnt_1s) + max(1, cnt_2s) + max(1, cnt_3s))
                }

                # 打印各指令下的表现
                for cmd, vals in cmd_stats.items():
                    avg_l2 = vals[0] / max(1, vals[1])
                    logger.info(f"Command {cmd} (0:L, 1:R, 2:F) -> Avg L2_3s: {avg_l2:.4f} (n={vals[1]})")

                logger.info(f"Trajectory Metrics: {eval_logs}")
                accelerator.log(eval_logs, step=global_step + i)

            # 保存结果到 JSON
            out_put_root = os.path.join(config.experiment.output_dir, "validation_only" if config.experiment.eval_only else f"step_{global_step+i}")
            os.makedirs(out_put_root, exist_ok=True)
            
            save_path = os.path.join(out_put_root, f"traj_pred_{accelerator.process_index}.json")
            with open(save_path, "w") as f:
                serializable_analys = []
                for p in action_pair:
                    entry = {}
                    for k, v in p.items():
                        if torch.is_tensor(v):
                            entry[k] = v.tolist()
                        else:
                            entry[k] = v
                    # 计算每对的 L2 存入 json，方便之后直接看哪个样本跑飞了
                    entry['l2_3s_sample'] = torch.norm(p['plan'][5] - p['gt'][5, :2]).item() if p['mask'][5, 0] > 0 else 0
                    serializable_analys.append(entry)
                    
                json.dump(serializable_analys, f, indent=4)
            logger.info(f"Saved debug trajectories to {save_path}")

    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()

    model.train()
    return eval_logs
if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    main()
