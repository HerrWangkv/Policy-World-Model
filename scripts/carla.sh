#!/bin/bash

# conda activate pwm
export NCCL_P2P_DISABLE=1
export PYTHONPATH=./:$PYTHONPATH
accelerate launch --config_file accelerate_configs/2_gpus_deepspeed_zero2.yaml --main_process_port=8954 training/eval_nuscenes_like_dataset.py
