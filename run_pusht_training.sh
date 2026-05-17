#!/bin/bash
set -e

cd /home/anthonyliang/Documents/vla_foundry_ant
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

torchrun --nproc_per_node=1 --master_port=29511 \
    vla_foundry/main.py \
    --config_path pusht_training_config.yaml \
    2>&1 | tee outputs/pusht_diffusion_v6_budget_warmup500_train.log
