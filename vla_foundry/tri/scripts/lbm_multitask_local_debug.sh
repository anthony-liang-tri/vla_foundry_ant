#!/usr/bin/env bash
# Local training with lbm_multitask_4cams config (2 GPUs, minimal steps for verification).
# Requires AWS credentials for S3 (manifests and stats). Run from repo root.
#
# Usage:
#   ./vla_foundry/tri/scripts/lbm_multitask_local_debug.sh
# Or with overrides:
#   ./vla_foundry/tri/scripts/lbm_multitask_local_debug.sh --wandb false
#
# For S3 access, uses AWS_PROFILE=sagemaker by default.

set -e

# Small batch and few samples so the run finishes quickly.
AWS_PROFILE=sagemaker uv run torchrun --nproc_per_node=2 --nnodes=1 vla_foundry/main.py \
  --config_path vla_foundry/config_presets/training_jobs/lbm_multitask_4cams.yaml \
  --total_train_samples 8192 \
  --num_checkpoints 1 \
  --hparams.per_gpu_batch_size 4 \
  --hparams.global_batch_size 8 \
  --wandb true \
  "$@"
