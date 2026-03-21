#!/bin/bash
# Train VLADiffusion model with SmolVLM encoder on robotics data

.venv/bin/torchrun --nproc_per_node=1 --nnodes=1 vla_foundry/main.py \
--config_path vla_foundry/config_presets/training_jobs/vla_diffusion_bellpepper_smolvlm_256m.yaml \
--remote_sync s3://tri-ml-datasets-uw2/lbm2_vla/model_checkpoints/vla_diffusion \
--num_checkpoints 5 \
--total_train_samples 1000 \
"$@"
