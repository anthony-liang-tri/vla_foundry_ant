#!/bin/bash
# Train VLADiffusion model with PaliGemma2 encoder on robotics data

.venv/bin/torchrun --nproc_per_node=2 --nnodes=1 vla_foundry/main.py \
--hparams.torchcompile True \
--config_path vla_foundry/config_presets/training_jobs/vla_diffusion_bellpepper.yaml \
--remote_sync s3://tri-ml-datasets-uw2/lbm2_vla/model_checkpoints/vla_diffusion \
--num_checkpoints 5 \
--total_train_samples 1000 \
"$@"
