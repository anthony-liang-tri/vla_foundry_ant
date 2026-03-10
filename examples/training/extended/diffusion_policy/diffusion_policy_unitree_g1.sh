.venv/bin/torchrun --nproc_per_node=2 --nnodes=1 vla_foundry/main.py \
--config_path vla_foundry/config_presets/training_jobs/diffusion_policy_unitree_g1.yaml \
--remote_sync s3://robotics-cam-checkpoints/platform/unitree_g1_dex3/model_checkpoints/diffusion_policy \
--remote_sync_fixed_path s3://robotics-cam-checkpoints/platform/unitree_g1_dex3/model_checkpoints_fixed/diffusion_policy \
--num_checkpoints 10 \
--total_train_samples 1000000
