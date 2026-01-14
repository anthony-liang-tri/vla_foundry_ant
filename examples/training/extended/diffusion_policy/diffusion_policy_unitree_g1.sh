.venv/bin/torchrun --nproc_per_node=2 --nnodes=1 vla_foundry/main.py \
--config_path vla_foundry/config_presets/training_jobs/diffusion_policy_unitree_g1.yaml \
--remote_sync_fixed_path s3://tri-ml-datasets/vla_foundry_scratch/tmp/aykut/model_checkpoints/diffusion_policy \
--num_checkpoints 2 \
--total_train_samples 10000
