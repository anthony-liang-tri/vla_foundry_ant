.venv/bin/torchrun --nproc_per_node=2 --nnodes=1 vla_foundry/main.py \
--config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml \
--remote_sync s3://tri-ml-datasets-uw2/lbm2_vla/model_checkpoints/diffusion_policy \
--num_checkpoints 5 \
--total_train_samples 100000 \
--data.dataset_manifest "[
's3://tri-ml-datasets-uw2/vla_foundry_datasets/stage3_singletask_sim/BimanualHangMugsOnMugHolderFromDryingRack/shards/manifest.jsonl',
's3://tri-ml-datasets-uw2/vla_foundry_datasets/stage3_singletask_sim/BimanualLayCerealBoxOnCuttingBoardFromTopShelf/shards/manifest.jsonl',
]" \
--data.dataset_statistics "[
's3://tri-ml-datasets-uw2/vla_foundry_datasets/stage3_singletask_sim/BimanualHangMugsOnMugHolderFromDryingRack/shards/stats.json',
's3://tri-ml-datasets-uw2/vla_foundry_datasets/stage3_singletask_sim/BimanualLayCerealBoxOnCuttingBoardFromTopShelf/shards/stats.json',
]" \
--data.dataset_modality "['robotics', 'robotics']" \
--data.dataset_weighting "[1.0, 1.0]" \
--data.camera_names "['scene_right_0', 'scene_left_0', 'wrist_left_minus', 'wrist_left_plus', 'wrist_right_minus', 'wrist_right_plus']" \
--data.image_indices "[-1, 0]" \
--data.pad_missing_images True \
--data.mask_padded_images True \