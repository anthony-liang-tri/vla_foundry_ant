uv run --group=sagemaker sagemaker/launch_training.py \
--sagemaker.user sedrick.keh \
--sagemaker.instance_type p5 \
--sagemaker.queue_name ml \
--sagemaker.priority 1 \
--config_path vla_foundry/config_presets/training_jobs/lbm_hparams_6cams.yaml \
--remote_sync s3://tri-ml-datasets-uw2/lbm2_vla/model_checkpoints/diffusion_policy/BimanualPlaceGluegunOnPegboard \
--num_checkpoints 5 \
--total_train_samples 100_000_000 \
--data.dataset_manifest "['s3://tri-ml-datasets-uw2/vla_foundry_datasets/toolhang_202602/BimanualPlaceGluegunOnPegboard/shards/manifest.jsonl',]" \
--data.dataset_statistics "['s3://tri-ml-datasets-uw2/vla_foundry_datasets/toolhang_202602/BimanualPlaceGluegunOnPegboard/shards/stats.json',]" \
--data.dataset_modality "['robotics']" \
--data.dataset_weighting "[1.0]" \
"$@"
