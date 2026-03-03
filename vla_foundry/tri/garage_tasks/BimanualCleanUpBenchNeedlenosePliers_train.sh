AWS_PROFILE=sagemaker uv run --group=sagemaker sagemaker/launch_training.py \
--sagemaker.user katherine.liu \
--sagemaker.instance_type p5 \
--sagemaker.profile sagemaker \
--sagemaker.queue_name ml \
--sagemaker.priority 1 \
--config_path vla_foundry/config_presets/training_jobs/lbm_hparams_6cams.yaml \
--remote_sync s3://tri-ml-datasets-uw2/lbm2_vla/model_checkpoints/diffusion_policy/BimanualCleanUpBenchNeedlenosePliers \
--num_checkpoints 5 \
--total_train_samples 100_000_000 \
--data.dataset_manifest "['s3://tri-ml-datasets-uw2/vla_foundry_datasets/garage_02242026/BimanualCleanUpBenchNeedlenosePliers/shards/manifest.jsonl',]" \
--data.dataset_statistics "['s3://tri-ml-datasets-uw2/vla_foundry_datasets/garage_02242026/BimanualCleanUpBenchNeedlenosePliers/shards/stats.json',]" \
--data.dataset_modality "['robotics']" \
--data.dataset_weighting "[1.0]"

# FINETUNE
AWS_PROFILE=sagemaker uv run --group=sagemaker sagemaker/launch_training.py \
--sagemaker.user katherine.liu \
--sagemaker.instance_type p5 \
--sagemaker.profile sagemaker \
--sagemaker.queue_name ml \
--sagemaker.priority 1 \
--config_path vla_foundry/config_presets/training_jobs/lbm_hparams_6cams.yaml \
--remote_sync s3://tri-ml-datasets-uw2/lbm2_vla/model_checkpoints/diffusion_policy/BimanualCleanUpBenchNeedlenosePliers \
--num_checkpoints 5 \
--total_train_samples 5_000_000 \
--data.dataset_manifest "['s3://tri-ml-datasets-uw2/vla_foundry_datasets/garage_02242026/BimanualCleanUpBenchNeedlenosePliers/shards/manifest.jsonl',]" \
--data.dataset_statistics "['s3://tri-ml-datasets-uw2/vla_foundry_datasets/v0.3/stage3_singletask_sim/stats.json',]" \
--data.dataset_modality "['robotics']" \
--data.dataset_weighting "[1.0]" \
--model.resume_from_checkpoint "s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/diffusion_policy/ablations/multitask/100m/2026_01_07-23_38_39-model_diffusion_policy-lr_5e-05-bsz_1024_converted/checkpoints/checkpoint_3.pt" \
--model.resume_weights_only True \
--ema.enabled False \
--hparams.warmup 0 \
--hparams.lr 2e-5