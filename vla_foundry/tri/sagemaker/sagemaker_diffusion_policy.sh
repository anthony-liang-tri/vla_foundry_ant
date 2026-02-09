# --data.cache_dir "/opt/ml/checkpoints/datacache" \
# --data.action_fields '["robot__actual__poses__left::panda__xyz", "robot__actual__poses__right::panda__xyz", "robot__actual__poses__left::panda__rot_6d", "robot__actual__poses__right::panda__rot_6d", "robot__action__grippers__left::panda_hand", "robot__action__grippers__right::panda_hand"]' \
#pos_rel
#seq_rel
#delta
# --data.dataset_manifest '["s3://tri-ml-datasets-uw2/vlm_datasets_v3/preprocess_384_past1_future8_relative/lbm/BimanualPutRedBellPepperInBin/riverway/sim/shards/manifest.jsonl"]' \
# --data.dataset_statistics '["s3://tri-ml-datasets-uw2/vlm_datasets_v3/preprocess_384_past1_future8_relative/lbm/BimanualPutRedBellPepperInBin/riverway/sim/shards/stats.json"]' \

# --data.dataset_manifest '["s3://tri-ml-datasets/scratch/jean.mercat/tmp/lbmdata/bimanualputredbellpepperinbin3/shards/manifest.jsonl"]' \
# --data.dataset_statistics '["s3://tri-ml-datasets/scratch/jean.mercat/tmp/lbmdata/bimanualputredbellpepperinbin3/shards/stats.json"]' \
# --data.dataset_manifest '["s3://tri-ml-datasets-uw2/vlm_datasets/bellpepper_v2/preprocess_384_past1_future8_relative/lbm/BimanualPutRedBellPepperInBin/riverway/sim/shards/manifest.jsonl"]' \
# --data.dataset_statistics '["s3://tri-ml-datasets-uw2/vlm_datasets/bellpepper_v2/preprocess_384_past1_future8_relative/lbm/BimanualPutRedBellPepperInBin/riverway/sim/shards/stats.json"]' \

uv run --group sagemaker sagemaker/launch_training.py \
--sagemaker.user jean.mercat \
--sagemaker.profile sagemaker \
--sagemaker.queue_name ml \
--sagemaker.instance_count 1 \
--sagemaker.priority 1 \
--sagemaker.instance_type p5 \
--sagemaker.arn $SAGEMAKER_ARN \
--sagemaker.max_run 2 \
--sagemaker.name_prefix "test_proprioception" \
\
--model "include vla_foundry/config_presets/models/diffusion_policy.yaml" \
--model.transformer "include vla_foundry/config_presets/models/transformer_410m.yaml" \
--model.transformer.is_causal True \
--model.clip.hf_pretrained openai/clip-vit-base-patch32 \
--model.clip.freeze_text_encoder False \
--model.clip.freeze_image_encoder False \
--model.noise_scheduler.num_timesteps 1000 \
--model.noise_scheduler.clamp_range "[-3, 3]" \
\
--data "include vla_foundry/config_presets/data/lbm/lbm_data_params.yaml" \
--data.normalization.scope "global" \
--data.normalization.method "percentile_1_99" \
--data.normalization.centered_norm True \
--data.normalization.epsilon 1e-6 \
--data.augmentation.enabled True \
--data.augmentation.image.color_jitter.enabled True \
--data.augmentation.image.color_jitter.brightness 0.2 \
--data.augmentation.image.color_jitter.contrast 0.4 \
--data.augmentation.image.color_jitter.saturation 0.2 \
--data.augmentation.image.color_jitter.hue "[-0.05, 0.05]" \
--data.augmentation.image.random_crop.enabled True \
--data.augmentation.image.random_crop.shape "[224, 224]" \
--data.dataset_weighting [1.0] \
--data.dataset_modality ["robotics"] \
--data.dataset_manifest '["s3://tri-ml-datasets-uw2/vla_foundry_datasets/stage3_singletask_sim/BimanualPutRedBellPepperInBin/shards/manifest.jsonl"]' \
--data.dataset_statistics '["s3://tri-ml-datasets-uw2/vla_foundry_datasets/stage3_singletask_sim/BimanualPutRedBellPepperInBin/shards/stats.json"]' \
--data.action_fields '["robot__action__poses__left::panda__xyz", "robot__action__poses__right::panda__xyz", "robot__action__poses__left::panda__rot_6d", "robot__action__poses__right::panda__rot_6d", "robot__action__grippers__left::panda_hand", "robot__action__grippers__right::panda_hand"]' \
--data.proprioception_fields '["robot__actual__poses__left::panda__xyz", "robot__actual__poses__right::panda__xyz", "robot__actual__poses__left::panda__rot_6d", "robot__actual__poses__right::panda__rot_6d"]' \
--data.camera_names '["scene_right_0", "scene_left_0", "wrist_left_minus", "wrist_left_plus", "wrist_right_minus", "wrist_right_plus"]' \
--data.image_indices '[-1, 0]' \
--data.lowdim_past_timesteps 1 \
--data.lowdim_future_timesteps 8 \
--data.image_size 224 \
--data.processor openai/clip-vit-base-patch32 \
--data.allow_multiple_epochs True \
--data.num_workers 24 \
--data.seq_len 2048 \
\
--distributed.fsdp False \
--distributed.use_distributed True \
\
--hparams.loss_function mse \
--hparams.torchcompile False \
--hparams.precision amp_bf16 \
--hparams.per_gpu_batch_size 32 \
--hparams.global_batch_size 1024 \
--hparams.grad_clip_norm 10.0 \
--hparams.lr 1e-4 \
--hparams.lr_cooldown_end 1e-6 \
--hparams.wd 1e-6 \
\
--wandb True \
--wandb_tags "[\"BimanualPutRedBellPepperInBin\", \"diffusion_policy\", \"stage3_sim_singletask\", \"relative\"]" \
--remote_sync s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/diffusion_policy \
--num_checkpoints 5 \
--total_train_samples 30_000_000
