#!/bin/bash

# Array of all tasks from the S3 bucket
TASKS=(
    # "BimanualHangMugsOnMugHolderFromDryingRack"
    # "BimanualHangMugsOnMugHolderFromTable"
    # "BimanualLayCerealBoxOnCuttingBoardFromTopShelf"
    # "BimanualLayCerealBoxOnCuttingBoardFromUnderShelf"
    # "BimanualPlaceAppleFromBowlIntoBin"
    # "BimanualPlaceAppleFromBowlOnCuttingBoard"
    # "BimanualPlaceAvocadoFromBowlOnCuttingBoard"
    # "BimanualPlaceFruitFromBowlIntoBin"
    # "BimanualPlaceFruitFromBowlOnCuttingBoard"
    # "BimanualPlacePearFromBowlIntoBin"
    # "BimanualPlacePearFromBowlOnCuttingBoard"
    # "BimanualPutMugsOnPlatesFromDryingRack"
    # "BimanualPutMugsOnPlatesFromTable"
    # "BimanualPutRedBellPepperInBin"
    # "BimanualPutSpatulaOnPlateFromDryingRack"
    # "BimanualPutSpatulaOnPlateFromTable"
    # "BimanualPutSpatulaOnTableFromDryingRack"
    # "BimanualPutSpatulaOnTableFromUtensilCrock"
    # "BimanualStackPlatesOnTableFromDryingRack"
    # "BimanualStackPlatesOnTableFromTable"
    # "BimanualStoreCerealBoxUnderShelf"
    # "PickAndPlaceBox"
    # "PlaceCupByCoaster"
    # "PlaceCupOnCoaster"
    # "PushCoasterToCenterOfTable"
    # "PushCoasterToMug"
    # "PutBananaInCenterOfTable"
    # "PutBananaOnSaucer"
    # "PutCupInCenterOfTable"
    # "PutCupOnSaucer"
    # "PutGreenAppleInCenterOfTable"
    # "PutGreenAppleOnSaucer"
    # "PutKiwiInCenterOfTable"
    # "PutKiwiOnSaucer"
    # "PutMugOnSaucer"
    # "PutOrangeInCenterOfTable"
    # "PutOrangeOnSaucer"
    # "PutSpatulaInUtensilCrock"
    # "PutSpatulaInUtensilCrockFromDryingRack"
    # "TurnCupUpsideDown"
    # "TurnMugRightsideUp"
)

# 16 selected tasks for LBM1
TASKS=(
    # "BimanualPlaceAppleFromBowlIntoBin" #
    # "BimanualPlaceFruitFromBowlIntoBin" #
    # "BimanualPutRedBellPepperInBin" #
    # "BimanualPutSpatulaOnPlateFromDryingRack" #
    # "BimanualPutSpatulaOnPlateFromTable" #
    # "BimanualStackPlatesOnTableFromDryingRack" #
    # "BimanualStoreCerealBoxUnderShelf" #
    # "PlaceCupByCoaster" 
    # "PushCoasterToCenterOfTable" #
    # "PushCoasterToMug" #
    # "PutBananaOnSaucer" #
    # "PutKiwiInCenterOfTable" #
    # "PutMugOnSaucer" #
    # "PutSpatulaInUtensilCrock"  #
    "TurnCupUpsideDown"
    # "TurnMugRightsideUp" #
)

# Base S3 path
BASE_S3_PATH="s3://tri-ml-datasets-uw2/vla_foundry_datasets/stage3_singletask_sim"

# Loop through each task and launch training
for TASK in "${TASKS[@]}"; do
    echo "=========================================="
    echo "Launching training for task: $TASK"
    echo "=========================================="
    
    # Construct paths
    MANIFEST_PATH="${BASE_S3_PATH}/${TASK}/shards/manifest.jsonl"
    STATS_PATH="${BASE_S3_PATH}/${TASK}/shards/stats.json"
    
    # Launch training
    uv run --group sagemaker sagemaker/launch_training.py \
    --sagemaker.user jean.mercat \
    --sagemaker.profile sagemaker \
    --sagemaker.queue_name vla \
    --sagemaker.instance_count 4 \
    --sagemaker.priority 1 \
    --sagemaker.instance_type p4de \
    --sagemaker.arn $SAGEMAKER_ARN \
    --sagemaker.max_run 5 \
    --sagemaker.name_prefix "stage3_${TASK}" \
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
    --data.dataset_manifest "[\"${MANIFEST_PATH}\"]" \
    --data.dataset_statistics "[\"${STATS_PATH}\"]" \
    --data.augmentation.image.color_jitter.hue "[-0.05, 0.05]" \
    --data.augmentation.image.random_crop.enabled True \
    --data.augmentation.image.random_crop.shape "[224, 224]" \
    --data.dataset_weighting [1.0] \
    --data.dataset_modality ["robotics"] \
    --data.action_fields '["robot__action__poses__left::panda__xyz", "robot__action__poses__right::panda__xyz", "robot__action__poses__left::panda__rot_6d", "robot__action__poses__right::panda__rot_6d", "robot__action__grippers__left::panda_hand", "robot__action__grippers__right::panda_hand"]' \
    --data.proprioception_fields '[]' \
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
    --hparams.lr 5e-5 \
    --hparams.lr_cooldown_end 1e-7 \
    --hparams.wd 1e-6 \
    \
    --wandb True \
    --wandb_tags "[\"${TASK}\",\"diffusion_policy\", \"stage3_sim_singletask\"]" \
    --remote_sync s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/diffusion_policy/${TASK} \
    --num_checkpoints 5 \
    --total_train_samples 30_000_000
    
    echo ""
    echo "Training launched for $TASK"
    echo ""
    
    # Optional: Add a small delay between launches to avoid overwhelming the system
    sleep 2
done

echo "=========================================="
echo "All training jobs launched!"
echo "=========================================="

