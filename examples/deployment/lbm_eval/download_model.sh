#!/bin/bash

# Check if experiment name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment_name> [checkpoint_number]"
    echo "If checkpoint_number is not provided, the highest checkpoint will be downloaded"
    exit 1
fi

EXPERIMENT_NAME=$1
CHECKPOINT_NUMBER=$2
S3_BASE_PATH="s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/diffusion_policy/$EXPERIMENT_NAME"
FILES_TO_DOWNLOAD=(
    "config.yaml"
    "config_normalizer.yaml"
    "config_processor.yaml"
    "stats_normalizer.json"
    "preprocessing_configs.yaml"
)

# Create local directories
mkdir -p "experiments/$EXPERIMENT_NAME/checkpoints"

# Download config file
echo "Downloading config file..."
for file in "${FILES_TO_DOWNLOAD[@]}"; do
    aws s3 cp "$S3_BASE_PATH/$file" "experiments/$EXPERIMENT_NAME/$file" --profile sagemaker
done

# If checkpoint number is provided, use it; otherwise find the highest
if [ -n "$CHECKPOINT_NUMBER" ]; then
    echo "Using provided checkpoint number: $CHECKPOINT_NUMBER"
    SELECTED_CHECKPOINT=$CHECKPOINT_NUMBER
else
    echo "Finding highest checkpoint number..."
    HIGHEST_CHECKPOINT=$(aws s3 ls "$S3_BASE_PATH/checkpoints/" --profile sagemaker | grep "checkpoint_.*\.pt" | sed 's/.*checkpoint_\([0-9]*\)\.pt.*/\1/' | sort -n | tail -1)

    if [ -z "$HIGHEST_CHECKPOINT" ]; then
        echo "No checkpoint files found in S3 directory"
        exit 1
    fi

    echo "Found highest checkpoint: $HIGHEST_CHECKPOINT"
    SELECTED_CHECKPOINT=$HIGHEST_CHECKPOINT
fi

# Download the selected checkpoint
echo "Downloading checkpoint_$SELECTED_CHECKPOINT.pt..."
aws s3 sync "$S3_BASE_PATH/checkpoints" "experiments/$EXPERIMENT_NAME/checkpoints" --exclude "*" --include "checkpoint_$SELECTED_CHECKPOINT.pt" --profile sagemaker

echo "Download complete!"