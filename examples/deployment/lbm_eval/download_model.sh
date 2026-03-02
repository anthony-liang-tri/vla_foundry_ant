#!/bin/bash

# Check if experiment name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment_name> [checkpoint_number] [destination_directory] [aws_profile]"
    echo "If checkpoint_number is not provided, the highest checkpoint will be downloaded"
    echo "If destination_directory is not provided, defaults to current directory"
    echo "If aws_profile is not provided, AWS_PROFILE from the environment is used"
    exit 1
fi

EXPERIMENT_NAME=$1
CHECKPOINT_NUMBER=$2
DESTINATION_DIR=${3:-"."}
AWS_PROFILE_ARG=${4:-${AWS_PROFILE:-}}
AWS_PROFILE_ARGS=()
if [ -n "$AWS_PROFILE_ARG" ]; then
    AWS_PROFILE_ARGS=(--profile "$AWS_PROFILE_ARG")
fi
S3_BASE_PATH="s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/diffusion_policy/$EXPERIMENT_NAME"
FILES_TO_DOWNLOAD=(
    "config.yaml"
    "config_normalizer.yaml"
    "config_processor.yaml"
    "stats.json"
    "preprocessing_configs.yaml"
)

# Create local directories
mkdir -p "$DESTINATION_DIR/experiments/$EXPERIMENT_NAME/checkpoints"

# Download config file
echo "Downloading config file..."
for file in "${FILES_TO_DOWNLOAD[@]}"; do
    aws s3 cp "$S3_BASE_PATH/$file" "$DESTINATION_DIR/experiments/$EXPERIMENT_NAME/$file" "${AWS_PROFILE_ARGS[@]}"
done

#TODO Jean, These two downloads are there temporarily to handle old naming conventions. 
# Remove them once we do not need to support old naming conventions.
aws s3 cp "$S3_BASE_PATH/stats_normalizer.json" "$DESTINATION_DIR/experiments/$EXPERIMENT_NAME/stats.json" "${AWS_PROFILE_ARGS[@]}"
aws s3 cp "$S3_BASE_PATH/preprocessing_configs.yaml" "$DESTINATION_DIR/experiments/$EXPERIMENT_NAME/preprocessing_config.yaml" "${AWS_PROFILE_ARGS[@]}"

# If checkpoint number is provided, use it; otherwise find the highest
if [ -n "$CHECKPOINT_NUMBER" ]; then
    echo "Using provided checkpoint number: $CHECKPOINT_NUMBER"
    SELECTED_CHECKPOINT=$CHECKPOINT_NUMBER
else
    echo "Finding highest checkpoint number..."
    HIGHEST_CHECKPOINT=$(aws s3 ls "$S3_BASE_PATH/checkpoints/" "${AWS_PROFILE_ARGS[@]}" | grep "checkpoint_.*\.pt" | sed 's/.*checkpoint_\([0-9]*\)\.pt.*/\1/' | sort -n | tail -1)

    if [ -z "$HIGHEST_CHECKPOINT" ]; then
        echo "No checkpoint files found in S3 directory"
        exit 1
    fi

    echo "Found highest checkpoint: $HIGHEST_CHECKPOINT"
    SELECTED_CHECKPOINT=$HIGHEST_CHECKPOINT
fi

# Download the selected checkpoint
echo "Downloading checkpoint_$SELECTED_CHECKPOINT.pt..."
aws s3 sync "$S3_BASE_PATH/checkpoints" "$DESTINATION_DIR/experiments/$EXPERIMENT_NAME/checkpoints" --exclude "*" --include "checkpoint_$SELECTED_CHECKPOINT.pt" "${AWS_PROFILE_ARGS[@]}"
aws s3 sync "$S3_BASE_PATH/checkpoints" "$DESTINATION_DIR/experiments/$EXPERIMENT_NAME/checkpoints" --exclude "*" --include "ema_$SELECTED_CHECKPOINT.pt" "${AWS_PROFILE_ARGS[@]}"

echo "Download complete!"
