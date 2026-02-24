#!/bin/bash
# Generic LBM dataset visualizer that works with any S3 robotics dataset
#
# Usage:
#   ./visualize_data.sh s3://tri-ml-datasets-uw2/vla_foundry_datasets/toolhang_202602/BimanualPlaceTtoolOnPegboard 5
#   ./visualize_data.sh --ordered s3://tri-ml-datasets-uw2/vla_foundry_datasets/toolhang_202602/BimanualPlaceTtoolOnPegboard 5

set -e

# Separate flags (forwarded to Python) from positional args
FLAGS=()
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --*) FLAGS+=("$arg") ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

if [ ${#POSITIONAL[@]} -lt 2 ]; then
    echo "Usage: $0 [--ordered] <dataset_path> <num_samples>"
    exit 1
fi

DATASET_PATH="${POSITIONAL[0]%/}"
NUM_SAMPLES="${POSITIONAL[1]}"

# Determine if shards directory exists in dataset path
if [[ "$DATASET_PATH" == *"/shards" ]]; then
    CONFIG_DIR="$DATASET_PATH"
else
    CONFIG_DIR="$DATASET_PATH/shards"
fi

# Load preprocessing config dynamically
CONFIG_FILE="$CONFIG_DIR/preprocessing_config.yaml"

# Parse YAML to extract config
if [[ "$CONFIG_FILE" == s3://* ]]; then
    # Use aws s3 cp to download YAML temporarily
    TEMP_CONFIG=$(mktemp)
    aws s3 cp "$CONFIG_FILE" "$TEMP_CONFIG" > /dev/null
    CONFIG_FILE="$TEMP_CONFIG"
    CLEANUP=true
else
    CLEANUP=false
fi

# Extract values from YAML using grep/awk
extract_yaml_list() {
    local key=$1
    local file=$2
    # Use sed range to capture only list items between this key and the next YAML key
    sed -n "/^${key}:/,/^[^ -]/{/^- /p}" "$file" | sed 's/^- //' | sed 's/ *$//' | paste -sd ',' -
}

extract_yaml_single() {
    local key=$1
    local file=$2
    grep "^$key:" "$file" | cut -d':' -f2- | sed 's/^ *//' | sed 's/ *$//'
}

CAMERAS=$(extract_yaml_list "camera_names" "$CONFIG_FILE")
IMAGE_INDICES=$(extract_yaml_list "image_indices" "$CONFIG_FILE")

# Convert to proper format for draccus
IFS=',' read -ra CAMERA_ARRAY <<< "$CAMERAS"
CAMERA_STR=$(printf '%s, ' "${CAMERA_ARRAY[@]}" | sed 's/, $//')

IFS=',' read -ra INDICES_ARRAY <<< "$IMAGE_INDICES"
INDICES_STR=$(printf '%s, ' "${INDICES_ARRAY[@]}" | sed 's/, $//')

# Generate image names from cameras and indices
IMAGE_NAMES=()
for camera in "${CAMERA_ARRAY[@]}"; do
    for idx in "${INDICES_ARRAY[@]}"; do
        if [ "$idx" == "-1" ]; then
            IMAGE_NAMES+=("${camera}_t-1")
        else
            IMAGE_NAMES+=("${camera}_t${idx}")
        fi
    done
done

CONFIG_PATH="examples/visualization/visualization_params.yaml"

# Generate image_names list string for draccus
IMAGE_NAMES_STR=$(printf '%s, ' "${IMAGE_NAMES[@]}" | sed 's/, $//')

# Always use shards paths; --ordered flag (forwarded to Python) rewrites manifest to episodes/
STATS_PATH="$DATASET_PATH/shards/stats.json"
MANIFEST_PATH="$DATASET_PATH/shards/manifest.jsonl"

# Build arguments dynamically from config
ARGS=(
    "--config_path=$CONFIG_PATH"
    "--data.dataset_manifest=[$MANIFEST_PATH]"
    "--data.dataset_statistics=[$STATS_PATH]"
    "--data.camera_names=[$CAMERA_STR]"
    "--data.image_indices=[$INDICES_STR]"
    "--data.image_names=[$IMAGE_NAMES_STR]"
    "--data.shuffle=true"
    "--total_train_samples=$NUM_SAMPLES"
    "--num_checkpoints=1"
    "--data.num_workers=0"
    "--data.normalization.enabled=false"
    "--data.normalization.field_configs={}"
)

# Cleanup temp file if created
if [ "$CLEANUP" = true ]; then
    rm "$TEMP_CONFIG"
fi

# Call lbm_vis.py (FLAGS like --ordered are forwarded to Python)
VISUALIZER="${VISUALIZER:-rerun}" uv run --group visualization vla_foundry/data/scripts/vis/lbm_vis.py "${FLAGS[@]}" "${ARGS[@]}"
