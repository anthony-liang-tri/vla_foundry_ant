#!/bin/bash
# Unitree G1 MCAP Preprocessing Script
#
# Converts MCAP ROS 2 recordings to WebDataset tar shards.
#
# Usage:
#   ./preprocess_robotics_data_mcap_g1.sh --source PATH --output PATH [OPTIONS]
#
# Required:
#   --source PATH       S3 or local path to MCAP episodes
#   --output PATH       S3 or local path for output tar shards
#
# Options:
#   --task-name NAME    Task name for language instruction (e.g. "stack cubes ordered")
#   --max-episodes N    Max episodes to process (-1 for all, default: -1)
#   --ray-cpus N        Ray CPUs (default: 32)
#   --dry-run           Print command without executing
#
# Examples:
#   ./preprocess_robotics_data_mcap_g1.sh \
#       --source s3://robotics-cam-data/platform/unitree_g1_dex3/mcap/stack_cubes_ordered/real/teleop/ \
#       --output s3://robotics-cam-data/platform/unitree_g1_dex3/tarfile/v1/stack_cubes_ordered/real/teleop/ \
#       --task-name "stack cubes ordered"
#
#   ./preprocess_robotics_data_mcap_g1.sh \
#       --source /local/data/mcap/test_task/ \
#       --output /local/data/tarfiles/test_task/ \
#       --task-name "place cup on plate" \
#       --max-episodes 5 --dry-run

set -euo pipefail

# -----------------------------------------------------------------------------
# Config Paths (relative to vla_foundry repo root)
# -----------------------------------------------------------------------------

CONFIG_PATH="vla_foundry/config_presets/data/robotics_preprocessing_params_1past_47future_30hz.yaml"
TOPICS_CONFIG="vla_foundry/config_presets/data/unitree_g1/g1_mcap_topics.yaml"
CAMERA_NAMES="vla_foundry/config_presets/data/unitree_g1/g1_data_camera_names.yaml"

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

SOURCE_PATH=""
OUTPUT_PATH=""
TASK_NAME=""
MAX_EPISODES=-1
RAY_CPUS=32
DRY_RUN=false

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------

print_help() {
    head -28 "$0" | tail -26
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --source=*) SOURCE_PATH="${1#*=}"; shift ;;
        --source) SOURCE_PATH="$2"; shift 2 ;;
        --output=*) OUTPUT_PATH="${1#*=}"; shift ;;
        --output) OUTPUT_PATH="$2"; shift 2 ;;
        --task-name=*) TASK_NAME="${1#*=}"; shift ;;
        --task-name) TASK_NAME="$2"; shift 2 ;;
        --max-episodes=*) MAX_EPISODES="${1#*=}"; shift ;;
        --max-episodes) MAX_EPISODES="$2"; shift 2 ;;
        --ray-cpus=*) RAY_CPUS="${1#*=}"; shift ;;
        --ray-cpus) RAY_CPUS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) print_help ;;
        *) echo "Error: Unknown option: $1"; exit 1 ;;
    esac
done

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

if [[ -z "$SOURCE_PATH" ]]; then
    echo "Error: --source is required"
    exit 1
fi

if [[ -z "$OUTPUT_PATH" ]]; then
    echo "Error: --output is required"
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Config not found: $CONFIG_PATH"
    exit 1
fi

if [[ ! -f "$TOPICS_CONFIG" ]]; then
    echo "Error: Topics config not found: $TOPICS_CONFIG"
    exit 1
fi

if [[ ! -f "$CAMERA_NAMES" ]]; then
    echo "Error: Camera names config not found: $CAMERA_NAMES"
    exit 1
fi

# -----------------------------------------------------------------------------
# Print Configuration
# -----------------------------------------------------------------------------

echo "=============================================="
echo "Unitree G1 MCAP Preprocessing (30Hz)"
echo "=============================================="
echo "Source:         ${SOURCE_PATH}"
echo "Output:         ${OUTPUT_PATH}"
echo "Task Name:      ${TASK_NAME:-<from config>}"
echo "Config:         ${CONFIG_PATH}"
echo "Topics:         ${TOPICS_CONFIG}"
echo "Camera Names:   ${CAMERA_NAMES}"
echo "Max Episodes:   ${MAX_EPISODES}"
echo "Ray CPUs:       ${RAY_CPUS}"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Build and Execute Command
# -----------------------------------------------------------------------------

CMD="uv run --group preprocessing python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
    --type mcap \
    --source_episodes \"['${SOURCE_PATH}']\" \
    --output_dir ${OUTPUT_PATH} \
    --config_path ${CONFIG_PATH} \
    --action_fields_config_path ${TOPICS_CONFIG} \
    --camera_names \"include ${CAMERA_NAMES}\" \
    --ray_address local \
    --ray_num_cpus ${RAY_CPUS} \
    --max_episodes_to_process ${MAX_EPISODES} \
    --output_dir_fixed_path ${OUTPUT_PATH}"

# Add task_name if specified
if [[ -n "$TASK_NAME" ]]; then
    CMD="$CMD \
    --task_name \"${TASK_NAME}\""
fi

if [[ "$DRY_RUN" = true ]]; then
    echo "[DRY RUN] Command:"
    echo ""
    echo "$CMD"
    exit 0
fi

echo "Starting preprocessing..."
START_TIME=$(date +%s)

if eval "$CMD"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ""
    echo "Complete: $((DURATION / 60))m $((DURATION % 60))s"
    echo "Output: ${OUTPUT_PATH}"
else
    echo "Error: Preprocessing failed"
    exit 1
fi