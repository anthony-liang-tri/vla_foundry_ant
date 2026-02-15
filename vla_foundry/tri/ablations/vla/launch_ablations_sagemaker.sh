#!/bin/bash
#
# Launch VLA ablation SageMaker training jobs from resolved config files.
# Distributes jobs across queues ml, vla, tri-cam-humanoid with ratio 2:2:1.
#
# Usage:
#   ./vla_foundry/tri/ablations/vla/launch_ablations_sagemaker.sh [--dry-run]
#   ./vla_foundry/tri/ablations/vla/launch_ablations_sagemaker.sh --task RedBell --dry-run
#   ./vla_foundry/tri/ablations/vla/launch_ablations_sagemaker.sh --ablation nominal
#

set -e

# Hardcoded SageMaker settings
SAGEMAKER_USER="sedrick.keh"
SAGEMAKER_PROFILE="sagemaker"
SAGEMAKER_INSTANCE_TYPE="p5"
SAGEMAKER_INSTANCE_COUNT="1"
SAGEMAKER_PRIORITY="1"
SAGEMAKER_MAX_RUN="1"

# Queue rotation: ml, ml, vla, vla, tri-cam-humanoid (ratio 2:2:1)
QUEUES=("ml" "ml" "vla" "vla" "tri-cam-humanoid")
QUEUE_INDEX=0

# Config directory (relative to repo root)
CONFIG_DIR="vla_foundry/tri/ablations/vla/ablation_configs"

# Parse arguments
DRY_RUN=false
FILTER_TASK=""
FILTER_ABLATION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --task)
            FILTER_TASK="$2"
            shift 2
            ;;
        --ablation)
            FILTER_ABLATION="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "Error: Directory not found: $CONFIG_DIR"
    exit 1
fi

if [[ -z "$SAGEMAKER_ARN" ]] && [[ "$DRY_RUN" == "false" ]]; then
    echo "Error: SAGEMAKER_ARN environment variable is required"
    echo "Set it with: export SAGEMAKER_ARN=arn:aws:iam::..."
    exit 1
fi

# Function to extract short name from config path
get_short_name() {
    local config_path="$1"
    local dir_path=$(dirname "$config_path")
    local task=$(basename "$dir_path")
    local ablation=$(basename "$(dirname "$dir_path")")

    # Create short task name
    local short_task=""
    case "$task" in
        "BimanualPutRedBellPepperInBin") short_task="RedPepper" ;;
        "BimanualPutSpatulaOnPlateFromTable") short_task="Spatula" ;;
        "TurnCupUpsideDown") short_task="TurnCup" ;;
        "PlaceCupByCoaster") short_task="CupCoaster" ;;
        *) short_task="${task:0:10}" ;;
    esac

    # Create short ablation name
    local short_ablation=""
    case "$ablation" in
        "nominal_10m") short_ablation="10m" ;;
        "nominal_30m") short_ablation="30m" ;;
        "relative") short_ablation="rel" ;;
        "relative_per_timestep") short_ablation="relPerTS" ;;
        "relative_with_proprioception") short_ablation="relProp" ;;
        "relative_per_timestep_with_proprioception") short_ablation="relPTSP" ;;
        "norm_range_1_5") short_ablation="clamp1.5" ;;
        "percentile_95") short_ablation="p95" ;;
        "std_normalization") short_ablation="std" ;;
        "no_rotation_norm") short_ablation="noRotN" ;;
        "with_proprioception") short_ablation="wProp" ;;
        "image_0") short_ablation="img0" ;;
        "past_timesteps_0") short_ablation="past0" ;;
        "past_timesteps_0_image_1") short_ablation="p0i1" ;;
        "per_timestep_norm") short_ablation="perStep" ;;
        "lr_sweep_lr_0.0002") short_ablation="lr2e4" ;;
        "lr_sweep_lr_0.0005") short_ablation="lr5e4" ;;
        "lr_sweep_lr_0.001") short_ablation="lr1e3" ;;
        "float32") short_ablation="fp32" ;;
        "vlm_layers_2") short_ablation="vlm2" ;;
        "vlm_layers_4") short_ablation="vlm4" ;;
        "freeze_vlm") short_ablation="frzVLM" ;;
        "causal_attention") short_ablation="causal" ;;
        "transformer_410m") short_ablation="t410m" ;;
        # "img_num_tokens_64") short_ablation="tok64" ;;
        # "img_num_tokens_128") short_ablation="tok128" ;;
        # "img_num_tokens_256") short_ablation="tok256" ;;
        *) short_ablation="${ablation:0:8}" ;;
    esac

    echo "vla-${short_task}-${short_ablation}"
}

# Find all resolved_config.yaml files
echo "Searching for configs in: $CONFIG_DIR"
CONFIG_FILES=$(find "$CONFIG_DIR" -name "resolved_config.yaml" -type f | sort)

if [[ -z "$CONFIG_FILES" ]]; then
    echo "No resolved_config.yaml files found in $CONFIG_DIR"
    exit 1
fi

NUM_CONFIGS=$(echo "$CONFIG_FILES" | wc -l)
echo "Found $NUM_CONFIGS config files"
echo "User: $SAGEMAKER_USER"
echo "Instance type: $SAGEMAKER_INSTANCE_TYPE"
echo "Queue rotation: ${QUEUES[*]} (ratio 2:2:1)"
echo "=========================================="

LAUNCHED=0
SKIPPED=0

for CONFIG_PATH in $CONFIG_FILES; do
    dir_path=$(dirname "$CONFIG_PATH")
    task=$(basename "$dir_path")
    ablation=$(basename "$(dirname "$dir_path")")

    # Apply filters (substring match)
    if [[ -n "$FILTER_TASK" ]] && [[ "$task" != *"$FILTER_TASK"* ]]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    # Ablation filter supports comma-separated exact matches or single substring match
    if [[ -n "$FILTER_ABLATION" ]]; then
        if [[ "$FILTER_ABLATION" == *","* ]]; then
            MATCH_FOUND=false
            IFS=',' read -ra ABLATION_PATTERNS <<< "$FILTER_ABLATION"
            for pattern in "${ABLATION_PATTERNS[@]}"; do
                if [[ "$ablation" == "$pattern" ]]; then
                    MATCH_FOUND=true
                    break
                fi
            done
            if [[ "$MATCH_FOUND" == "false" ]]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
        else
            if [[ "$ablation" != *"$FILTER_ABLATION"* ]]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
        fi
    fi

    # Round-robin queue selection
    CURRENT_QUEUE="${QUEUES[$QUEUE_INDEX]}"
    QUEUE_INDEX=$(( (QUEUE_INDEX + 1) % ${#QUEUES[@]} ))

    SHORT_NAME=$(get_short_name "$CONFIG_PATH")

    echo ""
    echo "Config: $CONFIG_PATH"
    echo "  Task: $task"
    echo "  Ablation: $ablation"
    echo "  Name prefix: $SHORT_NAME"
    echo "  Queue: $CURRENT_QUEUE"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would launch: $SHORT_NAME on queue $CURRENT_QUEUE"
    else
        echo "  Launching SageMaker job..."

        uv run --group sagemaker sagemaker/launch_training.py \
            --config_path "$CONFIG_PATH" \
            --sagemaker.user "$SAGEMAKER_USER" \
            --sagemaker.profile "$SAGEMAKER_PROFILE" \
            --sagemaker.queue_name "$CURRENT_QUEUE" \
            --sagemaker.instance_count "$SAGEMAKER_INSTANCE_COUNT" \
            --sagemaker.priority "$SAGEMAKER_PRIORITY" \
            --sagemaker.instance_type "$SAGEMAKER_INSTANCE_TYPE" \
            --sagemaker.arn "$SAGEMAKER_ARN" \
            --sagemaker.max_run "$SAGEMAKER_MAX_RUN" \
            --sagemaker.name_prefix "$SHORT_NAME" \
            --wandb True \
            --distributed.use_distributed True

        echo "  Launched!"
        sleep 2
    fi

    LAUNCHED=$((LAUNCHED + 1))
done

echo ""
echo "=========================================="
echo "Summary:"
echo "  Launched: $LAUNCHED"
echo "  Skipped: $SKIPPED (filtered out)"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "  (DRY RUN - no jobs were actually launched)"
fi
echo "=========================================="
