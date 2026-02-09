#!/bin/bash
#
# Launch SageMaker training jobs from resolved config files.
# Recursively searches for resolved_config.yaml files in the given path.
#
# Usage:
#   ./vla_foundry/tri/sagemaker/launch_from_configs.sh <config_dir> [--dry-run]
#
# Examples:
#   ./vla_foundry/tri/sagemaker/launch_from_configs.sh vla_foundry/tri/ablation_configs/
#   ./vla_foundry/tri/sagemaker/launch_from_configs.sh vla_foundry/tri/ablation_configs/ --dry-run
#

set -e

# Hardcoded SageMaker settings
SAGEMAKER_USER="jean.mercat"
SAGEMAKER_PROFILE="sagemaker"
SAGEMAKER_QUEUE="vla"
SAGEMAKER_INSTANCE_TYPE="p5"
SAGEMAKER_INSTANCE_COUNT="2"
SAGEMAKER_PRIORITY="1"
SAGEMAKER_MAX_RUN="1"

# Parse arguments
CONFIG_DIR=""
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
            if [[ -z "$CONFIG_DIR" ]]; then
                CONFIG_DIR="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$CONFIG_DIR" ]]; then
    echo "Error: Config directory is required"
    echo "Usage: $0 <config_dir> [options]"
    echo ""
    echo "Options:"
    echo "  --dry-run          Show what would be launched without launching"
    echo "  --task PATTERN     Filter configs by task name (substring match)"
    echo "  --ablation PATTERN Filter configs by ablation name (substring match)"
    echo ""
    echo "Examples:"
    echo "  $0 vla_foundry/tri/ablation_configs/ --ablation 10k"
    echo "  $0 vla_foundry/tri/ablation_configs/ --task RedBell --dry-run"
    exit 1
fi

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
        "nominal_10k") short_ablation="10k" ;;
        "nominal_30k") short_ablation="30k" ;;
        "norm_range_1_5") short_ablation="clamp1.5" ;;
        "percentile_95") short_ablation="p95" ;;
        "std_normalization") short_ablation="std" ;;
        "no_rotation_norm") short_ablation="noRotNorm" ;;
        "with_proprioception") short_ablation="withProp" ;;
        "no_proprioception") short_ablation="noProp" ;;
        "past_timesteps_1") short_ablation="past1" ;;
        "past_timesteps_0") short_ablation="past0" ;;
        "per_timestep_norm") short_ablation="perStep" ;;
        *) short_ablation="${ablation:0:8}" ;;
    esac

    echo "${short_task}_${short_ablation}"
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
            # Comma-separated list: require exact match with one of the values
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
            # Single pattern: substring match (backward compatible)
            if [[ "$ablation" != *"$FILTER_ABLATION"* ]]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
        fi
    fi

    SHORT_NAME=$(get_short_name "$CONFIG_PATH")

    echo ""
    echo "Config: $CONFIG_PATH"
    echo "  Task: $task"
    echo "  Ablation: $ablation"
    echo "  Name prefix: $SHORT_NAME"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would launch: $SHORT_NAME"
    else
        echo "  Launching SageMaker job..."

        uv run --group sagemaker sagemaker/launch_training.py \
            --config_path "$CONFIG_PATH" \
            --data.prefetch_factor 16 \
            --sagemaker.user "$SAGEMAKER_USER" \
            --sagemaker.profile "$SAGEMAKER_PROFILE" \
            --sagemaker.queue_name "$SAGEMAKER_QUEUE" \
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
echo "  Matched: $LAUNCHED"
echo "  Skipped: $SKIPPED (filtered out)"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "  (DRY RUN - no jobs were actually launched)"
fi
echo "=========================================="
