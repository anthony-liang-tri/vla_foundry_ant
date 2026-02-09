#!/usr/bin/env bash
#
# Runs the policy inference server (vla_foundry) as a standalone gRPC server.
# This is designed to be called from distributed evaluation where the simulator
# runs on a separate (CPU-only) node.
#
# Expected environment variables:
#   CHECKPOINT_DIR       - S3 or local path to model checkpoint (required)
#   NUM_FLOW_STEPS       - Number of flow steps (default: 8)
#   OPEN_LOOP_STEPS      - Number of open loop steps (default: 8)
#   DEVICE               - Device for inference (default: cuda)
#   GRPC_PORT            - Port to listen on (default: 50051)
#   GRPC_HOST            - Host to bind to (default: 0.0.0.0)
#   TASK_NAME            - Optional task name
#   VLA_FOUNDRY_HOME     - vla_foundry installation (default: /opt/vla_foundry)

set -euo pipefail

echo "Starting policy server v0.1.0"

# Required
if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
    echo "ERROR: CHECKPOINT_DIR must be set" >&2
    exit 1
fi

# Defaults
NUM_FLOW_STEPS="${NUM_FLOW_STEPS:-8}"
OPEN_LOOP_STEPS="${OPEN_LOOP_STEPS:-8}"
DEVICE="${DEVICE:-cuda}"
GRPC_PORT="${GRPC_PORT:-50051}"
GRPC_HOST="${GRPC_HOST:-0.0.0.0}"
VLA_FOUNDRY_HOME="${VLA_FOUNDRY_HOME:-/opt/vla_foundry}"

echo "Configuration:"
echo "  CHECKPOINT_DIR:  ${CHECKPOINT_DIR}"
echo "  NUM_FLOW_STEPS:  ${NUM_FLOW_STEPS}"
echo "  OPEN_LOOP_STEPS: ${OPEN_LOOP_STEPS}"
echo "  DEVICE:          ${DEVICE}"
echo "  GRPC_HOST:       ${GRPC_HOST}"
echo "  GRPC_PORT:       ${GRPC_PORT}"

# GPU health check
if [[ "${DEVICE}" == "cuda" ]]; then
    echo "===== GPU Health Check ====="
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found but DEVICE=cuda requested." >&2
        exit 1
    fi
    nvidia-smi
    echo "============================"
fi

# Ensure cache directories exist
mkdir -p "${HOME}/.cache" 2>/dev/null || true

# Download checkpoint if from S3
LOCAL_CHECKPOINT_DIR="${CHECKPOINT_DIR}"
if [[ "${CHECKPOINT_DIR}" == s3://* ]]; then
    echo "Downloading checkpoint from S3..."
    LOCAL_CHECKPOINT_DIR="/tmp/checkpoint"
    rm -rf "${LOCAL_CHECKPOINT_DIR}"
    mkdir -p "${LOCAL_CHECKPOINT_DIR}/checkpoints"

    aws_cmd=(aws)
    if [[ -n "${AWS_PROFILE:-}" ]]; then
        aws_cmd+=(--profile "${AWS_PROFILE}")
    fi

    s3_root="${CHECKPOINT_DIR%/}"
    explicit_checkpoint_path=""
    explicit_checkpoint_is_ckpt=0

    if [[ "${s3_root}" == *.pt ]]; then
        explicit_checkpoint_path="${s3_root}"
        s3_root="${s3_root%/checkpoints/*}"
    elif [[ "${s3_root}" == *.ckpt ]]; then
        explicit_checkpoint_path="${s3_root}"
        s3_root="${s3_root%/*}"
        explicit_checkpoint_is_ckpt=1
    fi

    # Download config files
    if [[ "${explicit_checkpoint_is_ckpt}" -eq 1 ]]; then
        # Optional for .ckpt; some inference scripts don't need it.
        "${aws_cmd[@]}" s3 cp "${s3_root}/config.yaml" "${LOCAL_CHECKPOINT_DIR}/config.yaml" 2>/dev/null || true
    else
        "${aws_cmd[@]}" s3 cp "${s3_root}/config.yaml" "${LOCAL_CHECKPOINT_DIR}/config.yaml" || true
    fi
    "${aws_cmd[@]}" s3 cp "${s3_root}/config_normalizer.yaml" "${LOCAL_CHECKPOINT_DIR}/config_normalizer.yaml" 2>/dev/null || true
    "${aws_cmd[@]}" s3 cp "${s3_root}/config_processor.yaml" "${LOCAL_CHECKPOINT_DIR}/config_processor.yaml" 2>/dev/null || true
    "${aws_cmd[@]}" s3 cp "${s3_root}/stats.json" "${LOCAL_CHECKPOINT_DIR}/stats.json" 2>/dev/null || \
        "${aws_cmd[@]}" s3 cp "${s3_root}/stats_normalizer.json" "${LOCAL_CHECKPOINT_DIR}/stats.json" 2>/dev/null || true
    "${aws_cmd[@]}" s3 cp "${s3_root}/preprocessing_config.yaml" "${LOCAL_CHECKPOINT_DIR}/preprocessing_config.yaml" 2>/dev/null || \
        "${aws_cmd[@]}" s3 cp "${s3_root}/preprocessing_configs.yaml" "${LOCAL_CHECKPOINT_DIR}/preprocessing_config.yaml" 2>/dev/null || true

    # Download checkpoint file
    if [[ -n "${explicit_checkpoint_path}" ]]; then
        checkpoint_basename="$(basename "${explicit_checkpoint_path}")"
    else
        checkpoint_basename=$("${aws_cmd[@]}" s3 ls "${s3_root}/checkpoints/" | awk '{print $4}' | grep '^checkpoint_[0-9]\+\.pt$' | sort -t_ -k2,2n | tail -1)
        explicit_checkpoint_path="${s3_root}/checkpoints/${checkpoint_basename}"
    fi

    echo "Downloading checkpoint: ${checkpoint_basename}"
    if [[ "${explicit_checkpoint_is_ckpt}" -eq 1 ]]; then
        "${aws_cmd[@]}" s3 cp "${explicit_checkpoint_path}" "${LOCAL_CHECKPOINT_DIR}/${checkpoint_basename}"
        export CHECKPOINT_FILE="${LOCAL_CHECKPOINT_DIR}/${checkpoint_basename}"
    else
        "${aws_cmd[@]}" s3 cp "${explicit_checkpoint_path}" "${LOCAL_CHECKPOINT_DIR}/checkpoints/${checkpoint_basename}"
    fi
    echo "Checkpoint downloaded to ${LOCAL_CHECKPOINT_DIR}"
fi

# Change to vla_foundry directory
cd "${VLA_FOUNDRY_HOME}"

# Build inference command
inference_cmd=(
    env -u PYTHONPATH -u VIRTUAL_ENV
)

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    inference_cmd+=(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}")
fi

inference_cmd+=(
    uv run --group inference --group visualization
    python vla_foundry/inference/robotics/inference_policy.py
    --checkpoint_directory "${LOCAL_CHECKPOINT_DIR}"
    --num_flow_steps "${NUM_FLOW_STEPS}"
    --open_loop_steps "${OPEN_LOOP_STEPS}"
    --device "${DEVICE}"
    --host "${GRPC_HOST}"
    --port "${GRPC_PORT}"
)

if [[ -n "${TASK_NAME:-}" ]]; then
    inference_cmd+=(--task_name "${TASK_NAME}")
fi

echo "Starting inference server: ${inference_cmd[*]}"
exec "${inference_cmd[@]}"
