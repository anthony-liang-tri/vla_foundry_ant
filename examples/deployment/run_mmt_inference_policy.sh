#!/bin/bash
# ============================================================
# ZZK Policy Inference Configuration & Runner
# ============================================================
# This script serves as both configuration and execution wrapper.
# All settings can be overridden via environment variables or CLI args.
#
# Usage:
#   # Default settings
#   bash examples/deployment/run_mmt_inference_policy.sh
#
#   # Override via environment variables
#   NUM_EPISODES=5 DEVICE=cpu bash examples/deployment/run_mmt_inference_policy.sh
#
#   # Override via CLI arguments (highest priority)
#   bash examples/deployment/run_mmt_inference_policy.sh --num_episodes 3 --device cpu
# ============================================================

# ============================================================
# Model Settings
# ============================================================
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/PATH/TO/CHECKPOINTS}"  # Must be set by user
CHECKPOINT_NAME="${CHECKPOINT_NAME:-}"  # Empty = auto-detect latest
DEVICE="${DEVICE:-cuda}"
NUM_FLOW_STEPS="${NUM_FLOW_STEPS:-10}"
OPEN_LOOP_STEPS="${OPEN_LOOP_STEPS:-4}"

# ============================================================
# Robot Settings
# ============================================================
ROBOT_HOSTNAME="${ROBOT_HOSTNAME:-localhost}"
ROBOT_PORT="${ROBOT_PORT:-8888}"

# ============================================================
# Task Settings
# ============================================================
LANGUAGE_INSTRUCTION="${LANGUAGE_INSTRUCTION:-You are a helpful robot assistant finishing tasks to help people daily lives.}"
NUM_EPISODES="${NUM_EPISODES:-10}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-500}"

# ============================================================
# ZZK API Paths (Customize for your environment)
# ============================================================
ZZK_API_CLIENT_PY_PATH="${ZZK_API_CLIENT_PY_PATH:-/PATH/TO/ZZK_API_CLIENT.py}"  # Must be set by user
ZZK_API_CTYPES_LIBRARY_PATH="${ZZK_API_CTYPES_LIBRARY_PATH:-/PATH/TO/ZZK_API_CTYPES_LIBRARY.so}"  # Must be set by user

# ============================================================
# Logging
# ============================================================
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# ============================================================
# Run Inference
# ============================================================
# Build command with all arguments
CMD="uv run python vla_foundry/inference/robotics/mmt/mmt_inference_policy.py"
CMD="$CMD --checkpoint_dir \"${CHECKPOINT_DIR}\""
CMD="$CMD --device \"${DEVICE}\""
CMD="$CMD --num_flow_steps ${NUM_FLOW_STEPS}"
CMD="$CMD --open_loop_steps ${OPEN_LOOP_STEPS}"
CMD="$CMD --robot_hostname \"${ROBOT_HOSTNAME}\""
CMD="$CMD --robot_port ${ROBOT_PORT}"
CMD="$CMD --language_instruction \"${LANGUAGE_INSTRUCTION}\""
CMD="$CMD --num_episodes ${NUM_EPISODES}"
CMD="$CMD --max_steps_per_episode ${MAX_STEPS_PER_EPISODE}"
CMD="$CMD --zzk_api_client_ctypes_library_path \"${ZZK_API_CTYPES_LIBRARY_PATH}\""
CMD="$CMD --log_level \"${LOG_LEVEL}\""

# Add checkpoint name if specified
if [ -n "$CHECKPOINT_NAME" ]; then
    CMD="$CMD --checkpoint_name \"${CHECKPOINT_NAME}\""
fi

# Add ZZK API client path if specified
if [ -n "$ZZK_API_CLIENT_PY_PATH" ]; then
    CMD="$CMD --zzk_api_client_py_path \"${ZZK_API_CLIENT_PY_PATH}\""
fi

# Append any CLI arguments (highest priority)
if [ $# -gt 0 ]; then
    CMD="$CMD $@"
fi

# Print configuration
echo "============================================================"
echo "ZZK Policy Inference"
echo "============================================================"
echo "Checkpoint:     ${CHECKPOINT_DIR}"
echo "Device:         ${DEVICE}"
echo "Robot:          ${ROBOT_HOSTNAME}:${ROBOT_PORT}"
echo "Episodes:       ${NUM_EPISODES} x ${MAX_STEPS_PER_EPISODE} steps"
echo "Flow steps:     ${NUM_FLOW_STEPS}"
echo "Open loop:      ${OPEN_LOOP_STEPS}"
echo "Log level:      ${LOG_LEVEL}"
echo "============================================================"
echo ""

# Execute
eval $CMD
