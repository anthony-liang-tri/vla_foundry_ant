#!/usr/bin/env bash

# CRITICAL: Unset PYTHONPATH inherited from run_inference_bundle.sh
# The inference side sets PYTHONPATH to include /opt/vla_foundry/packages/grpc-workspace/src
# which contains proto files generated with grpcio>=1.71.2, but Bazel's venv has grpcio==1.71.0.
# This causes version mismatch errors when the demonstrate binary imports grpc_workspace.
unset PYTHONPATH
unset VIRTUAL_ENV

source ./venv/bin/activate
touch ./venv/COLCON_IGNORE
export PYTHONPATH=`pwd`/venv/lib/python3.12/site-packages

export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# CRITICAL: Do NOT set DISPLAY for headless GPU rendering
# Drake/VTK will automatically use EGL for GPU-accelerated rendering when DISPLAY is unset
# Setting DISPLAY=1 (invalid) causes fallback to slow CPU software rendering (llvmpipe/Mesa)
unset DISPLAY

pkill ^bazel

terminate_inference() {
  # Kill all inference-related processes that might cause UUID state mismatches on retry
  local patterns=(
    "vla_foundry/inference/robotics/inference_policy.py"
    "diffusion_policy_server.py"
    "diffusion_policy/policy_wrapper"
    "grpc.*policy"
  )

  for pattern in "${patterns[@]}"; do
    if pgrep -f "${pattern}" >/dev/null 2>&1; then
      echo "Stopping inference processes matching ${pattern}"
      pkill -f "${pattern}" >/dev/null 2>&1 || true
    fi
  done

  # Give processes time to terminate gracefully
  sleep 2

  # Force kill any remaining processes
  for pattern in "${patterns[@]}"; do
    if pgrep -f "${pattern}" >/dev/null 2>&1; then
      echo "Force killing remaining processes matching ${pattern}"
      pkill -9 -f "${pattern}" >/dev/null 2>&1 || true
    fi
  done
}

# Find completed demonstration indices by checking for summary.yaml files
# Args: $1 = save_dir
# Output: comma-separated list of completed indices (e.g., "100,101,103")
find_completed_indices() {
  local save_dir="$1"
  python3 - "$save_dir" <<'PY'
import sys
import os
from pathlib import Path

save_dir = Path(sys.argv[1])
completed = []

if save_dir.exists():
    for d in save_dir.iterdir():
        if d.is_dir() and d.name.startswith("demonstration_"):
            summary_file = d / "summary.yaml"
            if summary_file.exists():
                # Extract index from directory name
                try:
                    idx = int(d.name.split("_")[1])
                    completed.append(idx)
                except (IndexError, ValueError):
                    pass

# Print sorted comma-separated list
print(",".join(str(i) for i in sorted(completed)))
PY
}

# Compute remaining indices from a range, excluding completed ones
# Args: $1 = indices range (e.g., "100:200"), $2 = comma-separated completed indices
# Output: indices string for remaining work (space-separated individual indices or range)
# Format is compatible with demonstrate.py's --demonstration_indices which uses nargs="+"
compute_remaining_indices() {
  local indices_range="$1"
  local completed_str="$2"
  python3 - "$indices_range" "$completed_str" <<'PY'
import sys

range_str = sys.argv[1]
completed_str = sys.argv[2] if len(sys.argv) > 2 else ""

# Parse the range
start, end = map(int, range_str.split(":"))
requested = set(range(start, end))

# Parse completed indices
completed = set()
if completed_str:
    for idx_str in completed_str.split(","):
        idx_str = idx_str.strip()
        if idx_str:
            try:
                completed.add(int(idx_str))
            except ValueError:
                pass

# Compute remaining
remaining = sorted(requested - completed)

if not remaining:
    print("")  # All done
    sys.exit(0)

# Output in format compatible with demonstrate.py --demonstration_indices
# Check if contiguous, then use range format; otherwise use space-separated indices
if remaining == list(range(remaining[0], remaining[-1] + 1)):
    # Contiguous range - use single range format
    print(f"{remaining[0]}:{remaining[-1] + 1}")
else:
    # Non-contiguous: output as space-separated individual indices
    # demonstrate.py's parse_indices() can handle individual integer strings
    print(" ".join(str(i) for i in remaining))
PY
}

pascal_to_snake() {
  python - "$1" <<'PY'
import re
import sys

name = sys.argv[1]
if not name:
    raise SystemExit("Task name must be provided.")
snake = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', snake).lower()
print(snake)
PY
}

resolve_task_config() {
  local snake_task="$1"
  local config_dir="`pwd`/intuitive/visuomotor/config"
  local preferred=()
  if [ -n "${LAUNCH_STATION:-}" ]; then
    preferred+=("$(echo "${LAUNCH_STATION}" | tr '[:upper:]' '[:lower:]')")
  fi
  preferred+=("riverway" "cabot" "wollaston")

  for station in "${preferred[@]}"; do
    local candidate="${config_dir}/${snake_task}_${station}.yaml"
    if [ -f "${candidate}" ]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done

  echo "Unable to find a config file for task ${snake_task}." >&2
  exit 1
}

lookup_t_max() {
  python - "$1" <<'PY'
import ast
import pathlib
import sys

snake = sys.argv[1]
evaluate_path = pathlib.Path("eval_campaigns/ray_policy_runner.py/evaluate.py")
tree = ast.parse(evaluate_path.read_text(), filename=str(evaluate_path))

t_max = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "scenario_t_max":
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id.startswith("scenario_")
                and isinstance(stmt.value, ast.Dict)
            ):
                for key, value in zip(stmt.value.keys, stmt.value.values):
                    if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                        t_max[key.value] = value.value
        break

if snake not in t_max:
    raise SystemExit(f"Task '{snake}' missing from scenario_t_max in evaluate.py")

print(t_max[snake])
PY
}

#  --config_file `pwd`/intuitive/visuomotor/config/pick_and_place_box_cabot.yaml \
#  --config_file `pwd`/intuitive/visuomotor/config/put_spatula_in_utensil_crock_from_drying_rack_riverway.yaml \
#  --config_file `pwd`/intuitive/visuomotor/config/put_kiwi_in_center_of_table_cabot.yaml \

if [ $# -gt 0 ]; then
  LAUNCH_TASK_NAME="$1"
  shift
else
  LAUNCH_TASK_NAME="${LAUNCH_TASK_NAME:-BimanualPutRedBellPepperInBin}"
fi

TASK_SNAKE_NAME="$(pascal_to_snake "${LAUNCH_TASK_NAME}")"

LAUNCH_CONFIG_FILE="${LAUNCH_CONFIG_FILE:-$(resolve_task_config "${TASK_SNAKE_NAME}")}"
LAUNCH_SCENARIO="${LAUNCH_SCENARIO:-GrpcServerToSim}"
LAUNCH_DEMONSTRATION_INDICES="${LAUNCH_DEMONSTRATION_INDICES:-100:200}"
LAUNCH_T_MAX="${LAUNCH_T_MAX:-$(lookup_t_max "${TASK_SNAKE_NAME}")}"
LAUNCH_SAVE_DIR="${LAUNCH_SAVE_DIR:-/tmp/lbm/rollouts/}"
# Unify outputs: write summaries into the rollouts directory so we only need one S3 upload prefix.
LAUNCH_SUMMARY_DIR="${LAUNCH_SUMMARY_DIR:-${LAUNCH_SAVE_DIR}}"
USE_EVAL_SEED="${USE_EVAL_SEED:-1}"

# Retry configuration for crash recovery
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-5}"  # seconds between retries

# Per-episode timeout to detect stuck simulations (in seconds)
# Default: 15 minutes per episode is generous (most episodes complete in 2-3 minutes)
EPISODE_TIMEOUT="${EPISODE_TIMEOUT:-900}"

# Build --use_eval_seed argument if enabled
if [[ "${USE_EVAL_SEED}" == "1" ]]; then
  EVAL_SEED_ARG="--use_eval_seed"
else
  EVAL_SEED_ARG=""
fi

# Ensure CUDA_VISIBLE_DEVICES is set (default to 0 if unset, as we typically have 1 GPU per container)
export CUDA_VISIBLE_DEVICES="${LAUNCH_CUDA_VISIBLE_DEVICES:-0}"

# Verify GPU access before starting Bazel
if command -v nvidia-smi &> /dev/null; then
  echo "Verifying GPU access..."
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found"
fi

set -x

# Monitor progress during demonstrate execution
# Kills the demonstrate process if no new episode is completed within EPISODE_TIMEOUT
# Args: $1 = demonstrate PID, $2 = save_dir, $3 = timeout_seconds
monitor_demonstrate_progress() {
  local pid="$1"
  local save_dir="$2"
  local timeout_seconds="$3"
  local check_interval=30
  local seconds_without_progress=0
  local last_count=-1

  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${check_interval}"

    # Count completed episodes
    local current_count=0
    if [[ -d "${save_dir}" ]]; then
      current_count=$(find "${save_dir}" -name "summary.yaml" 2>/dev/null | wc -l)
    fi

    if [[ "${current_count}" -gt "${last_count}" ]]; then
      # Progress made
      seconds_without_progress=0
      last_count="${current_count}"
      echo "[demonstrate-monitor] Progress: ${current_count} episodes completed"
    else
      # No progress
      seconds_without_progress=$((seconds_without_progress + check_interval))

      if [[ "${seconds_without_progress}" -ge "${timeout_seconds}" ]]; then
        echo "[demonstrate-monitor] STUCK: No progress for ${seconds_without_progress}s - killing demonstrate process"
        # Kill the demonstrate process and its children
        pkill -TERM -P "${pid}" 2>/dev/null || true
        kill -TERM "${pid}" 2>/dev/null || true
        sleep 2
        pkill -KILL -P "${pid}" 2>/dev/null || true
        kill -KILL "${pid}" 2>/dev/null || true
        return 124  # Return timeout exit code
      fi
    fi
  done

  # Process finished naturally, get its exit code
  wait "${pid}" 2>/dev/null
  return $?
}

# Function to run the demonstrate command
# $1 = indices (can be "100:200" or space-separated like "105 107 110")
run_demonstrate() {
  local indices="$1"
  # Convert indices string to array for proper argument handling
  # This handles both "100:200" and "105 107 110" formats
  read -ra INDICES_ARGS <<< "${indices}"

  local demonstrate_pid
  local exit_code

  if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    echo "Using pre-built binary (SKIP_BUILD=1)..."
    ./bazel-bin/intuitive/visuomotor/demonstrate \
      --config_file "${LAUNCH_CONFIG_FILE}" \
      --scenario "${LAUNCH_SCENARIO}" \
      --demonstration_indices "${INDICES_ARGS[@]}" \
      --t_max "${LAUNCH_T_MAX}" \
      --save_dir="${LAUNCH_SAVE_DIR}" \
      --summary_dir="${LAUNCH_SUMMARY_DIR}" \
      ${EVAL_SEED_ARG} &
    demonstrate_pid=$!
  else
    echo "Running bazel run..."
    bazel run \
      --verbose_failures \
      //intuitive/visuomotor:demonstrate -- \
      --config_file "${LAUNCH_CONFIG_FILE}" \
      --scenario "${LAUNCH_SCENARIO}" \
      --demonstration_indices "${INDICES_ARGS[@]}" \
      --t_max "${LAUNCH_T_MAX}" \
      --save_dir="${LAUNCH_SAVE_DIR}" \
      --summary_dir="${LAUNCH_SUMMARY_DIR}" \
      ${EVAL_SEED_ARG} &
    demonstrate_pid=$!
  fi

  # Monitor progress with timeout
  echo "Starting demonstrate monitor (PID: ${demonstrate_pid}, timeout: ${EPISODE_TIMEOUT}s per episode)"
  monitor_demonstrate_progress "${demonstrate_pid}" "${LAUNCH_SAVE_DIR}" "${EPISODE_TIMEOUT}"
  exit_code=$?

  if [[ "${exit_code}" -eq 124 ]]; then
    echo "Demonstrate timed out (no progress for ${EPISODE_TIMEOUT}s)"
  fi

  return "${exit_code}"
}

# Retry loop with crash recovery
ORIGINAL_INDICES="${LAUNCH_DEMONSTRATION_INDICES}"
ATTEMPT=0
FINAL_EXIT_CODE=0

while true; do
  ATTEMPT=$((ATTEMPT + 1))

  # Check which demonstrations are already completed
  COMPLETED_INDICES=$(find_completed_indices "${LAUNCH_SAVE_DIR}")
  if [ -n "${COMPLETED_INDICES}" ]; then
    echo "Found completed demonstrations: ${COMPLETED_INDICES}"
  fi

  # Compute remaining indices
  REMAINING_INDICES=$(compute_remaining_indices "${ORIGINAL_INDICES}" "${COMPLETED_INDICES}")

  if [ -z "${REMAINING_INDICES}" ]; then
    echo "All demonstrations completed successfully!"
    FINAL_EXIT_CODE=0
    break
  fi

  echo "=========================================="
  echo "Attempt ${ATTEMPT}/${MAX_RETRIES}: Running demonstrations ${REMAINING_INDICES}"
  echo "=========================================="

  # Run the demonstrate command
  set +e  # Don't exit on error
  run_demonstrate "${REMAINING_INDICES}"
  EXIT_CODE=$?
  set -e

  if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Demonstrate command completed successfully"
    FINAL_EXIT_CODE=0
    break
  fi

  echo "Demonstrate command failed with exit code ${EXIT_CODE}"

  # Check if we have more retries
  if [ ${ATTEMPT} -ge ${MAX_RETRIES} ]; then
    echo "Maximum retries (${MAX_RETRIES}) reached. Giving up."
    FINAL_EXIT_CODE=${EXIT_CODE}
    break
  fi

  # Check progress - if we made progress, reset attempt counter consideration
  NEW_COMPLETED=$(find_completed_indices "${LAUNCH_SAVE_DIR}")
  if [ "${NEW_COMPLETED}" != "${COMPLETED_INDICES}" ]; then
    echo "Progress made since last attempt. Continuing with remaining work."
  else
    echo "No progress made. Will retry in ${RETRY_DELAY} seconds..."
    sleep ${RETRY_DELAY}
  fi

  # Clean up any hanging processes before retry
  terminate_inference

  echo "Retrying..."
done

echo "=========================================="
echo "Final status: Attempt ${ATTEMPT}, Exit code ${FINAL_EXIT_CODE}"
FINAL_COMPLETED=$(find_completed_indices "${LAUNCH_SAVE_DIR}")
if [ -n "${FINAL_COMPLETED}" ]; then
  COMPLETED_COUNT=$(echo "${FINAL_COMPLETED}" | tr ',' '\n' | wc -l)
  echo "Completed demonstrations: ${COMPLETED_COUNT}"
fi
echo "=========================================="

# upload the rollouts to s3 at the models path
if [ -n "${CHECKPOINT_DIR:-}" ]; then
  # Strip trailing slashes to avoid double-slash paths in S3
  CHECKPOINT_DIR_CLEAN="${CHECKPOINT_DIR%/}"
  # Strip checkpoint filename if present (e.g., checkpoint.ckpt or checkpoint_12345.pt)
  # to get the parent directory for cleaner S3 paths
  if [[ "${CHECKPOINT_DIR_CLEAN}" == *.ckpt ]] || [[ "${CHECKPOINT_DIR_CLEAN}" == *.pt ]]; then
    CHECKPOINT_DIR_CLEAN="${CHECKPOINT_DIR_CLEAN%/*}"
  fi

  # Build S3 path components
  # Optional subfolder for organizing results from different campaigns (e.g., "oss", "stage3")
  if [ -n "${LAUNCH_EVALUATION_SUBFOLDER:-}" ]; then
    S3_SUBFOLDER_PREFIX="${LAUNCH_EVALUATION_SUBFOLDER}/"
  else
    S3_SUBFOLDER_PREFIX=""
  fi

  # Unique eval ID to prevent result overwrites across eval runs
  if [ -n "${LAUNCH_EVAL_ID:-}" ]; then
    S3_EVAL_ID_PREFIX="${LAUNCH_EVAL_ID}/"
  else
    S3_EVAL_ID_PREFIX=""
  fi

  # Include task name in S3 path to prevent different tasks from overwriting each other
  # when evaluating multiple tasks against the same checkpoint
  if [ -n "${LAUNCH_TASK_NAME:-}" ]; then
    S3_TASK_PREFIX="${LAUNCH_TASK_NAME}/"
  else
    S3_TASK_PREFIX=""
  fi

  # Final path: {checkpoint}/evaluation/{subfolder}/{eval_id}/{task}/rollouts/
  S3_UPLOAD_PATH="${CHECKPOINT_DIR_CLEAN}/evaluation/${S3_SUBFOLDER_PREFIX}${S3_EVAL_ID_PREFIX}${S3_TASK_PREFIX}rollouts/"
  echo "Uploading rollouts to s3 at ${S3_UPLOAD_PATH}"
  if env -u AWS_PROFILE aws s3 sync "${LAUNCH_SAVE_DIR}" "${S3_UPLOAD_PATH}"; then
    echo "Rollouts uploaded successfully, cleaning up local copy..."
    # Write success flag BEFORE cleaning up local files
    # This signals to run_inference_bundle.sh that the job completed successfully
    # even if local episode files are deleted
    SUCCESS_FLAG_DIR="$(dirname "${LAUNCH_SAVE_DIR}")"
    echo "S3_UPLOAD_SUCCESS" > "${SUCCESS_FLAG_DIR}/s3_upload_success"
    echo "SUCCESS_PATH=${S3_UPLOAD_PATH}" >> "${SUCCESS_FLAG_DIR}/s3_upload_success"
    rm -rf "${LAUNCH_SAVE_DIR}"/* 2>/dev/null || true
  else
    echo "WARNING: Rollout upload failed, keeping local copy"
  fi
else
  echo "Skipping S3 upload: CHECKPOINT_DIR is not set"
fi

terminate_inference

# ./run --build '' //intuitive/visuomotor:create_video_mosaic \
#     --episode_globs /tmp/lbm/rollouts/*.pkl \
#     --fps 10 --num_process 5 --mode individual_mosaic