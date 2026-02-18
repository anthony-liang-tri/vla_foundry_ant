#!/usr/bin/env bash
#
# Orchestrates a single evaluation by running the Bazel launch script and the
# uv-based inference client in parallel. Logs for each process are written to
# ${LOG_DIR}/<job_name>/{bazel,inference}.log.
#
# Expected environment variables:
#   JOB_NAME                      - Friendly name for log folders (default: job-<pid>).
#   CHECKPOINT_DIR                - S3 or local path passed to inference_policy.py.
#   TASK_NAME                     - Optional task name forwarded to inference_policy.py.
#   NUM_FLOW_STEPS                - Optional, defaults to 8.
#   OPEN_LOOP_STEPS               - Optional, defaults to 8.
#   DEVICE                        - Optional, defaults to "cuda".
#   INFERENCE_EXTRA_ARGS          - Optional string appended to inference command.
#   LAUNCH_SCRIPT                 - launch script to execute (default: launch_sim.sh).
#   LAUNCH_WORKDIR                - Directory containing launch scripts (default: /opt/anzu).
#   INFERENCE_WORKDIR             - Directory uv should use for inference (default: /opt/vla_foundry).
#   INFERENCE_USE_WRITABLE_COPY   - Set to 1 to rsync INFERENCE_WORKDIR into a
#                                   writable temp dir before running uv (default: 0).
#                                   When disabled, uv is executed directly from
#                                   INFERENCE_WORKDIR (e.g., /opt/vla_foundry).
#   LAUNCH_CONFIG_FILE, LAUNCH_SCENARIO,
#   LAUNCH_DEMONSTRATION_INDICES, LAUNCH_T_MAX, LAUNCH_SAVE_DIR, LAUNCH_SUMMARY_DIR,
#   LAUNCH_CUDA_VISIBLE_DEVICES   - forwarded to the launch script when set.
#   LOG_DIR                       - Base directory for logs (default: /tmp/lbm/logs).
#   VLA_FOUNDRY_HOME              - Directory containing the vla_foundry repo (default: /opt/vla_foundry).
#   VLA_FOUNDRY_REF               - Optional branch/tag/commit to align before syncing (default: current branch or main).
#   VLA_FOUNDRY_AUTO_UPDATE       - Set to 0 to skip auto git fetch/uv sync (default: 1).
#   VLA_FOUNDRY_REMOTE_URL        - Optional remote URL to use when updating (default: git@github.com:TRI-ML/vla_foundry.git).
#   VLA_FOUNDRY_REMOTE_NAME       - Remote name to update/fetch (default: origin).
#   VLA_FOUNDRY_GIT_TOKEN         - Personal access token for HTTPS remotes (default: GITHUB_TOKEN if set).
#   VLA_FOUNDRY_GIT_USER          - Username for HTTPS token auth (default: x-access-token).
#   VLA_FOUNDRY_GIT_SSH_COMMAND   - Override ssh invocation when using SSH remotes (default enables ssh-agent forwarding).
#   VLA_FOUNDRY_REQUIREMENTS      - Space-separated runtime tooling upgrades enforced post-sync (default: "setuptools>=68 wheel>=0.41 build>=0.10").
#   MAX_RETRIES                   - Maximum number of retries on SimFailure/OOM errors (default: 3).
#   RETRY_DELAY                   - Delay in seconds between retries (default: 5).
#   MAX_MEMORY_USAGE_PERCENT      - Maximum memory usage percentage before triggering restart (default: 94).
#                                   Set just below Ray's 95% threshold to preempt OOM kills.
#   MEMORY_CHECK_INTERVAL         - How often to check memory in seconds (default: 10).
#   INFERENCE_SCRIPT              - Python script to run for inference (default: vla_foundry/inference/robotics/inference_policy.py).
#                                   Use this to run a different inference script (e.g., grpc_workspace/diffusion_policy_server.py).
#   INFERENCE_SCRIPT_ARGS         - Additional arguments for the inference script.
#                                   These are appended after the default args. Use this to pass custom arguments.
#   INFERENCE_CMD_OVERRIDE        - Complete override for the inference command.
#                                   If set, replaces the entire inference command (ignores INFERENCE_SCRIPT, etc).
#                                   Supports placeholders: {checkpoint}, {num_flow_steps}, {open_loop_steps}, {device}.

echo "Running inference bundle v0.4.0"
set -euo pipefail

# Prevent uv hardlink race conditions when multiple jobs run on same node
export UV_LINK_MODE=copy

# Global flag to track if we received SIGTERM
RECEIVED_SIGTERM=0

JOB_NAME="${JOB_NAME:-job-$$}"
LAUNCH_WORKDIR="${LAUNCH_WORKDIR:-/opt/anzu}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-launch_sim.sh}"
INFERENCE_WORKDIR="${INFERENCE_WORKDIR:-/opt/vla_foundry}"
INFERENCE_USE_WRITABLE_COPY="${INFERENCE_USE_WRITABLE_COPY:-0}"
LOG_DIR="${LOG_DIR:-/tmp/lbm/logs}"
NUM_FLOW_STEPS="${NUM_FLOW_STEPS:-8}"
OPEN_LOOP_STEPS="${OPEN_LOOP_STEPS:-8}"
DEVICE="${DEVICE:-cuda}"
VLA_FOUNDRY_HOME="${VLA_FOUNDRY_HOME:-/opt/vla_foundry}"
VLA_FOUNDRY_REF="${VLA_FOUNDRY_REF:-}"
VLA_FOUNDRY_AUTO_UPDATE="${VLA_FOUNDRY_AUTO_UPDATE:-1}"
VLA_FOUNDRY_REMOTE_URL="${VLA_FOUNDRY_REMOTE_URL:-git@github.com:jmercat/vla_foundry.git}"
VLA_FOUNDRY_REMOTE_NAME="${VLA_FOUNDRY_REMOTE_NAME:-origin}"
VLA_FOUNDRY_GIT_TOKEN="${VLA_FOUNDRY_GIT_TOKEN:-${GITHUB_TOKEN:-}}"
VLA_FOUNDRY_GIT_USER="${VLA_FOUNDRY_GIT_USER:-x-access-token}"
VLA_FOUNDRY_GIT_SSH_COMMAND="${VLA_FOUNDRY_GIT_SSH_COMMAND:-}"
VLA_FOUNDRY_REQUIREMENTS="${VLA_FOUNDRY_REQUIREMENTS:-setuptools>=68 wheel>=0.41 build>=0.10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-5}"
MAX_MEMORY_USAGE_PERCENT="${MAX_MEMORY_USAGE_PERCENT:-90}"
MEMORY_CHECK_INTERVAL="${MEMORY_CHECK_INTERVAL:-1}"
# Progress watchdog - detects stuck simulations (no new episode files)
PROGRESS_TIMEOUT="${PROGRESS_TIMEOUT:-1200}"  # 20 minutes without progress = stuck
PROGRESS_CHECK_INTERVAL="${PROGRESS_CHECK_INTERVAL:-30}"  # Check every 30 seconds
# Custom inference command configuration
INFERENCE_SCRIPT="${INFERENCE_SCRIPT:-vla_foundry/inference/robotics/inference_policy.py}"
INFERENCE_SCRIPT_ARGS="${INFERENCE_SCRIPT_ARGS:-}"
INFERENCE_CMD_OVERRIDE="${INFERENCE_CMD_OVERRIDE:-}"

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  echo "CHECKPOINT_DIR must be set" >&2
  exit 1
fi

# Ensure cache directories exist for the container user
# When running as --user 1000:1000, HOME is set to /tmp, so create cache there
if [[ -n "${XDG_CACHE_HOME:-}" ]]; then
  mkdir -p "${XDG_CACHE_HOME}" 2>/dev/null || true
fi
# Also ensure HOME/.cache exists if HOME is set
if [[ -n "${HOME:-}" ]] && [[ "${HOME}" != "/" ]]; then
  mkdir -p "${HOME}/.cache" "${HOME}/.local" 2>/dev/null || true
fi

mkdir -p "${LOG_DIR}/${JOB_NAME}"
BAZEL_LOG="${LOG_DIR}/${JOB_NAME}/bazel.log"
INFERENCE_LOG="${LOG_DIR}/${JOB_NAME}/inference.log"
SETUP_LOG="${LOG_DIR}/${JOB_NAME}/setup.log"

cd "${LAUNCH_WORKDIR}"

echo "Launch script logs: ${BAZEL_LOG}"
echo "Inference logs:     ${INFERENCE_LOG}"
echo "Setup logs:         ${SETUP_LOG}"

touch "${BAZEL_LOG}" "${INFERENCE_LOG}" "${SETUP_LOG}"

# Redirect all setup output to setup log
exec 1> >(tee -a "${SETUP_LOG}")
exec 2> >(tee -a "${SETUP_LOG}" >&2)

# Start memory monitoring
MEMORY_LOG="${LOG_DIR}/${JOB_NAME}/memory_profile.log"
LOW_MEMORY_FLAG="${LOG_DIR}/${JOB_NAME}/low_memory_flag"
memory_monitor_pid=""

# Get memory usage as a percentage (0-100)
get_memory_usage_percent() {
  if command -v free >/dev/null 2>&1; then
    # Calculate: (total - available) / total * 100
    free | awk '/^Mem:/ {printf "%.0f", ($2 - $7) / $2 * 100}'
  else
    echo "0"  # Return 0 if free command not available (won't trigger kill)
  fi
}

# Get memory info for logging (total, used, available in GB)
get_memory_info() {
  if command -v free >/dev/null 2>&1; then
    free -g | awk '/^Mem:/ {printf "Total: %d GB, Used: %d GB, Available: %d GB", $2, $3, $7}'
  else
    echo "Memory info unavailable"
  fi
}

monitor_memory() {
  local max_usage_percent="$1"
  local check_interval="$2"
  local log_interval_seconds=30
  local last_log_time=0

  while true; do
    local usage_percent
    local mem_info
    local current_time
    
    usage_percent=$(get_memory_usage_percent)
    current_time=$(date +%s)
    
    # Log if enough time has passed OR if memory is getting high (> 80% or close to threshold)
    # We always log if we are within 5% of the threshold or above 80% to help debugging spikes.
    local should_log=0
    if (( current_time - last_log_time >= log_interval_seconds )); then
      should_log=1
    elif [[ "${usage_percent}" -ge 80 ]] || [[ "${usage_percent}" -ge $((max_usage_percent - 5)) ]]; then
      should_log=1
    fi

    if [[ "${should_log}" -eq 1 ]]; then
      mem_info=$(get_memory_info)
      {
        echo "===== Memory snapshot at $(date '+%Y-%m-%d %H:%M:%S') ====="
        echo "Memory usage: ${usage_percent}% (threshold: ${max_usage_percent}%)"
        echo "${mem_info}"
        ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -n 11
        echo ""
      } >> "${MEMORY_LOG}" 2>&1
      last_log_time=${current_time}
    fi

    # Check if memory usage exceeds threshold (before Ray's 95% kills us)
    if [[ "${usage_percent}" -ge "${max_usage_percent}" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] MEMORY CRITICAL: ${usage_percent}% usage >= ${max_usage_percent}% threshold - KILLING PROCESSES" >> "${MEMORY_LOG}"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] MEMORY CRITICAL: ${usage_percent}% usage >= ${max_usage_percent}% threshold - KILLING PROCESSES" >&2
      echo "LOW_MEMORY" > "${LOW_MEMORY_FLAG}"

      # Actively kill the simulation and inference processes to trigger restart
      # This allows the retry logic to resume from the last completed episode
      pkill -TERM -f "demonstrate" 2>/dev/null || true
      pkill -TERM -f "inference_policy.py" 2>/dev/null || true
      pkill -TERM -f "diffusion_policy_server.py" 2>/dev/null || true
      pkill -TERM -f "diffusion_policy/policy_wrapper" 2>/dev/null || true
      sleep 2
      pkill -KILL -f "demonstrate" 2>/dev/null || true
      pkill -KILL -f "inference_policy.py" 2>/dev/null || true
      pkill -KILL -f "diffusion_policy_server.py" 2>/dev/null || true
      pkill -KILL -f "diffusion_policy/policy_wrapper" 2>/dev/null || true

      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processes killed due to high memory usage. Retry logic will resume from last completed episode." >> "${MEMORY_LOG}"
    fi

    sleep "${check_interval}"
  done
}

# Remove any stale low memory flag
rm -f "${LOW_MEMORY_FLAG}"

# Progress watchdog - detects stuck simulations
PROGRESS_LOG="${LOG_DIR}/${JOB_NAME}/progress.log"
STUCK_FLAG="${LOG_DIR}/${JOB_NAME}/stuck_flag"
progress_watchdog_pid=""
last_episode_count_file="${LOG_DIR}/${JOB_NAME}/last_episode_count"

monitor_progress() {
  local save_dir="$1"
  local start_idx="$2"
  local end_idx="$3"
  local timeout_seconds="$4"
  local check_interval="$5"
  local log_file="$6"

  local seconds_without_progress=0
  local last_count=-1

  while true; do
    sleep "${check_interval}"

    # Count completed episodes
    local current_count=0
    if [[ -d "${save_dir}" ]]; then
      for idx in $(seq "${start_idx}" "$((end_idx - 1))"); do
        if [[ -f "${save_dir}/episode_${idx}.pkl" ]]; then
          current_count=$((current_count + 1))
        fi
      done
    fi

    # Store count for other processes to read
    echo "${current_count}" > "${last_episode_count_file}"

    if [[ "${current_count}" -gt "${last_count}" ]]; then
      # Progress made
      seconds_without_progress=0
      last_count="${current_count}"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Progress: ${current_count}/$((end_idx - start_idx)) episodes completed" >> "${log_file}"
    else
      # No progress
      seconds_without_progress=$((seconds_without_progress + check_interval))

      if [[ "${seconds_without_progress}" -ge "${timeout_seconds}" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] STUCK DETECTED: No progress for ${seconds_without_progress}s (threshold: ${timeout_seconds}s)" >> "${log_file}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] STUCK DETECTED: No progress for ${seconds_without_progress}s - killing processes" >&2
        echo "STUCK" > "${STUCK_FLAG}"

        # Kill simulation processes to trigger restart
        pkill -TERM -f "demonstrate" 2>/dev/null || true
        pkill -TERM -f "inference_policy.py" 2>/dev/null || true
        pkill -TERM -f "diffusion_policy_server.py" 2>/dev/null || true
        # Also kill Drake-related processes that might hold GPU resources
        pkill -TERM -f "drake" 2>/dev/null || true
        pkill -TERM -f "bazel-bin" 2>/dev/null || true
        sleep 2
        pkill -KILL -f "demonstrate" 2>/dev/null || true
        pkill -KILL -f "inference_policy.py" 2>/dev/null || true
        pkill -KILL -f "diffusion_policy_server.py" 2>/dev/null || true
        pkill -KILL -f "drake" 2>/dev/null || true
        pkill -KILL -f "bazel-bin" 2>/dev/null || true

        # Force release GPU memory by killing any remaining GPU processes
        if command -v nvidia-smi &>/dev/null; then
          local gpu_pids
          gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
          if [[ -n "${gpu_pids}" ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Killing GPU processes to free memory: ${gpu_pids}" >> "${log_file}"
            for pid in ${gpu_pids}; do
              kill -9 "${pid}" 2>/dev/null || true
            done
            sleep 2
          fi
        fi

        # Reset counter for next attempt
        seconds_without_progress=0
      elif [[ "$((seconds_without_progress % 120))" -eq 0 ]] && [[ "${seconds_without_progress}" -gt 0 ]]; then
        # Log warning every 2 minutes
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: No progress for ${seconds_without_progress}s (timeout at ${timeout_seconds}s)" >> "${log_file}"
      fi
    fi
  done
}

# SIGTERM handler - check if work is done before exiting
handle_sigterm() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Received SIGTERM signal"
  RECEIVED_SIGTERM=1

  # First check for S3 upload success flag (most reliable indicator)
  if [[ -n "${CURRENT_SAVE_DIR:-}" ]]; then
    local s3_success_flag
    s3_success_flag="$(dirname "${CURRENT_SAVE_DIR}")/s3_upload_success"
    if [[ -f "${s3_success_flag}" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] S3 upload success flag found - job already completed successfully before SIGTERM"
      cat "${s3_success_flag}"
      # Run cleanup
      cleanup_launched_processes 2>/dev/null || true
      upload_and_cleanup 2>/dev/null || true
      cleanup_orphaned_processes 2>/dev/null || true
      exit 0
    fi
  fi

  # Fall back to checking local episode files
  if [[ -n "${CURRENT_SAVE_DIR:-}" ]] && [[ -n "${ORIGINAL_START_IDX:-}" ]] && [[ -n "${ORIGINAL_END_IDX:-}" ]]; then
    local completed_idx
    completed_idx=$(get_next_episode_index "${CURRENT_SAVE_DIR}" "${ORIGINAL_START_IDX}" "${ORIGINAL_END_IDX}")

    if [[ "${completed_idx}" -ge "${ORIGINAL_END_IDX}" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] All episodes completed successfully before SIGTERM - exiting with success"
      # Run cleanup
      cleanup_launched_processes 2>/dev/null || true
      upload_and_cleanup 2>/dev/null || true
      cleanup_orphaned_processes 2>/dev/null || true
      exit 0
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Only $((completed_idx - ORIGINAL_START_IDX))/$((ORIGINAL_END_IDX - ORIGINAL_START_IDX)) episodes completed before SIGTERM"
    fi
  fi

  # If work not done, run cleanup and exit with SIGTERM status
  cleanup_launched_processes 2>/dev/null || true
  upload_and_cleanup 2>/dev/null || true
  cleanup_orphaned_processes 2>/dev/null || true
  exit 143
}

# Set up SIGTERM trap (will be refined after variables are initialized)
trap handle_sigterm TERM

echo "Starting memory monitor (output: ${MEMORY_LOG}, threshold: ${MAX_MEMORY_USAGE_PERCENT}%)"
if command -v ps >/dev/null 2>&1 && command -v free >/dev/null 2>&1; then
  monitor_memory "${MAX_MEMORY_USAGE_PERCENT}" "${MEMORY_CHECK_INTERVAL}" &
  memory_monitor_pid=$!
else
  echo "Warning: 'ps' or 'free' command not found. Memory monitoring disabled." >> "${SETUP_LOG}"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting inference bundle setup for ${JOB_NAME}"

# --- Health Checks & Diagnostics ---
echo "===== Node Diagnostics ====="
echo "Hostname: $(hostname)"
echo "Uptime: $(uptime)"
echo "Memory:"
free -h
echo "Process Count: $(ps -e | wc -l)"
echo "Disk Space:"
df -h /tmp
echo "============================"

if [[ "${DEVICE}" == "cuda" ]]; then
  echo "===== GPU Health Check ====="
  if ! command -v nvidia-smi &> /dev/null; then
     echo "ERROR: nvidia-smi not found but DEVICE=cuda requested." >&2
     exit 1
  fi
  nvidia-smi
  
  # Verify PyTorch sees the GPU
  # We use the python from the venv or system to check
  echo "Verifying PyTorch CUDA access..."
  if ! python3 -c "import torch; print(f'Torch version: {torch.__version__}'); assert torch.cuda.is_available(), 'PyTorch cannot see CUDA device'" 2>/dev/null; then
     echo "WARNING: System python3 could not verify CUDA. Trying inside uv/venv later."
  else
     echo "System PyTorch sees CUDA."
  fi
  echo "============================"
fi
# -----------------------------------

log_stream_pids=()
inference_pid=""

stream_log() {
  local label="$1"
  local path="$2"
  stdbuf -oL tail -n +1 -F "${path}" | sed -e "s/^/[${label}] /" &
  log_stream_pids+=("$!")
}

stream_log "LAUNCH" "${BAZEL_LOG}"
stream_log "INFER" "${INFERENCE_LOG}"

env_for_launch=(
  ANZU_ROOT="${LAUNCH_WORKDIR}"
  CHECKPOINT_DIR="${CHECKPOINT_DIR}"
  AWS_PROFILE="sagemaker"
)

# Forward optional environment variables to the launch script if provided.
forward_if_set() {
  local key="$1"
  local value="${!key:-}"
  if [[ -n "${value}" ]]; then
    env_for_launch+=("${key}=${value}")
  fi
}

forward_if_set LAUNCH_CONFIG_FILE
forward_if_set LAUNCH_SCENARIO
forward_if_set LAUNCH_DEMONSTRATION_INDICES
forward_if_set LAUNCH_T_MAX
forward_if_set LAUNCH_SAVE_DIR
forward_if_set LAUNCH_SUMMARY_DIR
forward_if_set LAUNCH_CUDA_VISIBLE_DEVICES
forward_if_set LAUNCH_TASK_NAME

# Ray sets CUDA_VISIBLE_DEVICES to isolate the GPU for this job.
# We explicitly forward it to LAUNCH_CUDA_VISIBLE_DEVICES so launch_sim.sh
# respects this isolation when running the simulator.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ -z "${LAUNCH_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export LAUNCH_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi
forward_if_set LAUNCH_CUDA_VISIBLE_DEVICES

update_vla_foundry_checkout() {
  if [[ "${VLA_FOUNDRY_AUTO_UPDATE}" == "0" ]]; then
    echo "Skipping vla_foundry auto-update (disabled via VLA_FOUNDRY_AUTO_UPDATE=0)"
    return 0
  fi

  if [[ ! -d "${VLA_FOUNDRY_HOME}/.git" ]]; then
    echo "Skipping vla_foundry auto-update: ${VLA_FOUNDRY_HOME} is not a git repository"
    return 0
  fi

  local remote_name="${VLA_FOUNDRY_REMOTE_NAME}"
  if [[ -n "${VLA_FOUNDRY_REMOTE_URL}" ]]; then
    if git -C "${VLA_FOUNDRY_HOME}" remote get-url "${remote_name}" >/dev/null 2>&1; then
      git -C "${VLA_FOUNDRY_HOME}" remote set-url "${remote_name}" "${VLA_FOUNDRY_REMOTE_URL}"
    else
      git -C "${VLA_FOUNDRY_HOME}" remote add "${remote_name}" "${VLA_FOUNDRY_REMOTE_URL}"
    fi
  fi

  local remote_url
  remote_url=$(git -C "${VLA_FOUNDRY_HOME}" remote get-url "${remote_name}" 2>/dev/null || true)
  if [[ -z "${remote_url}" ]]; then
    echo "Skipping vla_foundry auto-update: remote '${remote_name}' is not configured" >&2
    return 0
  fi

  local target_ref="${VLA_FOUNDRY_REF}"
  if [[ -z "${target_ref}" || "${target_ref}" == "HEAD" ]]; then
    target_ref="$(cd "${VLA_FOUNDRY_HOME}" && git symbolic-ref --short HEAD 2>/dev/null || true)"
  fi
  if [[ -z "${target_ref}" ]]; then
    target_ref="jean/inference_no_clip"
  fi

  echo "Auto-updating vla_foundry at ${VLA_FOUNDRY_HOME} (remote: ${remote_name} -> ${remote_url}, ref: ${target_ref})"
  (
    set -euo pipefail
    cd "${VLA_FOUNDRY_HOME}"
    local askpass_script=""
    local temp_known_hosts=""
    local git_env=()
    cleanup_auto_update_artifacts() {
      if [[ -n "${askpass_script}" && -f "${askpass_script}" ]]; then
        rm -f "${askpass_script}"
      fi
      if [[ -n "${temp_known_hosts}" && -f "${temp_known_hosts}" ]]; then
        rm -f "${temp_known_hosts}"
      fi
    }
    trap cleanup_auto_update_artifacts EXIT
    if [[ "${remote_url}" == http* ]] && [[ -n "${VLA_FOUNDRY_GIT_TOKEN}" ]]; then
      askpass_script="$(mktemp)"
      cat <<'EOF' > "${askpass_script}"
#!/usr/bin/env bash
case "$1" in
  *Username*) printf "%s\n" "${VLA_FOUNDRY_GIT_USER:-x-access-token}" ;;
  *Password*) printf "%s\n" "${VLA_FOUNDRY_GIT_TOKEN:?missing token for HTTPS git authentication}" ;;
  *) printf "\n" ;;
esac
EOF
      chmod 700 "${askpass_script}"
      git_env+=("GIT_ASKPASS=${askpass_script}" "GIT_TERMINAL_PROMPT=0" "VLA_FOUNDRY_GIT_USER=${VLA_FOUNDRY_GIT_USER}" "VLA_FOUNDRY_GIT_TOKEN=${VLA_FOUNDRY_GIT_TOKEN}")
    else
      local ssh_cmd="${VLA_FOUNDRY_GIT_SSH_COMMAND:-}"
      local ssh_known_hosts_default="${HOME:-/tmp}/.ssh/known_hosts"
      local ssh_known_hosts="${VLA_FOUNDRY_KNOWN_HOSTS:-${ssh_known_hosts_default}}"
      local remote_host=""
      if [[ "${remote_url}" == git@* ]]; then
        remote_host="${remote_url#*@}"
        remote_host="${remote_host%%[:/]*}"
      elif [[ "${remote_url}" == ssh://* ]]; then
        remote_host="${remote_url#ssh://}"
        remote_host="${remote_host%%[:/]*}"
      fi
      if [[ -z "${remote_host}" ]]; then
        remote_host="github.com"
      fi
      if [[ -z "${ssh_cmd}" ]]; then
        local ssh_known_hosts_dir
        ssh_known_hosts_dir="$(dirname "${ssh_known_hosts}")"
        if ! install -d -m 0700 "${ssh_known_hosts_dir}" >/dev/null 2>&1; then
          ssh_known_hosts="$(mktemp /tmp/vla_foundry_known_hosts.XXXXXX)"
          temp_known_hosts="${ssh_known_hosts}"
        fi
        if ! touch "${ssh_known_hosts}" >/dev/null 2>&1; then
          ssh_known_hosts="$(mktemp /tmp/vla_foundry_known_hosts.XXXXXX)"
          temp_known_hosts="${ssh_known_hosts}"
        fi
        chmod 600 "${ssh_known_hosts}" >/dev/null 2>&1 || true
        if [[ -n "${remote_host}" ]] && ! ssh-keygen -F "${remote_host}" -f "${ssh_known_hosts}" >/dev/null 2>&1; then
          ssh-keyscan "${remote_host}" >> "${ssh_known_hosts}" 2>/dev/null || true
        fi
        ssh_cmd="ssh -o StrictHostKeyChecking=yes -o UserKnownHostsFile=${ssh_known_hosts}"
      fi
      git_env+=("GIT_SSH_COMMAND=${ssh_cmd}")
      if [[ -n "${SSH_AUTH_SOCK:-}" ]]; then
        git_env+=("SSH_AUTH_SOCK=${SSH_AUTH_SOCK}")
      else
        echo "WARNING: SSH_AUTH_SOCK is not set; git fetch may fail if credentials are required" >&2
      fi
    fi
    run_git() {
      if [[ "${#git_env[@]}" -gt 0 ]]; then
        env "${git_env[@]}" "$@"
      else
        "$@"
      fi
    }

    # run_git git fetch --tags "${remote_name}"
    if ! git checkout "${target_ref}" >/dev/null 2>&1; then
      if run_git git rev-parse --verify --quiet "${remote_name}/${target_ref}" >/dev/null; then
        git checkout -B "${target_ref}" "${remote_name}/${target_ref}"
      else
        git checkout "${target_ref}"
      fi
    fi
    local reset_target="${remote_name}/${target_ref}"
    if ! run_git git rev-parse --verify --quiet "${reset_target}" >/dev/null; then
      reset_target="${target_ref}"
    fi
    git reset --hard "${reset_target}"
    uv sync --python /usr/bin/python3 --link-mode=copy
    if [[ -n "${VLA_FOUNDRY_REQUIREMENTS}" ]]; then
      echo "Ensuring vla_foundry tooling packages: ${VLA_FOUNDRY_REQUIREMENTS}"
      uv pip install --python "${VLA_FOUNDRY_VENV}/bin/python" --upgrade ${VLA_FOUNDRY_REQUIREMENTS}
    fi
    echo "vla_foundry now at commit $(git rev-parse HEAD) on branch $(git rev-parse --abbrev-ref HEAD)"
  )
}

if ! update_vla_foundry_checkout; then
  echo "WARNING: Failed to auto-update vla_foundry; continuing with existing checkout" >&2
fi

prepare_checkpoint_directory() {
  if [[ "${CHECKPOINT_DIR}" != s3://* ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checkpoint is local path, skipping S3 download"
    return
  fi

  local job_ckpt_dir="/tmp/vla_foundry_checkpoints/${JOB_NAME}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Preparing checkpoint workspace at ${job_ckpt_dir}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Downloading from ${CHECKPOINT_DIR}"
  rm -rf "${job_ckpt_dir}"
  mkdir -p "${job_ckpt_dir}/checkpoints"

  local aws_cmd=(aws)
  if [[ -n "${AWS_PROFILE:-}" ]]; then
    aws_cmd+=(--profile "${AWS_PROFILE}")
  fi

  local s3_root="${CHECKPOINT_DIR%/}"
  local explicit_checkpoint_path=""
  local explicit_checkpoint_is_ckpt=0

  if [[ "${s3_root}" == *.pt ]]; then
    explicit_checkpoint_path="${s3_root}"
    s3_root="${s3_root%/checkpoints/*}"
  elif [[ "${s3_root}" == *.ckpt ]]; then
    # LBM-style checkpoints are single files (.ckpt) rather than vla_foundry checkpoint directories.
    # Treat the provided path as the checkpoint file itself, and do not require config.yaml.
    explicit_checkpoint_path="${s3_root}"
    s3_root="${s3_root%/*}"
    explicit_checkpoint_is_ckpt=1
  fi

  copy_required() {
    local source="$1"
    local dest="$2"
    local friendly="$3"
    echo "aws s3 cp ${source} -> ${dest} (${friendly})"
    if ! "${aws_cmd[@]}" s3 cp "${source}" "${dest}"; then
      echo "ERROR: Failed to download ${friendly} from ${source}" >&2
      exit 1
    fi
  }

  copy_optional() {
    local source="$1"
    local dest="$2"
    echo "aws s3 cp ${source} -> ${dest} (optional)"
    if "${aws_cmd[@]}" s3 cp "${source}" "${dest}" >/dev/null 2>&1; then
      return 0
    fi
    return 1
  }

  # Core config files
  if [[ "${explicit_checkpoint_is_ckpt}" -eq 1 ]]; then
    # Optional for .ckpt: the inference script may not need it.
    copy_optional "${s3_root}/config.yaml" "${job_ckpt_dir}/config.yaml" || true
  else
    copy_required "${s3_root}/config.yaml" "${job_ckpt_dir}/config.yaml" "config.yaml"
  fi
  copy_optional "${s3_root}/config_normalizer.yaml" "${job_ckpt_dir}/config_normalizer.yaml" || true
  copy_optional "${s3_root}/config_processor.yaml" "${job_ckpt_dir}/config_processor.yaml" || true

  if ! copy_optional "${s3_root}/stats.json" "${job_ckpt_dir}/stats.json"; then
    if ! copy_optional "${s3_root}/stats_normalizer.json" "${job_ckpt_dir}/stats.json"; then
      echo "WARNING: Could not find stats.json or stats_normalizer.json under ${s3_root}"
    fi
  fi

  if ! copy_optional "${s3_root}/preprocessing_config.yaml" "${job_ckpt_dir}/preprocessing_config.yaml"; then
    if ! copy_optional "${s3_root}/preprocessing_configs.yaml" "${job_ckpt_dir}/preprocessing_config.yaml"; then
      echo "WARNING: Could not find preprocessing_config.yaml or preprocessing_configs.yaml under ${s3_root}"
    fi
  fi

  if ! copy_optional "${s3_root}/preprocessing_config.yaml" "${job_ckpt_dir}/preprocessing_configs.yaml"; then
    if ! copy_optional "${s3_root}/preprocessing_configs.yaml" "${job_ckpt_dir}/preprocessing_configs.yaml"; then
      echo "WARNING: Could not find preprocessing_config.yaml or preprocessing_configs.yaml under ${s3_root}"
    fi
  fi

  # Keep the original filenames if consumers expect them.
  copy_optional "${s3_root}/stats_normalizer.json" "${job_ckpt_dir}/stats_normalizer.json" || true
  copy_optional "${s3_root}/preprocessing_configs.yaml" "${job_ckpt_dir}/preprocessing_configs.yaml" || true

  local checkpoint_number=""
  local checkpoint_basename=""
  if [[ -n "${explicit_checkpoint_path}" ]]; then
    checkpoint_basename="$(basename "${explicit_checkpoint_path}")"
    # Extract number from explicit checkpoint path
    if [[ "${checkpoint_basename}" =~ checkpoint_([0-9]+)\.pt$ ]]; then
      checkpoint_number="${BASH_REMATCH[1]}"
    fi
    if [[ "${checkpoint_basename}" =~ \.ckpt$ ]]; then
      checkpoint_number=""
    fi
  else
    echo "Listing checkpoints under ${s3_root}/checkpoints/"
    local latest_checkpoint
    latest_checkpoint=$("${aws_cmd[@]}" s3 ls "${s3_root}/checkpoints/" | awk '{print $4}' | grep '^checkpoint_[0-9]\+\.pt$' | sort -t_ -k2,2n | tail -1)
    if [[ -z "${latest_checkpoint}" ]]; then
      echo "ERROR: Could not find any checkpoint_*.pt under ${s3_root}/checkpoints/" >&2
      exit 1
    fi
    checkpoint_basename="${latest_checkpoint}"
    explicit_checkpoint_path="${s3_root}/checkpoints/${checkpoint_basename}"
    # Extract number from latest checkpoint
    if [[ "${checkpoint_basename}" =~ checkpoint_([0-9]+)\.pt$ ]]; then
      checkpoint_number="${BASH_REMATCH[1]}"
    fi
  fi

  echo "Selected checkpoint file ${checkpoint_basename} (number: ${checkpoint_number})"
  echo "Downloading checkpoint ${checkpoint_basename}"
  if [[ "${explicit_checkpoint_is_ckpt}" -eq 1 ]]; then
    copy_required "${explicit_checkpoint_path}" "${job_ckpt_dir}/${checkpoint_basename}" "${checkpoint_basename}"
    export CHECKPOINT_FILE="${job_ckpt_dir}/${checkpoint_basename}"
    echo "Checkpoint stored at ${CHECKPOINT_FILE}"
  else
    copy_required "${explicit_checkpoint_path}" "${job_ckpt_dir}/checkpoints/${checkpoint_basename}" "${checkpoint_basename}"
    echo "Checkpoint stored at ${job_ckpt_dir}/checkpoints/${checkpoint_basename}"
  fi

  # Try to download corresponding EMA file if we have a checkpoint number
  if [[ -n "${checkpoint_number}" ]]; then
    local ema_basename="ema_${checkpoint_number}.pt"
    local ema_path="${s3_root}/checkpoints/${ema_basename}"
    echo "Checking for EMA file ${ema_basename}"
    if copy_optional "${ema_path}" "${job_ckpt_dir}/checkpoints/${ema_basename}"; then
      echo "EMA file stored at ${job_ckpt_dir}/checkpoints/${ema_basename}"
    else
      echo "No EMA file found at ${ema_path} (optional, continuing without it)"
    fi
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checkpoint download completed successfully"

  CHECKPOINT_DIR="${job_ckpt_dir}"
}

# Handle absolute paths for LAUNCH_SCRIPT (e.g., /opt/vla_foundry/launch_sim.sh)
# or relative paths (e.g., launch_sim.sh)
# CRITICAL: Use -u PYTHONPATH -u VIRTUAL_ENV to isolate the Bazel environment from
# the uv environment. The inference side sets PYTHONPATH to include grpc-workspace
# proto files that require grpcio>=1.71.2, but Bazel's venv may have an older version.
if [[ "${LAUNCH_SCRIPT}" == /* ]]; then
  # Absolute path - use as-is
  launch_cmd=(env -u PYTHONPATH -u VIRTUAL_ENV "${env_for_launch[@]}" bash "${LAUNCH_SCRIPT}")
else
  # Relative path - look in LAUNCH_WORKDIR
  launch_cmd=(env -u PYTHONPATH -u VIRTUAL_ENV "${env_for_launch[@]}" bash "./${LAUNCH_SCRIPT}")
fi

# Check if a SimFailure occurred by scanning the log files
check_sim_failure() {
  local log_file="$1"
  if [[ -f "${log_file}" ]]; then
    if grep -q "SimFailure\|The initial guess for line search is NaN" "${log_file}"; then
      return 0  # SimFailure detected
    fi
  fi
  return 1  # No SimFailure
}

# Check if an OOM (Out of Memory) error occurred
check_oom_failure() {
  local log_file="$1"
  if [[ -f "${log_file}" ]]; then
    if grep -q "killed due to the node running low on memory\|memory usage threshold\|out of memory\|OOM\|MemoryError" "${log_file}"; then
      return 0  # OOM detected
    fi
  fi
  return 1  # No OOM
}

# Check if the low memory flag was set by the memory monitor
check_low_memory_flag() {
  if [[ -f "${LOW_MEMORY_FLAG}" ]]; then
    return 0  # Low memory detected
  fi
  return 1
}

# Check if the stuck flag was set by the progress watchdog
check_stuck_flag() {
  if [[ -f "${STUCK_FLAG}" ]]; then
    return 0  # Stuck detected
  fi
  return 1
}

# Check if any retryable error occurred (SimFailure, OOM, low memory, etc.)
check_retryable_error() {
  local log_file="$1"
  local error_type=""

  if check_sim_failure "${log_file}"; then
    error_type="SimFailure"
  elif check_oom_failure "${log_file}"; then
    error_type="OOM"
  fi

  if [[ -n "${error_type}" ]]; then
    echo "${error_type}"
    return 0
  fi
  return 1
}

# Count completed episodes in the save directory and return the next episode index
# Args: save_dir, original_start_index, original_end_index
# Returns: new_start_index (the next episode to run)
get_next_episode_index() {
  local save_dir="$1"
  local start_idx="$2"
  local end_idx="$3"

  if [[ ! -d "${save_dir}" ]]; then
    echo "${start_idx}"
    return
  fi

  # Find the highest completed episode index within our range
  local highest_completed=-1
  for idx in $(seq "${start_idx}" "$((end_idx - 1))"); do
    if [[ -f "${save_dir}/episode_${idx}.pkl" ]]; then
      highest_completed="${idx}"
    fi
  done

  if [[ "${highest_completed}" -ge "${start_idx}" ]]; then
    # Return the next index after the highest completed
    echo "$((highest_completed + 1))"
  else
    echo "${start_idx}"
  fi
}

# Parse demonstration indices (format: "start:end" or just a number)
parse_demonstration_indices() {
  local indices="$1"
  if [[ "${indices}" == *:* ]]; then
    echo "${indices}" | tr ':' ' '
  else
    echo "${indices} ${indices}"
  fi
}

# Forcefully kill all related processes
kill_all_processes() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Killing all simulation and inference processes..."

  # Kill bazel process and its children
  if [[ -n "${bazel_pid:-}" ]] && kill -0 "${bazel_pid}" 2>/dev/null; then
    echo "Killing launch script process tree (pid ${bazel_pid})"
    pkill -TERM -P "${bazel_pid}" 2>/dev/null || true
    kill -TERM "${bazel_pid}" 2>/dev/null || true
    sleep 1
    pkill -KILL -P "${bazel_pid}" 2>/dev/null || true
    kill -KILL "${bazel_pid}" 2>/dev/null || true
    wait "${bazel_pid}" 2>/dev/null || true
  fi
  bazel_pid=""

  # Kill inference process and its children
  if [[ -n "${inference_pid:-}" ]] && kill -0 "${inference_pid}" 2>/dev/null; then
    echo "Killing inference process tree (pid ${inference_pid})"
    pkill -TERM -P "${inference_pid}" 2>/dev/null || true
    kill -TERM "${inference_pid}" 2>/dev/null || true
    sleep 1
    pkill -KILL -P "${inference_pid}" 2>/dev/null || true
    kill -KILL "${inference_pid}" 2>/dev/null || true
    wait "${inference_pid}" 2>/dev/null || true
  fi
  inference_pid=""

  # Kill any remaining demonstrate or inference_policy processes
  pkill -f "demonstrate" 2>/dev/null || true
  pkill -f "inference_policy.py" 2>/dev/null || true
  # Kill diffusion policy server processes (LBM campaigns)
  pkill -f "diffusion_policy_server.py" 2>/dev/null || true
  pkill -f "diffusion_policy/policy_wrapper" 2>/dev/null || true
  pkill -f "grpc.*policy" 2>/dev/null || true

  # Give processes time to clean up
  sleep 2
}

# Save the original S3 checkpoint path before prepare_checkpoint_directory modifies it
# This is needed for uploading rollouts back to S3
CHECKPOINT_DIR_S3="${CHECKPOINT_DIR}"

prepare_checkpoint_directory

cleanup_launched_processes() {
  if [[ -n "${memory_monitor_pid:-}" ]] && kill -0 "${memory_monitor_pid}" >/dev/null 2>&1; then
    echo "Stopping memory monitor (pid ${memory_monitor_pid})"
    kill "${memory_monitor_pid}" >/dev/null 2>&1 || true
    wait "${memory_monitor_pid}" 2>/dev/null || true
    memory_monitor_pid=""
  fi
  if [[ -n "${progress_watchdog_pid:-}" ]] && kill -0 "${progress_watchdog_pid}" >/dev/null 2>&1; then
    echo "Stopping progress watchdog (pid ${progress_watchdog_pid})"
    kill "${progress_watchdog_pid}" >/dev/null 2>&1 || true
    wait "${progress_watchdog_pid}" 2>/dev/null || true
    progress_watchdog_pid=""
  fi
  if [[ -n "${bazel_pid:-}" ]] && kill -0 "${bazel_pid}" >/dev/null 2>&1; then
    echo "Stopping launch script (pid ${bazel_pid})"
    kill "${bazel_pid}" >/dev/null 2>&1 || true
    wait "${bazel_pid}" 2>/dev/null || true
    bazel_pid=""
  fi
  if [[ -n "${inference_pid:-}" ]] && kill -0 "${inference_pid}" >/dev/null 2>&1; then
    echo "Stopping inference command (pid ${inference_pid})"
    kill "${inference_pid}" >/dev/null 2>&1 || true
    wait "${inference_pid}" 2>/dev/null || true
    inference_pid=""
  fi
  if [[ "${#log_stream_pids[@]}" -gt 0 ]]; then
    for tail_pid in "${log_stream_pids[@]}"; do
      if kill -0 "${tail_pid}" >/dev/null 2>&1; then
        # Kill the process and its children (the tail -F in the pipeline)
        pkill -TERM -P "${tail_pid}" 2>/dev/null || true
        kill "${tail_pid}" >/dev/null 2>&1 || true
        wait "${tail_pid}" 2>/dev/null || true
      fi
    done
  fi
  # Kill any remaining tail processes watching our log files
  pkill -f "tail.*${LOG_DIR}/${JOB_NAME}" 2>/dev/null || true
}

trap cleanup_launched_processes EXIT INT TERM

# Build inference command (done once, outside the retry loop)
# Supports three modes:
# 1. INFERENCE_CMD_OVERRIDE: Complete custom command with placeholder substitution
# 2. INFERENCE_SCRIPT: Custom script path with standard uv run wrapper
# 3. Default: vla_foundry/inference/robotics/inference_policy.py

build_inference_command() {
  local cmd=()

  if [[ -n "${INFERENCE_CMD_OVERRIDE:-}" ]]; then
    # Mode 1: Complete override with placeholder substitution
    echo "Using custom inference command override" >&2
    local override_cmd="${INFERENCE_CMD_OVERRIDE}"
    # Prefer a concrete checkpoint file path when available (e.g., LBM .ckpt),
    # otherwise fall back to CHECKPOINT_DIR (vla_foundry checkpoint directory).
    override_cmd="${override_cmd//\{checkpoint\}/${CHECKPOINT_FILE:-${CHECKPOINT_DIR}}}"
    override_cmd="${override_cmd//\{num_flow_steps\}/${NUM_FLOW_STEPS}}"
    override_cmd="${override_cmd//\{open_loop_steps\}/${OPEN_LOOP_STEPS}}"
    override_cmd="${override_cmd//\{device\}/${DEVICE}}"
    # For override mode, just return the command string (will be executed via eval)
    echo "${override_cmd}"
    return
  fi

  # Build standard command with env prefix
  cmd=(env -u PYTHONPATH -u VIRTUAL_ENV)

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    cmd+=(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}")
  fi

  # Mode 2/3: Use INFERENCE_SCRIPT (custom or default vla_foundry script)
  local script="${INFERENCE_SCRIPT}"
  echo "Using inference script: ${script}" >&2

  cmd+=(
    uv run --group inference --group visualization
    python "${script}"
  )

  # Add default arguments based on script type
  # Check if this looks like the vla_foundry inference_policy.py
  if [[ "${script}" == *"inference_policy.py"* ]]; then
    cmd+=(
      --checkpoint_directory "${CHECKPOINT_DIR}"
      --num_flow_steps "${NUM_FLOW_STEPS}"
      --open_loop_steps "${OPEN_LOOP_STEPS}"
      --device "${DEVICE}"
    )

    if [[ -n "${TASK_NAME:-}" ]]; then
      cmd+=(--task_name "${TASK_NAME}")
    fi
  else
    # For other scripts (like LBM's diffusion_policy_server.py), use more generic args
    # Users should pass specific args via INFERENCE_SCRIPT_ARGS
    echo "Note: Non-standard inference script detected. Use INFERENCE_SCRIPT_ARGS for custom arguments." >&2
  fi

  # Add extra args (legacy)
  if [[ -n "${INFERENCE_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2086
    cmd+=(${INFERENCE_EXTRA_ARGS})
  fi

  # Add custom script args
  if [[ -n "${INFERENCE_SCRIPT_ARGS:-}" ]]; then
    # shellcheck disable=SC2086
    cmd+=(${INFERENCE_SCRIPT_ARGS})
  fi

  # Return command as a quoted string for later use
  printf '%q ' "${cmd[@]}"
}

# Store the built command (as a string since it may be an override or array)
BUILT_INFERENCE_CMD="$(build_inference_command)"
echo "Built inference command: ${BUILT_INFERENCE_CMD}"

if [[ ! -d "${INFERENCE_WORKDIR}" ]]; then
  echo "ERROR: INFERENCE_WORKDIR '${INFERENCE_WORKDIR}' does not exist" >&2
  exit 1
fi

if [[ ! -f "${INFERENCE_WORKDIR}/pyproject.toml" ]]; then
  echo "WARNING: INFERENCE_WORKDIR '${INFERENCE_WORKDIR}' is missing pyproject.toml" >&2
  # Don't exit - custom repos may not need pyproject.toml
fi

# Only check for default vla_foundry script if not using custom script/override
if [[ -z "${INFERENCE_CMD_OVERRIDE:-}" ]] && [[ "${INFERENCE_SCRIPT}" == "vla_foundry/inference/robotics/inference_policy.py" ]]; then
  if [[ ! -f "${INFERENCE_WORKDIR}/${INFERENCE_SCRIPT}" ]]; then
    echo "ERROR: INFERENCE_WORKDIR '${INFERENCE_WORKDIR}' does not contain ${INFERENCE_SCRIPT}" >&2
    exit 1
  fi
fi


# When using the baked-in /opt/vla_foundry checkout, copy it to a writable
# location so editable installs (uv run) can update egg-info metadata.
INFERENCE_WORKDIR_RUNTIME="${INFERENCE_WORKDIR}"
if [[ "${INFERENCE_USE_WRITABLE_COPY}" != "1" ]]; then
  workspace_writable=0
  if [[ -w "${INFERENCE_WORKDIR}" ]]; then
    workspace_writable=1
  fi
  if [[ -d "${INFERENCE_WORKDIR}/.venv" && ! -w "${INFERENCE_WORKDIR}/.venv" ]]; then
    workspace_writable=0
  fi
  if [[ "${workspace_writable}" -eq 0 ]]; then
    echo "Inference workspace ${INFERENCE_WORKDIR} is not writable; enabling writable copy."
    INFERENCE_USE_WRITABLE_COPY=1
  fi
fi

if [[ "${INFERENCE_USE_WRITABLE_COPY}" == "1" ]]; then
  INFERENCE_WORKDIR_RUNTIME="${INFERENCE_WRITABLE_COPY_DIR:-/tmp/vla_foundry_runtime}"
  echo "Preparing writable inference workspace at ${INFERENCE_WORKDIR_RUNTIME}"
  rm -rf "${INFERENCE_WORKDIR_RUNTIME}"
  mkdir -p "${INFERENCE_WORKDIR_RUNTIME}"
  # OPTIMIZATION: Exclude unnecessary files and allow partial copy (ignore unreadable temp files)
  rsync -a --ignore-errors --no-perms --no-owner --no-group \
    --exclude ".git" \
    --exclude ".venv" \
    --exclude "__pycache__" \
    --exclude "tests" \
    --exclude "*.pyc" \
    "${INFERENCE_WORKDIR}/" "${INFERENCE_WORKDIR_RUNTIME}/" || true
fi

# If the inference command is fully overridden (e.g., LBM policy server), uv may
# not install vla_foundry's inference dependency group. Ensure the bundled
# workspace packages are importable by adding them to PYTHONPATH.
#
# This is a no-op for the default vla_foundry path because the standard command
# starts with `env -u PYTHONPATH ...`.
pythonpath_prefix=""
if [[ -d "/opt/vla_foundry/packages/robot-gym/src" ]]; then
  pythonpath_prefix="/opt/vla_foundry/packages/robot-gym/src"
fi
if [[ -d "/opt/vla_foundry/packages/grpc-workspace/src" ]]; then
  if [[ -n "${pythonpath_prefix}" ]]; then
    pythonpath_prefix="${pythonpath_prefix}:/opt/vla_foundry/packages/grpc-workspace/src"
  else
    pythonpath_prefix="/opt/vla_foundry/packages/grpc-workspace/src"
  fi
fi
if [[ -n "${pythonpath_prefix}" ]]; then
  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${pythonpath_prefix}:${PYTHONPATH}"
  else
    export PYTHONPATH="${pythonpath_prefix}"
  fi
fi

# Wrap the built command in an array for execution.
# NOTE: This must happen AFTER INFERENCE_WORKDIR_RUNTIME is finalized so uv runs
# from a writable workspace (otherwise it may try to create /opt/vla_foundry/.venv).
inference_cmd=(bash -c "
  cd '${INFERENCE_WORKDIR_RUNTIME}'
  TORCH_LIB_PATH=\"${INFERENCE_WORKDIR_RUNTIME}/.venv/lib/python3.12/site-packages/torch/lib\"
  export LD_LIBRARY_PATH=\"\${TORCH_LIB_PATH}:\${LD_LIBRARY_PATH:-}\"
  ${BUILT_INFERENCE_CMD}"
)

# Debug: Show which vla_foundry code is being used
echo "===== vla_foundry code verification ====="
echo "INFERENCE_WORKDIR: ${INFERENCE_WORKDIR}"
echo "INFERENCE_WORKDIR_RUNTIME: ${INFERENCE_WORKDIR_RUNTIME}"
if [[ -d "${INFERENCE_WORKDIR}/.git" ]]; then
  echo "Git info at INFERENCE_WORKDIR:"
  git -C "${INFERENCE_WORKDIR}" log -1 --oneline 2>/dev/null || echo "  (could not read git log)"
  git -C "${INFERENCE_WORKDIR}" branch --show-current 2>/dev/null || echo "  (could not read branch)"
else
  echo "INFERENCE_WORKDIR is not a git repo (likely a mount without .git)"
fi
echo "Files in INFERENCE_WORKDIR_RUNTIME:"
ls -la "${INFERENCE_WORKDIR_RUNTIME}/" 2>/dev/null | head -10 || echo "  (could not list files)"
echo "========================================="

print_log_file() {
  local label="$1"
  local path="$2"
  if [[ -f "${path}" ]]; then
    echo
    echo "===== ${label} log (${path}) ====="
    cat "${path}"
    echo "===== end ${label} log ====="
  else
    echo "WARNING: ${label} log not found at ${path}"
  fi
}

# Main execution with retry loop for SimFailure errors
retry_count=0
final_status=0

# Store original demonstration indices for calculating resume points
ORIGINAL_DEMONSTRATION_INDICES="${LAUNCH_DEMONSTRATION_INDICES:-100:200}"
read -r ORIGINAL_START_IDX ORIGINAL_END_IDX <<< "$(parse_demonstration_indices "${ORIGINAL_DEMONSTRATION_INDICES}")"
CURRENT_DEMONSTRATION_INDICES="${ORIGINAL_DEMONSTRATION_INDICES}"

# Get save directory for checking completed episodes
CURRENT_SAVE_DIR="${LAUNCH_SAVE_DIR:-/tmp/lbm/rollouts/}"

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting attempt $((retry_count + 1)) of $((MAX_RETRIES + 1))"

  # Clear any stale flags at the start of each attempt
  rm -f "${LOW_MEMORY_FLAG}"
  rm -f "${STUCK_FLAG}"

  # Stop any previous progress watchdog
  if [[ -n "${progress_watchdog_pid:-}" ]] && kill -0 "${progress_watchdog_pid}" 2>/dev/null; then
    kill "${progress_watchdog_pid}" 2>/dev/null || true
    wait "${progress_watchdog_pid}" 2>/dev/null || true
    progress_watchdog_pid=""
  fi

  # On retry, check for completed episodes and update start index
  if [[ "${retry_count}" -gt 0 ]]; then
    # Backup previous logs before clearing
    cp "${BAZEL_LOG}" "${BAZEL_LOG}.attempt${retry_count}" 2>/dev/null || true
    cp "${INFERENCE_LOG}" "${INFERENCE_LOG}.attempt${retry_count}" 2>/dev/null || true

    # Find the next episode to run based on completed episodes
    NEW_START_IDX=$(get_next_episode_index "${CURRENT_SAVE_DIR}" "${ORIGINAL_START_IDX}" "${ORIGINAL_END_IDX}")

    if [[ "${NEW_START_IDX}" -ge "${ORIGINAL_END_IDX}" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] All episodes (${ORIGINAL_START_IDX}:${ORIGINAL_END_IDX}) already completed!"
      final_status=0
      break
    fi

    CURRENT_DEMONSTRATION_INDICES="${NEW_START_IDX}:${ORIGINAL_END_IDX}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resuming from episode ${NEW_START_IDX} (originally ${ORIGINAL_START_IDX}:${ORIGINAL_END_IDX})"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed episodes: $((NEW_START_IDX - ORIGINAL_START_IDX)) of $((ORIGINAL_END_IDX - ORIGINAL_START_IDX))"
  fi
  : > "${BAZEL_LOG}"
  : > "${INFERENCE_LOG}"

  # Build launch command with potentially updated demonstration indices
  current_env_for_launch=(
    ANZU_ROOT="${LAUNCH_WORKDIR}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR_S3}"
    AWS_PROFILE="sagemaker"
    LAUNCH_DEMONSTRATION_INDICES="${CURRENT_DEMONSTRATION_INDICES}"
  )

  # Forward optional environment variables (except LAUNCH_DEMONSTRATION_INDICES which we handle above)
  for key in LAUNCH_CONFIG_FILE LAUNCH_SCENARIO LAUNCH_T_MAX LAUNCH_SAVE_DIR LAUNCH_SUMMARY_DIR LAUNCH_CUDA_VISIBLE_DEVICES LAUNCH_TASK_NAME; do
    local_value="${!key:-}"
    if [[ -n "${local_value}" ]]; then
      current_env_for_launch+=("${key}=${local_value}")
    fi
  done

  # Build launch command (with environment isolation)
  if [[ "${LAUNCH_SCRIPT}" == /* ]]; then
    current_launch_cmd=(env -u PYTHONPATH -u VIRTUAL_ENV "${current_env_for_launch[@]}" bash "${LAUNCH_SCRIPT}")
  else
    current_launch_cmd=(env -u PYTHONPATH -u VIRTUAL_ENV "${current_env_for_launch[@]}" bash "./${LAUNCH_SCRIPT}")
  fi

  echo "Starting launch script: ${current_launch_cmd[*]}"
  (
    "${current_launch_cmd[@]}" > "${BAZEL_LOG}" 2>&1
  ) &
  bazel_pid=$!

  echo "Starting inference command: ${inference_cmd[*]}"
  (
    cd "${INFERENCE_WORKDIR_RUNTIME}"
    "${inference_cmd[@]}" > "${INFERENCE_LOG}" 2>&1
  ) &
  inference_pid=$!

  # Start progress watchdog to detect stuck simulations
  echo "Starting progress watchdog (timeout: ${PROGRESS_TIMEOUT}s, check interval: ${PROGRESS_CHECK_INTERVAL}s)"
  # Parse current demonstration indices for watchdog
  read -r WATCH_START_IDX WATCH_END_IDX <<< "$(parse_demonstration_indices "${CURRENT_DEMONSTRATION_INDICES}")"
  monitor_progress "${CURRENT_SAVE_DIR}" "${WATCH_START_IDX}" "${WATCH_END_IDX}" \
    "${PROGRESS_TIMEOUT}" "${PROGRESS_CHECK_INTERVAL}" "${PROGRESS_LOG}" &
  progress_watchdog_pid=$!

  inference_status=""
  inference_reaped=0
  bazel_status=""
  bazel_reaped=0
  first_finished=""
  finished_pid=""

  set +e
  while true; do
    if [[ "${bazel_reaped}" -eq 0 && "${inference_reaped}" -eq 0 ]]; then
      wait -n -p finished_pid -- "${bazel_pid}" "${inference_pid}"
      status=$?
    elif [[ "${bazel_reaped}" -eq 0 ]]; then
      wait -p finished_pid "${bazel_pid}"
      status=$?
    elif [[ "${inference_reaped}" -eq 0 ]]; then
      wait -p finished_pid "${inference_pid}"
      status=$?
    else
      break
    fi

    # If wait was interrupted by a signal (status > 128), exit the loop
    # The cleanup handler will take care of terminating processes
    if [[ $status -gt 128 ]]; then
      echo "Wait interrupted by signal (status $status), exiting process wait loop"
      # Set inference_status so final_status has a valid value
      inference_status=$status
      break
    fi

    if [[ "${finished_pid:-}" == "${bazel_pid}" ]]; then
      bazel_status=$status
      bazel_reaped=1
      bazel_pid=""
      if [[ -z "${first_finished}" ]]; then
        first_finished="launch"
      fi
      break
    elif [[ "${finished_pid:-}" == "${inference_pid}" ]]; then
      inference_status=$status
      inference_reaped=1
      inference_pid=""
      if [[ -z "${first_finished}" ]]; then
        first_finished="inference"
      fi
      break
    fi
  done

  if [[ "${first_finished}" == "launch" ]]; then
    echo "Launch script completed (status ${bazel_status}). Stopping inference command."
    if [[ "${inference_reaped}" -eq 0 && -n "${inference_pid:-}" ]]; then
      kill "${inference_pid}" >/dev/null 2>&1 || true
      wait "${inference_pid}" >/dev/null 2>&1
      inference_status=$?
      inference_reaped=1
      inference_pid=""
    fi
    if [[ "${bazel_status}" -eq 0 ]]; then
      inference_status=0
    fi
  else
    echo "Inference command completed (status ${inference_status}). Stopping launch script."
    if [[ "${bazel_reaped}" -eq 0 && -n "${bazel_pid:-}" ]]; then
      kill "${bazel_pid}" >/dev/null 2>&1 || true
      wait "${bazel_pid}" >/dev/null 2>&1
      bazel_status=$?
      bazel_reaped=1
      bazel_pid=""
    fi
  fi
  set -e

  # Stop progress watchdog for this attempt (will restart on retry)
  if [[ -n "${progress_watchdog_pid:-}" ]] && kill -0 "${progress_watchdog_pid}" 2>/dev/null; then
    kill "${progress_watchdog_pid}" 2>/dev/null || true
    wait "${progress_watchdog_pid}" 2>/dev/null || true
    progress_watchdog_pid=""
  fi

  # Check if a retryable error occurred (SimFailure, OOM, low memory, stuck, etc.)
  retryable_error=""

  # First check the stuck flag (progress watchdog detection)
  if check_stuck_flag; then
    retryable_error="Stuck"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stuck simulation detected by progress watchdog!"
    # Clear the flag for next iteration
    rm -f "${STUCK_FLAG}"
  # Then check the low memory flag (proactive detection)
  elif check_low_memory_flag; then
    retryable_error="LowMemory"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Low memory condition detected by memory watchdog!"
    # Clear the flag for next iteration
    rm -f "${LOW_MEMORY_FLAG}"
  # Then check logs for errors
  elif error_type=$(check_retryable_error "${BAZEL_LOG}"); then
    retryable_error="${error_type}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${retryable_error} error detected in launch script log!"
  elif error_type=$(check_retryable_error "${INFERENCE_LOG}"); then
    retryable_error="${error_type}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${retryable_error} error detected in inference log!"
  fi

  if [[ -n "${retryable_error}" && "${retry_count}" -lt "${MAX_RETRIES}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${retryable_error} detected. Attempt $((retry_count + 1)) of $((MAX_RETRIES + 1)) failed."
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Will check for completed episodes and retry in ${RETRY_DELAY} seconds..."

    # Kill any remaining processes forcefully
    kill_all_processes

    retry_count=$((retry_count + 1))
    sleep "${RETRY_DELAY}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting simulation and inference (attempt $((retry_count + 1))) - will resume from last completed episode..."
    continue
  fi

  # No retryable error or max retries reached
  if [[ -n "${retryable_error}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${retryable_error} persisted after $((retry_count + 1)) attempts. Giving up."
    final_status=1
  else
    final_status="${inference_status}"
  fi
  break
done

cleanup_launched_processes

if [[ -n "${bazel_pid:-}" ]]; then
  wait "${bazel_pid}" 2>/dev/null || true
fi

print_log_file "Launch script" "${BAZEL_LOG}"
print_log_file "Inference" "${INFERENCE_LOG}"

if [[ "${retry_count}" -gt 0 ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total retry attempts: ${retry_count}"
fi

# Check if all episodes completed successfully (override final_status if so)
# This handles cases where inference exits non-zero but all work was done

# First check for S3 upload success flag (written by launch_sim.sh after successful upload)
# This is the most reliable indicator because local files may be cleaned up
S3_SUCCESS_FLAG="$(dirname "${CURRENT_SAVE_DIR}")/s3_upload_success"
if [[ -f "${S3_SUCCESS_FLAG}" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] S3 upload success flag found - job completed successfully!"
  cat "${S3_SUCCESS_FLAG}"
  final_status=0
else
  # Fall back to checking local files
  COMPLETED_EPISODE_IDX=$(get_next_episode_index "${CURRENT_SAVE_DIR}" "${ORIGINAL_START_IDX}" "${ORIGINAL_END_IDX}")
  if [[ "${COMPLETED_EPISODE_IDX}" -ge "${ORIGINAL_END_IDX}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] All episodes (${ORIGINAL_START_IDX}:${ORIGINAL_END_IDX}) completed successfully!"
    final_status=0
  elif [[ "${final_status}" -ne 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Job failed with status ${final_status}. Completed episodes: $((COMPLETED_EPISODE_IDX - ORIGINAL_START_IDX)) of $((ORIGINAL_END_IDX - ORIGINAL_START_IDX))"
  fi
fi

# Upload rollouts to S3 and clean up disk space
upload_and_cleanup() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting post-job upload and cleanup..."

  local aws_cmd=(aws)
  if [[ -n "${AWS_PROFILE:-}" ]]; then
    aws_cmd+=(--profile "${AWS_PROFILE}")
  fi

  # Upload rollouts to S3 if checkpoint was from S3
  if [[ "${CHECKPOINT_DIR_S3}" == s3://* ]] && [[ -d "${CURRENT_SAVE_DIR}" ]]; then
    # Strip checkpoint filename to get the base checkpoint path
    local s3_base="${CHECKPOINT_DIR_S3%/}"
    if [[ "${s3_base}" == *.pt ]]; then
      s3_base="${s3_base%/checkpoints/*}"
    elif [[ "${s3_base}" == *.ckpt ]]; then
      s3_base="${s3_base%/*}"
    fi

    # Include task name in path to support multi-task evaluation on same checkpoint
    if [[ -n "${LAUNCH_TASK_NAME:-}" ]]; then
      local s3_dest="${s3_base}/evaluation/${LAUNCH_TASK_NAME}/rollouts/"
    else
      local s3_dest="${s3_base}/evaluation/rollouts/"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Uploading rollouts to ${s3_dest}"

    if "${aws_cmd[@]}" s3 sync "${CURRENT_SAVE_DIR}" "${s3_dest}" --quiet; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rollouts uploaded successfully"

      # Clean up rollouts directory after successful upload
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up rollouts directory: ${CURRENT_SAVE_DIR}"
      rm -rf "${CURRENT_SAVE_DIR}"
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Failed to upload rollouts to S3, keeping local copy"
    fi
  fi

  # Clean up downloaded checkpoint (always, as it's a copy from S3)
  local job_ckpt_dir="/tmp/vla_foundry_checkpoints/${JOB_NAME}"
  if [[ -d "${job_ckpt_dir}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up checkpoint directory: ${job_ckpt_dir}"
    rm -rf "${job_ckpt_dir}"
  fi

  # Clean up writable inference workspace copy
  if [[ "${INFERENCE_USE_WRITABLE_COPY}" == "1" ]] && [[ -d "${INFERENCE_WORKDIR_RUNTIME}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up inference workspace: ${INFERENCE_WORKDIR_RUNTIME}"
    rm -rf "${INFERENCE_WORKDIR_RUNTIME}"
  fi

  # Job-scoped disk cleanup (safe when multiple jobs run on same node)
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running job-scoped disk cleanup for ${JOB_NAME}..."

  # Clean up only THIS job's checkpoint directory (not other jobs')
  local job_ckpt_dir="/tmp/vla_foundry_checkpoints/${JOB_NAME}"
  if [[ -d "${job_ckpt_dir}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up checkpoint directory: ${job_ckpt_dir}"
    rm -rf "${job_ckpt_dir}"
  fi

  # Clean up only THIS job's rollout directory
  if [[ -n "${CURRENT_SAVE_DIR:-}" ]] && [[ -d "${CURRENT_SAVE_DIR}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up rollout directory: ${CURRENT_SAVE_DIR}"
    rm -rf "${CURRENT_SAVE_DIR}"
  fi

  # Clean up only THIS job's runtime directory (if it exists)
  if [[ -d "/tmp/vla_foundry_runtime_${JOB_NAME}" ]]; then
    rm -rf "/tmp/vla_foundry_runtime_${JOB_NAME}"
  fi

  # Clean up pip/cache directories that may accumulate
  rm -rf /tmp/pip-* 2>/dev/null || true

  # Clean up any large core dumps
  find /tmp -maxdepth 1 -name "core.*" -delete 2>/dev/null || true

  # Docker cleanup only works if we have socket access (typically only on host, not inside container)
  # This is a no-op inside containers but useful if script runs on host
  if command -v docker &>/dev/null && [[ -S /var/run/docker.sock ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up Docker resources..."
    docker container prune -f 2>/dev/null || true
    docker builder prune -f 2>/dev/null || true
    # Remove dangling images only (not tagged base images)
    docker image prune -f 2>/dev/null || true
  fi

  # Clean up HuggingFace cache if using local storage (models stay cached in /mnt/local_storage)
  # Don't clean /mnt/local_storage/cache as it's shared across jobs for efficiency
  # But clean any temp HF downloads in /tmp
  rm -rf /tmp/transformers_cache 2>/dev/null || true
  rm -rf /tmp/huggingface* 2>/dev/null || true

  # Report disk usage after cleanup
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Disk usage after cleanup:"
  df -h /tmp 2>/dev/null || true

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Post-job cleanup completed"
}

# Kill any orphaned processes that might hold GPU memory or other resources
cleanup_orphaned_processes() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up orphaned processes..."

  # Kill any remaining demonstrate/inference processes
  pkill -9 -f "demonstrate" 2>/dev/null || true
  pkill -9 -f "inference_policy.py" 2>/dev/null || true
  pkill -9 -f "drake" 2>/dev/null || true
  # Kill diffusion policy server processes (LBM campaigns)
  pkill -9 -f "diffusion_policy_server.py" 2>/dev/null || true
  pkill -9 -f "diffusion_policy/policy_wrapper" 2>/dev/null || true
  pkill -9 -f "grpc.*policy" 2>/dev/null || true

  # Clear CUDA/GPU memory by killing any python processes using GPU
  # This is aggressive but ensures GPU memory is released
  if command -v nvidia-smi &>/dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU state before cleanup:"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv 2>/dev/null || true

    # Get PIDs of processes using GPU and kill them (except this script's python)
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
    if [[ -n "${gpu_pids}" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Killing GPU processes: ${gpu_pids}"
      for pid in ${gpu_pids}; do
        kill -9 "${pid}" 2>/dev/null || true
      done
    fi

    # Wait a moment for GPU memory to be released
    sleep 2

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU state after cleanup:"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv 2>/dev/null || true
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Orphaned process cleanup completed"
}

# Run upload and cleanup (best effort - don't fail the job if cleanup fails)
upload_and_cleanup || echo "WARNING: Upload/cleanup encountered errors but continuing"

# Clean up orphaned processes to release GPU memory and allow node to become idle
cleanup_orphaned_processes || echo "WARNING: Process cleanup encountered errors but continuing"

exit "${final_status}"

