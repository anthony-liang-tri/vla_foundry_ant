#!/usr/bin/env bash
#
# Download OSS eval results from S3 into a layout the dashboard can read.
#
# Downloads results for one or more checkpoints and organizes them as:
#   rollouts/{TaskName}/{checkpoint_name}/results-*.json
#
# This lets the dashboard show each checkpoint as a separate "model" for
# side-by-side comparison.
#
# Usage:
#   # Single checkpoint
#   ./download_results.sh s3://bucket/.../checkpoint_A/evaluation/oss_eval
#
#   # Multiple checkpoints (for comparison)
#   ./download_results.sh \
#     s3://bucket/.../checkpoint_A/evaluation/oss_eval \
#     s3://bucket/.../checkpoint_B/evaluation/oss_eval
#
#   # Custom output directory
#   ./download_results.sh --output-dir my_rollouts s3://...
#
#   # Then view with the dashboard
#   uv run --group eval-viewer python vla_foundry/eval/results_explorer.py rollouts/
#
set -euo pipefail

OUTPUT_DIR="rollouts"
AWS_PROFILE="${AWS_PROFILE:-sagemaker}"
S3_PATHS=()

usage() {
cat <<'EOF'
Usage: download_results.sh [options] <s3_eval_path> [<s3_eval_path> ...]

Each s3_eval_path should point to the evaluation subfolder, e.g.:
  s3://bucket/.../checkpoint/evaluation/oss_eval

Options:
  --output-dir DIR    Local output directory (default: rollouts)
  --help              Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --help|-h)    usage; exit 0 ;;
    s3://*)       S3_PATHS+=("$1"); shift ;;
    *)            echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ${#S3_PATHS[@]} -eq 0 ]]; then
  echo "ERROR: No S3 paths provided." >&2
  usage
  exit 1
fi

for s3_eval_path in "${S3_PATHS[@]}"; do
  s3_eval_path="${s3_eval_path%/}"

  # Extract checkpoint name from the S3 path
  # e.g., s3://bucket/.../2026_01_07-model_name/evaluation/oss_eval
  #   → checkpoint_name = "2026_01_07-model_name"
  # Strip /evaluation/... suffix to get the checkpoint base
  checkpoint_base="${s3_eval_path%/evaluation/*}"
  checkpoint_name="${checkpoint_base##*/}"

  echo "============================================"
  echo "Downloading: ${checkpoint_name}"
  echo "  From: ${s3_eval_path}"
  echo "============================================"

  # List tasks under this eval path
  tasks=$(AWS_PROFILE="${AWS_PROFILE}" aws s3 ls "${s3_eval_path}/" 2>/dev/null \
    | awk '/PRE/ {gsub(/\/$/, "", $2); print $2}')

  if [[ -z "${tasks}" ]]; then
    echo "  WARNING: No tasks found at ${s3_eval_path}/"
    continue
  fi

  for task in ${tasks}; do
    local_dir="${OUTPUT_DIR}/${task}/${checkpoint_name}"
    echo "  ${task} → ${local_dir}"
    mkdir -p "${local_dir}"

    AWS_PROFILE="${AWS_PROFILE}" aws s3 sync \
      "${s3_eval_path}/${task}/rollouts/" \
      "${local_dir}/" \
      --exclude "*.pkl" \
      --quiet
  done

  echo ""
done

echo "============================================"
echo "Results downloaded to: ${OUTPUT_DIR}/"
echo ""
echo "View with:"
echo "  uv run --group eval-viewer python vla_foundry/eval/results_explorer.py ${OUTPUT_DIR}/"
echo "============================================"
