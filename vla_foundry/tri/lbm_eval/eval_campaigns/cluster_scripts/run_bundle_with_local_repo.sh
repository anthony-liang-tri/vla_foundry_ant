#!/usr/bin/env bash
#
# Clones (or updates) a vla_foundry checkout on the host and runs the
# run_inference_bundle.sh inside the Docker image with that checkout bind-mounted
# into /opt/vla_foundry. This avoids in-container git auth and lets you iterate on
# a local clone while reusing the containerized runtime environment.
#
# Example:
#   CHECKPOINT_DIR=s3://... \
#   ./docker/vla_foundry/run_bundle_with_local_repo.sh \
#       --ref main \
#       -- docker/vla_foundry/run_inference_bundle.sh specific flags...
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_IMAGE="anzu-vla-foundry:stage3"
DEFAULT_REPO_URL="git@github.com:jmercat/vla_foundry.git"
DEFAULT_REPO_DIR="${HOME}/.cache/vla_foundry/local_clone"
DEFAULT_REF="main"
DEFAULT_REMOTE="origin"
BUNDLE_SCRIPT="/usr/local/bin/run_inference_bundle.sh"

IMAGE="${DEFAULT_IMAGE}"
REPO_URL="${DEFAULT_REPO_URL}"
REPO_DIR="${DEFAULT_REPO_DIR}"
REF="${DEFAULT_REF}"
REMOTE="${DEFAULT_REMOTE}"
DOCKER_ARGS=(--gpus all)
BUNDLE_ARGS=()

usage() {
cat <<'EOF'
Usage: run_bundle_with_local_repo.sh [options] -- [run_inference_bundle args...]

Options:
  --image IMAGE          Docker image tag to run (default: anzu-vla-foundry:stage3)
  --repo-url URL         Git URL to clone (default: git@github.com:TRI-ML/vla_foundry.git)
  --repo-dir PATH        Directory to store the local clone (default: ~/.cache/vla_foundry/local_clone)
  --ref REF              Branch/tag/commit to check out before running (default: main)
  --remote NAME          Remote name to fetch/reset from (default: origin)
  --docker-arg ARG       Additional docker run argument (repeatable)
  --help                 Show this message

Arguments after -- are passed directly to run_inference_bundle.sh inside the container.
Set bundle env vars (e.g. CHECKPOINT_DIR) in your shell before invoking this helper.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"; shift 2 ;;
    --repo-url)
      REPO_URL="$2"; shift 2 ;;
    --repo-dir)
      REPO_DIR="$2"; shift 2 ;;
    --ref)
      REF="$2"; shift 2 ;;
    --remote)
      REMOTE="$2"; shift 2 ;;
    --docker-arg)
      DOCKER_ARGS+=("$2"); shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    --)
      shift
      BUNDLE_ARGS+=("$@")
      break ;;
    *)
      BUNDLE_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -d "${REPO_DIR}" ]]; then
  mkdir -p "${REPO_DIR}"
fi
REPO_DIR="$(cd "${REPO_DIR}" && pwd)"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "Cloning ${REPO_URL} into ${REPO_DIR}"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

echo "Syncing ${REPO_DIR} to ${REF} (${REMOTE})"
(
  cd "${REPO_DIR}"
  # git fetch --tags "${REMOTE}"
  if git rev-parse --verify --quiet "${REF}" >/dev/null 2>&1; then
    git checkout "${REF}"
  elif git rev-parse --verify --quiet "${REMOTE}/${REF}" >/dev/null 2>&1; then
    git checkout -B "${REF}" "${REMOTE}/${REF}"
  else
    git checkout "${REF}"
  fi
  if git rev-parse --verify --quiet "${REMOTE}/${REF}" >/dev/null 2>&1; then
    git reset --hard "${REMOTE}/${REF}"
  fi
)

mkdir -p "${REPO_DIR}/.venv"

docker_cmd=(
  docker run --rm
  "${DOCKER_ARGS[@]}"
  -v "${REPO_DIR}:/opt/vla_foundry"
  -v "${REPO_DIR}/.venv:/opt/vla_foundry/.venv"
  -e VLA_FOUNDRY_HOME=/opt/vla_foundry
  -e INFERENCE_WORKDIR=/opt/vla_foundry
  -e VLA_FOUNDRY_AUTO_UPDATE=0
  "${IMAGE}"
  bash "${BUNDLE_SCRIPT}"
)

if [[ "${#BUNDLE_ARGS[@]}" -gt 0 ]]; then
  docker_cmd+=("${BUNDLE_ARGS[@]}")
fi

echo "Running: ${docker_cmd[*]}"
"${docker_cmd[@]}"

