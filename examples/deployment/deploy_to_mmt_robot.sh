#!/bin/bash
set -euo pipefail

# Deploy vla_foundry_internal to a robot via rsync
# Usage: ./deploy_to_mmt_robot.sh [--dry-run] <robot_hostname>
# Example: ./deploy_to_mmt_robot.sh tzkmc0003.local
# Example: ./deploy_to_mmt_robot.sh --dry-run tzkmc0003.local

ROBOT_USER="user"
REMOTE_DIR="/home/${ROBOT_USER}/code/vla_foundry_internal"
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            ROBOT_HOST="$1"
            shift
            ;;
    esac
done

if [ -z "${ROBOT_HOST:-}" ]; then
    echo "Usage: $0 [--dry-run] <robot_hostname>"
    echo "Example: $0 tzkmc0003.local"
    exit 1
fi

# Resolve the vla_foundry_internal root directory relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Deploying: ${LOCAL_DIR}"
echo "Target:    ${ROBOT_USER}@${ROBOT_HOST}:${REMOTE_DIR}"

# Ensure remote directory exists for initial deployment
if [ -z "${DRY_RUN}" ]; then
    ssh "${ROBOT_USER}@${ROBOT_HOST}" "mkdir -p ${REMOTE_DIR}"
fi

# Sync the code
# --filter=':- .gitignore' excludes files matched by .gitignore
# --exclude=.git and --exclude=logs/ are not in .gitignore so added explicitly
rsync -avz --progress --delete --delete-excluded ${DRY_RUN} \
    --filter=':- .gitignore' \
    --exclude=.git \
    --exclude=logs/ \
    "${LOCAL_DIR}/" "${ROBOT_USER}@${ROBOT_HOST}:${REMOTE_DIR}/"

echo "Deploy complete."
