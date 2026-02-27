#!/bin/bash
# Standalone evaluation campaign launcher
# Usage: ./launch_campaign.sh [config.yaml] [additional args...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to example_campaign.yaml if no argument is provided or if the first argument is a flag
if [[ -z "$1" ]] || [[ "$1" == --* ]]; then
    CONFIG="${SCRIPT_DIR}/campaigns/example_campaign.yaml"
else
    CONFIG="$1"
    shift
fi

# Use AWS profile if specified (default to manip-cluster for backward compat)
if [[ -z "$AWS_PROFILE" ]]; then
    export AWS_PROFILE="manip-cluster"
fi

# Set EVAL_CAMPAIGN_ROOT to the actual repository root
# SCRIPT_DIR is eval_campaigns, so repo root is 4 levels up:
# eval_campaigns -> lbm_eval -> tri -> vla_foundry -> repo_root
export EVAL_CAMPAIGN_ROOT="${SCRIPT_DIR}/../../../.."

# Run the campaign orchestrator
cd "${EVAL_CAMPAIGN_ROOT}"
uv run python -m vla_foundry.tri.lbm_eval.eval_campaigns.run_evaluation_campaign "$CONFIG" "$@"
