#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
mkdir -p outputs

# Robosuite evals and WebDataset workers can exceed the shell default of 1024.
ulimit -n 1048576 || ulimit -n 65536 || true

if ! python - <<'PY'
import importlib.util
import sys

missing = [pkg for pkg in ("robosuite", "robomimic", "mujoco", "imageio") if importlib.util.find_spec(pkg) is None]
sys.exit(1 if missing else 0)
PY
then
    uv pip install --python "$(which python)" \
        'robosuite==1.4.0' \
        'robomimic==0.2.0' \
        'mujoco>=2.3.0,<4.0.0' \
        'imageio[ffmpeg]>=2.37.0'
fi

LEROBOT_SOURCE_ROOT="${LEROBOT_SOURCE_ROOT:-$HOME/.cache/huggingface/lerobot/TRI-ML/PretrainFinetune}"
if [ ! -f "$LEROBOT_SOURCE_ROOT/data/chunk-000/file-000.parquet" ]; then
    HF_HUB_SNAPSHOTS="$HOME/.cache/huggingface/lerobot/hub/datasets--TRI-ML--PretrainFinetune/snapshots"
    if [ -d "$HF_HUB_SNAPSHOTS" ]; then
        LEROBOT_SOURCE_ROOT="$(find "$HF_HUB_SNAPSHOTS" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
    fi
fi

if [ ! -f data/robosuite_square_full_horizon32/manifest.jsonl ]; then
    python scripts/materialize_lerobot_lift_wds.py \
        --source-root "$LEROBOT_SOURCE_ROOT" \
        --task "pick up the square nut and place it on the square peg" \
        --output-dir data/robosuite_square_full_horizon32 \
        --past-lowdim-steps 1 \
        --future-lowdim-steps 30
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1

torchrun --nproc_per_node=1 --master_port=29520 \
    vla_foundry/main.py \
    --config_path train_robosuite_flow_square_full_state_vision_ft_lerobot_style_100k.yaml \
    2>&1 | tee -a outputs/robosuite_flow_square_full_state_vision_ft_lerobot_style_100k_train.log
