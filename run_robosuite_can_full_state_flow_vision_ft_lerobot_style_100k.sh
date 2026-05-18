#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate

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

if [ ! -f data/robosuite_can_full_horizon32/manifest.jsonl ]; then
    python scripts/materialize_lerobot_lift_wds.py \
        --task "pick the coke can and place it in the bin" \
        --output-dir data/robosuite_can_full_horizon32 \
        --past-lowdim-steps 1 \
        --future-lowdim-steps 30
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1

torchrun --nproc_per_node=1 --master_port=29519 \
    vla_foundry/main.py \
    --config_path train_robosuite_flow_can_full_state_vision_ft_lerobot_style_100k.yaml \
    2>&1 | tee -a outputs/robosuite_flow_can_full_state_vision_ft_lerobot_style_100k_train.log
