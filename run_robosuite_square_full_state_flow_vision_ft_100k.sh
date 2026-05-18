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

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1

torchrun --nproc_per_node=1 --master_port=29517 \
    vla_foundry/main.py \
    --config_path train_robosuite_flow_square_full_state_vision_ft_100k.yaml \
    2>&1 | tee -a outputs/robosuite_flow_square_full_state_vision_ft_100k_train.log
