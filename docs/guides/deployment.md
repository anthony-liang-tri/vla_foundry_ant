# Deployment

This guide covers deploying and evaluating LBM robotics policies using the evaluation utilities provided in VLA Foundry.

## Overview

The `examples/deployment/lbm_eval/` directory provides lightweight scripts for evaluating LBM robotics policies against a gRPC client. Each script is ready to run with `uv` so that you can reuse the repository's managed environment.

### Contents

| Script | Description |
|---|---|
| `launch_wave_policy.sh` | Launches a dummy gRPC policy server that waves the robot end-effectors in a simple sinusoidal pattern. Helpful for verifying the evaluation pipeline. |
| `launch_inference_policy.sh` | Launches a gRPC policy server that uses the DiffusionPolicy model to generate actions. The experiment path (e.g., model checkpoint) should be modified in the script or passed as an argument. |

## Prerequisites

Before running the deployment scripts, ensure the following:

- Complete the project setup in the repository root (see main README for `uv sync --frozen` instructions).
- Provide any required credentials (e.g., AWS, W&B, Hugging Face tokens) for accessing checkpoints or datasets referenced in your configuration.
- Ensure your shell is in the repository root before running the scripts.

## Docker Setup for Anzu

!!! warning "Ubuntu 22.04 requirement"
    Anzu's `lbm_eval_0_5` branch only supports Ubuntu 22.04. You must run it in Docker when using Ubuntu 24.04. The inference policy itself can run outside of Docker.

### 1. Log in to Docker with ECR credentials

You need the `manip-cluster` profile to access the ECR repository:

```bash
aws ecr get-login-password --region us-east-1 --profile manip-cluster | docker login \
  --username AWS \
  --password-stdin 682769330988.dkr.ecr.us-east-1.amazonaws.com
```

### 2. Pull the Anzu image from ECR

```bash
docker pull 682769330988.dkr.ecr.us-east-1.amazonaws.com/anzu-vla-foundry:latest
```

### 3. Run the Anzu image

```bash
docker run --rm -it --network host \
    --runtime=nvidia \
    --gpus all \
    --device /dev/dri \
    --group-add video \
    --group-add $(stat -c '%g' /dev/dri/renderD128) \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e SKIP_BUILD=1 \
    -v $SSH_AUTH_SOCK:/ssh-agent \
    -e SSH_AUTH_SOCK=/ssh-agent \
    -v ${HOME}/.aws:/home/anzu/.aws \
    682769330988.dkr.ecr.us-east-1.amazonaws.com/anzu-vla-foundry:latest \
    bash /opt/anzu/launch_sim.sh BimanualPutRedBellPepperInBin
```

## Wave Policy Demo

The wave policy is a deterministic scripted policy that validates the gRPC stack.

### Launch the wave policy server

```bash
bash examples/deployment/lbm_eval/launch_wave_policy.sh
```

### Run the client from Anzu (in simulation)

From the Anzu repository:

```bash
bazel run //intuitive/visuomotor:demonstrate -- \
  --config_file `pwd`/lbm_eval/scenarios/3_cabot_breakfast/put_cup_in_center_of_table.yaml \
  --scenario GrpcServerToSim \
  --demonstration_indices 1000:1005 \
  --t_max 10.0 \
  --save_dir=/tmp/lbm/rollouts/ \
  --summary_dir=/tmp/lbm/rollouts/
```

Key behavior:

- Starts a gRPC server defined in `packages/grpc-workspace/src/grpc_workspace/wave_around_policy_server.py`.
- Streams sinusoidal joint poses to connected clients until interrupted.

## Inference Policy Demo

Follow these steps to download a trained policy checkpoint and run the evaluation demo.

### Step 1: Download the policy checkpoint

From the repository root, download by W&B run name:

```bash
python examples/deployment/lbm_eval/download_model_from_wandb.py \
  --run-name "<RUN_NAME>"
```

For example: `2025_11_05-21_34_11-model_diffusion_policy-lr_5e-05-bsz_1024`

Alternatively, use a W&B URL directly:

```bash
python examples/deployment/lbm_eval/download_model_from_wandb.py \
  --run-url "https://wandb.ai/entity/project/runs/abc123"
```

To search for available runs:

```bash
python examples/deployment/lbm_eval/download_model_from_wandb.py --search "diffusion"
```

To download a specific checkpoint number (default is latest):

```bash
python examples/deployment/lbm_eval/download_model_from_wandb.py \
  --run-name "<RUN_NAME>" \
  --checkpoint 5
```

### Step 2: Launch the inference policy service

From the repository root:

```bash
bash examples/deployment/lbm_eval/launch_inference_policy.sh
```

### Step 3: Run the client demo

From the Anzu repository root (on the `lbm_eval_0_5` branch):

```bash
source .venv/bin/activate
touch .venv/COLCON_IGNORE
export PYTHONPATH=`pwd`/venv/lib/python3.12/site-packages:$PYTHONPATH
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export DISPLAY=1

CUDA_VISIBLE_DEVICES=1 xvfb-run -a bazel run //intuitive/visuomotor:demonstrate -- \
  --config_file `pwd`/intuitive/visuomotor/config/bimanual_put_red_bell_pepper_in_bin_riverway.yaml \
  --scenario GrpcServerToSim \
  --demonstration_indices 0:50 \
  --t_max 45.0 \
  --save_dir=/tmp/lbm/rollouts/ \
  --summary_dir=/tmp/lbm/rollouts/
```

The demo connects to the policy server started in Step 2 and saves rollouts under `/tmp/lbm/rollouts/`.
