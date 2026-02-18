## LBM Evaluation Utilities

This directory provides lightweight scripts for evaluating LBM robotics policies
against a gRPC client. Each script is ready to run with `uv` so that you can
reuse the repository's managed environment.

### (Optional) Set up Anzu using Docker
Since Anzu’s `lbm_eval_0_5` branch only supports Ubuntu 22.04, you need to run it in Docker when using Ubuntu 24.04. Note that you can run the inference policy outside of Docker.

1. Log in to Docker with your ECR credentials (you need the manip-cluster profile to access the ECR repository)

```bash
aws ecr get-login-password --region us-east-1 --profile manip-cluster | docker login \
  --username AWS \
  --password-stdin 682769330988.dkr.ecr.us-east-1.amazonaws.com
```

2. Pull the Anzu image from ECR

```bash
docker pull 682769330988.dkr.ecr.us-east-1.amazonaws.com/anzu-vla-foundry:latest
```

3. Run the Anzu image on Docker

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

### Contents
- `launch_wave_policy.sh` – launches a dummy gRPC policy server that waves the
  robot end-effectors in a simple sinusoidal pattern. Helpful for verifying the
  evaluation pipeline.
- `launch_inference_policy.sh` – launches a gRPC policy server that uses the DiffusionPolicy model to generate actions. The experiment path (e.g., model checkpoint) should be modified in the script or passed as an argument as needed.

### Prerequisites
- Complete the project setup in the repository root (see main README for
  `uv sync --frozen` instructions).
- Provide any required credentials (e.g., AWS, WANDB, Hugging Face tokens) for
  accessing checkpoints or datasets referenced in your custom configuration.
- Ensure your shell is in the repository root before running the scripts.

### Running the Wave Policy Demo
The wave policy is a deterministic scripted policy that validates the gRPC
stack.

```bash
bash examples/deployment/lbm_eval/launch_wave_policy.sh
```

From the Anzu repo, in simulation, you can run the following command to launch the wave policy:
```bash
bazel run //intuitive/visuomotor:demonstrate -- \
--config_file `pwd`/lbm_eval/scenarios/3_cabot_breakfast/put_cup_in_center_of_table.yaml \ --scenario GrpcServerToSim \
--demonstration_indices 1000:1005 \
--t_max 10.0 \
--save_dir=/tmp/lbm/rollouts/ \
--summary_dir=/tmp/lbm/rollouts/
```

Key behavior:
- Starts a gRPC server defined in
  `packages/grpc-workspace/src/grpc_workspace/wave_around_policy_server.py`.
- Streams sinusoidal joint poses to connected clients until interrupted.

### Running the Inference Policy Demo

Follow these steps to download the trained policy and run the evaluation demo.

#### 1. Download the policy checkpoint (from repo root)

Download by W&B run name:
```bash
python examples/deployment/lbm_eval/download_model_from_wandb.py --run-name "<RUN_NAME>"
```
Example: `2025_11_05-21_34_11-model_diffusion_policy-lr_5e-05-bsz_1024`

Or use a W&B URL directly:
```bash
python examples/deployment/lbm_eval/download_model_from_wandb.py --run-url "https://wandb.ai/entity/project/runs/abc123"
```

To search for available runs:
```bash
python examples/deployment/lbm_eval/download_model_from_wandb.py --search "diffusion"
```

To download a specific checkpoint number (default is latest):
```bash
python examples/deployment/lbm_eval/download_model_from_wandb.py --run-name "<RUN_NAME>" --checkpoint 5
```

#### 2. Launch the inference policy service (from repo root)
```bash
bash examples/deployment/lbm_eval/launch_inference_policy.sh
```

#### 3. Run the client demo on anzu (`lbm_eval_0_5` branch) (from anzu root)
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

The demo connects to the policy server started in step 2 and saves rollouts under `/tmp/lbm/rollouts/`.
