# Running Simulation Evaluation Locally

This tutorial walks through evaluating a trained policy in the LBM simulation environment released in the [original repo](https://github.com/ToyotaResearchInstitute/lbm_eval) on your local machine. The simulation runs inside a Docker container, while the policy server runs on the host. The two communicate over gRPC. We have provided a docker container that exposes additional options such as generating videos for your convenience.

## Prerequisites
- NVIDIA GPU
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed
- Docker installed
- `uv` installed (see [main README](../../README.md#installation))
- Project dependencies synced: `uv sync --group inference`
- Sufficient local disk space for rollout data

## Architecture

When getting started, it is easiest to run evaluation in **two terminals**: one for the policy server (host) and one for the simulation (Docker). They communicate over gRPC on `localhost:50051`:

```
Terminal 1 (Host)                Terminal 2 (Docker)
┌──────────────────────┐        ┌──────────────────────┐
│  inference_policy.py  │  gRPC  │  Drake simulation    │
│  (policy server)      │◄──────►│  (lbm_eval Docker)   │
│                       │        │                      │
│  Loads checkpoint,    │        │  Runs evaluation     │
│  generates actions    │        │  episodes, checks    │
│  from observations    │        │  success criteria    │
└──────────────────────┘        └──────────────────────┘
```

## Step 1: Download a Model Checkpoint

TODO: tri folks, you can try this checkpoint
s3://tri-ml-datasets-uw2/lbm2_vla/model_checkpoints/vla_multitask/vlm_loading_ablation/bsz1024_300m_samples_300k_steps_lr5e-5_floor0_llmvlmvla/2026_03_10-19_39_52-model_diffusion_policy-lr_5e-05-bsz_1024/

Use the provided download script to fetch a checkpoint from S3. This downloads only the files needed for inference (config, normalization stats, and model weights):

```bash
uv run python examples/deployment/lbm_eval/download_model_from_wandb.py \
    --s3-path <S3_PATH_TO_CHECKPOINT>
```

This creates an `experiments/` directory with the checkpoint. To specify a particular checkpoint number or output directory:

```bash
uv run python examples/deployment/lbm_eval/download_model_from_wandb.py \
    --s3-path <S3_PATH_TO_CHECKPOINT> \
    --checkpoint 5 \
    --output experiments/my_model
```

For the rest of this tutorial, we assume the checkpoint is at:
```bash
CHECKPOINT_DIR="experiments/<your_checkpoint>/"
```

## Step 2: Launch the Policy Server

In **Terminal 1**, start the gRPC policy server. This loads the model and waits for observations from the simulation.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group inference \
    python vla_foundry/inference/robotics/inference_policy.py \
    --checkpoint_directory $CHECKPOINT_DIR \
    --num_flow_steps 8 \
    --open_loop_steps 8 \
    --device cuda
```

Wait for `LBMDiffusionPolicy initialized with model on cuda` before proceeding. The server listens on `localhost:50051`.

**Key parameters:**
| Parameter | Description | Paper default |
|---|---|---|
| `--num_flow_steps` | Diffusion denoising steps | 8 |
| `--open_loop_steps` | Actions executed before re-planning | 8 |
| `--device` | `cuda` or `cpu` | `cuda` |

## Step 3: Pull the Evaluation Docker Image

```bash
docker pull 124224456861.dkr.ecr.us-west-2.amazonaws.com/lbm-eval-oss:latest
```


## Step 4: Run Evaluation Episodes

In **Terminal 2**, launch the simulation container. The following will run 5 rollouts for the `PutMugOnSaucer` task:

```bash
mkdir -p rollouts && chmod 777 rollouts
docker run --rm -it --network host \
    --runtime=nvidia \
    --gpus all \
    --device /dev/dri \
    --group-add video \
    --group-add $(stat -c '%g' /dev/dri/renderD128) \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e LAUNCH_DEMONSTRATION_INDICES=100:105 \
    -e RECORD_VIDEO=1 \
    -v $(pwd)/rollouts:/tmp/lbm/rollouts \
    682769330988.dkr.ecr.us-east-1.amazonaws.com/lbm-eval-oss:latest \
    bash launch_sim.sh PutMugOnSaucer
```

Replace `PutMugOnSaucer` with any task from the [list of available tasks](https://github.com/ToyotaResearchInstitute/lbm_eval/tree/main/anzu/intuitive/skill_types) in the original repo.

The simulation connects to the policy server in Terminal 1, runs each episode, and writes results to the mounted `rollouts/` directory. You should see gRPC activity in the policy server terminal once the first episode starts.

> **Interactive shell:** To explore the Docker container instead of running the evaluation, replace `bash launch_sim.sh PutMugOnSaucer` with just `bash`. You can then run `bash launch_sim.sh PutMugOnSaucer` manually from inside.

**Controlling the number of episodes:**
`LAUNCH_DEMONSTRATION_INDICES` controls which episode seeds to run. The format is `start:end` (exclusive). For example, `100:150` runs 50 episodes with seeds 100-149. For paper-comparable results, use `0:200` (200 episodes per task).

**Docker environment variables:**
| Variable | Description | Default |
|---|---|---|
| `LAUNCH_DEMONSTRATION_INDICES` | Episode seed range (`start:end`) | `100:200` |
| `NUM_PROCESSES` | Parallel evaluation processes inside the container | `1` |
| `POLICY_HOST` | gRPC policy server hostname | `localhost` |
| `POLICY_PORT` | gRPC policy server port | `50051` |
| `USE_EVAL_SEED` | Deterministic seeds (`1` = yes) | `1` |
| `RECORD_VIDEO` | Save MP4 videos (`1` = yes) | `0` |
| `VIDEO_CAMERA` | Comma-separated camera list for recording (empty = all cameras) | *(all)* |
| `VIDEO_FPS` | Video frame rate | `10` |

**Rollout directory structure:**
The evaluation writes a `results-*.json` summary and per-episode data:
```
rollouts/
├── results-2026-03-18T20:11:57.json    <- success/failure for all episodes
└── put_mug_on_saucer/                  <- per-episode recordings and metadata
    ├── demonstration_100/
    │   ├── recording.html              <- 3D replay (open in browser)
    │   └── ...
    └── demonstration_101/
        └── ...
```

## Step 5: View Results

### Gradio dashboard

For interactive exploration with charts, video playback, and 3D replays, install the viewer dependency group (one-time) and launch:

```bash
uv sync --group eval-viewer
uv run --group eval-viewer python vla_foundry/eval/results_explorer.py rollouts/
```

This starts a Gradio server at `http://localhost:8505` with summary table, bar/violin/spider charts, and a paginated episode recording viewer. Filter by task or model with the dropdowns, and click **Refresh** to pick up new results while evaluation is running.

> **Browser compatibility:** The dashboard has been tested on Chrome. Video playback in the Episode Recordings tab does not work properly on Safari.

To compare multiple models, run evaluations with different `--model_name` values (see [Evaluating All Tasks](#evaluating-all-tasks)) and point the dashboard at the same output directory. The dashboard auto-discovers all models.

## Evaluating All Tasks

> **Important:** Full evaluation takes many hours. We suggest running inside `tmux` or `screen` to prevent SSH disconnects from killing the policy servers.

[`run_evaluation.py`](./run_evaluation.py) automates Steps 2–4 above: it launches policy servers (one per GPU), distributes tasks across GPUs with no idle time, and prints progress. When a GPU finishes a task it immediately picks up the next one.

```bash
uv run python tutorials/sim_evaluation/run_evaluation.py $CHECKPOINT_DIR
```

Run specific tasks only:
```bash
uv run python tutorials/sim_evaluation/run_evaluation.py $CHECKPOINT_DIR \
    --tasks PutMugOnSaucer TurnCupUpsideDown
```

Parallelize across multiple GPUs:
```bash
uv run python tutorials/sim_evaluation/run_evaluation.py $CHECKPOINT_DIR --num_gpus 3
```

Compare two models:
```bash
uv run python tutorials/sim_evaluation/run_evaluation.py $CHECKPOINT_A --model_name model_a
uv run python tutorials/sim_evaluation/run_evaluation.py $CHECKPOINT_B --model_name model_b
```

See all options with `--help`. Key flags: `--num_gpus`, `--tasks_per_gpu`, `--num_processes`, `--model_name`, `--num_episodes` (seed range, default `0:200`).

**Speeding up evaluation:** The simulation (physics + rendering) is the bottleneck, not the policy server — GPU utilization is typically low. Two ways to increase throughput:

- `--num_processes N` — run N parallel episodes *within* each Docker container, sharing one policy server. Start with 5 and increase while monitoring `nvidia-smi`. This is the simplest way to speed things up.
- `--tasks_per_gpu N` — run N tasks concurrently on each GPU, each with its own policy server and Docker container. Each policy server uses ~7-10 GB of GPU memory. With 49 GB GPUs you can fit ~5 concurrent tasks per GPU.

These can be combined. For example, with 3 GPUs:
```bash
uv run python tutorials/sim_evaluation/run_evaluation.py $CHECKPOINT_DIR \
    --num_gpus 3 --tasks_per_gpu 3 --num_processes 5
```
This runs 9 tasks concurrently (3 per GPU), each with 5 parallel episodes = 45 episodes in flight.

## Troubleshooting

- **Docker container logs:** Simulation output is saved to `${OUTPUT_DIR}/${TASK}/${MODEL_NAME}/.docker.log`. Check these if a task fails or hangs.
- **Connection refused / simulation hangs:** Ensure the policy server is fully initialized (`LBMDiffusionPolicy initialized` in logs) before starting Docker. Check `${OUTPUT_DIR}/.policy_server_gpu0.log` for errors.
- **GPU rendering errors (EGL):** Verify NVIDIA Container Toolkit is installed and `/dev/dri` is accessible.
- **Permission denied on rollouts:** Run `mkdir -p rollouts && chmod 777 rollouts` before starting.
- **Slow first episode:** Normal — Drake downloads model packages and compiles the scene on the first run.
- **OOM:** Policy server and simulation share the GPU. Try reducing `--num_flow_steps` (e.g., 8 → 4).
- **Script errors:** If `run_evaluation.py` fails, check the policy server and Docker logs listed in the output.
- **Stale containers:** `docker kill $(docker ps -q)` to clean up.
- **Reproducing paper numbers:** Use 200 episodes (`NUM_EPISODES=0:200`). Local results may differ from paper numbers by a few percentage points due to GPU hardware differences and non-deterministic CUDA operations.
- **Diagnosing crashes:** Check `rollouts/**/results-*.json` — episodes with `total_time: 0` and a gRPC traceback in `failure_message` are infrastructure crashes, not eval failures.
