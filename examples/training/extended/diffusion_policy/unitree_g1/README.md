# Unitree G1 Diffusion Policy Training

`train_g1_policy.py` is the single entry point for training a diffusion policy on Unitree G1
data, both locally (via torchrun) and on SageMaker. It builds all draccus overrides from
structured arguments, prints a job summary, then either launches torchrun directly or delegates
to `sagemaker/launch_training.py`. Pass `--dry-run` to inspect the exact command without
executing it.

## Prerequisites

Data must be preprocessed to WebDataset tar shards on one of the dedicated EC2 preprocessing
instances. The shards directory must contain `manifest.jsonl` and `stats.json`.

> Note: a PR for remotely launching preprocessing jobs without manual SSH is in progress.

### EC2 preprocessing instances

| Host | IP |
|------|----|
| `humanoid_data_1` | `10.161.51.208` |
| `humanoid_data_2` | `10.161.51.87` |

SSH config (`~/.ssh/config`):
```
Host humanoid_data_1
  Hostname 10.161.51.208
  ForwardAgent yes
  IdentityFile ~/.ssh/manip-perception.pem
  User ubuntu

Host humanoid_data_2
  Hostname 10.161.51.87
  ForwardAgent yes
  IdentityFile ~/.ssh/manip-perception.pem
  User ubuntu
```

### Running preprocessing

Example command for sim ZED mini data (4 cameras, 64 workers):

```bash
ssh humanoid_data_1   # or humanoid_data_2
cd ~/vla_foundry
git pull origin main  # ensure latest configs

uv run --group preprocessing vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
    --type mcap \
    --source_episodes "['s3://robotics-cam-data/platform/unitree_g1_dex3/mcap']" \
    --output_dir "s3://robotics-cam-data/platform/unitree_g1_dex3/tarfiles/v3.2/" \
    --output_dir_fixed_path "s3://robotics-cam-data/platform/unitree_g1_dex3/dataset_fixed/" \
    --config_path "vla_foundry/config_presets/data/unitree_g1/g1_preprocessing_params_1past_47future_30hz.yaml" \
    --topics_to_fields_path "vla_foundry/config_presets/data/unitree_g1/g1_mcap_topics.yaml" \
    --camera_names "include vla_foundry/config_presets/data/unitree_g1/camera_names/zedm_head_d405_wrists.yaml" \
    --task_filter '["move_block_on_plate"]' \
    --domain_filter '["sim"]' \
    --source_filter '["teleop"]' \
    --samples_per_shard 100 \
    --ray_num_cpus 64
```

Key flags:
- `--task_filter` / `--domain_filter` / `--source_filter` — narrow which episodes to convert
- `--ray_num_cpus 64` — verified stable on these instances with up to 4 cameras (ZED stereo + wrist ×2)
- `--output_dir` — bump the version (`v1` → `v2` etc.) when re-preprocessing to avoid overwriting

Output lands at:
```
s3://robotics-cam-data/platform/unitree_g1_dex3/tarfiles/v3.2/{task}/{domain}/{source}/
```

Pass that path as `--data-root` to `train_g1_policy.py`.

### Whole-body preprocessing (sonic teleop)

Whole-body MCAPs have additional topics (`/smpl_joint_points`,
`/reference_motion`) and produce a 98D action layout. Use the
`unitree_g1_wholebody/sonic/` configs and write to a separate
`tarfiles/v0_wholebody/` track:

```bash
ssh humanoid_data_1   # or humanoid_data_2
cd ~/vla_foundry
git pull origin main

uv run --group preprocessing vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
    --type mcap \
    --source_episodes "['s3://robotics-cam-data/platform/unitree_g1_dex3/mcap']" \
    --output_dir "s3://robotics-cam-data/platform/unitree_g1_dex3/tarfiles/v0_wholebody/" \
    --output_dir_fixed_path "s3://robotics-cam-data/platform/unitree_g1_dex3/dataset_fixed/" \
    --config_path "vla_foundry/config_presets/data/unitree_g1_wholebody/sonic/sonic_preprocessing_params.yaml" \
    --topics_to_fields_path "vla_foundry/config_presets/data/unitree_g1_wholebody/sonic/sonic_topics_to_fields.yaml" \
    --action_fields_config_path "vla_foundry/config_presets/data/unitree_g1_wholebody/sonic/sonic_action_fields.yaml" \
    --language_annotations_path "vla_foundry/config_presets/data/unitree_g1/g1_language_annotations.yaml" \
    --camera_names "include vla_foundry/config_presets/data/unitree_g1/camera_names/zedm_head.yaml" \
    --task_filter '["put_cap_in_laundry_basket"]' \
    --domain_filter '["real"]' \
    --source_filter '["teleop"]' \
    --samples_per_shard 100 \
    --ray_num_cpus 64
```

### Camera-name presets

Camera-name lists live under `vla_foundry/config_presets/data/unitree_g1/camera_names/`:

| Preset | Cameras |
|--------|---------|
| `d435_head.yaml` | `head` |
| `d435_head_d405_wrists.yaml` | `head, left_wrist, right_wrist` |
| `zedm_head.yaml` | `stereo_head_left, stereo_head_right` |
| `zedm_head_d405_wrists.yaml` | `stereo_head_left, stereo_head_right, left_wrist, right_wrist` |

Reference one of these from `--camera_names "include …"` (preprocessing) or
`--camera-config <preset>` (training launcher).

## Quick start

```bash
# Local (2 GPUs)
uv run python examples/training/extended/diffusion_policy/unitree_g1/train_g1_policy.py \
    move_block_on_plate --obs vision_propio --domain sim

# SageMaker (tri-cam-humanoid queue, p5)
uv run python examples/training/extended/diffusion_policy/unitree_g1/train_g1_policy.py \
    move_block_on_plate --obs vision_propio --domain sim \
    --sagemaker --user firstname.lastname

# Whole-body (sonic teleop demos), ZED Mini head-only
uv run python examples/training/extended/diffusion_policy/unitree_g1/train_g1_policy.py \
    put_cap_in_laundry_basket --obs wholebody --domain real --camera-config zedm_head
```

## Usage

```
uv run python train_g1_policy.py TASK --obs MODE [OPTIONS]
```

### Observation modes (`--obs`)

| Mode | Fields |
|------|--------|
| `vision_propio` | Vision + EE pose (18D) + finger joint positions (14D) |
| `vision_propio_tactile` | Vision + EE pose + finger joint positions + joint torque (14D) + pressure |
| `wholebody` | Vision + 43 G1 joint positions |

Default action space (vision_propio, vision_only, vision_propio_tactile):
absolute EE pose (18D) + Dex3 finger joint positions (14D) = 32D.

`wholebody` overrides this with a 98D action: 72 SMPL joint positions
(24 joints × xyz) + 6D root orientation + 6D wrist references + 14D Dex3 hand
joints. Default data root is
`s3://robotics-cam-data/platform/unitree_g1_dex3/tarfiles/v0_wholebody`.

### All options

| Option | Default | Description |
|--------|---------|-------------|
| `--obs` | required | Observation mode (see above) |
| `--model` | `410m` | Transformer size: `100m` or `410m` |
| `--domain` | `real` | Data domain: `real` or `sim` |
| `--cameras` | `auto` | Camera subset: `auto` (all), `head`, `wrist`, or JSON list |
| `--gpu` | `0,1` | CUDA device IDs (local only) |
| `--samples` | `10000000` | Total training samples |
| `--steps` | — | Training steps instead of samples; requires `--global-batch` |
| `--batch` | config default | Per-GPU batch size override |
| `--global-batch` | config default | Global batch size override |
| `--data-root` | S3 path from task name | Override dataset shards directory |
| `--val-samples` | off | Enable validation visualizations |
| `--sagemaker` | false | Submit to SageMaker instead of running locally |
| `--user` | required w/ `--sagemaker` | `firstname.lastname` for job naming and ECR image |
| `--queue` | `tri-cam-humanoid` | Queue family (`fss-ml` or `tri-cam-humanoid`) |
| `--instance` | `p5` | Instance shorthand (`p5`, `p5en`, `p6`) |
| `--dry-run` | false | Print the exact launch command without executing |

## Data paths

Training data is read from:
```
s3://robotics-cam-data/platform/unitree_g1_dex3/tarfiles/v3.2/{TASK}/{DOMAIN}/teleop/shards/
```

Tactile configs (`vision_propio_tactile`) default to `v2/` — requires data preprocessed with
dex3 torque/pressure topics.

Checkpoints are written to:
```
s3://robotics-cam-checkpoints/unitree_g1/{TASK}_{OBS}_{SAMPLES}
```

Override with `--data-root`.

## Batch size

`--batch` sets per-GPU batch size; `--global-batch` sets the total across all GPUs.
Keep `global_batch_size` consistent regardless of GPU count to preserve effective learning rate.

> Starting-point guidelines for `vision_propio` (4 cameras, float32). Adding tactile
> modalities increases memory usage — reduce `--batch` if you hit OOM.

| Hardware | `--batch` | `--global-batch` | Notes |
|----------|-----------|-----------------|-------|
| 2x RTX 4090 (24 GB, float32) | `8` | `16` | Reduce to `4` with tactile obs |
| p5 / 8x H100 (80 GB, float32) | `32` | `256` | |

Use `--steps` with `--global-batch` to target a fixed number of optimizer steps:
```bash
uv run python train_g1_policy.py move_block_on_plate --obs vision_propio --domain sim \
    --sagemaker --user firstname.lastname \
    --steps 100000 --global-batch 256 --batch 32
```

## SageMaker

### Authentication

```bash
aws sso login --profile sagemaker
```

On a new machine, run the smoke test before your first launch to verify all prerequisites
(Docker, ECR login, Python deps, secrets.env):

```bash
uv run --group sagemaker python sagemaker/smoke_test.py --user firstname.lastname
```

The script verifies credentials against the target ARN before building the Docker image.
On submission it:
1. Builds and pushes `<account>.dkr.ecr.<region>.amazonaws.com/<user>-vla_foundry:latest`
2. Serialises hyperparameters to `sagemaker/configs/hyperparameters_<uuid>.yaml`
3. Captures a `git diff` of local changes in the image for reproducibility
4. Submits to the SageMaker Batch queue

### Available queues

| `--queue` | `--instance` | Queue name |
|-----------|--------------|------------|
| `tri-cam-humanoid` (default) | `p5` | `fss-tri-cam-humanoid-p5-48xlarge-us-west-2` |
| `ml` | `p5` | `fss-ml-p5-48xlarge-us-west-2` |
| `ml` | `p5en` | `fss-ml-p5en-48xlarge-us-west-2` |

### Monitoring with sagey / batchy

sagey and batchy are TRI CLI tools for managing SageMaker and AWS Batch jobs.
Install once (requires SSH access to `github.com/TRI-ML/sagey`):

```bash
pip install "sagetui @ git+ssh://git@github.com/TRI-ML/sagey.git#subdirectory=sagetui"
```

This installs four CLI tools: `sagey`, `batchy`, `sagetui`, `batchtui`.

Monitor a running SageMaker job (the job name is printed at submission):

```bash
sagey ls                   # list active jobs
sagey watch <job-name>     # stream training logs live
sagetui                    # interactive TUI (browse, stop, view logs)
```

One-liner to tail logs for a known job:

```bash
sagey watch <job-name>
```

For AWS Batch jobs (e.g. preprocessing):

```bash
batchy ls                  # list batch queues and jobs
batchtui                   # interactive TUI
```

## Training configs

`train_g1_policy.py` selects a YAML config based on `--obs` and `--model`. The configs live in:

```
vla_foundry/config_presets/training_jobs/unitree_g1/
```

Dataset versions:
- **v1** — vision + proprioception only (ZED mini stereo head + D435 wrist cameras)
- **v2** — v1 + dex3 tactile sensors (joint torque + pressure); required for `vision_propio_tactile`

The configs inherit shared data params from `vla_foundry/config_presets/data/unitree_g1/g1_data_params.yaml`,
so edits to obs/action fields take effect without touching the job configs.
