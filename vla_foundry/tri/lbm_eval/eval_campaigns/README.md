# Cluster Evaluation Campaign Guide

This guide documents the end-to-end workflow for running LBM policy evaluations on a Ray cluster.

## Overview

The evaluation pipeline consists of:
1. **Exporting training runs from W&B** - Get a CSV of model checkpoints
2. **Converting to task tuples** - Transform CSV to the required format
3. **Creating a campaign config** - Configure cluster and evaluation parameters
4. **Launching the campaign** - Spin up cluster and run evaluations
5. **Collecting results** - Download and analyze success rates

---

## Prerequisites

- AWS credentials configured with `manip-cluster` profile (see below)
- Access to the Ray cluster AMI and ECR images
- W&B access for exporting training runs

### AWS SSO Login with manip-cluster Profile

1. **Configure AWS SSO** by adding the following to `~/.aws/config`:

```ini
[sso-session sso]
sso_region = us-east-1
sso_start_url = https://tri-sso.awsapps.com/start/#
sso_registration_scopes = sso:account:access
output = json
region = us-east-1

[profile manip-cluster]
sso_session = sso
sso_account_id = 682769330988
sso_role_name = Robotics-LBM-PowerUserAccess
```

2. **Log in** using AWS SSO:

```bash
aws sso login --profile manip-cluster
```

This will open a browser window for authentication. Once authenticated, your credentials will be cached for subsequent AWS operations.

3. **Verify** your credentials are working:

```bash
aws sts get-caller-identity --profile manip-cluster
```

---

## Step 1: Export Training Runs from W&B

### Option A: Automated Export from Ablation Configs

```bash
python eval_campaigns/export_wandb_ablation_and_make_task_list.py \
    vla_foundry/tri/ablations/ablation_configs/my_ablation/
```

| Flag | Description |
|------|-------------|
| `--states` | Run states to include (default: `finished`). Use `--states finished running` for early eval |
| `--max-runs-per-task` | Max runs per task (default: 1 = latest only). Use 0 for all |
| `--export-name` | Custom output basename (default: ablation directory name) |
| `--entity` | W&B entity (default: `$WANDB_ENTITY`) |
| `--project` | W&B project (default: from `resolved_config.yaml`) |

Outputs: `wandb_exports/<name>.csv` and `task_lists/<name>.txt`

### Option B: Manual Export from W&B UI

1. Go to your W&B project containing training runs
2. Select the runs you want to evaluate
3. Click **Export** → **CSV**
4. Ensure the export includes these columns:
   - `Name` - Run name (used to build S3 path)
   - `Tags` - First tag should be the task name
   - `remote_sync` - S3 base path for checkpoints

5. Save the CSV file (e.g., `wandb_export.csv`)

---

## Step 2: Convert CSV to Task Tuples

Use `csv_to_tuples.py` to convert the W&B export to the tuple format:

```bash
# Basic usage - outputs to stdout
python eval_campaigns/csv_to_tuples.py wandb_export.csv

# Save to file
python eval_campaigns/csv_to_tuples.py wandb_export.csv eval_campaigns/task_lists/my_tasks.txt

# Filter to specific tasks
python eval_campaigns/csv_to_tuples.py wandb_export.csv my_tasks.txt \
    --filter-tasks BimanualPutRedBellPepperInBin PlaceCupByCoaster

# Filter using a task list file
python eval_campaigns/csv_to_tuples.py wandb_export.csv my_tasks.txt \
    --filter tasks_list.txt

# Exclude runs already evaluated (from another CSV)
python eval_campaigns/csv_to_tuples.py wandb_export.csv my_tasks.txt \
    --exclude-csv already_evaluated.csv
```

### Output Format

The output file will contain tuples like:
```python
(
    "stage3_singletask_sim_BimanualPutRedBellPepperInBin",
    "BimanualPutRedBellPepperInBin",
    f"s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/.../",
),
```

---

## Step 3: Create Campaign Configuration

Create a YAML configuration file (copy from `campaigns/example_campaign.yaml`):

```yaml
# eval_campaigns/campaigns/my_campaign.yaml

# Campaign metadata
campaign_name: "my_evaluation_dec2025"
description: "Description of what you're evaluating"

# Cluster configuration
cluster:
  config_file: "eval_campaigns/ray_policy_runner_cluster.yaml"
  num_workers: 128          # Number of parallel workers
  aws_profile: "manip-cluster"
  teardown_on_completion: false
  teardown_on_failure: false

# Evaluation configuration
evaluation:
  # Option 1: Use a tasks file (recommended for multiple tasks)
  tasks_file: "eval_campaigns/task_lists/my_tasks.txt"

  # Option 2: Single task with multiple checkpoints
  # task: "BimanualPutRedBellPepperInBin"
  # checkpoints:
  #   - "s3://..."
  #   - "s3://..."

  num_samples: 100          # Rollouts per checkpoint
  start_index: 100          # Starting demo index
  jobs_per_gpu: 1           # Jobs sharing each GPU

  # Launch settings
  launch_scenario: "GrpcServerToSim"
  launch_script: "launch_sim.sh"

  # Inference parameters
  num_flow_steps: 8
  open_loop_steps: 8
  device: "cuda"
  entrypoint_num_gpus: 1
  entrypoint_memory: 62277025792  # ~58GB per job

  # AWS profile for loading checkpoints from S3
  inference_aws_profile: "sagemaker"

# Docker configuration
docker:
  image: "682769330988.dkr.ecr.us-east-1.amazonaws.com/anzu-vla-foundry:latest"

# vla_foundry source (for inference code)
vla_foundry:
  repo_url: "git@github.com:TRI-ML/vla_foundry.git"
  ref: "main"
  local_dir: "~/.cache/vla_foundry/ray_mount"
  use_remote_repo: true
  remote_dir: "/home/ubuntu/vla_foundry_mount"
  mount_target: "/opt/vla_foundry"

# (OPTIONAL) Custom inference command - for alternative policy repos
# See campaigns/lbm_ablation_campaign.yaml for a full example
inference:
  # Option 1: Complete command override with placeholders
  cmd_override: "uv run python my_script.py --checkpoint {checkpoint}"
  # Option 2: Custom script with default wrapper
  # script: "path/to/inference_script.py"
  # args: "--extra-arg value"

# Output configuration
output:
  results_dir: "results/my_evaluation"
  save_job_logs: true
```

### Key Configuration Options

| Option | Description |
|--------|-------------|
| `num_workers` | Total parallel evaluation workers |
| `num_samples` | Rollouts per task/checkpoint |
| `jobs_per_gpu` | GPU sharing (1 = exclusive, 4 = 4 jobs/GPU) |
| `entrypoint_memory` | Memory limit per job (prevents OOM) |
| `inference_aws_profile` | AWS profile for S3 checkpoint access |
| `inference.cmd_override` | Complete custom inference command with placeholders |
| `inference.script` | Custom inference script path (uses uv run wrapper) |
| `inference.args` | Extra arguments for custom inference script |

---

## Step 4: Launch the Campaign

```bash
# Launch with your campaign config
./eval_campaigns/launch_campaign.sh eval_campaigns/campaigns/my_campaign.yaml

# Or use the default example_campaign.yaml
./eval_campaigns/launch_campaign.sh
```

### What Happens

1. **Cluster Spinup**: Ray cluster is created/connected via `ray up`
2. **Job Submission**: Evaluation jobs are submitted to the cluster
3. **Docker Containers**: Each worker runs simulations in Docker with:
   - `--runtime=nvidia` for GPU access
   - `--device /dev/dri` for EGL rendering
   - Proper group permissions for hardware rendering
4. **Progress Monitoring**: Jobs are tracked and logged
5. **Results Collection**: Success metrics are saved to S3 and locally

### Monitoring

```bash
# Check Ray dashboard (get URL from cluster output)
# Usually: http://<head-node-ip>:8265

# Monitor job status
ray job list --address http://<cluster-url>:8265

# Check specific job logs
ray job logs <job-id> --address http://<cluster-url>:8265
```

### Cluster Teardown

To manually tear down the cluster after your evaluation completes:

```bash
AWS_PROFILE=manip-cluster ray down vla_foundry/tri/lbm_eval/eval_campaigns/ray_policy_runner_cluster.yaml --yes
```

Alternatively, set `teardown_on_completion: true` in your campaign config for automatic teardown.

---

## Step 5: Collect and Analyze Results

### Download Success Metrics

```bash
python eval_campaigns/compute_success_rates_from_rollouts.py \
    eval_campaigns/task_lists/my_tasks.txt \
    --campaign-name my_evaluation_nov2025 \
    --output-dir ./rollout_summaries
```

This will:
1. Download `summary.yaml` files from S3 for each task
2. Parse success/failure from each rollout
3. Compute success rates per task
4. Save results to `rollout_summaries/success_rates.json`

### Output Example

```
[1/16] BimanualPutRedBellPepperInBin
  ✓ Found 100 summaries total
  Success: 76/100 = 76.0%

================================================================================
SUMMARY - 16 tasks processed
================================================================================
  BimanualPutRedBellPepperInBin                       :  76/100 =  76.0%
  PlaceCupByCoaster                                   :  82/100 =  82.0%
  ...
```

---

## Using Alternative Policy Repos (e.g., LBM)

The campaign system supports running evaluations with different policy repositories and inference commands.
This is useful for ablation studies with repositories like LBM that have a different inference interface.

### Configuration Example

See `campaigns/lbm_ablation_campaign.yaml` for a complete example. Key differences:

```yaml
# Use a different policy repo
vla_foundry:
  repo_url: "git@github.shared-services.aws.tri.global:jean-mercat/lbm.git"
  ref: "jean/compatible_with_anzu_vla_foundry"
  mount_target: "/opt/lbm"  # Different mount point

# Override the inference command
inference:
  # Use placeholders: {checkpoint}, {num_flow_steps}, {open_loop_steps}, {device}
  cmd_override: >-
    uv run python grpc_workspace/diffusion_policy_server.py
    --checkpoint-file {checkpoint}
    --server-uri 0.0.0.0:50051
```

### Inference Configuration Options

| Option | Description |
|--------|-------------|
| `inference.cmd_override` | Complete command override with placeholder substitution. Replaces the entire inference command. |
| `inference.script` | Path to custom inference script (relative to repo root). Uses standard `uv run python <script>` wrapper. |
| `inference.args` | Additional arguments appended to the inference command. |

### Placeholders for `cmd_override`

| Placeholder | Description |
|-------------|-------------|
| `{checkpoint}` | Replaced with the checkpoint path (from task file or downloaded location) |
| `{num_flow_steps}` | Number of flow steps from evaluation config |
| `{open_loop_steps}` | Number of open-loop steps from evaluation config |
| `{device}` | Device (cuda/cpu) from evaluation config |

### Notes

- The simulation side (Bazel/launch_sim.sh) remains unchanged - only the inference command is customized
- Your policy repo must be compatible with the simulation's gRPC interface
- Environment setup (uv sync) uses your repo's `pyproject.toml`

---

## Troubleshooting

### Slow Simulation Performance

If simulations are running slowly in Docker, ensure:

1. **NVIDIA runtime is configured**:
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

2. **EGL is working** (check inside container):
   ```bash
   eglinfo | grep -i nvidia
   # Should show "EGL vendor string: NVIDIA"
   ```

3. **Render group access**:
   ```bash
   ls -la /dev/dri/
   # Check renderD* group ownership
   ```

### Common Issues

| Issue | Solution |
|-------|----------|
| `Permission denied /dev/dri/renderD*` | Add render group: `--group-add $(stat -c '%g' /dev/dri/renderD128)` |
| `eglInitialize failed` | Missing `--runtime=nvidia` or NVIDIA EGL vendor file |
| Jobs stuck pending | Check cluster capacity, increase `num_workers` in cluster config |
| OOM errors | Reduce `jobs_per_gpu` or increase `entrypoint_memory` |

---

## Directory Structure

```
eval_campaigns/
├── campaigns/              # Campaign YAML configs
│   ├── example_campaign.yaml
│   ├── ablation_campaign.yaml
│   └── proprio_campaign.yaml
├── task_lists/             # Task list files (.txt)
│   ├── ablation.txt
│   ├── ablation_subset.txt
│   └── ...
├── cluster_scripts/        # Scripts for cluster/Docker execution
│   ├── launch_sim.sh
│   └── run_inference_bundle.sh
├── wandb_exports/          # W&B CSV exports
│   └── merge_stats.csv
├── launch_campaign.sh      # Main entry point
├── ray_policy_runner.py    # Ray job submission logic
├── run_evaluation_campaign.py  # Campaign orchestrator
├── ray_policy_runner_cluster.yaml  # Ray cluster config
├── task_metadata.yaml      # Task skill/station mappings
└── *.py                    # Utility scripts
```

## File Reference

| File | Description |
|------|-------------|
| `export_wandb_ablation_and_make_task_list.py` | Auto-export W&B runs from ablation configs |
| `csv_to_tuples.py` | Convert W&B CSV to task tuples |
| `campaigns/example_campaign.yaml` | Template campaign configuration |
| `compute_success_rates_from_rollouts.py` | Download and analyze results |
| `launch_campaign.sh` | Main entry point for launching campaigns |
| `run_evaluation_campaign.py` | Campaign orchestrator |
| `ray_policy_runner.py` | Ray job submission logic |
| `ray_policy_runner_cluster.yaml` | Ray cluster configuration |

---

## Quick Start Checklist

- [ ] Export CSV from W&B with Name, Tags, remote_sync columns
- [ ] Run `csv_to_tuples.py` to generate tasks file
- [ ] Copy and edit `campaigns/example_campaign.yaml`
- [ ] Set correct `tasks_file` path (use `task_lists/` prefix)
- [ ] Verify `docker.image` is correct
- [ ] Check `vla_foundry` settings match your inference code
- [ ] Run `./launch_campaign.sh your_campaign.yaml`
- [ ] Monitor via Ray dashboard
- [ ] Collect results with `compute_success_rates_from_rollouts.py`