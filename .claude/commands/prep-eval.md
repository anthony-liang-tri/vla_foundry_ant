Discover checkpoint paths from completed training runs and generate evaluation campaign configs.

The argument specifies the S3 base path or training run identifier, e.g.:
- `/prep-eval s3://tri-ml-datasets-uw2/vla_foundry_checkpoints/qwen3_mt_ft_2ksteps/` — discover checkpoints under this path
- `/prep-eval qwen3_ft` — search for recent completed training jobs matching this pattern and build eval from their checkpoints

## Steps

### Step 1: Discover checkpoint paths

**Option A: From S3 base path**
If an S3 path is given, list subdirectories to find per-task checkpoints:
```bash
aws s3 ls "<s3_base_path>/" --profile sagemaker
```

Each task subdirectory should contain a run directory (with timestamp) that has checkpoints. Find the run directory:
```bash
aws s3 ls "<s3_base_path>/<task>/" --profile sagemaker | grep "PRE " | head -1
```

**Option B: From SageMaker job pattern**
If a name pattern is given, find completed jobs and extract their `remote_sync` paths:
```bash
aws sagemaker list-training-jobs --profile sagemaker --region us-west-2 \
  --max-results 100 --status-equals Completed --sort-by CreationTime --sort-order Descending \
  --output json
```
Filter by pattern, then for each job, describe it to get the hyperparameters which contain the config with `remote_sync`.

### Step 2: Verify checkpoints exist

For each discovered task checkpoint path, verify it has actual checkpoint files:
```bash
aws s3 ls "<checkpoint_path>/checkpoints/" --profile sagemaker | grep checkpoint_ | grep -v optimizer
```

Report any tasks with missing checkpoints.

### Step 3: Generate task list file

Create a task list file at `vla_foundry/tri/lbm_eval/eval_campaigns/task_lists/<name>.txt`:
```
# Task list for <description>
# Generated on <date>
# Total tasks: <N>

("<TaskName>", "<TaskName>", "<s3_checkpoint_path>"),
```

The format is: `("job_name", "task_name", "checkpoint_s3_path"),` — one per line.

### Step 4: Generate eval campaign YAML

Create a campaign config at `vla_foundry/tri/lbm_eval/eval_campaigns/campaigns/my_<name>_campaign.yaml` based on the template:

```yaml
campaign_name: "<name>_eval"
description: "<description>"

cluster:
  config_file: "vla_foundry/tri/lbm_eval/eval_campaigns/ray_policy_runner_cluster.yaml"
  num_workers: 256
  aws_profile: "manip-cluster"
  owner_email: "jean.mercat@tri.global"
  cluster_name: "vla_eval_jean_mercat"
  teardown_on_completion: true
  teardown_on_failure: true

evaluation:
  jobs_per_gpu: 1
  tasks_file: "vla_foundry/tri/lbm_eval/eval_campaigns/task_lists/<name>.txt"
  num_samples: 200
  start_index: 0
  launch_scenario: "GrpcServerToSim"
  launch_script: "launch_sim.sh"
  num_flow_steps: 8
  open_loop_steps: 8
  device: "cuda"
  entrypoint_num_gpus: 1
  entrypoint_memory: 62277025792
  inference_aws_profile: "sagemaker"

docker:
  image: "682769330988.dkr.ecr.us-east-1.amazonaws.com/anzu-vla-foundry:latest"

vla_foundry:
  repo_url: "git@github.com:jmercat/vla_foundry.git"
  ref: "jean/merge_qwen+uuid"
  local_dir: "~/.cache/vla_foundry/ray_mount"
  use_remote_repo: true
  remote_dir: "/home/ubuntu/vla_foundry_mount"
  mount_target: "/opt/vla_foundry"

output:
  results_dir: "results/<name>"
  save_job_logs: true
```

### Step 5: Confirm and report

- Show the generated task list (task names + checkpoint paths)
- Show the campaign config path
- Report any issues (missing checkpoints, incomplete training)
- Suggest next steps: `/eval launch campaigns/my_<name>_campaign.yaml`

## Key details

- Checkpoint paths from `remote_sync` are the base; actual run dirs have timestamps like `2026_03_28-13_00_07-model_diffusion_policy-lr_5e-05-bsz_384/`
- The eval system expects the checkpoint directory (containing `checkpoints/` subfolder), NOT a specific `.pt` file
- Always verify that checkpoints actually exist before generating the eval config
- The `vla_foundry.ref` in the campaign config should match the branch the model was trained with
- Default eval parameters: 200 samples, 8 flow steps, 8 open loop steps
- Use `AWS_PROFILE=sagemaker` for S3 access to checkpoint paths
