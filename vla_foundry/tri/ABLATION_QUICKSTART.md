# Training to Evaluation Quickstart

End-to-end guide: train models → run evaluations → get results.

## Prerequisites

- AWS SSO configured with `manip-cluster` profile (see [eval campaigns README](vla_foundry/tri/lbm_eval/eval_campaigns/README.md#aws-sso-login-with-manip-cluster-profile))
- W&B entity set: `export WANDB_ENTITY=your_entity`

## 1. Train Models

Each command below launches training for a single ablation. To see all available ablations and create custom experiments, see [vla_foundry/tri/ablations/README.md](vla_foundry/tri/ablations/README.md). 
IMPORTANT: Before running you must edit the ```user``` field in the ```*_nominal_config```.

**Finetuning from Checkpoints**: When finetuning, use the checkpoint's `s3://config.yaml` as your `config_path` to automatically inherit all data parameters (including `dataset_statistics` for normalization consistency). Then override only what changes for finetuning.

### Example A: Sim 16 Tasks
```bash
# 208 total training runs = 13 configs per task × 16 tasks
# (1 from_scratch + 12 finetune_sweep: 2 checkpoints × 3 lrs × 2 samples)
uv run python vla_foundry/tri/ablations/launch_ablation_sagemaker.py \
  vla_foundry/tri/ablations/ft_ablations.yaml \
  --nominal-config vla_foundry/tri/ablations/sim_16_nominal_config.yaml

```

### Example B: Real-World Toolhang (6 Tasks)
```bash
# 78 total training runs = 13 configs per task × 6 tasks
uv run python vla_foundry/tri/ablations/launch_ablation_sagemaker.py \
  vla_foundry/tri/ablations/ft_ablations.yaml \
  --nominal-config vla_foundry/tri/ablations/realworld_toolhang_nominal_config.yaml
```
To preview what would be launch without starting jobs, add the ```--dry-run``` option; to pick just a single ablation try ```--ablation from_scratch_10m```. In each case, note the runset hash printed, for example: ```runset_20260212_143052_a3f9d8e1```. All runs are tagged with the runset hash in wandb for easy filtering.

## 2. Generate Evaluation Campaign

Once training completes, for the sim_16 tasks (single task for now)

```bash
# Minimal: just use the runset hash from training; 200 samples and last checkpoint
python vla_foundry/tri/lbm_eval/eval_campaigns/auto_generate_campaign.py \
  --tags runset_20260212_143052_a3f9d8e1 \
  --owner-email your.name@tri.global
```

**Required**: `--owner-email` must be your @tri.global email (for AWS cluster tagging). If you did not
generate the runs with this workflow, you'll have to manually generate the configuration files from
```csv_to_tuples.py``` instead. By default, this will use campaigns/example_campaign.yaml. Set a different template to overwrite via ```--template```.

**Options**:
- `--checkpoint-num`: Which checkpoint to evaluate (3 = checkpoint_3.pt, 5 = checkpoint_5.pt). Omit to use latest checkpoint directory.
- `--num-samples`: Number of rollouts per task (default: 200)
- `--name`: Campaign name (default: uses the runset hash)

This creates:
- `vla_foundry/tri/lbm_eval/eval_campaigns/campaigns/runset_20260212_143052_a3f9d8e1.yaml`
- `vla_foundry/tri/lbm_eval/eval_campaigns/task_lists/runset_20260212_143052_a3f9d8e1.txt`
The yaml will point to the text file.

## 3. Launch Evaluation

```bash
./vla_foundry/tri/lbm_eval/eval_campaigns/launch_campaign.sh \
  vla_foundry/tri/lbm_eval/eval_campaigns/campaigns/runset_20260212_143052_a3f9d8e1.yaml
```

The cluster will:
- Spin up Ray workers
- Download checkpoints from S3
- Save results to `results/runset_20260212_143052_a3f9d8e1/`

## 4. Check Results

After the evaluation completes, compute success rates:

```bash
# Compute success rates from rollouts
python vla_foundry/tri/lbm_eval/eval_campaigns/compute_success_rates_from_rollouts.py \
  vla_foundry/tri/lbm_eval/eval_campaigns/task_lists/runset_20260212_143052_a3f9d8e1.txt \
  --campaign-name runset_20260212_143052_a3f9d8e1 \
  --output-dir results/runset_20260212_143052_a3f9d8e1

# View success rates
cat results/runset_20260212_143052_a3f9d8e1/success_metrics.json

# Or check the campaign report
cat results/runset_20260212_143052_a3f9d8e1/campaign_report.txt

# Or use interactive viewer on local rollouts
python vla_foundry/tri/utils/gather_results.py results/runset_20260212_143052_a3f9d8e1/rollouts/rollouts/
```