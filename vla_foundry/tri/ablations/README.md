# Ablation Experiments

This directory contains scripts and configuration files for running ablation experiments on the VLA Foundry training pipeline.

## Directory Structure

```
ablations/
├── README.md                      # This file
├── nominal_config.yaml            # Base configuration (tasks, paths, default params)
├── ablations.yaml                 # Ablation definitions with sweep support
├── generate_ablation_configs.py   # Python script to generate configs
├── launch_ablation_sagemaker.py   # Wrapper: generate configs + launch SageMaker from results
├── manage_ablations.py            # Script to check ablation run status from wandb
└── ablation_configs/              # Generated config files
    ├── <ablation_name>/
    │   └── <task_name>/
    │       └── resolved_config.yaml
    └── ...
```

## Quick Start

```bash
# Generate configs from an ablations YAML, then launch SageMaker jobs from the results
uv run python vla_foundry/tri/ablations/launch_ablation_sagemaker.py \
  vla_foundry/tri/ablations/ablations.yaml

# Dry-run (prints what would be launched; does not require SAGEMAKER_ARN)
uv run python vla_foundry/tri/ablations/launch_ablation_sagemaker.py \
  vla_foundry/tri/ablations/ablations.yaml --dry-run
```

### Create your ablations YAML

Create a new YAML file (e.g. `vla_foundry/tri/ablations/my_ablations.yaml`) with a top-level `ablations:` mapping.
Each key is an **ablation name**, and the value is a dict of **overrides** applied on top of `nominal_config.yaml`’s `base_args`.

You can start by copying `vla_foundry/tri/ablations/ablations.yaml` (or `my_ablations.yaml`) and editing it.

Example:

```yaml
# In your ablations YAML (tasks + parameter overrides / sweeps)
tasks:
  - BimanualPutRedBellPepperInBin
  - TurnCupUpsideDown

ablations:
  nominal_10m: {}

  fp32_30m:
    hparams.precision: float32
    total_train_samples: 30_000_000

  lr_sweep:
    hparams.lr: [0.00002, 0.00005, 0.0001]
```

Sweep syntax supported:
- `[v1, v2, v3]`
- `linspace(start, end, n_steps)`
- `logspace(start, end, n_steps)`

---

## Configuration Files

### `nominal_config.yaml`

Contains base configuration shared by all ablations:
- **tasks**: List of task names to run
- **base_s3_path**: S3 path for datasets
- **checkpoint_base**: S3 path for checkpoints
- **base_args**: Default hyperparameters

### `ablations.yaml`

Defines ablation experiments with **sweep support**:

```yaml
ablations:
  # Simple ablation
  nominal_10m:
    total_train_samples: 10_000_000

  # Sweep over a list of values (generates 3 ablations)
  lr_sweep:
    total_train_samples: 10_000_000
    hparams.lr: [0.00002, 0.00005, 0.0001]

  # Linear spacing: linspace(start, end, n_steps)
  lr_linspace:
    hparams.lr: linspace(0.00001, 0.0001, 5)

  # Log spacing: logspace(start, end, n_steps)
  lr_logspace:
    hparams.lr: logspace(0.00001, 0.0001, 5)
```

**Sweep syntax:**
- `[v1, v2, v3]` - Explicit list of values
- `linspace(start, end, n)` - n linearly spaced values
- `logspace(start, end, n)` - n logarithmically spaced values

Auto-naming: `lr_sweep` with `lr=[2e-5, 5e-5]` generates `lr_sweep_lr_2.00e-05`, `lr_sweep_lr_5.00e-05`

---

## Scripts

### Generate Ablation Configs

```bash
# Generate all configs
uv run python vla_foundry/tri/ablations/generate_ablation_configs.py

# Dry run (preview without generating)
uv run python vla_foundry/tri/ablations/generate_ablation_configs.py --dry-run

# Custom configs
uv run python vla_foundry/tri/ablations/generate_ablation_configs.py \
    --nominal-config path/to/nominal.yaml \
    --ablations-config path/to/ablations.yaml
```


---

### Manage Ablations

Check the status of ablation runs from wandb:

```bash
# Basic status view
uv run python vla_foundry/tri/ablations/manage_ablations.py

# Verbose mode (shows step count and run names)
uv run python vla_foundry/tri/ablations/manage_ablations.py -v

# Custom wandb entity/project
uv run python vla_foundry/tri/ablations/manage_ablations.py --entity <entity> --project <project>
```

**Output:**
- 🟢 **finished** - Training completed
- 🔵 **running** - Currently training
- 🔴 **failed/crashed** - Run encountered an error
- ⬜ **not_launched** - No matching run in wandb

---

## Launching Training

### Local Training

```bash
uv run python vla_foundry/main.py \
    --config vla_foundry/tri/ablations/ablation_configs/<ablation>/<task>/resolved_config.yaml \
    --wandb True
```

### SageMaker Training

```bash
# Set your SageMaker ARN
export SAGEMAKER_ARN=arn:aws:iam::...

# Launch all configs
./vla_foundry/tri/sagemaker/launch_from_configs.sh vla_foundry/tri/ablations/ablation_configs/

# Launch a specific ablation (all 4 tasks)
./vla_foundry/tri/sagemaker/launch_from_configs.sh vla_foundry/tri/ablations/ablation_configs/nominal_10m/

# Launch with filtering
./vla_foundry/tri/sagemaker/launch_from_configs.sh vla_foundry/tri/ablations/ablation_configs/ --ablation 10m
./vla_foundry/tri/sagemaker/launch_from_configs.sh vla_foundry/tri/ablations/ablation_configs/ --task RedBell

# Dry run (preview without launching)
./vla_foundry/tri/sagemaker/launch_from_configs.sh vla_foundry/tri/ablations/ablation_configs/ --dry-run
```
