# TRI-specific files and READMEs

## Dashboard and Leaderboard

See [leaderboard/README.md](frontend/README.md) for documentation on running the leaderboard.
You can access the leaderboard at:
[http://10.242.10.206:8080/leaderboard/](http://10.242.10.206:8080/leaderboard/)

And the dashboard at:
[http://10.242.10.206:8080/](http://10.242.10.206:8080/)

## Shared Directories
- Clean directory: `s3://tri-ml-datasets-uw2/vla_foundry_datasets/`
    - Please do not add anything here until it has already been tested.
    - More specifically, it would be nice if every folder/file in that bucket has a corresponding script in this repo that can be ran to reproduce that dataset.
    - Eventually we probably want to protect this bucket to avoid overwriting. We're small enough now that it's fine.
- Messy directory for scratch: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/`

---

## Data Preprocessing (Ray)

`data_generation_ray.py` preprocesses sim/real robotics data into WebDataset shards on S3, using a Ray cluster for parallelism.

### Basic usage

```bash
# Run on a Ray cluster (starts cluster, submits job, streams logs)
uv run vla_foundry/tri/data_generation_ray.py \
    --tasks-file vla_foundry/tri/stage3_singletask_real/stage3_real_filenames.txt \
    --output-prefix s3://tri-ml-datasets-uw2/vla_foundry_datasets_test/v0.4.3.7/

# Run locally (single machine)
uv run vla_foundry/tri/data_generation_ray.py \
    --tasks-file vla_foundry/tri/stage3_singletask_sim/stage3_sim_filenames.txt --local
```

### Overwrite modes

By default, the script errors out if the output directory already has data (completed or partial). Use these flags to control behavior:

| Flag | Completed task | Partial/crashed data | No data |
|------|---------------|---------------------|---------|
| (none) | Error | Error | Process |
| `--continue` | Skip | Clean up + retry | Process |
| `--force` | Overwrite | Overwrite | Process |

### Options

| Flag | Description |
|------|-------------|
| `--tasks-file FILE` | **(required)** File with S3 episode paths, one per line |
| `--output-prefix S3_URI` | S3 prefix for output (overrides config file) |
| `--force` | Overwrite all existing data, including completed tasks |
| `--continue` | Skip completed tasks, clean up and retry partial/crashed ones |
| `--batch-size N` | Max tasks to run in parallel (default: all at once) |
| `--data-type {sim,real,all}` | Filter which data types to process (default: all) |
| `--camera-names CAM1,CAM2` | Override camera names (comma-separated) |
| `--skip-missing-cameras` | Skip episodes missing requested cameras |
| `--no-episode-shards` | Only create random shards, skip per-episode shards |
| `--config FILE` | Preprocessing config YAML (default: `vla_foundry/tri/preprocessing_config.yaml`) |
| `--cluster-config FILE` | Ray cluster config YAML (default: `vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml`) |
| `--local` | Run on local machine instead of a Ray cluster |
| `--on-head` | Run on the cluster head node (used internally by job submission) |

---

## Ablation Experiments

See [ablations/README.md](ablations/README.md) for documentation on running ablation experiments.

