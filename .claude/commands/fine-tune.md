Generate per-task fine-tuning configs from a base multitask model and launch training on SageMaker.

The argument specifies the base model or task list, e.g.:
- `/fine-tune vla_foundry/tri/lbm_eval/eval_campaigns/task_lists/qwen3_multitasks.txt` — fine-tune using tasks and checkpoint from a task list file
- `/fine-tune s3://path/to/checkpoint/ --tasks BimanualPutRedBellPepperInBin,PlaceCupByCoaster` — fine-tune specific tasks from a checkpoint

## Workflow

### Step 1: Identify base model and tasks

- If a **task list file** is given (`.txt`), parse it to extract:
  - The base checkpoint S3 path (from the comment header or first entry)
  - The list of task names (from each line's first field)
- If a **checkpoint path** is given with `--tasks`, use those directly
- If no argument, list available task lists in `vla_foundry/tri/lbm_eval/eval_campaigns/task_lists/` and ask the user to pick

### Step 2: Identify model architecture

Read the base model's config to determine key architecture parameters. Check for a `config.yaml` in the checkpoint's S3 experiment directory:
```
aws s3 ls <checkpoint_base_path>/
```

Key parameters to extract from the base model config:
- **Model type**: `vlm_backbone` vs `vlm_foundry_backbone`, `hf_pretrained` vs separate VLM checkpoint
- **Architecture**: `hidden_dim`, `n_layers`, `n_heads`, `is_causal`, `vocab_size`
- **Batch size**: `per_gpu_batch_size`, `global_batch_size`
- **Qwen-specific**: `num_action_head_repeats`, `processor`, `processor_kwargs`, `diffusion_step_conditioning`
- **Data config**: `img_num_tokens`, `proprioception_dim`, `proprioception_fields`, `lowdim_future_timesteps`, normalization settings, extrinsics/intrinsics fields
- **Noise/flow**: `input_noise_std`, `clamp_range`, `use_flow_matching_scheduler`

### Step 3: Confirm fine-tuning hyperparameters with user

Present defaults and ask for confirmation:
- **total_train_samples**: `1_000_000` (default, ~2-3k steps depending on batch size)
- **lr**: Same as base training (default) or lower for fine-tuning
- **warmup**: `300` steps (default for fine-tuning; must be << total steps)
- **lr_scheduler**: `cosine` with `lr_cooldown_end: 0.0`
- **EMA**: disabled (default for fine-tuning)
- **per_gpu_batch_size**: Same as base model training
- **global_batch_size**: Same as base model training (adjusted for instance count)
- **num_checkpoints**: `10`
- **Instance type**: `p5` (default), `p5en`, `p4de`
- **Instance count**: `1` (default for fine-tuning)
- **Queue**: `vla` (default)
- **max_run**: `3` hours (default)

### Step 4: Generate configs

Create a Python generation script at `vla_foundry/tri/ablations/vla/generate_<model_name>_ft_configs.py` that:
- Templates a `resolved_config.yaml` for each task
- Uses per-task dataset manifests: `s3://tri-ml-datasets-uw2/vla_foundry_datasets_test/<version>/sim/<task>/shards/manifest.jsonl`
- Uses the global stats file for normalization
- Sets `model.resume_from_checkpoint` to the latest checkpoint `.pt` file from the base model
- Sets `model.resume_weights_only: true`
- Sets unique `remote_sync` paths per task: `s3://tri-ml-datasets-uw2/vla_foundry_checkpoints/<model_name>_ft_<steps>/<task>/`
- Sets appropriate `wandb_tags` for tracking

Output directory structure:
```
vla_foundry/tri/ablations/vla/ablation_configs_<model>_ft/<model>_ft_<steps>/<task>/resolved_config.yaml
```

Run the generation script and verify the output.

### Step 5: Launch on SageMaker

1. Do a **dry run** first:
   ```
   ./vla_foundry/tri/sagemaker/launch_from_configs.sh <config_dir> --dry-run --sagemaker.user jmercat --sagemaker.instance_type <type> --sagemaker.instance_count <count> --sagemaker.queue_name <queue> --sagemaker.max_run <hours>
   ```
2. Show the dry run output and confirm with user
3. Launch for real (same command without `--dry-run`)
4. Set up a monitoring cron with `/training-status` pattern

### Step 6: Prepare evaluation

After launching, automatically create:
- A task list generator script at `vla_foundry/tri/lbm_eval/eval_campaigns/task_lists/generate_<model>_ft_task_list.sh`
- An eval campaign YAML at `vla_foundry/tri/lbm_eval/eval_campaigns/campaigns/my_<model>_ft_campaign.yaml`

Tell the user to run `/prep-eval` once training completes.

## Key details

- The latest checkpoint is usually the highest numbered `checkpoint_N.pt` in `<s3_path>/checkpoints/`
- Dataset manifests are at `s3://tri-ml-datasets-uw2/vla_foundry_datasets_test/<version>/sim/<task>/shards/manifest.jsonl`
- Stats are at `s3://tri-ml-datasets-uw2/vla_foundry_datasets_test/<version>/stats.json`
- The dataset version should match what the base model was trained on
- For Qwen models: `per_gpu_batch_size=16`, `num_action_head_repeats=8`, `vlm_backbone` type with `hf_pretrained`
- For SmolVLM models: `per_gpu_batch_size=16`, `vlm_foundry_backbone` type with separate VLM checkpoint
- Always use `resume_weights_only: true` for fine-tuning to reset optimizer state
