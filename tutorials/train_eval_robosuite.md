# Multi-Task DiT Training With Eval Rollouts

This guide covers converting the robosuite and PushT datasets to VLA Foundry WebDataset shards, then training with checkpoint-time rollout evals.

The robosuite example follows the same broad policy shape as [LeRobot's Multi-Task DiT policy](https://huggingface.co/docs/lerobot/en/multi_task_dit): a diffusion transformer predicts action chunks conditioned on image tokens, language instructions, and optional proprioceptive state. The provided robosuite config uses `openai/clip-vit-base-patch16` for both vision and text conditioning, fine-tunes the vision encoder, freezes the text encoder, and trains a 6-layer, 512-hidden-dim transformer head with rotary position embeddings.

For robosuite, action generation uses the flow-matching path rather than DDPM sampling. Training samples a continuous time `tau`, interpolates between Gaussian noise and normalized actions, and predicts the velocity target `actions - noise`. The default robosuite config uses beta timestep sampling with `alpha=1.5`, `beta=1.0`, `s=0.999`, continuous time embeddings, `sigma_min=0.0`, and a valid-action denoising mask so both past and future valid low-dimensional slots are handled consistently. Evaluation then integrates the learned velocity field for `num_inference_steps` steps and executes `action_window` actions per policy chunk.

The example configs live in:

- `tutorials/configs/multitask_dit_robosuite.yaml`
- `tutorials/configs/pusht.yaml`

Eval rollout metrics are logged to W&B as:

- `eval_rollouts/<task>/success_rate`
- `eval_rollouts/<task>/video_grid`

## Robosuite Lift/Can/Square

Install the optional dependencies:

```bash
uv sync --group preprocessing --group robosuite
```

Download the LeRobot-format robosuite dataset:

```bash
hf download TRI-ML/robosuite_ph \
  --repo-type dataset \
  --local-dir data/lerobot/TRI-ML/robosuite_ph
```

Convert each task to WDS shards with the standard robotics preprocessing entrypoint. The output paths below match the default robosuite config.

```bash
uv run --group preprocessing python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
  --type lerobot \
  --source_episodes "['data/lerobot/TRI-ML/robosuite_ph']" \
  --output_dir data/robosuite_lift_full_horizon32 \
  --camera_names "['observation.images.agentview', 'observation.images.wrist']" \
  --observation_keys "['observation.state']" \
  --action_keys "['action']" \
  --lowdim_key_remap "{'observation.state': 'state', 'action': 'actions'}" \
  --resize_images_size "[256, 256]" \
  --samples_per_shard 1000 \
  --past_lowdim_steps 1 \
  --future_lowdim_steps 30 \
  --max_padding_left 1 \
  --max_padding_right 30 \
  --task_filter "lift the red cube off the table" \
  --skip_git_tagging true \
  --db_logging false

uv run --group preprocessing python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
  --type lerobot \
  --source_episodes "['data/lerobot/TRI-ML/robosuite_ph']" \
  --output_dir data/robosuite_can_full_horizon32 \
  --camera_names "['observation.images.agentview', 'observation.images.wrist']" \
  --observation_keys "['observation.state']" \
  --action_keys "['action']" \
  --lowdim_key_remap "{'observation.state': 'state', 'action': 'actions'}" \
  --resize_images_size "[256, 256]" \
  --samples_per_shard 1000 \
  --past_lowdim_steps 1 \
  --future_lowdim_steps 30 \
  --max_padding_left 1 \
  --max_padding_right 30 \
  --task_filter "pick the coke can and place it in the bin" \
  --skip_git_tagging true \
  --db_logging false

uv run --group preprocessing python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
  --type lerobot \
  --source_episodes "['data/lerobot/TRI-ML/robosuite_ph']" \
  --output_dir data/robosuite_square_full_horizon32 \
  --camera_names "['observation.images.agentview', 'observation.images.wrist']" \
  --observation_keys "['observation.state']" \
  --action_keys "['action']" \
  --lowdim_key_remap "{'observation.state': 'state', 'action': 'actions'}" \
  --resize_images_size "[256, 256]" \
  --samples_per_shard 1000 \
  --past_lowdim_steps 1 \
  --future_lowdim_steps 30 \
  --max_padding_left 1 \
  --max_padding_right 30 \
  --task_filter "pick up the square nut and place it on the square peg" \
  --skip_git_tagging true \
  --db_logging false
```

Train the multi-task policy:

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc_per_node=1 \
  vla_foundry/main.py \
  --config_path tutorials/configs/multitask_dit_robosuite.yaml
```

The config trains on all three task datasets with equal task weighting, proprioceptive state, language instructions, both `agentview` and wrist cameras, and min-max normalization for `state` and `actions`.

## PushT

Install the optional dependencies:

```bash
uv sync --group preprocessing --group tutorials
```

Download the LeRobot PushT dataset:

```bash
hf download lerobot/pusht \
  --repo-type dataset \
  --local-dir tutorials/data/lerobot/pusht
```

Convert it to WDS shards:

```bash
uv run --group preprocessing python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
  --type lerobot \
  --source_episodes "['tutorials/data/lerobot/pusht']" \
  --output_dir tutorials/data/preprocessed/pusht \
  --camera_names "['observation.image']" \
  --observation_keys "['observation.state']" \
  --action_keys "['action']" \
  --samples_per_shard 1000 \
  --config_path vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml \
  --skip_git_tagging true
```

Train the PushT policy:

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc_per_node=1 \
  vla_foundry/main.py \
  --config_path tutorials/configs/pusht.yaml
```

The PushT eval runner logs rollout success rate and a 2x2 rollout video grid. The eval result JSONs and videos are also written under `outputs/eval_rollouts/`.
