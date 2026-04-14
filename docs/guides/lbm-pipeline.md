# LBM Pipeline

This guide walks through the end-to-end process for LBM (Learned Behavior Models): preprocessing Spartan data, training a diffusion policy, and finetuning from pretrained weights.

## Preprocessing

VLA Foundry uses the script `vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py` to convert raw Spartan data into webdataset tar shards.

!!! note "Ray parallelization"
    The preprocessing scripts use Ray for parallelization. Ray can run either locally or on EC2 nodes. See the [Data Preprocessing](data-preprocessing.md) guide for instructions on setting up and using Ray.

!!! note "Dependencies"
    Preprocessing may require additional dependencies. Install them with:

    ```bash
    uv sync --group=preprocessing
    ```

### Example Preprocessing Command

The following example (from `examples/preprocessing/preprocess_robotics_data_lbm.sh`) processes BimanualPutRedBellPepperInBin Spartan shards and converts them to tar shards in the specified `output_dir`:

```bash
python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
  --type "spartan" \
  --source_episodes "[
      's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-02T10-49-28-05-00/diffusion_spartan/',
      's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-02T14-21-19-05-00/diffusion_spartan/',
      's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-06T08-58-31-05-00/diffusion_spartan/',
      ]" \
  --output_dir s3://tri-ml-datasets/vla_foundry_scratch/models/spartan_datasets/bimanualputredbellpepperinbin3/ \
  --camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_4cameras.yaml" \
  --language_annotations_path vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml \
  --action_fields_config_path vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml \
  --data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
  --samples_per_shard 100 \
  --config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml"
```

After preprocessing completes, check the `output_dir` folder. There should be a `shards` subfolder containing the files required for training (manifest, stats, and tar shards).

## Training

The training script below is adapted from `examples/training/diffusion_policy.sh`. The preprocessed dataset is referenced through the `dataset_manifest` and `dataset_statistics` fields inside the config at `config_path`.

```bash
.venv/bin/torchrun --nproc_per_node=2 --nnodes=1 vla_foundry/main.py \
  --config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml \
  --remote_sync s3://tri-ml-datasets-uw2/lbm2_vla/model_checkpoints/diffusion_policy \
  --num_checkpoints 5 \
  --total_train_samples 100000
```

### Resolving Configs

Use the `--resolve_configs` and `--resolve_configs_path` flags to inspect the fully resolved configuration before running training.

- Setting `--resolve_configs=True` prints the resolved config to stdout.
- Setting `--resolve_configs_path` saves the resolved config to `{resolve_configs_path}/resolved_config.yaml`.

```bash
.venv/bin/torchrun --nproc_per_node=2 --nnodes=1 vla_foundry/main.py \
  --config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml \
  --remote_sync s3://tri-ml-datasets-uw2/lbm2_vla/model_checkpoints/diffusion_policy \
  --num_checkpoints 5 \
  --total_train_samples 100000 \
  --resolve_configs True \
  --resolve_configs_path ./
```

### SageMaker Training

For training on SageMaker, use the `sagemaker/launch_training.py` script. Its argument parser wraps the `vla_foundry/main.py` parser, so all training arguments can be reused. SageMaker-specific arguments use the `sagemaker.` prefix:

```bash
uv run --group=sagemaker sagemaker/launch_training.py \
  --sagemaker.user firstname.lastname \
  --sagemaker.instance_count 1 \
  --sagemaker.instance_type p4de \
  --sagemaker.queue_name vla \
  # copy-paste other training arguments here as-is, e.g., --data.something
```

!!! note
    Use `uv run --group=sagemaker` to launch this script. There is no need for `torchrun` when using SageMaker.

## Finetuning from Pretrained Weights

To finetune a model from a pretrained checkpoint instead of training from scratch, use the `--model.resume_from_checkpoint` and `--model.resume_weights_only` flags. This loads only the model weights without resuming optimizer state or training progress.

```bash
uv run --group=sagemaker sagemaker/launch_training.py \
  --sagemaker.user firstname.lastname \
  --sagemaker.instance_count 1 \
  --sagemaker.instance_type p4de \
  --sagemaker.queue_name vla \
  --config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml \
  --remote_sync s3://tri-ml-datasets-uw2/lbm2_vla/model_checkpoints/finetuned \
  --model.resume_from_checkpoint s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/diffusion_policy/ablations/multitask/100m/2026_01_07-23_38_39-model_diffusion_policy-lr_5e-05-bsz_1024/checkpoints/checkpoint_3.pt \
  --model.resume_weights_only True
```

!!! tip "Config-based approach"
    It is usually preferable to place `resume_from_checkpoint` and `resume_weights_only` directly into the config file referenced by `--config_path` rather than passing them on the command line.

Key details:

- The checkpoint path is validated early in training to fail fast with a clear error if the path is invalid.
- Use `--model.resume_weights_only=False` (the default) to fully resume training including optimizer state.
