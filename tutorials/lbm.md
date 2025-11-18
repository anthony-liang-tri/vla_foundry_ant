# LBM preprocessing, training, inference
This tutorial takes users through the process of preprocessing LBM Spartan data, then using the data to train, then loading and inferencing the model.

## Preprocessing
We use the scripts [vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py](/vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py) to convert from raw Spartan data to VLA Foundry webdataset tar shards. 

These scripts use Ray. Ray can work either locally or on EC2 nodes. See [vla_foundry/data/preprocessing/README.md](/vla_foundry/data/preprocessing/README.md) for instructions on how to use Ray. 

This script below is taken from `examples/preprocessing/preprocess_robotics_data_lbm.sh`. It processes BimanualPutRedBellPepperInBin Spartan shards and converts them to tar shards in the given `output_dir`. 

```bash
python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--source_type "spartan" \
--source_episodes "[
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-02T10-49-28-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-02T14-21-19-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-06T08-58-31-05-00/diffusion_spartan/',
]" \
--output_dir s3://tri-ml-datasets/vla_foundry_scratch/tmp/insert.user.name.here/BimanualPutRedBellPepperInBin \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_4cameras.yaml" \
--language_annotations_path vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml \
--action_fields_config_path vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml"
```

(Note: This might require using some preprocessing dependencies which might require `uv sync --group=preprocessing`.)


## Training
After running the preprocessing, you can check the `output_dir` folder that you provided as an argument in the preprocessing script. There should be a subfolder `shards` in that folder, which contains the files required for training (e.g., manifest, stats, tar shards).

The training script below is taken from `examples/training/diffusion_policy.sh`. Here, dataset you preprocessed goes in `dataset_manifest` and `dataset_statistics`. 

```bash
.venv/bin/torchrun --nproc_per_node=2 --nnodes=1 vla_foundry/main.py \
--config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml \
--remote_sync s3://tri-ml-datasets-uw2/lbm2_vla/model_checkpoints/diffusion_policy \
--num_checkpoints 5 \
--total_train_samples 100000
```

For training with SageMaker, we train with [sagemaker/launch_training.py](/sagemaker/launch_training.py). The argument parser of this script is a wrapper around the argument parser of [vla_foundry/main.py](/vla_foundry/main.py), so we can reuse the same arguments. The SageMaker-specific arguments can be supplied with the `sagemaker.` prefix, as shown below:
```bash
uv run --group=sagemaker sagemaker/launch_training.py \
--sagemaker.user (firstname.lastname) \
--sagemaker.instance_count 1 \
--sagemaker.instance_type p4de \
--sagemaker.queue_name vla \
(copy-paste other arguments here as-is, e.g., --data.something)
```

Note: We use `uv run --group=sagemaker` to launch this script. No need for torchrun here.

## Inference
(todo)
