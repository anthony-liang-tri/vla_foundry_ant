# Data Preprocessing

This guide covers the data preprocessing pipeline in VLA Foundry, including dependency setup, Ray parallelization, and conversion scripts for various dataset formats.

## Dependencies

The preprocessing scripts use dependencies that are isolated from the training code. Install them with:

```bash
uv sync --group=preprocessing
```

## Using Ray

Many preprocessing scripts use [Ray](https://docs.ray.io/) for parallelization. The general workflow is:

1. Create your Ray-compatible script.
2. Start a Ray instance -- either locally or on AWS EC2 clusters.
3. Run your script from within the Ray environment.

### Local Ray

For local usage, start Ray on your machine:

```bash
ray start --head
```

!!! tip "Ray Dashboard"
    Add `--include-dashboard=True` before `--head` to enable the Ray dashboard for diagnostics.

### AWS Ray Cluster

#### 1. Create the cluster

```bash
ray up vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml
```

!!! warning "Before running"
    You may need to edit the following in `ray_cluster_configs.yaml`:

    - `tags`, `username`, number of nodes, etc.
    - `file_mounts`: By default it copies your HF token from `~/.cache/huggingface/token`. Change this if your token is stored elsewhere.
    - `rsync_exclude`: Currently excludes `.venv` and `wandb`. Add additional paths you want to exclude (e.g., large checkpoints).

#### 2. Attach to the cluster

```bash
ray attach vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml
```

#### 3. Run your script inside the cluster

```bash
# Optionally start a persistent terminal like tmux
cd vla_foundry
python some_preprocessing_script.py
```

!!! note
    Ray scripts currently do **not** work well with `uv run`. Use `uv sync --group=preprocessing` (automatically done in `ray_cluster_configs.yaml`) and `source .venv/bin/activate` instead.

#### 4. Shut down the cluster

When finished, exit the cluster, then from your local machine:

```bash
ray down vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml
```

## Conversion Scripts

### Downloading a Hugging Face Dataset to S3

```bash
python vla_foundry/data/preprocessing/hf_utils/hf_dataset_downloader.py \
  --dataset IPEC-COMMUNITY/droid_lerobot \
  --mode s3 \
  --s3-output-path s3://tri-ml-datasets/hf_datasets/droid_lerobot \
  --local-output-dir /datasets/hf_datasets/droid_lerobot \
  --preserve-structure
```

### Converting VLM Hugging Face Captions to Tar Shards

This uses [img2dataset](https://github.com/rom1504/img2dataset) for image downloading and webdataset shard creation. The HF dataset must already be downloaded to S3 (see section above).

```bash
python vla_foundry/data/preprocessing/preprocess_captionshf_to_tar.py \
  --cluster ray \
  --input_path s3://tri-ml-datasets/vla_foundry_scratch/downloads/ \
  --output_path s3://tri-ml-datasets/vla_foundry_scratch/downloads2/ \
  --url_col images \
  --caption_col texts \
  --save_additional_columns metadata
```

### Converting Text Hugging Face Datasets to Tar Shards

```bash
python vla_foundry/data/preprocessing/preprocess_untokenized_to_tar.py \
  --s3_input_path s3://tri-ml-datasets/hf_datasets/fineweb-edu-350BT \
  --s3_output_path s3://tri-ml-datasets/lbm2_datasets/text/fineweb-edu-350BT \
  --tmp_dir /tmp/finewebshards
```

### Converting LeRobot to Tar Shards

The HF dataset must already be downloaded to S3.

```bash
source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
  --type "lerobot" \
  --source_episodes "['s3://tri-ml-datasets/hf_datasets/pi_libero/']" \
  --output_dir s3://tri-ml-datasets/vla_foundry_scratch/lerobotdata/pi_libero/ \
  --camera_names "['image', 'wrist_image']" \
  --samples_per_shard 100 \
  --config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml" \
  --observation_keys "['state']" \
  --action_keys "['actions']"
```

### Converting LBM Spartan Data to Tar Shards

```bash
python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
  --type "spartan" \
  --source_episodes "[
      's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-02T10-49-28-05-00/diffusion_spartan/',
      's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-02T14-21-19-05-00/diffusion_spartan/',
      's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-06T08-58-31-05-00/diffusion_spartan/',
      ]" \
  --output_dir s3://tri-ml-datasets/vla_foundry_scratch/spartan_datasets/bimanualputredbellpepperinbin3/ \
  --camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_4cameras.yaml" \
  --language_annotations_path vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml \
  --action_fields_config_path vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml \
  --data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
  --samples_per_shard 100 \
  --config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml"
```

!!! tip "Depth data"
    Set `--use_depth_data true` to also extract depth and point cloud data whenever depth information is available.

### Converting CAM MCAPs to Tar Shards

For robots such as the Unitree G1, the CAM TZK, or vendor-sourced UMI data, teleop/trainable data is available as ROS 2 MCAP files. Use the `--type mcap` argument and specify a `--action_fields_config_path` pointing to a YAML config file that lists topics.

Example for the Unitree G1:

```bash
python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
    --type mcap \
    --source_episodes <s3 or local/path/to/episodes> \
    --output_dir s3://<path_to_bucket>/ \
    --config_path vla_foundry/config_presets/data/unitree_g1/robotics_preprocessing_params_1past_47future_30hz.yaml \
    --action_fields_config_path vla_foundry/config_presets/data/unitree_g1/g1_mcap_topics.yaml \
    --camera_names "include vla_foundry/config_presets/data/unitree_g1/g1_data_camera_names.yaml" \
    --task_name "do_the_task"
```

A convenience wrapper script is also provided for the Unitree G1:

```bash
./examples/preprocessing/extended/preprocess_robotics_data_mcap_g1.sh \
    --source <s3:/src/s3/path> \
    --output <s3:/dest/s3/path> \
    --task-name <task name>
```

### Supported ROS 2 Message Types

The MCAP converter uses attribute inspection for message type detection, avoiding direct ROS 2 dependencies. The following message types are currently supported:

| Extraction Method | Message Type | Output |
|---|---|---|
| Structured | `sensor_msgs/JointState` | `__<joint_name>` per joint |
| Structured | `geometry_msgs/PoseStamped` | `__xyz`, `__rot_6d` |
| Structured | `geometry_msgs/Pose` | `__xyz`, `__rot_6d` |
| Flat | `sensor_msgs/Imu` | `[quat(4), angular_vel(3), linear_accel(3)]` |
| Flat | `geometry_msgs/WrenchStamped` | `[force(3), torque(3)]` |
| Flat | `geometry_msgs/Wrench` | `[force(3), torque(3)]` |
| Field Path | Custom messages | Config-driven via `field_extraction` in topics YAML |
| Image | `sensor_msgs/CompressedImage` | JPEG bytes (jpeg, png) |
| Image | `sensor_msgs/Image` | JPEG bytes (rgb8, bgr8, mono8) |

!!! info "Custom messages"
    For custom messages (such as Dex3 tactile, lowstate, or PolicyKeyframe), use the `field_extraction` config in the topics YAML to specify dot-separated field paths.
