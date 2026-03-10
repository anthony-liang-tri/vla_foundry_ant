# Dependencies
The scripts in this folder use certain preprocessing-specific dependencies, which we have isolated from the training code. 

To run these scripts locally, use
```
uv sync --group=preprocessing
```

# Using Ray
Many scripts use Ray for parallelization.

The general flow is as follows: (1) Create your script that's compatible with Ray. (2) Start a Ray instance on AWS clusters with `ray_cluster_configs.yaml`. This will start a head node with several worker instances. (3) From within the Ray cluster, run your script.

Alternatively, Ray also works on local instances, so Steps 2 and 3 can be skipped. 

1. Start ray (if not already running)
```bash
ray start --head
```

Add the --include-dashboard=True arg before the --head to include the ray dashboard for diagnostics.

2. [Optional] Create the ray cluster
```bash
####################
# IMPORTANT: You may need to edit the following in ray_cluster_configs.yaml before running.
# - tags, username, number of nodes, etc.
# - file_mounts: It copies your HF token from ~/.cache/huggingface/token. Change this if it's somewhere else.
# - rsync_exclude: It currently excludes rsyncing `.venv` and `wandb`. Add here if there are other paths you want to exclude (e.g. large checkpoints).
####################
ray up vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml
```

3. [Optional] Attach the ray cluster. This will take you "inside" the cluster.
```bash
ray attach vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml
```

4. Run your script inside the cluster. Note that ray scripts currently do **not** work well with `uv run`. The requirements can still be used with `uv sync --group=preprocessing` (automatically done in `ray_cluster_configs.yaml`) and `source .venv/bin/activate`.
```bash
# [optional] Start a persistent terminal like tmux
cd vla_foundry
python (some-script-here)
```

5. When finished, exit the cluster. Then, from your own machine, shut down the ray cluster with `ray down`.
```bash
ray down vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml
```

# Downloading a Hugging Face dataset to S3
```bash
python vla_foundry/data/preprocessing/hf_utils/hf_dataset_downloader.py --dataset IPEC-COMMUNITY/droid_lerobot --mode s3 --s3-output-path s3://tri-ml-datasets/hf_datasets/droid_lerobot --local-output-dir /datasets/hf_datasets/droid_lerobot --preserve-structure
```

# Converting VLM Hugging Face captions to tar shards

We use [img2dataset](https://github.com/rom1504/img2dataset) to handle image downloading and webdataset shard creation. 

This assumes that the HF dataset is already downloaded to S3 (see above section).

```bash
python vla_foundry/data/preprocessing/preprocess_captionshf_to_tar.py --cluster ray --input_path s3://tri-ml-datasets/vla_foundry_scratch/downloads/ --output_path s3://tri-ml-datasets/vla_foundry_scratch/downloads2/ --url_col images --caption_col texts --save_additional_columns metadata
```

# Converting a text Hugging Face dataset to tar shards
```bash
python vla_foundry/data/preprocessing/preprocess_untokenized_to_tar.py --s3_input_path s3://tri-ml-datasets/hf_datasets/fineweb-edu-350BT --s3_output_path s3://tri-ml-datasets/lbm2_datasets/text/fineweb-edu-350BT --tmp_dir /tmp/finewebshards
```

# Converting LeRobot to tar shards
This assumes that the HF dataset is already downloaded to S3.

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

# Converting LBM Spartan data to tar shards
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

You can set `--use_depth_data true` if you also want to extract depth and point cloud data whenever depth information is available.

# Converting MMT NPZ data to tar shards
Data ripped from MMT robots is stored as npz files, named as `ep\d{4}_t\d{4}\.npz`. Each npz file store all data from one time step.
```
# If you are stuck at `rpc_client.h:153: Failed to connect to GCS at address` error you might have stale Ray cluster state
# Remove /tmp/ray to remove the stale pointer
<AWS_PROFILE=your_profile> uv run --group=preprocessing vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type mmt_npz \
--source_episodes "[
    # Local path also supported
    's3://tri-mmt-data/lpp_data/20251028_paper_towel/npz_head/'
    ]" \
--output_dir s3://tri-mmt-data/lpp_data/vla_foundry/20251028_paper_towel \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/mmt_preprocessing_params_1past_14future.yaml"
```
`--output_dir` can be either S3 or a local filesystem path.
For `--type mmt_npz`, local output paths sanitize episode IDs for filesystem safety. S3 output keeps legacy naming.

# Converting CAM mcaps to tar shards
For robots such as the Unitree G1, the CAM TZK, or vendor-sourced UMI data, teleop/trainable data is available as ROS 2 MCAPs. To convert this data, use `--type mcap` and point to the relevant config files for topics, action fields, and language annotations. You can optionally filter by task, domain (`sim`/`real`), source (`teleop`/`filtered`), and episode index. The output directory will automatically include subdirectories reflecting the active filters (**NOTE** that converting more than one task produces an output directory called `multitask`).

Example Unitree G1 conversion:
```
uv run --group preprocessing vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
    --type mcap \
    --source_episodes <s3 or local/path/to/episodes> \
    --output_dir <s3 or local/path/to/output> \
    --output_dir_fixed_path <s3 path to fixed dataset bucket> \
    --config_path vla_foundry/config_presets/data/g1_preprocessing_params_1past_47future_30hz.yaml \
    --action_fields_config_path vla_foundry/config_presets/data/unitree_g1/g1_action_fields.yaml \
    --topics_to_fields_path vla_foundry/config_presets/data/unitree_g1/g1_mcap_topics.yaml \
    --camera_names "include vla_foundry/config_presets/data/unitree_g1/g1_data_camera_names.yaml" \
    --samples_per_shard 100 \
    --task_filter '["stack_cubes_ordered"]' \
    --domain_filter '["sim"]' \
    --source_filter '["teleop"]'
```
A ready-to-run example script that preprocesses Unitree G1 Dex3 teleop data for the `move_block_on_plate` sim task at 30 Hz (1 past + 47 future steps) is available at:
```
./examples/preprocessing/extended/preprocess_robotics_data_mcap_g1.sh
```

**NOTE**: Running this converter locally can be memory-intensive on your machine, so considering tuning the number of Ray workers (`--ray_num_cpus`) or defaulting to an EC2 instance.

##### Supported ROS 2 Message Types

The MCAP converter uses attribute inspection for message type detection (avoiding direct ROS 2 dependencies). The following message types are currently supported:

| Extraction Method | Message Type | Output |
|-------------------|--------------|--------|
| Structured | `sensor_msgs/JointState` | `__<joint_name>` per joint |
| Structured | `geometry_msgs/PoseStamped` | `__xyz`, `__rot_6d` |
| Structured | `geometry_msgs/Pose` | `__xyz`, `__rot_6d` |
| Flat | `sensor_msgs/Imu` | `[quat(4), angular_vel(3), linear_accel(3)]` |
| Flat | `geometry_msgs/WrenchStamped` | `[force(3), torque(3)]` |
| Flat | `geometry_msgs/Wrench` | `[force(3), torque(3)]` |
| Field Path | Custom messages | Config-driven via `field_extraction` in topics YAML |
| Image | `sensor_msgs/CompressedImage` | JPEG bytes (jpeg, png) |
| Image | `sensor_msgs/Image` | JPEG bytes (rgb8, bgr8, mono8) |

For custom messages (Dex3 tactile, lowstate, PolicyKeyframe), use `field_extraction` config in the topics YAML to specify dot-separated field paths.
