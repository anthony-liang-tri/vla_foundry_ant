# Dependencies
The scripts in this folder use certain preprocessing-specific dependencies, which we have isolated from the training code. 

To run these scripts locally, use
```
uv sync --groups=preprocessing
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
python vla_foundry/data/preprocessing/preprocess_captionshf_to_tar.py --cluster ray --input_path s3://tri-ml-datasets/scratch/sedrick.keh/downloads/ --output_path s3://tri-ml-datasets/scratch/sedrick.keh/downloads2/ --url_col images --caption_col texts --save_additional_columns metadata
```

# Converting a text Hugging Face dataset to tar shards
```bash
python vla_foundry/data/preprocessing/preprocess_untokenized_to_tar.py --s3_input_path s3://tri-ml-datasets/hf_datasets/fineweb-edu-350BT --s3_output_path s3://tri-ml-datasets/lbm2_datasets/text/fineweb-edu-350BT --tmp_dir /tmp/finewebshards
```

# Converting LeRobot to tar shards
This assumes that the HF dataset is already downloaded to S3.

```bash
source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--source_type "lerobot" \
--source_episodes "['s3://tri-ml-datasets/hf_datasets/pi_libero/']" \
--output_dir s3://tri-ml-datasets/scratch/sedrick.keh/tmp/lerobotdata/pi_libero/ \
--camera_names "['image', 'wrist_image']" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml"
```

# Converting LBM Spartan data to tar shards
```bash
python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--source_type "spartan" \
--source_episodes "[
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-02T10-49-28-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-02T14-21-19-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-06T08-58-31-05-00/diffusion_spartan/',
    ]" \
--output_dir s3://tri-ml-datasets/scratch/sedrick.keh/tmp/lbmdata/bimanualputredbellpepperinbin3/ \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_4cameras.yaml" \
--language_annotations_path vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml \
--action_fields_config_path vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml" \
```

# Converting MMT NPZ data to tar shards
Data ripped from MMT robots is stored as npz files, named as `ep\d{4}_t\d{4}\.npz`. Each npz file store all data from one time step.
```
<AWS_PROFILE=your_profile> uv run vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--source_episodes "[
    # Local path also supported
    's3://tri-mmt-data/lpp_data/20251028_paper_towel/npz_head/'
    ]" \
--output_dir s3://tri-mmt-data/lpp_data/vla_foundry/20251028_paper_towel \
--samples_per_shard 100
--config_path "vla_foundry/config_presets/data/mmt_preprocessing_params_1past_14future.yaml"
```