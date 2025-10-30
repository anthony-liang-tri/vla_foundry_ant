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

2. [Optional] Create the ray cluster
```bash
####################
# IMPORTANT: You may need to edit the following in ray_cluster_configs.yaml before running.
# - tags, username, number of nodes, etc.
# - file_mounts: It copies your HF token from ~/.cache/huggingface/token. Change this if it's somewhere else.
# - rsync_exclude: It currently excludes rsyncing `.venv` and `wandb`. Add here if there are other paths you want to exclude (e.g. large checkpoints).
####################
ray up vla_foundry/data/scripts/ray_cluster_configs.yaml
```

3. [Optional] Attach the ray cluster. This will take you "inside" the cluster.
```bash
ray attach vla_foundry/data/scripts/ray_cluster_configs.yaml
```

4. Run your script inside the cluster. Note that ray scripts currently do **not** work well with `uv run`. The requirements can still be used with `uv sync --group=preprocessing` (automatically done in `ray_cluster_configs.yaml`) and `source .venv/bin/activate`.
```bash
# [optional] Start a persistent terminal like tmux
cd vla_foundry
python (some-script-here)
```

5. When finished, exit the cluster. Then, from your own machine, shut down the ray cluster with `ray down`.
```bash
ray down vla_foundry/data/scripts/ray_cluster_configs.yaml
```

# Downloading a Hugging Face dataset to S3
```bash
python vla_foundry/data/scripts/hf_dataset_downloader.py --dataset IPEC-COMMUNITY/droid_lerobot --mode s3 --s3-output-path s3://tri-ml-datasets/hf_datasets/droid_lerobot --local-output-dir /datasets/hf_datasets/droid_lerobot --preserve-structure
```

# Converting VLM Hugging Face captions to tar shards

We use [img2dataset](https://github.com/rom1504/img2dataset) to handle image downloading and webdataset shard creation. 

This assumes that the HF dataset is already downloaded to S3 (see above section).

```bash
python vla_foundry/data/scripts/preprocessing/preprocess_captionshf_to_tar.py --cluster ray --input_path s3://tri-ml-datasets/scratch/sedrick.keh/downloads/ --output_path s3://tri-ml-datasets/scratch/sedrick.keh/downloads2/ --url_col images --caption_col texts --save_additional_columns metadata
```

# Converting a text Hugging Face dataset to tar shards
```bash
python vla_foundry/data/scripts/preprocessing/preprocess_untokenized_to_tar.py --s3_input_path s3://tri-ml-datasets/hf_datasets/fineweb-edu-350BT --s3_output_path s3://tri-ml-datasets/lbm2_datasets/text/fineweb-edu-350BT --tmp_dir /tmp/finewebshards
```

# Converting LeRobot to tar shards
This assumes that the HF dataset is already downloaded to S3.

```bash
python vla_foundry/data/scripts/preprocessing/preprocess_lerobot_to_tar.py --dataset_path s3://tri-ml-datasets/hf_datasets/oxe_lerobot/droid_lerobot --s3_output_path s3://tri-ml-datasets/lbm2_datasets/droid_lerobot --tmp_dir /tmp --shard_size 2048
```

# Converting LBM Spartan data to tar shards
```bash
python vla_foundry/data/scripts/preprocessing/preprocess_lbm_to_tar.py \
    --source_episodes "['s3://robotics-manip-lbm/efs/data/tasks/PickAndPlaceBox/cabot/sim/bc/teleop/2025-02-11T17-04-00-05-00/']" \
    --output_dir s3://tri-ml-datasets-uw2/preprocess_lbm_test/lbm/PickAndPlaceBox/cabot/sim/ \
    --language_annotations_path vla_foundry/config_presets/data/lbm_language_annotations.yaml \
    --camera_discard_keys "include vla_foundry/config_presets/data/lbm_data_discard_key.yaml" \
    --camera_names "include vla_foundry/config_presets/data/lbm_data_camera_names.yaml" \
    --past_lowdim_steps 1 \
    --future_lowdim_steps 14 \
    --image_indices "[-1, 0]" \
    --max_padding_left 1 \
    --max_padding_right 15 \
    --samples_per_shard 1 \
    --max_episodes_to_process 5 \
    --jpeg_quality 95 \
    --filter_still_samples False \
    --no_statistics False \
    --still_threshold 0.05 \
    --resize_images_size "[224, 224]"
```