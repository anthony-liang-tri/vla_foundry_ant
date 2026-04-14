# Preprocessing Examples

This page walks through the data preprocessing scripts that convert raw datasets into the WebDataset tar shard format used by VLA Foundry at training time. Most robotics preprocessing is handled by a single entrypoint (`preprocess_robotics_to_tar.py`) with a `--type` flag that selects the source format.

---

## HF Dataset Download to S3

Download a Hugging Face dataset to S3 using Ray for parallel transfers.

**Script:** `vla_foundry/data/preprocessing/hf_utils/hf_dataset_downloader.py`

```bash
source .venv/bin/activate && python \
    vla_foundry/data/preprocessing/hf_utils/hf_dataset_downloader.py \
    --repo_id <hf-dataset-repo-id> \               # (1)!
    --output_dir s3://your-bucket/hf_datasets/      # (2)!
```

1. The Hugging Face dataset repository ID (e.g., `lerobot/aloha_sim_insertion_human`).
2. S3 or local path where the downloaded dataset files are stored.

This utility downloads all files from a HF dataset repo and optionally uploads them to S3 in parallel using Ray workers. It is typically a first step before converting HF data into tar shards.

---

## VLM HF Captions to Tar Shards (img2dataset)

Convert a Hugging Face image-caption dataset (in Parquet format) into WebDataset tar shards using `img2dataset`.

**Script:** `vla_foundry/data/preprocessing/preprocess_captionshf_to_tar.py`

```bash
source .venv/bin/activate && python \
    vla_foundry/data/preprocessing/preprocess_captionshf_to_tar.py \
    --cluster local \                               # (1)!
    --input_path s3://your-bucket/hf_dataset/ \     # (2)!
    --output_path s3://your-bucket/shards/ \        # (3)!
    --url_col url \                                 # (4)!
    --caption_col re_caption                        # (5)!
```

1. Run locally (`local`) or distributed on a Ray cluster (`ray`).
2. S3 or local path to the Parquet files containing image URLs and captions.
3. Output path for the WebDataset tar shards.
4. Column name in the Parquet file that contains the image URLs.
5. Column name that contains the caption text.

Under the hood, this uses the `img2dataset` library to download images, resize them (keeping aspect ratio, max 512px), and encode them as WebP files in WebDataset shards of 512 samples each.

---

## Text HF to Tar Shards

Convert Parquet-format text datasets from S3 into WebDataset tar shards with JSON files.

**Script:** `vla_foundry/data/preprocessing/preprocess_text_untokenized_to_tar.py`

```bash
source .venv/bin/activate && python \
    vla_foundry/data/preprocessing/preprocess_text_untokenized_to_tar.py \
    --input_path s3://your-bucket/text_parquets/ \  # (1)!
    --output_path s3://your-bucket/text_shards/ \   # (2)!
    --samples_per_shard 512                         # (3)!
```

1. S3 path to the directory containing Parquet files with raw text data.
2. Output path for the tar shards. A `manifest.jsonl` is automatically created.
3. Number of text samples per tar shard.

Each Parquet row becomes a JSON file inside the tar shard, identified by a UUID filename. This format is consumed by the `text_untokenized` data type in VLA Foundry, which tokenizes on-the-fly during training.

---

## LeRobot to Tar Shards

Convert a LeRobot-format dataset into WebDataset tar shards for robotics training.

**Source:** `examples/preprocessing/preprocess_robotics_data_lerobot.sh`

```bash
source .venv/bin/activate && python \
    vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
    --type "lerobot" \                              # (1)!
    --source_episodes "['s3://your-bucket/hf_datasets/pi_libero/']" \ # (2)!
    --output_dir s3://your-bucket/vla_foundry_datasets/lerobot/pi_libero/ \ # (3)!
    --camera_names "['image', 'wrist_image']" \     # (4)!
    --samples_per_shard 100 \                       # (5)!
    --config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml" \ # (6)!
    --observation_keys "['state']" \                # (7)!
    --action_keys "['actions']"                     # (8)!
```

1. Source format type -- `lerobot` for LeRobot HF datasets.
2. List of S3 paths to LeRobot episode directories.
3. Output directory for the tar shards, stats, and manifest.
4. Camera names to extract from the dataset (must match the keys in the LeRobot dataset).
5. Number of trajectory samples per tar shard file.
6. Preprocessing params YAML that specifies the number of past/future timesteps, chunk sizes, etc.
7. Observation keys to extract (e.g., proprioceptive state).
8. Action keys to extract from the dataset.

---

## LBM Spartan to Tar Shards

Convert LBM Spartan-format simulation data into WebDataset tar shards.

**Source:** `examples/preprocessing/preprocess_robotics_data_lbm.sh`

```bash
source .venv/bin/activate && python \
    vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
    --type "spartan" \                              # (1)!
    --source_episodes "[
        's3://...diffusion_spartan/',
        's3://...diffusion_spartan/',
    ]" \                                            # (2)!
    --output_dir "s3://your-bucket/spartan_datasets/task_name/" \ # (3)!
    --camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_4cameras.yaml" \ # (4)!
    --language_annotations_path "vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml" \ # (5)!
    --action_fields_config_path "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml" \ # (6)!
    --data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \ # (7)!
    --samples_per_shard 100 \
    --config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml"
```

1. Source format type -- `spartan` for LBM Spartan simulation data.
2. List of S3 paths to Spartan episode directories. Multiple episodes can be listed.
3. Output directory for the processed tar shards.
4. Camera names loaded from a YAML config preset using the `include` syntax.
5. Path to a YAML file mapping tasks to language instructions.
6. Path to a YAML file specifying which action fields to extract.
7. Keys to discard from the raw data during preprocessing.

---

## CAM MCAPs to Tar Shards

Convert ROS 2 MCAP recordings (e.g., from the Unitree G1 humanoid) into WebDataset tar shards.

**Source:** `examples/preprocessing/extended/preprocess_robotics_data_mcap_g1.sh`

```bash
./examples/preprocessing/extended/preprocess_robotics_data_mcap_g1.sh \
    --source s3://your-bucket/mcap/stack_cubes_ordered/real/teleop/ \ # (1)!
    --output s3://your-bucket/tarfile/v1/stack_cubes_ordered/real/teleop/ \ # (2)!
    --task-name "stack cubes ordered" \             # (3)!
    --max-episodes 5 \                              # (4)!
    --ray-cpus 32                                   # (5)!
```

1. S3 or local path containing the MCAP episode recordings.
2. Output path for the generated tar shards.
3. Natural-language task name used as the language instruction annotation.
4. Maximum number of episodes to process (`-1` for all).
5. Number of Ray CPUs for parallel processing.

Under the hood, this wrapper script calls:

```bash
uv run --group preprocessing python \
    vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
    --type mcap \
    --source_episodes "['s3://...']" \
    --output_dir s3://... \
    --config_path vla_foundry/config_presets/data/robotics_preprocessing_params_1past_47future_30hz.yaml \
    --action_fields_config_path vla_foundry/config_presets/data/unitree_g1/g1_mcap_topics.yaml \
    --camera_names "include vla_foundry/config_presets/data/unitree_g1/g1_data_camera_names.yaml" \
    --ray_address local \
    --ray_num_cpus 32
```

!!! tip "Dry run"
    Use the `--dry-run` flag to print the full command without executing it, which is useful for verifying paths before a long preprocessing run.
