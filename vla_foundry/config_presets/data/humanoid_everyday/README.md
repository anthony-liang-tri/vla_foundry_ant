# HumanoidEveryday Dataset Converter and Training Config

End-to-end support for the **HumanoidEveryday** dataset (`s3://tri-ml-sandbox-16011-us-west-2-datasets/cv_downloaded/HumanoidEveryday/`) — from raw data conversion to webdataset format, through to training a diffusion policy. Supports **both G1 and H1 embodiments** with per-embodiment state/action dimensions and G1-only tactile sensor and odometry fields.

## Changes Overview

### New files
- **`vla_foundry/data/preprocessing/robotics/converters/humanoid_everyday.py`** — The main converter class (`HumanoidEverydayConverter`) that ingests the HumanoidEveryday raw format and outputs webdataset samples. Handles both G1 and H1 embodiments with dimension-aware lowdim extraction.
- **`vla_foundry/config_presets/data/humanoid_everyday/humanoid_everyday_data_params.yaml`** — Data parameters for training (camera names, proprioception/action fields, normalization).
- **`vla_foundry/config_presets/data/humanoid_everyday/humanoid_everyday_preprocessing_params.yaml`** — Preprocessing parameters for data conversion (image size, depth resolution, windowing, etc.).
- **`vla_foundry/config_presets/training_jobs/diffusion_policy_humanoid_everyday.yaml`** — Example training job config for diffusion policy on this dataset.
- **`tests/essential/data/test_humanoid_everyday_converter.py`** — Unit tests covering episode discovery (task/episode/embodiment filters) and end-to-end conversion smoke test with lowdim shape validation per embodiment.
- **`tests/essential/test_assets/small_truncated_humanoid_everyday_dataset/`** — Truncated test data (3 frames each) for both a G1 episode (`drag_a_white_board`) and an H1 episode (`pick_a_bag_of_fork_and_place_it_in_a_container`).

### Modified files
- **`vla_foundry/data/preprocessing/robotics/converters/__init__.py`** — Registers `HumanoidEverydayConverter` in the converter factory.
- **`vla_foundry/data/preprocessing/robotics/preprocess_params.py`** — Adds `HumanoidEverydayPreprocessParams` with `use_depth_data`, `resize_images_size`, `depth_resolution` (required, set via YAML config), and optional `task_filter`, `episode_filter`, `embodiment_filter` fields.
- **`vla_foundry/data/preprocessing/utils.py`** — Fixes local filesystem path bugs in `create_shard` and `create_episode_shard` (details below).

## Original Data Structure (HumanoidEveryday)

The raw dataset at `/data/HumanoidEveryday` is organized as:

```
/data/HumanoidEveryday/
├── <task_name>/                          # e.g., drag_a_white_board
│   ├── metadata/
│   │   └── metadata.json                 # Task-level language annotations (title, description)
│   ├── episode_0/
│   │   ├── data.json                     # Per-frame states, actions, and file references
│   │   ├── color/
│   │   │   ├── frame_000000.jpg          # Head-mounted RGB camera (varying resolution)
│   │   ├── depth/
│   │   │   ├── frame_000000.npy.lzma     # LZMA-compressed raw uint16 depth (640x480, mm)
│   │   └── lidar/                        # Point clouds (.pcd) — not used
│   ├── episode_1/
│   │   └── ...
```

Each frame in `data.json` contains:
- **States (both embodiments)**:
  - `arm_state` (14D), `imu` (quaternion 4D, accelerometer 3D, gyroscope 3D, RPY 3D)
  - `leg_state` (G1: 15D, H1: 13D), `hand_state` (G1: 14D, H1: 12D)
- **States (G1-only)**:
  - `hand_pressure_state` — tactile sensors: type-A (48D) and type-B (18D)
  - `odometry` — position (3D), velocity (3D), RPY (3D), quaternion (4D)
- **Actions (both embodiments)**: `sol_q` (14D), `tau_ff` (14D), `head_rmat` (9D), `left_pose` (16D), `right_pose` (16D)
  - `left_angles` / `right_angles`: G1=7D, H1=12D
- **Images**: RGB `.jpg` + depth `.npy.lzma`

## Converted Data Structure (WebDataset)

After conversion, the output follows the standard VLA Foundry webdataset format:

```
/data/webdataset/
├── shards/
│   ├── shard_000000.tar                  # Shuffled training shards
│   ├── manifest.jsonl                    # Shard manifest for dataloader
│   ├── stats.json                        # Dataset statistics for normalization
│   └── preprocessing_config.yaml         # Auto-detected camera names and image indices
├── frames/                               # Intermediate per-frame tars (can be cleaned up)
├── episodes/                             # Episode-ordered shards (for evaluation)
```

Each sample within a shard contains:
- `{uuid}.head_rgb_t-1.jpg` / `{uuid}.head_rgb_t0.jpg` — Previous + current frame RGB
- `{uuid}.head_rgb_depth_t-1.png` / `{uuid}.head_rgb_depth_t0.png` — Previous + current depth (uint16 PNG)
- `{uuid}.lowdim.npz` — State + action fields, windowed (1 past + 1 current + 14 future timesteps). G1-only fields (tactile sensors, odometry) are included only for G1 episodes.
- `{uuid}.metadata.json` — Sample metadata (camera names, episode length, task name, etc.)
- `{uuid}.language_instructions.json` — Task-level language instructions (original + alternative)

## Data Conversion Parameters

Run conversion with:
```bash
uv run python -m vla_foundry.data.preprocessing.preprocess_robotics_to_tar \
    --type humanoid_everyday \
    --config_path vla_foundry/config_presets/data/humanoid_everyday/humanoid_everyday_preprocessing_params.yaml \
    --source_episodes '["/data/HumanoidEveryday"]' \
    --output_dir /data/webdataset \
    --task_filter '["drag_a_white_board"]'
```

Key parameters in `HumanoidEverydayPreprocessParams`:

| Parameter | Default | Description |
|---|---|---|
| `type` | — | Must be `"humanoid_everyday"` |
| `source_episodes` | — | Path(s) to the raw dataset root |
| `output_dir` | — | Output directory for webdataset |
| `use_depth_data` | — | Whether to include depth images and point clouds (set via YAML config) |
| `resize_images_size` | — | Target image size `[H, W]` (set via YAML config, e.g. `[384, 384]`) |
| `depth_resolution` | — | Raw depth sensor resolution `[H, W]` (set via YAML config, e.g. `[480, 640]`) |
| `task_filter` | `None` | List of task names to process (e.g., `["drag_a_white_board"]`). `None` = all |
| `episode_filter` | `None` | List of episode names to process (e.g., `["episode_0"]`). `None` = all |
| `embodiment_filter` | `None` | List of robot types to process (e.g., `["g1"]`, `["h1"]`, or `["g1", "h1"]`). `None` = all |
| `samples_per_shard` | `128` | Number of samples per training shard |

## Training Parameters

Run training with:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m vla_foundry.main \
    --config_path vla_foundry/config_presets/training_jobs/diffusion_policy_humanoid_everyday.yaml \
    --wandb false \
    --save_path /data/training_output/humanoid_everyday_test
```

The example training config uses:

| Parameter | Value | Notes |
|---|---|---|
| Model | DiffusionPolicy (100M transformer) | CLIP ViT-B/32 for vision+language |
| Camera | `head_rgb` (t-1, t0) | Depth not yet supported by training pipeline |
| Proprioception | 69D for G1 (11 state fields incl. odometry) | Arm 14D, leg 15D, hand 14D, IMU 13D, odometry 13D. Dimensions differ for H1. |
| Actions | 32D (end-effector poses) | Left + right 4x4 homogeneous transforms, flattened |
| Batch size | 4 | Small for testing; increase for real runs |
| Training samples | 1000 | Small for testing |
| Normalization | percentile_1_99, centered | Applied globally to all proprioception and action fields |

## Bug Fixes in `utils.py`

Two pre-existing bugs fixed in the sharding utilities:

1. **`create_shard`: wrong directory name** — Was reading intermediate frame tars from `episodes/` instead of `frames/`. Phase 1 (`upload_sample_to_s3`) writes to `{output_dir}/frames/`, but `create_shard` was looking in `{output_dir}/episodes/`. This affects both S3 and local paths.

2. **`create_episode_shard`: no local filesystem support** — Was hardcoded for S3 only (unconditionally creating a boto3 client and parsing `output_dir` as `s3://bucket/prefix`). When given a local path, it produced an "Invalid bucket name" error. Added `is_s3` branching to support both S3 and local paths.

## Known Limitations

- **Depth images are converted but not used during training.** The training pipeline (`robotics.py`) only loads `.jpg` files, so depth `.png` files are stored in the webdataset but ignored. Pipeline changes needed to use depth as input.
- **Lidar data (`.pcd`) is not converted.** The converter skips point cloud data entirely.
- **Tactile sensor data is converted but not in the default training config.** The `hand_pressure_a` (48D) and `hand_pressure_b` (18D) fields are stored in the lowdim NPZ for G1 episodes but not included in the default `proprioception_fields` training config. Add them to the YAML to use as training input.
- **All conversion happens locally.** We need to manually download and update the data as needed. It's not hard to support cloud, but the S3 util lib is missing and some housekeeping is required beforehand; thus, saved for future improvement.
- **Lack of centralized key mapping mechanism.** Since there are multiple embodiments in this dataset, keys could be different. However, how to do a key-mapping for multiple embodiments across multiple datasets is still TBD. So we simply dump whatever key we have in the dataset for now and assume training only needs one embodiment source for now.
