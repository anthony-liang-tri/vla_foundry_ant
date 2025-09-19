The following README specifially discusses the file [preprocess_lbm_to_tar.py](preprocess_lbm_to_tar.py). For a more general README, please see the [README in the scripts folder](./../README.md).

# LBM Data Preprocessing

This folder containts python files to preprocess LBM data and convert them to various formats. In general, these scripts convert processed robotics demonstration data from S3 storage into WebDataset tar files or LeRobot formats for LBM2 training. This tool takes episodic robot data (images, actions, observations) and packages them into training-ready format with temporal sequences and language annotations. Note that this README and this folder is still under development and will be updated frequently with additional support and documentation being added. 

## What it does

This preprocessor transforms robotics episodes stored in S3 into structured training datasets by:

- **Creating temporal sequences (aka action chunks)**: Extracts past/future timesteps which can be flexibly defined
- **Multi-modal packaging**: Combines camera images, low-dimensional sensor data, and robot actions
- **Language annotation**: Adds task-specific natural language instructions from YAML files
- **Data filtering (Optional)**: Removes static/motionless samples and handles padding constraints
- **WebDataset format**: Outputs compressed tar shards compatible with LBM2 training pipeline
- **LeRobot format**: Outputs compressed tar shards compatible with LBM2 training pipeline

It works by running two sets of Ray parallel jobs.
1. First, it will process each file individually and create frames on s3. Each frame of each episode will have its own tar shard. These episodes will all be in the same bucket and be called episode\_{X}\_frame\_{Y}.tar. In terms of parallelism, each worker will be assigned one episode to process.
2. List out all tar shards generated in the previous step. Randomly shuffle and group together by shard\_size. Each worker is assigned one shard and will untar the component frames, combine the contents, then tar them back.

The `stats.json` is built up by the workers of Step 1, and the `manifest.jsonl` is built up by the workers of Step 2.

## Input Data Format

Expects processed episodes in S3 with this structure:
```
s3://bucket/path/to/episode/
└── processed/
    ├── metadata.yaml          # Episode metadata
    ├── observations.npz       # Multi-modal observations
    ├── actions.npz           # Robot actions (optional)
    ├── intrinsics.npz        # Camera calibration (optional)
    └── extrinsics.npz        # Camera extrinsics (optional)
```

## Output Format (LBM2 shards)

Generates WebDataset tar files:
```
output_directory/
├── episode_{X}_frame_{Y}.tar     # Intermediate tar shards
├── [...]
└── shards
    ├── shard_000000.tar          # Training data shards
    ├── shard_000001.tar
    ├── manifest.jsonl            # Shard index
    └── stats.json                # Dataset statistics
```

## Basic Usage

Note Ray doesn't work well with `uv run` right now. You can still use the uv requirements with with `source .venv/bin/activate`.

```bash
python lbm2/data/scripts/preprocessing/preprocess_lbm_to_tar.py \
    --source_episodes "['s3://robotics-manip-lbm/efs/data/tasks/PickAndPlaceBox/cabot/sim/bc/teleop/2025-02-11T17-04-00-05-00/']" \
    --output_dir s3://tri-ml-datasets-uw2/scratch/tmp/lbmpreprocess \
    --past_lowdim_steps 1 \
    --future_lowdim_steps 14 \
    --image_indices "[-1, 0]" \
    --stride 1 \
    --max_padding_left 3 \
    --max_padding_right 15 \
    --padding_strategy copy \
    --filter_still_samples False \
    --still_threshold 0.05 \
    --camera_discard_keys "include lbm2/config_presets/data/lbm_data_discard_key.yaml" \
    --camera_names "include lbm2/config_presets/data/lbm_data_camera_names.yaml" \
    --samples_per_shard 100 \
    --jpeg_quality 95 \
    --max_episodes_to_process -1 \
    --fail_on_nan True \
    --skip_git_tagging False \
    --resize_images_size "[224, 224]"
```

## Key Configuration

The full list of options and parameters for converting from *processed* LBM data to LBM2 shards can be found in `params.py`. 

### Temporal Windows
- `--past_lowdim_steps`: Past timesteps for low-dim data (default: 2)
- `--future_lowdim_steps`: Future timesteps for low-dim data (default: 14)  
- `--image_indices`: Image timestep offsets (default: [-2, 0])

### Data Processing
- `--resize_images_size`: Target image size [height, width] (default: [256, 342])
- `--padding_strategy`: How to pad sequences - "copy", "zero", "reflect" (default: "copy")
- `--filter_still_samples`: Remove static samples (default: True)
- `--samples_per_shard`: Samples per tar file (default: 1)

### Language Annotations

Language instructions are loaded from YAML:
```yaml
language_dict:
  TaskName:
    original:
      - "Pick up the box and place it on the table"
    randomized:
      - "Place the box on the table"
      - "Move the box to the table surface"
```

## Sample Structure

Each processed sample contains:
- **Images**: Multi-camera views at specified timesteps
- **Low-dimensional data**: Joint positions, poses, sensor readings in temporal sequences
- **Actions**: Robot action sequences
- **Language instructions**: Task descriptions in multiple formats
- **Metadata**: Timing, padding, and provenance information
- **Camera calibration**: Intrinsics/extrinsics for the sequence timespan