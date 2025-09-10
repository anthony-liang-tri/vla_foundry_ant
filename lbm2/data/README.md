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
├── shard_000000.tar          # Training data shards
├── shard_000001.tar
├── manifest.jsonl            # Shard index
├── dataset_statistics.json   # Dataset statistics
└── processing_metadata.json  # Processing provenance
```

## Basic Usage

```bash
python preprocess_lbm_data_optimized.py \
        --source_episodes \
          "[s3://robotics-manip-lbm/efs/data/tasks/PickAndPlaceBox/cabot/sim/bc/teleop/2025-02-11T17-04-00-05-00/,]" \
        --output_dir s3://my-bucket/processed-dataset/ \
        --language_annotations_path lbm2/data/preprocessing/lbm_language_annotations.yaml \
        --past_lowdim_steps 2 \
        --future_lowdim_steps 14 \
        --image_indices -2,0 \
        --max_padding_left 3 \
        --max_padding_right 16 \
        --padding_strategy copy \
        --filter_still_samples True \
        --still_threshold 0.01 \
        --samples_per_shard 1000 \
        --stride 1 \
        --jpeg_quality 95 \
        --num_workers 16 \
        --no_statistics True \
        --resize_images_size 256, 342 \
        --use_gpu_resize True \
        --resume False
```

## Key Configuration

### Temporal Windows
- `--past_lowdim_steps`: Past timesteps for low-dim data (default: 2)
- `--future_lowdim_steps`: Future timesteps for low-dim data (default: 14)  
- `--image_indices`: Image timestep offsets (default: [-2, 0])

### Data Processing
- `--resize_images_size`: Target image size [height, width] (default: [256, 342])
- `--padding_strategy`: How to pad sequences - "copy", "zero", "reflect" (default: "copy")
- `--filter_still_samples`: Remove static samples (default: True)
- `--samples_per_shard`: Samples per tar file (default: 1000)

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

## Resume and Recovery

The tool supports resuming interrupted processing:
```bash
python preprocess_lbm_data_optimized.py \
    --resume True \
    --enable_incremental_updates True \
    [other options]
```

Processing metadata includes git commit tracking and full reproducibility information.