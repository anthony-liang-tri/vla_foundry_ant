## Robotics Dataset Structure (Bare Minimum Requirements)

**S3 Path:**
```
dataset_directory_in_s3/
├── manifest.jsonl
├── stats.json
├── shard_00000.tar
├── shard_00001.tar
└── ...
```
- `stats.json` should be a dict with keys corresponding to camera names, states, observations, etc. (basically all tensors). These should themselves contain dicts with keys `mean`, `std`, etc.


**Inside each shard (e.g., shard_00000.tar):**
```
├── (unique_id_1).lowdim.npz
├── (unique_id_1).camera_name_{1,2,...n}.jpg
├── (unique_id_1).language_instructions.json
├── (unique_id_1).metadata.json
└── ...
```
- `language_instructions.json` should be a dict with keys in set ["original", "randomized", "verbose", "alternative"]
- np arrays in `lowdim.npz` should exist in `stats.json`