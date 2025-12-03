# Visualization scripts

## Testing the dataloader:
To look at the first few samples of a dataloader, run the script below:
```
python vla_foundry/data/scripts/vis/dataloader_vis.py (--arguments-here)
```

This uses the same argument parsing `draccus.parse(config_class=TrainExperimentParams)` as `main.py`, so you can take any script in `./examples` and copy paste the arguments exactly. It will load the dataloader the exact same way that `main.py` loads it.

## MMT visualizer:
To look at episodes in shards of MMT data. This script will visualize the following:
- RGB image
- Depth image
- RGB pointcloud
- left end effector action, right end effector action, lift action
- Camera frame, chest frame, left and right end effector frames.

Run the script as follows
```
VISUALIZER=<visualizer-backend> python mmt_vis.py --s3_path <s3-path-to-shards> --data_params <path-to-data-params-yaml>
```

Example command for rerun backend
```
VISUALIZER=rerun python mmt_vis.py --s3_path s3://tri-mmt-data/lpp_data/vla_foundry/20251028_paper_towel/shards/ \
        --data_params vla_foundry/config_presets/data/mmt/mmt_data_params.yaml
        
```