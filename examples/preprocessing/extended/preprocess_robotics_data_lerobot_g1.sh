# NOTE: Needs the StackOxoContainers task dataset to be updated to 
# include image data, does not work at the moment!
uv run --group preprocessing vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "lerobot" \
--source_episodes "['s3://robotics-cam-data/platform/unitree_g1_dex3/lerobot/StackOxoContainers/real/teleop']" \
--output_dir "s3://robotics-cam-data/platform/unitree_g1_dex3/tarfile/v0/StackOxoContainers/real/teleop/" \
--output_dir_fixed_path "s3://robotics-cam-data/platform/unitree_g1_dex3/dataset_fixed/" \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_47future_30hz.yaml" \
--camera_names "['observation.images.cam_left_high', 'observation.images.cam_right_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']" \
--observation_keys "['observation.state']" \
--action_keys "['actions']" \
--samples_per_shard 100 \
--ray_address local \
--ray_num_cpus 32
