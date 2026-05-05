# NOTE: Please bump up the --output_dir version in tarfiles if updating datasets
# or alternatively use the `tarfiles/scratch` directory if testing
uv run --group preprocessing vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "mcap" \
--source_episodes "['s3://robotics-cam-data/platform/unitree_g1_dex3/mcap']" \
--output_dir "s3://robotics-cam-data/platform/unitree_g1_dex3/tarfiles/v3.2/" \
--output_dir_fixed_path "s3://robotics-cam-data/platform/unitree_g1_dex3/dataset_fixed/" \
--config_path "vla_foundry/config_presets/data/unitree_g1/g1_preprocessing_params_1past_47future_30hz.yaml" \
--action_fields_config_path "vla_foundry/config_presets/data/unitree_g1/g1_action_fields.yaml" \
--topics_to_fields_path "vla_foundry/config_presets/data/unitree_g1/g1_mcap_topics.yaml" \
--camera_names "include vla_foundry/config_presets/data/unitree_g1/g1_data_camera_names_zed2mini.yaml" \
--samples_per_shard 100 \
--ray_num_cpus 32 \
--task_filter '["move_block_on_plate"]' \
--domain_filter '["real"]' \
--source_filter '["teleop"]'
