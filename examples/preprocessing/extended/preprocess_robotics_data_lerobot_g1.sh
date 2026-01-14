source .venv/bin/activate && python vla_foundry/data/scripts/preprocessing/preprocess_robotics_to_tar.py \
--source_type "lerobot" \
--source_episodes "['s3://tri-ml-datasets/vla_foundry_scratch/tmp/aykut/humanoid/unitree_g1_dex3/real/converted/StackOxoContainers/']" \
--output_dir s3://tri-ml-datasets/vla_foundry_scratch/tmp/aykut/StackOxoContainers \
--camera_names "include vla_foundry/config_presets/data/unitree_g1/g1_data_camera_names.yaml" \
--language_annotations_path vla_foundry/config_presets/data/unitree_g1/g1_language_annotations.yaml \
--action_fields_config_path vla_foundry/config_presets/data/unitree_g1/g1_action_fields.yaml \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml" \  # needs change for 30 Hz
--ray_address local
