source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "lerobot" \
--source_episodes "['s3://tri-ml-datasets/hf_datasets/pi_libero/']" \
--output_dir s3://tri-ml-datasets/vla_foundry_datasets/lerobot/pi_libero/ \
--camera_names "['image', 'wrist_image']" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml"