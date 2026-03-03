source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "spartan" \
--source_episodes "[
    's3://robotics-manip-lbm/efs/data/tasks/BimanualCleanUpBenchPhillipsScrewdriver/worcester/real/bc/teleop/2026-02-24T08-00-50-08-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualCleanUpBenchPhillipsScrewdriver/worcester/real/bc/teleop/2026-02-24T16-54-05-08-00/diffusion_spartan/',
    ]" \
--output_dir "s3://tri-ml-datasets-uw2/vla_foundry_datasets/garage_02242026/BimanualCleanUpBenchPhillipsScrewdriver" \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_6cameras.yaml" \
--language_annotations_path "vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml" \
--action_fields_config_path "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml" \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_5past_20future_lbmsize.yaml"