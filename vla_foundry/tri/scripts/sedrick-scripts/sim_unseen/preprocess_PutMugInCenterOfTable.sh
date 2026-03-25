source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "spartan" \
--source_episodes "[
    's3://robotics-manip-lbm/efs/data/tasks/PutMugInCenterOfTable/cabot/sim/bc/teleop/2024-11-18T11-44-34-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutMugInCenterOfTable/cabot/sim/bc/teleop/2024-11-18T15-19-08-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutMugInCenterOfTable/cabot/sim/bc/teleop/2024-11-18T15-45-47-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutMugInCenterOfTable/cabot/sim/bc/teleop/2024-11-18T18-12-41-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutMugInCenterOfTable/cabot/sim/bc/teleop/2024-11-19T09-42-15-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutMugInCenterOfTable/cabot/sim/bc/teleop/2024-12-19T10-14-42-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutMugInCenterOfTable/cabot/sim/bc/teleop/2024-12-19T11-21-29-05-00/diffusion_spartan/',
    ]" \
--output_dir "s3://tri-ml-datasets-uw2/vla_foundry_datasets/sim_unseen_03242026/PutMugInCenterOfTable" \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_6cameras.yaml" \
--language_annotations_path "vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml" \
--action_fields_config_path "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml" \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_5past_20future_lbmsize.yaml"
