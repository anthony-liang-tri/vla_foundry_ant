source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "spartan" \
--source_episodes "[
    's3://robotics-manip-lbm/efs/data/tasks/TurnLargeContainerUpsideDown/cabot/sim/bc/teleop/2025-04-02T18-42-52-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/TurnLargeContainerUpsideDown/cabot/sim/bc/teleop/2025-04-03T10-23-32-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/TurnLargeContainerUpsideDown/cabot/sim/bc/teleop/2025-04-03T11-29-04-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/TurnLargeContainerUpsideDown/cabot/sim/bc/teleop/2025-04-03T11-57-42-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/TurnLargeContainerUpsideDown/cabot/sim/bc/teleop/2025-04-03T13-39-18-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/TurnLargeContainerUpsideDown/cabot/sim/bc/teleop/2025-04-03T14-01-12-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/TurnLargeContainerUpsideDown/cabot/sim/bc/teleop/2025-04-03T14-19-56-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/TurnLargeContainerUpsideDown/cabot/sim/bc/teleop/2025-04-03T14-39-59-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/TurnLargeContainerUpsideDown/cabot/sim/bc/teleop/2025-04-29T18-58-14-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/TurnLargeContainerUpsideDown/cabot/sim/bc/teleop/2025-04-30T08-56-08-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/TurnLargeContainerUpsideDown/cabot/sim/bc/teleop/2025-04-30T11-02-11-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/TurnLargeContainerUpsideDown/cabot/sim/bc/teleop/2025-04-30T14-25-40-04-00/diffusion_spartan/',
    ]" \
--output_dir "s3://tri-ml-datasets-uw2/vla_foundry_datasets/sim_unseen_03242026/TurnLargeContainerUpsideDown" \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_6cameras.yaml" \
--language_annotations_path "vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml" \
--action_fields_config_path "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml" \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_5past_20future_lbmsize.yaml"
