source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "spartan" \
--source_episodes "[
    's3://robotics-manip-lbm/efs/data/tasks/PutContainersOnPlate/cabot/sim/bc/teleop/2025-04-01T15-19-14-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutContainersOnPlate/cabot/sim/bc/teleop/2025-04-01T20-16-49-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutContainersOnPlate/cabot/sim/bc/teleop/2025-04-02T11-37-12-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutContainersOnPlate/cabot/sim/bc/teleop/2025-04-02T14-02-27-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutContainersOnPlate/cabot/sim/bc/teleop/2025-04-02T14-55-59-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutContainersOnPlate/cabot/sim/bc/teleop/2025-04-02T18-37-50-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutContainersOnPlate/cabot/sim/bc/teleop/2025-04-30T08-27-43-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutContainersOnPlate/cabot/sim/bc/teleop/2025-04-30T11-30-06-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutContainersOnPlate/cabot/sim/bc/teleop/2025-05-01T08-37-22-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutContainersOnPlate/cabot/sim/bc/teleop/2025-05-01T14-46-17-04-00/diffusion_spartan/',
    ]" \
--output_dir "s3://tri-ml-datasets-uw2/vla_foundry_datasets/sim_unseen_03242026/PutContainersOnPlate" \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_6cameras.yaml" \
--language_annotations_path "vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml" \
--action_fields_config_path "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml" \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_5past_20future_lbmsize.yaml"
