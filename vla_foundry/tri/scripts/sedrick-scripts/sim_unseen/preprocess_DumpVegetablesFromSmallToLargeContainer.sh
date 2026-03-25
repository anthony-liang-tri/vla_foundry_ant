source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "spartan" \
--source_episodes "[
    's3://robotics-manip-lbm/efs/data/tasks/DumpVegetablesFromSmallToLargeContainer/cabot/sim/bc/teleop/2025-04-02T19-31-33-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/DumpVegetablesFromSmallToLargeContainer/cabot/sim/bc/teleop/2025-04-03T11-25-24-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/DumpVegetablesFromSmallToLargeContainer/cabot/sim/bc/teleop/2025-04-03T15-36-45-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/DumpVegetablesFromSmallToLargeContainer/cabot/sim/bc/teleop/2025-04-03T17-39-51-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/DumpVegetablesFromSmallToLargeContainer/cabot/sim/bc/teleop/2025-04-03T18-23-37-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/DumpVegetablesFromSmallToLargeContainer/cabot/sim/bc/teleop/2025-04-03T19-45-39-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/DumpVegetablesFromSmallToLargeContainer/cabot/sim/bc/teleop/2025-04-03T21-26-51-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/DumpVegetablesFromSmallToLargeContainer/cabot/sim/bc/teleop/2025-05-01T17-56-05-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/DumpVegetablesFromSmallToLargeContainer/cabot/sim/bc/teleop/2025-05-02T10-49-16-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/DumpVegetablesFromSmallToLargeContainer/cabot/sim/bc/teleop/2025-05-02T11-56-33-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/DumpVegetablesFromSmallToLargeContainer/cabot/sim/bc/teleop/2025-05-02T14-30-55-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/DumpVegetablesFromSmallToLargeContainer/cabot/sim/bc/teleop/2025-05-02T15-22-00-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/DumpVegetablesFromSmallToLargeContainer/cabot/sim/bc/teleop/2025-05-02T17-29-37-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/DumpVegetablesFromSmallToLargeContainer/cabot/sim/bc/teleop/2025-05-02T19-00-26-04-00/diffusion_spartan/',
    ]" \
--output_dir "s3://tri-ml-datasets-uw2/vla_foundry_datasets/sim_unseen_03242026/DumpVegetablesFromSmallToLargeContainer" \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_6cameras.yaml" \
--language_annotations_path "vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml" \
--action_fields_config_path "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml" \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_5past_20future_lbmsize.yaml"
