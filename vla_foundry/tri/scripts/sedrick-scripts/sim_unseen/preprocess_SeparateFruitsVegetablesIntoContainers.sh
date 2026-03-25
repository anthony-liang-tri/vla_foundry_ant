source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "spartan" \
--source_episodes "[
    's3://robotics-manip-lbm/efs/data/tasks/SeparateFruitsVegetablesIntoContainers/cabot/sim/bc/teleop/2025-04-01T16-05-01-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/SeparateFruitsVegetablesIntoContainers/cabot/sim/bc/teleop/2025-04-01T17-41-43-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/SeparateFruitsVegetablesIntoContainers/cabot/sim/bc/teleop/2025-04-01T18-18-22-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/SeparateFruitsVegetablesIntoContainers/cabot/sim/bc/teleop/2025-04-01T19-27-38-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/SeparateFruitsVegetablesIntoContainers/cabot/sim/bc/teleop/2025-04-01T19-54-32-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/SeparateFruitsVegetablesIntoContainers/cabot/sim/bc/teleop/2025-05-01T15-08-24-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/SeparateFruitsVegetablesIntoContainers/cabot/sim/bc/teleop/2025-05-01T17-21-03-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/SeparateFruitsVegetablesIntoContainers/cabot/sim/bc/teleop/2025-05-01T17-35-01-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/SeparateFruitsVegetablesIntoContainers/cabot/sim/bc/teleop/2025-05-02T11-33-23-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/SeparateFruitsVegetablesIntoContainers/cabot/sim/bc/teleop/2025-05-02T11-35-16-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/SeparateFruitsVegetablesIntoContainers/cabot/sim/bc/teleop/2025-05-02T14-26-30-04-00/diffusion_spartan/',
    ]" \
--output_dir "s3://tri-ml-datasets-uw2/vla_foundry_datasets/sim_unseen_03242026/SeparateFruitsVegetablesIntoContainers" \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_6cameras.yaml" \
--language_annotations_path "vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml" \
--action_fields_config_path "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml" \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_5past_20future_lbmsize.yaml"
