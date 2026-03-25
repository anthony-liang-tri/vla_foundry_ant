source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "spartan" \
--source_episodes "[
    's3://robotics-manip-lbm/efs/data/tasks/PutFruitInLargeContainerAndCoverWithPlate/cabot/sim/bc/teleop/2025-04-02T09-58-32-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutFruitInLargeContainerAndCoverWithPlate/cabot/sim/bc/teleop/2025-04-02T13-35-07-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutFruitInLargeContainerAndCoverWithPlate/cabot/sim/bc/teleop/2025-04-02T14-17-39-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutFruitInLargeContainerAndCoverWithPlate/cabot/sim/bc/teleop/2025-04-02T16-37-50-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutFruitInLargeContainerAndCoverWithPlate/cabot/sim/bc/teleop/2025-04-02T17-11-24-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutFruitInLargeContainerAndCoverWithPlate/cabot/sim/bc/teleop/2025-04-02T17-43-05-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutFruitInLargeContainerAndCoverWithPlate/cabot/sim/bc/teleop/2025-04-02T18-22-02-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutFruitInLargeContainerAndCoverWithPlate/cabot/sim/bc/teleop/2025-04-30T14-38-39-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutFruitInLargeContainerAndCoverWithPlate/cabot/sim/bc/teleop/2025-05-01T09-41-25-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutFruitInLargeContainerAndCoverWithPlate/cabot/sim/bc/teleop/2025-05-01T10-14-26-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/PutFruitInLargeContainerAndCoverWithPlate/cabot/sim/bc/teleop/2025-05-01T14-04-28-04-00/diffusion_spartan/',
    ]" \
--output_dir "s3://tri-ml-datasets-uw2/vla_foundry_datasets/sim_unseen_03242026/PutFruitInLargeContainerAndCoverWithPlate" \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_6cameras.yaml" \
--language_annotations_path "vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml" \
--action_fields_config_path "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml" \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_5past_20future_lbmsize.yaml"
