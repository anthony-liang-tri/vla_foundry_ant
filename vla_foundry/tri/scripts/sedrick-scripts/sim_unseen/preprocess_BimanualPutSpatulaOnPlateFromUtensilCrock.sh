source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "spartan" \
--source_episodes "[
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-06-20T10-26-00-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-06-20T11-19-17-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-06-20T11-28-22-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-06-20T13-32-15-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-06-20T14-05-12-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-06-20T14-20-05-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-06-20T15-32-10-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-06-20T16-28-32-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-12-17T13-49-50-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-12-17T17-47-54-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-12-18T13-59-30-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-12-18T19-14-46-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-12-19T16-30-24-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-12-19T17-35-46-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-12-19T18-08-04-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-12-19T18-25-39-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-12-19T18-30-10-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutSpatulaOnPlateFromUtensilCrock/riverway/sim/bc/teleop/2024-12-19T21-49-23-05-00/diffusion_spartan/',
    ]" \
--output_dir "s3://tri-ml-datasets-uw2/vla_foundry_datasets/sim_unseen_03242026/BimanualPutSpatulaOnPlateFromUtensilCrock" \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_6cameras.yaml" \
--language_annotations_path "vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml" \
--action_fields_config_path "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml" \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_5past_20future_lbmsize.yaml"
