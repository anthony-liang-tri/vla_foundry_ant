source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "spartan" \
--source_episodes "[
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-18T08-58-46-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-22T11-13-04-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-22T11-43-45-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-22T14-18-31-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-22T14-37-15-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-22T17-56-13-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-22T18-03-08-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-22T18-08-16-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-22T19-04-47-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-22T19-36-32-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-23T11-08-07-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-23T13-43-01-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-23T14-33-41-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-23T16-18-47-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-23T18-22-54-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-07-23T19-26-01-04-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-12-18T11-48-56-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-12-18T17-20-04-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-12-18T18-38-03-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-12-19T10-36-24-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-12-19T15-33-22-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPlaceAvocadoFromBowlIntoBin/riverway/sim/bc/teleop/2024-12-19T16-46-39-05-00/diffusion_spartan/',
    ]" \
--output_dir "s3://tri-ml-datasets-uw2/vla_foundry_datasets/sim_unseen_03242026/BimanualPlaceAvocadoFromBowlIntoBin" \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_6cameras.yaml" \
--language_annotations_path "vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml" \
--action_fields_config_path "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml" \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_5past_20future_lbmsize.yaml"