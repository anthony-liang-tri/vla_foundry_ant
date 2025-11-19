source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "spartan" \
--source_episodes "[
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-02T10-49-28-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-02T14-21-19-05-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-06T08-58-31-05-00/diffusion_spartan/',
    ]" \
--output_dir "s3://tri-ml-datasets/scratch/sedrick.keh/tmp/lbmdata/bimanualputredbellpepperinbin3/" \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_4cameras.yaml" \
--language_annotations_path "vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml" \
--action_fields_config_path "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml" \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml" \