source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "spartan" \
--source_episodes "[
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutScrewdriverOnPegboard/wollaston/real/bc/teleop/2025-09-23T11-05-22-07-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutScrewdriverOnPegboard/wollaston/real/bc/teleop/2025-09-24T07-52-18-07-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutScrewdriverOnPegboard/wollaston/real/bc/teleop/2025-10-01T13-50-23-07-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutScrewdriverOnPegboard/wollaston/real/bc/teleop/2025-10-02T08-11-02-07-00/diffusion_spartan/',
    's3://robotics-manip-lbm/efs/data/tasks/BimanualPutScrewdriverOnPegboard/wollaston/real/bc/teleop/2025-10-08T11-11-54-07-00/diffusion_spartan/',
    ]" \
--output_dir "s3://tri-ml-datasets-uw2/vla_foundry_datasets/toolhang_202602/BimanualPutScrewdriverOnPegboard" \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_6cameras.yaml" \
--language_annotations_path "vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml" \
--action_fields_config_path "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml" \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_5past_20future_lbmsize.yaml"
