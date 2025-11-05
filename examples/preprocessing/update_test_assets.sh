output_dir=s3://tri-ml-datasets-uw2/preprocess_lbm_test/lbm/PickAndPlaceBox/cabot/sim

aws s3 rm --recursive $output_dir

source .venv/bin/activate && python vla_foundry/data/scripts/preprocessing/preprocess_robotics_to_tar.py \
    --source_episodes "['s3://robotics-manip-lbm/efs/data/tasks/PickAndPlaceBox/cabot/sim/bc/teleop/2025-02-11T17-04-00-05-00/diffusion_spartan/']" \
    --source_type "spartan" \
    --output_dir $output_dir \
    --action_fields_config_path vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml \
    --language_annotations_path vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml \
    --data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
    --camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names.yaml" \
    --past_lowdim_steps 1 \
    --future_lowdim_steps 14 \
    --image_indices "[-1, 0]" \
    --stride 1 \
    --max_padding_left 3 \
    --max_padding_right 12 \
    --samples_per_shard 5 \
    --max_episodes_to_process 1 \
    --jpeg_quality 5 \
    --filter_still_samples False \
    --compute_statistics True \
    --still_threshold 0.05 \
    --resize_images_size "[224, 224]"
    
files_to_download=(
    $output_dir/shards/shard_000000.tar
    $output_dir/shards/stats.json
    $output_dir/shards/processing_metadata.json
    $output_dir/shards/preprocessing_config.yaml
)

test_assets_dir=tests/test_assets/small_lbm_dataset

for file in ${files_to_download[@]}; do
    aws s3 cp $file $test_assets_dir
done

# write manifest.jsonl
rm $test_assets_dir/manifest.jsonl
touch $test_assets_dir/manifest.jsonl
echo '{"shard": "shard_000000", "num_sequences": 5}' > $test_assets_dir/manifest.jsonl

echo "Downloaded files to $test_assets_dir"