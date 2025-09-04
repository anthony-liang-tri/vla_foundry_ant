#!/bin/bash

# Example script to convert LBM data to LeRobot format and push to Hugging Face Hub
# Make sure to set your HF_TOKEN environment variable or pass --token
user="jmercat"
name="lbm_bellpepper_test"

# For testing (small dataset)
uv run --group preprocessing python lbm2/data/preprocessing/preprocess_lbm_to_lerobot.py \
    --source_episodes "['s3://robotics-manip-lbm/efs/data/tasks/BimanualPutRedBellPepperInBin/riverway/sim/bc/teleop/2025-01-06T08-58-31-05-00/',]" \
    --output_dir ./lerobot_dataset_hub_test/ \
    --dataset_name "${name}" \
    --robot_type "lbm_bimanual_panda" \
    --fps 10 \
    --chunk_size 10 \
    --num_workers 8 \
    --max_episodes 2 \
    --preserve_depth False \
    --preserve_segmentation True \
    --preserve_calibration True \
    --push_to_hub True \
    --repo_id "${user}/${name}" \
    --private True

# For testing with list of episodes loaded from CSV
# uv run --group preprocessing python lbm2/data/preprocessing/preprocess_lbm_to_lerobot.py \
#     --source_eps_csv_path examples/preprocessing/s3_episodes_list.csv \
#     --output_dir ~/lbm2/lerobot/task_datasets/ \
#     --dataset_name "lbm_eval" \
#     --robot_type "lbm_bimanual_panda" \
#     --fps 10 \
#     --chunk_size 1000 \
#     --num_workers 16 \
#     --preserve_depth True \
#     --preserve_segmentation True \
#     --preserve_calibration True \
#     --push_to_hub True \
#     --repo_id "TRI-ML/lbm_eval" \
#     --private True 

echo "✅ Dataset conversion and upload complete!"
echo "📋 Next steps:"
echo "  1. Visit https://huggingface.co/datasets/${user}/${name}"
echo "  2. Update the dataset card with more details"
echo "  3. Test loading the dataset with: from datasets import load_dataset; ds = load_dataset('${user}/${name}')"
