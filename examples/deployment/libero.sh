python lbm2/eval/run_eval.py \
--env libero_goal \
--task 8 \
--model_path s3://tri-ml-datasets/lbm2/model_checkpoints/libero/PutBowlOnPlate/2025_10_08-18_44_40-model_diffusion_policy-lr_0.0005-bsz_2048/ \
--num_rollouts 1 \
--num_steps 20

