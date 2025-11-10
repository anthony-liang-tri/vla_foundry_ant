python lbm2/eval/run_eval.py \
--env robocasa \
--task CoffeeSetupMug \
--model_path s3://tri-ml-datasets/lbm2/model_checkpoints/robocasa/2025_10_02-22_31_12-model_diffusion_policy-lr_0.0005-bsz_2048/ \
--num_rollouts 1 \
--num_steps 50 