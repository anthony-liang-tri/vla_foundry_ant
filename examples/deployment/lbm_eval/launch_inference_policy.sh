experiment_path="experiments/2025_11_05-21_34_11-model_diffusion_policy-lr_5e-05-bsz_1024/"


CUDA_VISIBLE_DEVICES=0 uv run --group inference python vla_foundry/inference/robotics/inference_policy.py \
    --checkpoint_directory $experiment_path \
    --num_flow_steps 8 \
    --device cuda \
    --open_loop_steps 8 \
