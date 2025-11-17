source .venv/bin/activate && xvfb-run -a python vla_foundry/eval/run_eval.py \
    --env robosuite \
    --task Lift \
    --num_rollouts 10 \
    --num_steps 100 \
    --video_path ./lift_eval_test.mp4 \
    --action_window 2 \
    --model_path s3://tri-ml-datasets-uw2/richard/robosuite/lift/model_checkpoints/lift_right_mixed_800demos_clipL14/2025_11_14-03_43_29-model_diffusion_policy-lr_0.0003-bsz_512 \
