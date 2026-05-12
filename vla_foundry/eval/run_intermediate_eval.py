"""Subprocess entry point for intermediate eval during training.

Invoked by vla_foundry.intermediate_eval via:
    python -m vla_foundry.eval.run_intermediate_eval --experiment-path ... --checkpoint ...

Runs rollout eval using the VLA Foundry eval runner and writes a JSON result file.
"""

import argparse
import json
import logging
import os
import sys

import imageio
import torch
from tqdm import tqdm

from vla_foundry.eval.runners import get_eval_runner
from vla_foundry.file_utils import load_model_checkpoint
from vla_foundry.models import create_model
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Run intermediate eval for a VLA Foundry checkpoint.")
    parser.add_argument("--experiment-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--eval-name", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--action-window", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-episodes", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--result-json", type=str, required=True)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    experiment_path = args.experiment_path
    checkpoint_path = os.path.join(experiment_path, "checkpoints", f"checkpoint_{args.checkpoint}.pt")

    if not os.path.exists(checkpoint_path):
        logging.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    eval_dir = args.eval_dir
    os.makedirs(eval_dir, exist_ok=True)

    from types import SimpleNamespace
    eval_params = SimpleNamespace(
        image_names=None,
        model_path=experiment_path,
        env=args.env,
    )
    eval_runner = get_eval_runner(eval_params)

    model_cfg = load_experiment_params_from_yaml(
        os.path.join(experiment_path, "config.yaml"), localize_params=True
    )
    model = create_model(model_cfg.model, load_pretrained=False)
    load_model_checkpoint(model, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    eval_runner.model = model

    by_task: dict = {}
    episodes: list = []

    for task in args.tasks:
        task_successes = 0
        eval_runner.load_env(args.env, task, horizon=args.max_steps)

        for ep_idx in tqdm(range(args.episodes), desc=f"[eval] {task}"):
            should_save_video = args.save_videos and ep_idx < args.video_episodes
            video_writer = None
            video_path = None

            if should_save_video:
                video_dir = os.path.join(eval_dir, "videos", args.eval_name, task)
                os.makedirs(video_dir, exist_ok=True)
                video_path = os.path.join(video_dir, f"episode_{ep_idx:03d}.mp4")
                video_writer = imageio.get_writer(video_path, fps=args.video_fps)

            obs = eval_runner.env_reset()
            success = False
            steps = 0

            for _step_i in range(args.max_steps):
                obs_extracted = eval_runner.extract_from_obs(obs)
                model_input_tensors = {k: v for k, v in obs_extracted.items() if isinstance(v, torch.Tensor)}
                actions = eval_runner.model.generate_actions(
                    **model_input_tensors, num_inference_steps=args.num_inference_steps
                )
                actions = eval_runner.denormalize_actions(actions)

                for action_i in range(eval_runner.num_past_actions, eval_runner.num_past_actions + args.action_window):
                    obs = eval_runner.env_step(actions[action_i])
                    obs = eval_runner.get_obs_tensor(obs)
                    eval_runner.update_action_buffer(actions[action_i])
                    eval_runner.update_image_buffer(eval_runner.get_current_images())
                    steps += 1

                    if video_writer is not None:
                        video_writer.append_data(eval_runner.get_image_for_video())

                    if eval_runner.check_success() or eval_runner.check_finished():
                        break

                if eval_runner.check_success() or eval_runner.check_finished():
                    success = eval_runner.check_success()
                    break

            if video_writer is not None:
                video_writer.close()

            if success:
                task_successes += 1

            episode_entry = {
                "task": task,
                "episode": ep_idx,
                "success": bool(success),
                "steps": steps,
            }
            if video_path is not None:
                episode_entry["video_path"] = video_path
            episodes.append(episode_entry)

        task_success_rate = task_successes / args.episodes
        by_task[task] = {"success_rate": task_success_rate, "episodes": args.episodes, "successes": task_successes}
        logging.info("[INTERMEDIATE_EVAL] task=%s success_rate=%.2f", task, task_success_rate)

        eval_runner.env_close()

    total_successes = sum(t["successes"] for t in by_task.values())
    total_episodes = sum(t["episodes"] for t in by_task.values())
    overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0

    results = {
        "mode": "rollout",
        "by_task": by_task,
        "overall": {"success_rate": overall_success_rate, "episodes": total_episodes, "successes": total_successes},
        "episodes": episodes,
    }

    with open(args.result_json, "w") as f:
        json.dump(results, f, indent=2)
    logging.info("Wrote %s", args.result_json)


if __name__ == "__main__":
    main()
