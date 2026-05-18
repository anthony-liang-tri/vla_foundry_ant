"""Subprocess entry point for intermediate eval during training.

Invoked by vla_foundry.intermediate_eval via:
    python -m vla_foundry.eval.run_intermediate_eval --experiment-path ... --checkpoint ...

Runs rollout eval using the VLA Foundry eval runner and writes a JSON result file.
"""

import argparse
import json
import logging
import os
import random
import sys

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from vla_foundry.eval.runners import get_eval_runner
from vla_foundry.file_utils import load_ema_checkpoint, load_model_checkpoint
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
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-episodes", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-grid-rows", type=int, default=0)
    parser.add_argument("--video-grid-cols", type=int, default=0)
    parser.add_argument("--result-json", type=str, required=True)
    return parser.parse_args()


def _draw_label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    x, y = 8, 8
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255))
    return np.asarray(image)


def _write_video_grid(
    rollout_videos: list[dict],
    output_path: str,
    rows: int,
    cols: int,
    fps: int,
) -> None:
    selected = rollout_videos[: rows * cols]
    if not selected:
        return

    sample_frame = next((video["frames"][0] for video in selected if video["frames"]), None)
    if sample_frame is None:
        return

    blank = np.zeros_like(sample_frame)
    max_frames = max(len(video["frames"]) for video in selected)

    writer = imageio.get_writer(output_path, fps=fps)
    try:
        for frame_idx in range(max_frames):
            cells = []
            for cell_idx in range(rows * cols):
                if cell_idx < len(selected) and selected[cell_idx]["frames"]:
                    video = selected[cell_idx]
                    frames = video["frames"]
                    frame = frames[min(frame_idx, len(frames) - 1)]
                    label = f"ep {int(video['episode']):03d} success {int(video['success'])}"
                    frame = _draw_label(frame, label)
                else:
                    frame = blank
                cells.append(frame)

            row_images = []
            for row in range(rows):
                row_images.append(np.concatenate(cells[row * cols : (row + 1) * cols], axis=1))
            writer.append_data(np.concatenate(row_images, axis=0))
    finally:
        writer.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    if model_cfg.ema.enabled:
        ema_checkpoint_path = os.path.join(experiment_path, "checkpoints", f"ema_{args.checkpoint}.pt")
        if os.path.exists(ema_checkpoint_path):
            load_ema_checkpoint(model, ema_checkpoint_path)
        else:
            logging.warning(
                "EMA is enabled but %s does not exist; falling back to raw checkpoint.",
                ema_checkpoint_path,
            )
            load_model_checkpoint(model, checkpoint_path)
    else:
        load_model_checkpoint(model, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    eval_runner.model = model

    by_task: dict = {}
    episodes: list = []
    videos: list = []
    grid_rows = max(0, args.video_grid_rows)
    grid_cols = max(0, args.video_grid_cols)
    use_video_grid = args.save_videos and grid_rows > 0 and grid_cols > 0
    tracks_coverage = args.env == "pusht"

    for task in args.tasks:
        task_successes = 0
        eval_runner.load_env(args.env, task, horizon=args.max_steps)
        task_video_rollouts: list[dict] = []

        for ep_idx in tqdm(range(args.episodes), desc=f"[eval] {task}"):
            should_save_video = args.save_videos and ep_idx < args.video_episodes
            video_writer = None
            video_path = None
            video_frames: list[np.ndarray] | None = [] if should_save_video and use_video_grid else None

            if should_save_video and not use_video_grid:
                video_dir = os.path.join(eval_dir, "videos", args.eval_name, task)
                os.makedirs(video_dir, exist_ok=True)
                video_path = os.path.join(video_dir, f"episode_{ep_idx:03d}.mp4")
                video_writer = imageio.get_writer(video_path, fps=args.video_fps)

            obs = eval_runner.env_reset(seed=args.seed + ep_idx)
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
                    previous_images = eval_runner.get_current_images()
                    obs = eval_runner.env_step(actions[action_i])
                    obs = eval_runner.get_obs_tensor(obs)
                    eval_runner.update_action_buffer(actions[action_i])
                    eval_runner.update_image_buffer(previous_images)
                    steps += 1

                    if video_writer is not None:
                        video_writer.append_data(eval_runner.get_image_for_video())
                    elif video_frames is not None:
                        video_frames.append(eval_runner.get_image_for_video())

                    if eval_runner.check_success() or eval_runner.check_finished():
                        break

                if eval_runner.check_success() or eval_runner.check_finished():
                    success = eval_runner.check_success()
                    break

            if video_writer is not None:
                video_writer.close()

            if success:
                task_successes += 1

            if video_frames is not None:
                task_video_rollouts.append(
                    {
                        "episode": ep_idx,
                        "success": bool(success),
                        "frames": video_frames,
                    }
                )

            episode_entry = {
                "task": task,
                "episode": ep_idx,
                "success": bool(success),
                "steps": steps,
            }
            if tracks_coverage:
                max_coverage = eval_runner.get_max_coverage() if hasattr(eval_runner, "get_max_coverage") else 0.0
                episode_entry["max_coverage"] = float(max_coverage)
            if video_path is not None:
                episode_entry["video_path"] = video_path
            episodes.append(episode_entry)

        task_success_rate = task_successes / args.episodes
        by_task[task] = {
            "success_rate": task_success_rate,
            "episodes": args.episodes,
            "successes": task_successes,
        }
        if tracks_coverage:
            task_coverages = [e["max_coverage"] for e in episodes if e["task"] == task]
            task_mean_coverage = sum(task_coverages) / len(task_coverages) if task_coverages else 0.0
            by_task[task]["mean_max_coverage"] = task_mean_coverage
            logging.info(
                "[INTERMEDIATE_EVAL] task=%s success_rate=%.2f mean_coverage=%.4f",
                task,
                task_success_rate,
                task_mean_coverage,
            )
        else:
            logging.info("[INTERMEDIATE_EVAL] task=%s success_rate=%.2f", task, task_success_rate)

        if use_video_grid and task_video_rollouts:
            video_dir = os.path.join(eval_dir, "videos", args.eval_name, task)
            os.makedirs(video_dir, exist_ok=True)
            grid_path = os.path.join(video_dir, "rollouts.mp4")
            _write_video_grid(task_video_rollouts, grid_path, grid_rows, grid_cols, args.video_fps)
            videos.append(
                {
                    "task": task,
                    "name": "rollouts",
                    "video_path": grid_path,
                }
            )

        eval_runner.env_close()

    total_successes = sum(t["successes"] for t in by_task.values())
    total_episodes = sum(t["episodes"] for t in by_task.values())
    overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0

    results = {
        "mode": "rollout",
        "by_task": by_task,
        "overall": {
            "success_rate": overall_success_rate,
            "episodes": total_episodes,
            "successes": total_successes,
        },
        "episodes": episodes,
        "videos": videos,
    }
    if tracks_coverage:
        all_coverages = [e["max_coverage"] for e in episodes]
        results["overall"]["mean_max_coverage"] = sum(all_coverages) / len(all_coverages) if all_coverages else 0.0

    with open(args.result_json, "w") as f:
        json.dump(results, f, indent=2)
    logging.info("Wrote %s", args.result_json)


if __name__ == "__main__":
    main()
