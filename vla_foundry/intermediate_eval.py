import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def should_run_intermediate_eval(cfg, checkpoint_num: int) -> bool:
    eval_cfg = cfg.intermediate_eval
    if not eval_cfg.enabled:
        return False
    if eval_cfg.every_n_checkpoints <= 0:
        raise ValueError("intermediate_eval.every_n_checkpoints must be positive.")
    return checkpoint_num % eval_cfg.every_n_checkpoints == 0


def intermediate_eval_marker_path(experiment_path: str, checkpoint_num: int) -> Path:
    return Path(experiment_path) / f".intermediate_eval_checkpoint_{checkpoint_num}.done"


def clear_intermediate_eval_marker(experiment_path: str, checkpoint_num: int) -> None:
    intermediate_eval_marker_path(experiment_path, checkpoint_num).unlink(missing_ok=True)


def mark_intermediate_eval_complete(experiment_path: str, checkpoint_num: int) -> None:
    marker = intermediate_eval_marker_path(experiment_path, checkpoint_num)
    marker.write_text("done\n")


def wait_for_intermediate_eval_marker(
    experiment_path: str,
    checkpoint_num: int,
    timeout_seconds: int | None,
    poll_interval_seconds: float = 5.0,
) -> None:
    marker = intermediate_eval_marker_path(experiment_path, checkpoint_num)
    start = time.monotonic()
    while not marker.exists():
        if timeout_seconds is not None and time.monotonic() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for intermediate eval marker: {marker}")
        time.sleep(poll_interval_seconds)


def run_intermediate_eval(cfg, experiment_path: str, checkpoint_num: int, global_step: int) -> dict[str, Any]:
    eval_cfg = cfg.intermediate_eval
    eval_name = f"train_ckpt_{checkpoint_num:04d}_step_{global_step:08d}"
    eval_dir = Path(eval_cfg.eval_dir) / Path(experiment_path).name
    eval_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(experiment_path) / "checkpoints" / f"checkpoint_{checkpoint_num}.pt"
    if not checkpoint_path.exists():
        message = f"Checkpoint not found: {checkpoint_path}"
        if eval_cfg.fail_training_on_error:
            raise FileNotFoundError(message)
        logging.error("[INTERMEDIATE_EVAL] %s", message)
        return {"failed": True, "checkpoint_num": checkpoint_num, "global_step": global_step, "error": message}

    result_json_path = eval_dir / f"{eval_name}_eval.json"

    cmd = [
        _get_python_executable(),
        "-m", "vla_foundry.eval.run_intermediate_eval",
        "--experiment-path", experiment_path,
        "--checkpoint", str(checkpoint_num),
        "--eval-dir", str(eval_dir),
        "--eval-name", eval_name,
        "--env", eval_cfg.env,
        "--tasks", *eval_cfg.tasks,
        "--episodes", str(eval_cfg.episodes),
        "--max-steps", str(eval_cfg.max_steps or 150),
        "--action-window", str(eval_cfg.action_window),
        "--seed", str(eval_cfg.seed),
        "--result-json", str(result_json_path),
    ]

    if eval_cfg.num_inference_steps is not None:
        cmd.extend(["--num-inference-steps", str(eval_cfg.num_inference_steps)])
    if eval_cfg.save_videos:
        cmd.extend(
            [
                "--save-videos",
                "--video-episodes",
                str(eval_cfg.video_episodes),
                "--video-fps",
                str(eval_cfg.video_fps),
            ]
        )
        if eval_cfg.video_grid_rows is not None:
            cmd.extend(["--video-grid-rows", str(eval_cfg.video_grid_rows)])
        if eval_cfg.video_grid_cols is not None:
            cmd.extend(["--video-grid-cols", str(eval_cfg.video_grid_cols)])

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["TOKENIZERS_PARALLELISM"] = "false"

    logging.info("[INTERMEDIATE_EVAL] checkpoint=%s step=%s cmd=%s", checkpoint_num, global_step, " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=eval_cfg.timeout_seconds,
            check=False,
        )
    except Exception as exc:
        if eval_cfg.fail_training_on_error:
            raise
        logging.exception("[INTERMEDIATE_EVAL] launch failed: %s", exc)
        return {"failed": True, "checkpoint_num": checkpoint_num, "global_step": global_step, "error": str(exc)}

    stdout_tail = "\n".join(proc.stdout.splitlines()[-40:])
    stderr_tail = "\n".join(proc.stderr.splitlines()[-40:])
    if stdout_tail:
        logging.info("[INTERMEDIATE_EVAL] stdout tail:\n%s", stdout_tail)
    if stderr_tail:
        logging.warning("[INTERMEDIATE_EVAL] stderr tail:\n%s", stderr_tail)

    if proc.returncode != 0:
        message = f"eval command failed with returncode={proc.returncode}"
        if eval_cfg.fail_training_on_error:
            raise RuntimeError(f"{message}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        logging.error("[INTERMEDIATE_EVAL] %s", message)
        return {
            "failed": True,
            "checkpoint_num": checkpoint_num,
            "global_step": global_step,
            "returncode": proc.returncode,
        }

    if not result_json_path.exists():
        message = f"eval completed but no result JSON at {result_json_path}"
        if eval_cfg.fail_training_on_error:
            raise FileNotFoundError(message)
        logging.error("[INTERMEDIATE_EVAL] %s", message)
        return {"failed": True, "checkpoint_num": checkpoint_num, "global_step": global_step, "error": message}

    with result_json_path.open() as f:
        results = json.load(f)
    results["checkpoint_num"] = checkpoint_num
    results["global_step"] = global_step
    results["result_path"] = str(result_json_path)
    logging.info("[INTERMEDIATE_EVAL] loaded results from %s", result_json_path)
    return results


def flatten_intermediate_eval_metrics(results: dict[str, Any]) -> dict[str, float | int]:
    prefix = f"intermediate_eval/{results.get('mode', 'eval')}"
    flat: dict[str, float | int] = {
        "intermediate_eval/checkpoint_num": int(results.get("checkpoint_num", -1)),
        "intermediate_eval/global_step": int(results.get("global_step", -1)),
        "intermediate_eval/failed": int(bool(results.get("failed", False))),
    }

    for task, metrics in results.get("by_task", {}).items():
        _add_scalar_metrics(flat, f"{prefix}/{task}", metrics)
    _add_scalar_metrics(flat, f"{prefix}/overall", results.get("overall", {}))
    return flat


def build_intermediate_eval_wandb_log(results: dict[str, Any], video_fps: int = 20) -> dict[str, Any]:
    log_dict = flatten_intermediate_eval_metrics(results)
    episodes = results.get("episodes", [])
    videos = results.get("videos", [])
    mode = results.get("mode", "eval")

    if not episodes and not videos:
        return log_dict

    import wandb

    for video in videos:
        video_path = video.get("video_path")
        if not video_path:
            continue
        path = Path(video_path)
        if not path.exists():
            logging.warning("[INTERMEDIATE_EVAL] video path does not exist: %s", path)
            continue
        task = video.get("task", "unknown_task")
        log_dict[f"intermediate_eval/{mode}/{task}/video"] = wandb.Video(
            str(path),
            fps=video_fps,
            format="mp4",
        )

    for episode in episodes:
        video_path = episode.get("video_path")
        if not video_path:
            continue
        path = Path(video_path)
        if not path.exists():
            logging.warning("[INTERMEDIATE_EVAL] video path does not exist: %s", path)
            continue
        task = episode.get("task", "unknown_task")
        episode_name = f"ep_{episode.get('episode', 0):03d}"
        log_dict[f"intermediate_eval/{mode}/{task}/{episode_name}/video"] = wandb.Video(
            str(path),
            fps=video_fps,
            format="mp4",
        )

    return log_dict


def _get_python_executable() -> str:
    import sys
    return sys.executable


def _add_scalar_metrics(flat: dict[str, float | int], prefix: str, metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, bool):
            flat[f"{prefix}/{key}"] = int(value)
        elif isinstance(value, (int, float)):
            flat[f"{prefix}/{key}"] = value
