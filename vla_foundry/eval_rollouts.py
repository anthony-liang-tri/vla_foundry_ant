import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def should_run_eval_rollouts(cfg, checkpoint_num: int) -> bool:
    eval_cfg = cfg.eval_rollouts
    if not eval_cfg.enabled:
        return False
    if eval_cfg.every_n_checkpoints <= 0:
        raise ValueError("eval_rollouts.every_n_checkpoints must be positive.")
    return checkpoint_num % eval_cfg.every_n_checkpoints == 0


def eval_rollouts_marker_path(experiment_path: str, checkpoint_num: int) -> Path:
    return Path(experiment_path) / f".eval_rollouts_checkpoint_{checkpoint_num}.done"


def clear_eval_rollouts_marker(experiment_path: str, checkpoint_num: int) -> None:
    eval_rollouts_marker_path(experiment_path, checkpoint_num).unlink(missing_ok=True)


def mark_eval_rollouts_complete(experiment_path: str, checkpoint_num: int) -> None:
    marker = eval_rollouts_marker_path(experiment_path, checkpoint_num)
    marker.write_text("done\n")


def wait_for_eval_rollouts_marker(
    experiment_path: str,
    checkpoint_num: int,
    timeout_seconds: int | None,
    poll_interval_seconds: float = 5.0,
) -> None:
    marker = eval_rollouts_marker_path(experiment_path, checkpoint_num)
    start = time.monotonic()
    while not marker.exists():
        if timeout_seconds is not None and time.monotonic() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for eval rollout marker: {marker}")
        time.sleep(poll_interval_seconds)


def run_eval_rollouts(cfg, experiment_path: str, checkpoint_num: int, global_step: int) -> dict[str, Any]:
    eval_cfg = cfg.eval_rollouts
    eval_name = f"train_ckpt_{checkpoint_num:04d}_step_{global_step:08d}"
    eval_dir = Path(eval_cfg.eval_dir) / Path(experiment_path).name
    eval_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(experiment_path) / "checkpoints" / f"checkpoint_{checkpoint_num}.pt"
    if not checkpoint_path.exists():
        message = f"Checkpoint not found: {checkpoint_path}"
        if eval_cfg.fail_training_on_error:
            raise FileNotFoundError(message)
        logging.error("[EVAL_ROLLOUTS] %s", message)
        return {"failed": True, "checkpoint_num": checkpoint_num, "global_step": global_step, "error": message}

    result_json_path = eval_dir / f"{eval_name}_eval.json"

    cmd = [
        _get_python_executable(),
        "-m", "vla_foundry.eval.run_eval_rollouts",
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
    if getattr(eval_cfg, "robosuite_path", None):
        cmd.extend(["--robosuite-path", eval_cfg.robosuite_path])
    if getattr(eval_cfg, "robosuite_env_args_dir", None):
        cmd.extend(["--robosuite-env-args-dir", eval_cfg.robosuite_env_args_dir])
    if getattr(eval_cfg, "robosuite_env_args_paths", None):
        cmd.extend(["--robosuite-env-args-paths", *eval_cfg.robosuite_env_args_paths])
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
    if getattr(eval_cfg, "robosuite_path", None):
        env["VLAF_ROBOSUITE_EVAL_PATH"] = eval_cfg.robosuite_path
    if getattr(eval_cfg, "robosuite_env_args_dir", None):
        env["VLAF_ROBOSUITE_ENV_ARGS_DIR"] = eval_cfg.robosuite_env_args_dir

    logging.info("[EVAL_ROLLOUTS] checkpoint=%s step=%s cmd=%s", checkpoint_num, global_step, " ".join(cmd))
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
        logging.exception("[EVAL_ROLLOUTS] launch failed: %s", exc)
        return {"failed": True, "checkpoint_num": checkpoint_num, "global_step": global_step, "error": str(exc)}

    stdout_tail = "\n".join(proc.stdout.splitlines()[-40:])
    stderr_tail = "\n".join(proc.stderr.splitlines()[-40:])
    if stdout_tail:
        logging.info("[EVAL_ROLLOUTS] stdout tail:\n%s", stdout_tail)
    if stderr_tail:
        logging.warning("[EVAL_ROLLOUTS] stderr tail:\n%s", stderr_tail)

    if proc.returncode != 0:
        message = f"eval command failed with returncode={proc.returncode}"
        if eval_cfg.fail_training_on_error:
            raise RuntimeError(f"{message}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        logging.error("[EVAL_ROLLOUTS] %s", message)
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
        logging.error("[EVAL_ROLLOUTS] %s", message)
        return {"failed": True, "checkpoint_num": checkpoint_num, "global_step": global_step, "error": message}

    with result_json_path.open() as f:
        results = json.load(f)
    results["checkpoint_num"] = checkpoint_num
    results["global_step"] = global_step
    results["result_path"] = str(result_json_path)
    logging.info("[EVAL_ROLLOUTS] loaded results from %s", result_json_path)
    return results


def flatten_eval_rollouts_metrics(results: dict[str, Any]) -> dict[str, float | int]:
    flat: dict[str, float | int] = {}

    for task, metrics in results.get("by_task", {}).items():
        success_rate = metrics.get("success_rate")
        if isinstance(success_rate, (int, float)):
            flat[f"eval_rollouts/{task}/success_rate"] = success_rate
        mean_max_coverage = metrics.get("mean_max_coverage")
        if isinstance(mean_max_coverage, (int, float)):
            flat[f"eval_rollouts/{task}/mean_max_coverage"] = mean_max_coverage
    return flat


def build_eval_rollouts_wandb_log(results: dict[str, Any], video_fps: int = 20) -> dict[str, Any]:
    log_dict = flatten_eval_rollouts_metrics(results)
    videos = results.get("videos", [])

    if not log_dict and not videos:
        return log_dict

    import wandb

    for video in videos:
        video_path = video.get("video_path")
        if not video_path:
            continue
        path = Path(video_path)
        if not path.exists():
            logging.warning("[EVAL_ROLLOUTS] video path does not exist: %s", path)
            continue
        task = video.get("task", "unknown_task")
        log_dict[f"eval_rollouts/{task}/video_grid"] = wandb.Video(
            str(path),
            fps=video_fps,
            format="mp4",
        )

    return log_dict


def _get_python_executable() -> str:
    import sys
    return sys.executable
