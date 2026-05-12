import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


_WROTE_JSON_RE = re.compile(r"Wrote\s+(?P<path>.+_eval\.json)")


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

    cmd = [
        eval_cfg.python,
        eval_cfg.script_path,
        "--output-dir",
        experiment_path,
        "--data-dir",
        eval_cfg.data_dir,
        "--eval-dir",
        str(eval_dir),
        "--checkpoint",
        str(checkpoint_num),
        "--mode",
        eval_cfg.mode,
        "--tasks",
        *eval_cfg.tasks,
        "--episodes",
        str(eval_cfg.episodes),
        "--seed",
        str(eval_cfg.seed),
        "--eval-name",
        eval_name,
    ]

    if eval_cfg.max_steps is not None:
        cmd.extend(["--max-steps", str(eval_cfg.max_steps)])
    if eval_cfg.num_inference_steps is not None:
        cmd.extend(["--num-inference-steps", str(eval_cfg.num_inference_steps)])
    if eval_cfg.mode == "rollout":
        cmd.extend(
            [
                "--image-convention",
                eval_cfg.image_convention,
                "--expected-image-convention",
                eval_cfg.expected_image_convention,
            ]
        )
        if not eval_cfg.validate_obs_format:
            cmd.append("--no-validate-obs-format")
        if eval_cfg.save_videos:
            cmd.extend(
                [
                    "--save-videos",
                    "--video-episodes",
                    str(eval_cfg.video_episodes),
                    "--video-fps",
                    str(eval_cfg.video_fps),
                    "--video-codec",
                    eval_cfg.video_codec,
                ]
            )

    cmd.extend(eval_cfg.extra_args)

    env = os.environ.copy()
    if eval_cfg.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = eval_cfg.cuda_visible_devices
    env["MUJOCO_GL"] = eval_cfg.mujoco_gl
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLA_FOUNDRY_DIR"] = eval_cfg.vla_foundry_dir
    env["NUMBA_CACHE_DIR"] = eval_cfg.numba_cache_dir
    pythonpath_parts = [eval_cfg.vla_foundry_dir, eval_cfg.rfm_rl_dir]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    logging.info("[INTERMEDIATE_EVAL] checkpoint=%s step=%s cmd=%s", checkpoint_num, global_step, " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=eval_cfg.rfm_rl_dir,
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
        return {
            "failed": True,
            "checkpoint_num": checkpoint_num,
            "global_step": global_step,
            "error": str(exc),
        }

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

    result_path = _extract_result_path(proc.stdout)
    if result_path is None:
        candidates = sorted(eval_dir.glob(f"*{eval_name}*_eval.json"), key=lambda path: path.stat().st_mtime)
        result_path = candidates[-1] if candidates else None
    if result_path is None or not result_path.exists():
        message = f"eval completed but no result JSON was found under {eval_dir}"
        if eval_cfg.fail_training_on_error:
            raise FileNotFoundError(message)
        logging.error("[INTERMEDIATE_EVAL] %s", message)
        return {
            "failed": True,
            "checkpoint_num": checkpoint_num,
            "global_step": global_step,
            "error": message,
        }

    with result_path.open() as f:
        results = json.load(f)
    results["checkpoint_num"] = checkpoint_num
    results["global_step"] = global_step
    results["result_path"] = str(result_path)
    logging.info("[INTERMEDIATE_EVAL] loaded results from %s", result_path)
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
    mode = results.get("mode", "eval")

    if not episodes:
        return log_dict

    import wandb

    for episode in episodes:
        video_path = episode.get("video_path")
        if not video_path:
            continue
        path = Path(video_path)
        if not path.exists():
            logging.warning("[INTERMEDIATE_EVAL] video path does not exist: %s", path)
            continue
        task = episode.get("task", "unknown_task")
        episode_name = episode.get("episode", path.stem)
        log_dict[f"intermediate_eval/{mode}/{task}/{episode_name}/video"] = wandb.Video(
            str(path),
            fps=video_fps,
            format="mp4",
        )

    return log_dict


def _extract_result_path(stdout: str) -> Path | None:
    for line in reversed(stdout.splitlines()):
        match = _WROTE_JSON_RE.search(line)
        if match:
            return Path(match.group("path"))
    return None


def _add_scalar_metrics(flat: dict[str, float | int], prefix: str, metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, bool):
            flat[f"{prefix}/{key}"] = int(value)
        elif isinstance(value, (int, float)):
            flat[f"{prefix}/{key}"] = value
