#!/usr/bin/env python3
"""Submit inference workloads to a Ray cluster using Dockerized vla_foundry image."""

import argparse
import ast
import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import ray
import requests
from ray.autoscaler.sdk import request_resources
from ray.job_submission import JobStatus, JobSubmissionClient

# Environment variables for Ray jobs to prevent premature timeouts
JOB_ENVVARS = {
    "RAY_JOB_START_TIMEOUT_SECONDS": "999999999999",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
}

MIN_SAMPLES_PER_WORKER = 10
EVAL_DOCKER_LABEL = "anzu_lbm_eval"
EVAL_DOCKER_JOB_LABEL = "anzu_lbm_eval_job"

# Global task metadata registry - can be populated from config files
# Format: {"snake_case_task_name": {"t_max": float, "station": str, "config_file": str}}
TASK_METADATA: dict = {}


def load_task_metadata_from_yaml(yaml_path: Path) -> None:
    """Load task metadata from a YAML file into the global registry.

    The YAML file format is:
        tasks:
          TaskNameCamelCase:
            skill_name: "task_name_snake_case"
            t_max: 30.0
            station: "cabot"

    This function populates TASK_METADATA with snake_case keys.
    """
    import yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Support both 'tasks' and 'task_metadata' keys
    tasks_data = data.get("tasks") or data.get("task_metadata") or {}

    for camel_name, meta in tasks_data.items():
        # Use skill_name if present, otherwise convert CamelCase to snake_case
        snake_name = meta["skill_name"] if "skill_name" in meta else pascal_to_snake(camel_name)

        TASK_METADATA[snake_name] = {
            "t_max": meta.get("t_max"),
            "station": meta.get("station"),
            "config_file": meta.get("config_file"),
            "camel_name": camel_name,  # Store original name for reference
        }


def _auto_load_task_metadata():
    """Auto-load task_metadata.yaml from the same directory as this script."""
    script_dir = Path(__file__).resolve().parent
    metadata_path = script_dir / "task_metadata.yaml"
    if metadata_path.exists():
        try:
            load_task_metadata_from_yaml(metadata_path)
        except Exception as e:
            print(f"Warning: Failed to auto-load task_metadata.yaml: {e}")


# Auto-load task metadata on module import
_auto_load_task_metadata()


def register_task_metadata(
    task_name: str,
    t_max: float,
    station: str,
    config_file: Optional[str] = None,
) -> None:
    """Register metadata for a task programmatically."""
    TASK_METADATA[task_name] = {
        "t_max": t_max,
        "station": station,
        "config_file": config_file,
    }


@dataclass(frozen=True)
class JobSpec:
    name: str
    task: str
    checkpoint: str
    demo_indices: str
    launch_config_file: Optional[str]
    launch_scenario: str
    launch_script: str
    launch_task_name: str = ""
    mount_src: Optional[str] = None  # Per-job vla_foundry mount source path


def parse_range(range_text):
    start_text, end_text = range_text.split(":")
    start = int(start_text)
    end = int(end_text)
    if end <= start:
        raise ValueError(f"Invalid demo range {range_text}")
    return start, end


def shift_range(range_text, repetition):
    start, end = parse_range(range_text)
    span = end - start
    shift = repetition * span
    new_start = start + shift
    new_end = new_start + span
    return f"{new_start}:{new_end}"


def split_jobs_by_max_samples(jobs: List[JobSpec], max_samples_per_job: int) -> List[JobSpec]:
    """Split jobs so each job evaluates at most max_samples_per_job demonstrations.

    This operates on JobSpec.demo_indices ranges (start:end), producing multiple
    jobs per input job if the range span exceeds max_samples_per_job.
    """
    if max_samples_per_job <= 0:
        raise ValueError(f"max_samples_per_job must be > 0, got {max_samples_per_job}")

    split: List[JobSpec] = []
    for job in jobs:
        start, end = parse_range(job.demo_indices)
        span = end - start
        if span <= max_samples_per_job:
            split.append(job)
            continue

        cur = start
        while cur < end:
            chunk_end = min(end, cur + max_samples_per_job)
            # Ensure each chunk is uniquely identifiable for logs + output paths.
            chunk_name = f"{job.name}-d{cur}_{chunk_end}"
            split.append(
                JobSpec(
                    name=chunk_name,
                    task=job.task,
                    checkpoint=job.checkpoint,
                    demo_indices=f"{cur}:{chunk_end}",
                    launch_config_file=job.launch_config_file,
                    launch_scenario=job.launch_scenario,
                    launch_script=job.launch_script,
                    launch_task_name=job.launch_task_name,
                    mount_src=job.mount_src,
                )
            )
            cur = chunk_end

    return split


def pascal_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def snake_to_pascal(name):
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in name.split("_"))


def lookup_t_max(snake_name):
    """Look up t_max for a task from the metadata registry or evaluate.py fallback."""
    # First check the task metadata registry
    if snake_name in TASK_METADATA:
        return TASK_METADATA[snake_name].get("t_max")

    # Fuzzy match in registry
    for key, meta in TASK_METADATA.items():
        if key.startswith(snake_name) or snake_name.startswith(key):
            print(f"Found fuzzy match for t_max in registry: '{snake_name}' matches '{key}'")
            return meta.get("t_max")

    # Fallback: try to parse evaluate.py (for backward compatibility)
    return _lookup_t_max_from_evaluate_py(snake_name)


def _lookup_t_max_from_evaluate_py(snake_name):
    """Legacy fallback: parse evaluate.py to find t_max values."""
    current_dir = Path(__file__).parent
    script_path = Path(__file__).resolve()

    candidates = [
        current_dir / "evaluate.py",
        Path.cwd() / "intuitive" / "lbm_eval_dev" / "evaluate.py",
    ]

    # Handle Bazel runfiles
    script_path_str = str(script_path)
    if ".runfiles" in script_path_str:
        parts = script_path_str.split(".runfiles/")
        if len(parts) >= 2:
            after_runfiles = parts[1]
            workspace_parts = after_runfiles.split("/")
            if len(workspace_parts) >= 1:
                workspace_name = workspace_parts[0]
                runfiles_root = Path(parts[0]) / ".runfiles" / workspace_name
                candidates.insert(0, runfiles_root / "intuitive" / "lbm_eval_dev" / "evaluate.py")

    evaluate_path = None
    for candidate in candidates:
        if candidate.exists():
            evaluate_path = candidate
            break

    if not evaluate_path:
        print("DEBUG: evaluate.py not found, and task not in TASK_METADATA registry")
        return None

    try:
        content = evaluate_path.read_text()
        tree = ast.parse(content, filename=str(evaluate_path))

        t_max_dict = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "scenario_t_max":
                for stmt in node.body:
                    if (
                        isinstance(stmt, ast.Assign)
                        and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id.startswith("scenario_")
                        and isinstance(stmt.value, ast.Dict)
                    ):
                        for key, value in zip(stmt.value.keys, stmt.value.values, strict=False):
                            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                                t_max_dict[key.value] = value.value
                break

        if snake_name in t_max_dict:
            return t_max_dict[snake_name]

        for key, value in t_max_dict.items():
            if key.startswith(snake_name) or snake_name.startswith(key):
                print(f"Found fuzzy match for t_max: '{snake_name}' matches '{key}'")
                return value

        print(f"Warning: Task '{snake_name}' not found in scenario_t_max in evaluate.py")
        return None

    except Exception as e:
        print(f"Warning: Failed to parse evaluate.py for t_max: {e}")
        return None


def lookup_skill_info(snake_name):
    """Look up the full skill name and station for a given (possibly truncated) snake_case skill name.

    Returns a tuple (full_skill_name, station_name) or (None, None) if not found.
    """
    # First check the task metadata registry
    if snake_name in TASK_METADATA:
        meta = TASK_METADATA[snake_name]
        return snake_name, meta.get("station")

    # Fuzzy match in registry
    for key, meta in TASK_METADATA.items():
        if key.startswith(snake_name) or snake_name.startswith(key):
            print(f"Found fuzzy match in registry: '{snake_name}' matches '{key}'")
            return key, meta.get("station")

    # Fallback: try to parse evaluate.py (for backward compatibility)
    return _lookup_skill_info_from_evaluate_py(snake_name)


def _lookup_skill_info_from_evaluate_py(snake_name):
    """Legacy fallback: parse evaluate.py to find skill info."""
    current_dir = Path(__file__).parent
    script_path = Path(__file__).resolve()

    candidates = [
        current_dir / "evaluate.py",
        Path.cwd() / "intuitive" / "lbm_eval_dev" / "evaluate.py",
    ]

    # Handle Bazel runfiles
    script_path_str = str(script_path)
    if ".runfiles" in script_path_str:
        parts = script_path_str.split(".runfiles/")
        if len(parts) >= 2:
            after_runfiles = parts[1]
            workspace_parts = after_runfiles.split("/")
            if len(workspace_parts) >= 1:
                workspace_name = workspace_parts[0]
                runfiles_root = Path(parts[0]) / ".runfiles" / workspace_name
                candidates.insert(0, runfiles_root / "intuitive" / "lbm_eval_dev" / "evaluate.py")

    evaluate_path = None
    for candidate in candidates:
        if candidate.exists():
            evaluate_path = candidate
            break

    if not evaluate_path:
        print(f"Warning: evaluate.py not found and task not in TASK_METADATA for '{snake_name}'")
        return None, None

    # Parse evaluate.py to find which scenario_*_skills dict contains this skill
    content = evaluate_path.read_text()
    tree = ast.parse(content, filename=str(evaluate_path))

    cabot_skills = set()  # scenario_0, scenario_3, scenario_4
    riverway_skills = set()  # scenario_1, scenario_2

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "scenario_t_max":
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id.startswith("scenario_")
                    and isinstance(stmt.value, ast.Dict)
                ):
                    var_name = stmt.targets[0].id
                    scenario_num = None
                    parts = var_name.split("_")
                    if len(parts) >= 2 and parts[1].isdigit():
                        scenario_num = int(parts[1])

                    for key in stmt.value.keys:
                        if isinstance(key, ast.Constant):
                            skill_name = key.value
                            if scenario_num in {0, 3, 4}:
                                cabot_skills.add(skill_name)
                            elif scenario_num in {1, 2}:
                                riverway_skills.add(skill_name)
            break

    if snake_name in cabot_skills:
        return snake_name, "cabot"
    if snake_name in riverway_skills:
        return snake_name, "riverway"

    for skill in cabot_skills:
        if skill.startswith(snake_name) or snake_name.startswith(skill):
            print(f"Found fuzzy match: '{snake_name}' matches full skill '{skill}' -> cabot")
            return skill, "cabot"
    for skill in riverway_skills:
        if skill.startswith(snake_name) or snake_name.startswith(skill):
            print(f"Found fuzzy match: '{snake_name}' matches full skill '{skill}' -> riverway")
            return skill, "riverway"

    print(f"Warning: Could not find skill matching '{snake_name}'")
    return None, None


@dataclass(frozen=True)
class TaskSpec:
    """Specification for a task with its checkpoint and task name."""

    name: str  # Job name (e.g., stage3_singletask_sim_BimanualPlaceAppleFromBowlIntoBin)
    task_name: str  # Task name (e.g., BimanualPlaceAppleFromBowlIntoBin)
    checkpoint: str  # S3 path or local path
    vla_ref: Optional[str] = None  # Optional vla_foundry git ref (branch/tag/commit)
    mount_src: Optional[str] = None  # Optional per-task vla_foundry mount source path
    demo_indices: Optional[str] = None  # Optional per-task demo range (start:end)


def load_tasks_from_file(file_path: Path) -> List[TaskSpec]:
    """
    Load task specifications from a text file.

    Each line should be a tuple like:
    ("job_name", "TaskName", "s3://path/to/checkpoint")

    Or with optional vla_foundry ref (4th element):
    ("job_name", "TaskName", "s3://path/to/checkpoint", "branch_name")

    Or with optional mount_src (5th element, set by campaign orchestrator):
    ("job_name", "TaskName", "s3://path/to/checkpoint", "branch_name", "/path/to/mount")

    Or with optional demo_indices (6th element, used for re-runs of missing chunks):
    ("job_name", "TaskName", "s3://path/to/checkpoint", "branch_name", "/path/to/mount", "120:140")

    Or a simpler CSV format:
    job_name,TaskName,s3://path/to/checkpoint
    job_name,TaskName,s3://path/to/checkpoint,branch_name
    job_name,TaskName,s3://path/to/checkpoint,branch_name,/path/to/mount
    job_name,TaskName,s3://path/to/checkpoint,branch_name,/path/to/mount,120:140
    """

    def _parse_missing_indices_ranges(text: str) -> List[Tuple[int, int]]:
        """Parse '10-12,15,20-21' into [(10,13), (15,16), (20,22)] (end exclusive)."""
        s = text.strip()
        if not s:
            return []
        ranges: List[Tuple[int, int]] = []
        for chunk in s.split(","):
            token = chunk.strip()
            if not token:
                continue
            if "-" in token:
                a_str, b_str = [p.strip() for p in token.split("-", 1)]
                a = int(a_str)
                b = int(b_str)
                if b < a:
                    raise ValueError(f"Invalid missing index range '{token}' (end < start)")
                ranges.append((a, b + 1))
            else:
                idx = int(token)
                ranges.append((idx, idx + 1))
        return ranges

    tasks: List[TaskSpec] = []
    lines = file_path.read_text().splitlines()
    pending_missing_ranges: Optional[List[Tuple[int, int]]] = None

    buf: List[str] = []
    paren_depth = 0

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue

        if paren_depth == 0 and stripped.startswith("#"):
            # Optional "rerun only missing" hints (used by verify_campaign_completion.py output).
            prefix = "# Missing indices:"
            if stripped.startswith(prefix):
                pending_missing_ranges = _parse_missing_indices_ranges(stripped[len(prefix) :].strip())
            continue

        # CSV-ish single-line format (only when not in a tuple block).
        if paren_depth == 0 and not stripped.startswith("(") and not stripped.startswith("["):
            parts = [p.strip().strip("\"'") for p in stripped.split(",")]
            if len(parts) < 3:
                raise ValueError(f"Invalid task line (expected >=3 fields): {raw}")
            name = parts[0]
            task_name = parts[1]
            checkpoint = parts[2]
            vla_ref = parts[3] if len(parts) > 3 else None
            mount_src = parts[4] if len(parts) > 4 else None
            demo_indices = parts[5] if len(parts) > 5 else None
            if vla_ref == "":
                vla_ref = None
            if mount_src == "":
                mount_src = None
            if demo_indices == "":
                demo_indices = None
            tasks.append(
                TaskSpec(
                    name=name,
                    task_name=task_name,
                    checkpoint=checkpoint,
                    vla_ref=vla_ref,
                    mount_src=mount_src,
                    demo_indices=demo_indices,
                )
            )
            continue

        # Tuple blocks (may span multiple lines).
        buf.append(raw)
        paren_depth += raw.count("(") - raw.count(")")
        if paren_depth != 0:
            continue

        block = "\n".join(buf).strip().rstrip(",")
        buf = []
        # Replace f-strings with regular strings for literal_eval
        block = block.replace('f"', '"').replace("f'", "'")
        item = ast.literal_eval(block)
        if isinstance(item, list):
            item = tuple(item)
        if not isinstance(item, tuple) or len(item) < 3:
            raise ValueError(f"Invalid task spec (expected tuple len>=3): {item}")

        name = item[0]
        task_name = item[1]
        checkpoint = item[2]
        vla_ref = item[3] if len(item) > 3 else None
        mount_src = item[4] if len(item) > 4 else None
        demo_indices = item[5] if len(item) > 5 else None
        if vla_ref == "":
            vla_ref = None
        if mount_src == "":
            mount_src = None
        if demo_indices == "":
            demo_indices = None

        # If the file provides missing index hints, expand into per-range tasks.
        if pending_missing_ranges:
            for start, end in pending_missing_ranges:
                suffix = f"missing_{start}_{end}"
                tasks.append(
                    TaskSpec(
                        name=f"{name}-{suffix}",
                        task_name=task_name,
                        checkpoint=checkpoint,
                        vla_ref=vla_ref,
                        mount_src=mount_src,
                        demo_indices=f"{start}:{end}",
                    )
                )
            pending_missing_ranges = None
        else:
            tasks.append(
                TaskSpec(
                    name=name,
                    task_name=task_name,
                    checkpoint=checkpoint,
                    vla_ref=vla_ref,
                    mount_src=mount_src,
                    demo_indices=demo_indices,
                )
            )

    if paren_depth != 0:
        raise ValueError("Unbalanced parentheses while parsing tasks file.")
    if buf:
        raise ValueError("Trailing unterminated task block while parsing tasks file.")

    return tasks


def distribute_tasks(
    tasks: List[TaskSpec], num_samples: int, num_workers: int, start_index: int, defaults
) -> List[JobSpec]:
    """
    Distribute task specifications and sample ranges across workers optimally.

    Strategy:
    - If num_workers <= len(tasks): each worker gets a different task with full range
    - If num_workers > len(tasks): split sample range evenly across workers for each task

    Args:
        tasks: List of TaskSpec objects
        num_samples: Total number of evaluation samples per task
        num_workers: Number of workers to distribute across
        start_index: Starting demonstration index
        defaults: Default JobSpec for other parameters

    Returns:
        List of JobSpec objects, one per job to submit
    """
    jobs = []

    # If tasks specify explicit demo_indices, treat them as pre-specified jobs.
    # Concurrency is handled by the submission loop (target_parallelism=num_workers).
    if any(t.demo_indices for t in tasks):
        for task_spec in tasks:
            demo_indices = task_spec.demo_indices or f"{start_index}:{start_index + num_samples}"
            jobs.append(
                JobSpec(
                    name=task_spec.name,
                    task=task_spec.task_name,
                    checkpoint=task_spec.checkpoint,
                    demo_indices=demo_indices,
                    launch_config_file=defaults.launch_config_file,
                    launch_scenario=defaults.launch_scenario,
                    launch_script=defaults.launch_script,
                    launch_task_name=task_spec.task_name,
                    mount_src=task_spec.mount_src,
                )
            )
        return jobs

    if num_workers <= len(tasks):
        # More tasks than workers: create one job per task with full range.
        # The submission loop will queue jobs and keep at most num_workers active.
        for _i, task_spec in enumerate(tasks):
            demo_indices = f"{start_index}:{start_index + num_samples}"
            jobs.append(
                JobSpec(
                    name=task_spec.name,
                    task=task_spec.task_name,
                    checkpoint=task_spec.checkpoint,
                    demo_indices=demo_indices,
                    launch_config_file=defaults.launch_config_file,
                    launch_scenario=defaults.launch_scenario,
                    launch_script=defaults.launch_script,
                    launch_task_name=task_spec.task_name,
                    mount_src=task_spec.mount_src,
                )
            )
    else:
        # More workers than tasks: split sample range for each task
        workers_per_task = num_workers // len(tasks)
        extra_workers = num_workers % len(tasks)
        max_workers_allowed_for_task = max(1, num_samples // MIN_SAMPLES_PER_WORKER) if num_samples else 1

        worker_idx = 0
        for task_idx, task_spec in enumerate(tasks):
            # Some tasks get an extra worker if there's a remainder
            num_workers_for_task = workers_per_task + (1 if task_idx < extra_workers else 0)

            if num_workers_for_task == 0:
                continue

            # Avoid over-splitting a checkpoint: enforce a minimum number of samples per worker.
            if num_workers_for_task > max_workers_allowed_for_task:
                num_workers_for_task = max_workers_allowed_for_task

            # Split the sample range evenly across workers for this task
            samples_per_worker = num_samples // num_workers_for_task
            remainder = num_samples % num_workers_for_task

            current_start = start_index
            for worker_offset in range(num_workers_for_task):
                # Distribute remainder samples to first few workers
                worker_samples = samples_per_worker + (1 if worker_offset < remainder else 0)
                current_end = current_start + worker_samples

                job_name = f"{task_spec.name}-w{worker_idx}"
                demo_indices = f"{current_start}:{current_end}"
                jobs.append(
                    JobSpec(
                        name=job_name,
                        task=task_spec.task_name,
                        checkpoint=task_spec.checkpoint,
                        demo_indices=demo_indices,
                        launch_config_file=defaults.launch_config_file,
                        launch_scenario=defaults.launch_scenario,
                        launch_script=defaults.launch_script,
                        launch_task_name=task_spec.task_name,
                        mount_src=task_spec.mount_src,
                    )
                )

                current_start = current_end
                worker_idx += 1

    return jobs


def load_jobs_from_file(path, defaults):
    items = json.loads(Path(path).read_text())
    jobs = []
    for item in items:
        job = JobSpec(
            name=item["name"],
            task=item["task"],
            checkpoint=item["checkpoint"],
            demo_indices=item.get("demo_indices", defaults.demo_indices),
            launch_config_file=item.get("launch_config_file", defaults.launch_config_file),
            launch_scenario=item.get("launch_scenario", defaults.launch_scenario),
            launch_script=item.get("launch_script", defaults.launch_script),
            launch_task_name=item.get("launch_task_name", defaults.launch_task_name),
        )
        jobs.append(job)
    return jobs


def parse_inline_jobs(values, defaults):
    jobs = []
    for value in values:
        parts = value.split(",")
        if len(parts) != 3:
            raise ValueError(f"Inline jobs must use name,task,checkpoint format. Received: {value}")
        job = JobSpec(
            name=parts[0],
            task=parts[1],
            checkpoint=parts[2],
            demo_indices=defaults.demo_indices,
            launch_config_file=defaults.launch_config_file,
            launch_scenario=defaults.launch_scenario,
            launch_script=defaults.launch_script,
            launch_task_name=defaults.launch_task_name,
        )
        jobs.append(job)
    return jobs


def cleanup_docker_containers(args, final_cleanup: bool = False):
    """
    Clean up Docker containers on all Ray nodes to free memory.
    This helps the autoscaler detect nodes as idle for termination.

    Args:
        args: Command line arguments
        final_cleanup: If True, kill ALL inference containers (only safe when all jobs done)
    """
    if args.dry_run:
        return

    try:
        import ray

        ray.init(
            address=args.cluster_url.replace("http://", "ray://").replace(":8265", ":10001"), ignore_reinit_error=True
        )
        nodes = ray.nodes()
        ray.shutdown()
    except Exception as e:
        print(f"  Could not connect to Ray cluster for cleanup: {e}")
        return

    # Get list of alive node IPs (excluding head node for SSH cleanup)
    worker_node_ips = []
    head_node_ip = None
    for node in nodes:
        if node.get("Alive", False):
            ip = node.get("NodeManagerAddress", "")
            if ip:
                # Check if this is the head node
                if node.get("Resources", {}).get("node:__internal_head__"):
                    head_node_ip = ip
                else:
                    worker_node_ips.append(ip)

    if not worker_node_ips and not head_node_ip:
        return

    # Build cleanup script based on cleanup type
    if final_cleanup:
        cleanup_script = """
# Prune stopped containers
sudo docker container prune -f >/dev/null 2>&1 || true

# Kill any remaining inference containers (safe because all jobs are done)
orphan_containers=$(sudo docker ps -q --filter "label=anzu_lbm_eval=1" 2>/dev/null || true)
if [ -n "$orphan_containers" ]; then
    echo "Killing remaining inference containers: $orphan_containers"
    sudo docker kill $orphan_containers >/dev/null 2>&1 || true
fi

# Clear Python bytecode cache to free memory
find /tmp -name "*.pyc" -delete 2>/dev/null || true
find /tmp -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Drop filesystem caches to free memory (requires root)
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

echo "Final cleanup completed"
"""
    else:
        # Periodic cleanup - only prune stopped containers, don't kill running ones
        # Running containers might belong to active jobs
        cleanup_script = """
# Prune stopped containers only
sudo docker container prune -f >/dev/null 2>&1 || true

# Drop filesystem caches to free memory (requires root)
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

echo "Periodic cleanup completed"
"""

    cleanup_type = "final" if final_cleanup else "periodic"
    total_nodes = len(worker_node_ips) + (1 if head_node_ip else 0)
    print(f"  Running {cleanup_type} container cleanup on {total_nodes} nodes...")

    # Submit cleanup jobs to ALL nodes by using Ray tasks with node affinity
    # This ensures cleanup runs on every node, not just one
    try:
        import ray

        ray.init(
            address=args.cluster_url.replace("http://", "ray://").replace(":8265", ":10001"), ignore_reinit_error=True
        )

        @ray.remote(num_cpus=0.01)
        def run_cleanup_on_node(script: str):
            """Run cleanup script on the node where this task is scheduled."""
            import subprocess

            result = subprocess.run(
                ["bash", "-c", script],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return result.stdout + result.stderr

        # Get all node IDs for scheduling
        all_nodes = ray.nodes()
        cleanup_tasks = []

        for node in all_nodes:
            if not node.get("Alive", False):
                continue
            node_id = node.get("NodeID")
            if not node_id:
                continue

            # Schedule cleanup task with node affinity to ensure it runs on this specific node
            task = run_cleanup_on_node.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                )
            ).remote(cleanup_script)
            cleanup_tasks.append(task)

        if cleanup_tasks:
            print(f"  Submitted {len(cleanup_tasks)} cleanup tasks across all nodes")
            # Wait briefly for cleanup to complete (with timeout to avoid blocking)
            try:
                ready, not_ready = ray.wait(cleanup_tasks, timeout=15.0, num_returns=len(cleanup_tasks))
                print(f"  Cleanup completed on {len(ready)}/{len(cleanup_tasks)} nodes")
                if not_ready:
                    print(f"  Canceling {len(not_ready)} slow cleanup tasks to free node resources")
                    # Cancel pending tasks to prevent them from keeping nodes "busy"
                    # This is critical for autoscaler scale-down: pending/running tasks
                    # reserve resources (num_cpus=0.01) which prevents nodes from being idle.
                    # This only cancels our specific cleanup tasks, not other jobs on the node.
                    for task_ref in not_ready:
                        with contextlib.suppress(Exception):
                            ray.cancel(task_ref, force=True)
            except Exception as wait_err:
                print(f"  Could not wait for cleanup tasks: {wait_err}")

        ray.shutdown()

    except Exception as e:
        print(f"  Failed to submit cleanup tasks via Ray: {e}")
        # Fallback: try submitting a single job (better than nothing)
        try:
            client = JobSubmissionClient(args.cluster_url)
            cleanup_id = client.submit_job(
                entrypoint=f"bash -c {shlex.quote(cleanup_script)}",
                runtime_env={"env_vars": JOB_ENVVARS},
                entrypoint_num_cpus=0.1,
            )
            print(f"  Fallback cleanup job submitted: {cleanup_id[:8]}")
        except Exception as e2:
            print(f"  Fallback cleanup also failed: {e2}")


def verify_job_success_via_s3(
    checkpoint: str, task_name: str, demo_indices: str, evaluation_subfolder: Optional[str] = None
) -> Tuple[bool, int, int]:
    """
    Verify if a job actually succeeded by checking if results exist in S3.

    When a container is killed during cleanup, Ray may mark the job as FAILED/STOPPED
    even though the job had already completed and uploaded results to S3.

    This function checks S3 to see if the expected results are present.

    Args:
        checkpoint: The S3 checkpoint path (e.g., s3://bucket/path/to/checkpoint/)
        task_name: The task name (e.g., BimanualPutRedBellPepperInBin)
        demo_indices: The demo range (e.g., "100:200")
        evaluation_subfolder: Optional subfolder in evaluation path (e.g., "oss")

    Returns:
        Tuple of (success, found_count, expected_count):
        - success: True if job results exist in S3
        - found_count: Number of rollout summaries found
        - expected_count: Number of expected rollouts based on demo_indices
    """
    # Parse expected demo count from indices
    expected_indices: Set[int] = set()
    try:
        start, end = demo_indices.split(":")
        start_i = int(start)
        end_i = int(end)
        expected_count = end_i - start_i
        if expected_count > 0:
            expected_indices = set(range(start_i, end_i))
    except (ValueError, AttributeError):
        expected_count = 0

    # Build S3 path to check
    # vla_foundry uploads to: {checkpoint}/evaluation/{task_name}/rollouts/
    # lbm-eval-oss uploads to: {checkpoint}/evaluation/{subfolder}/{task_name}/results/
    # Remove trailing slash from checkpoint for consistent path building
    checkpoint_base = checkpoint.rstrip("/")
    # Strip checkpoint filename if present (e.g., checkpoint.ckpt or checkpoint_12345.pt)
    if checkpoint_base.endswith(".ckpt") or checkpoint_base.endswith(".pt"):
        checkpoint_base = checkpoint_base.rsplit("/", 1)[0]

    # Determine S3 path for verification
    # Path format: {checkpoint}/evaluation/{subfolder}/{task}/rollouts/
    # subfolder is optional (only used when evaluation_subfolder is set)
    if evaluation_subfolder:
        s3_path = f"{checkpoint_base}/evaluation/{evaluation_subfolder}/{task_name}/rollouts/"
    else:
        s3_path = f"{checkpoint_base}/evaluation/{task_name}/rollouts/"

    try:
        # Use aws s3 ls to check for result files
        print(f"    S3 verification checking: {s3_path}")
        cmd = ["aws", "s3", "ls", s3_path, "--recursive"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            # Path doesn't exist or no access
            print("    S3 path not found or no access")
            return False, 0, expected_count

        # Count summary.yaml files (each rollout has one)
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        # Match either summary.yaml (vla_foundry) or resolved_scenario.yaml (lbm-eval-oss)
        pattern = re.compile(r"demonstration_(\d+)/(summary\.yaml|resolved_scenario\.yaml)")
        found_indices: Set[int] = set()
        for line in lines:
            match = pattern.search(line)
            if match:
                found_indices.add(int(match.group(1)))
        matching_indices = found_indices.intersection(expected_indices) if expected_indices else set()
        summary_count = len(matching_indices)

        print(f"    Expected indices: {sorted(expected_indices) if expected_indices else 'none'}")
        print(f"    Found indices in S3: {sorted(found_indices) if found_indices else 'none'}")
        print(f"    Matching indices: {sorted(matching_indices) if matching_indices else 'none'}")

        # Consider success if we found at least some results
        # We use a threshold of 50% to account for partial uploads or job splits
        min_required = max(1, expected_count // 2) if expected_count > 0 else 1
        success = summary_count >= min_required

        return success, summary_count, expected_count

    except subprocess.TimeoutExpired:
        print(f"  WARNING: S3 verification timed out for {task_name}")
        return False, 0, expected_count
    except Exception as e:
        print(f"  WARNING: S3 verification failed for {task_name}: {e}")
        return False, 0, expected_count


def build_docker_command(args, job, repetition=0):
    job_suffix = f"{job.name}-rep{repetition}" if repetition > 0 else job.name
    demo_indices = shift_range(job.demo_indices, repetition) if repetition > 0 else job.demo_indices
    launch_script = job.launch_script
    launch_workdir = None
    if launch_script == "launch_sim.sh" or launch_script.endswith("/launch_sim.sh"):
        launch_script = "/opt/anzu/launch_sim.sh"
        launch_workdir = "/opt/anzu"

    # Use per-job mount_src if available, otherwise fall back to args
    mount_src = job.mount_src or args.vla_foundry_mount_src
    inference_workdir = args.vla_foundry_mount_target if mount_src else "/opt/vla_foundry"
    # Use unique save directories per job to prevent cross-contamination
    # when different ablations of the same task run on the same node.
    # The job_suffix includes the full job name (e.g., "task_ablation-w0")
    # which uniquely identifies each checkpoint being evaluated.
    job_save_dir = f"/tmp/lbm/{job_suffix}/rollouts/"
    job_summary_dir = f"/tmp/lbm/{job_suffix}/rollouts"
    envs = {
        "JOB_NAME": job_suffix,
        # TASK_NAME removed - inference_policy.py doesn't accept --task_name argument
        "CHECKPOINT_DIR": job.checkpoint,
        "LAUNCH_DEMONSTRATION_INDICES": demo_indices,
        "LAUNCH_SCENARIO": job.launch_scenario,
        "LAUNCH_SCRIPT": launch_script,
        "LAUNCH_SAVE_DIR": job_save_dir,
        "LAUNCH_SUMMARY_DIR": job_summary_dir,
        "NUM_FLOW_STEPS": str(args.num_flow_steps),
        "OPEN_LOOP_STEPS": str(args.open_loop_steps),
        "DEVICE": args.device,
        "INFERENCE_WORKDIR": inference_workdir,
        "HOME": "/home/anzu",
        "XDG_CACHE_HOME": "/home/anzu/.cache",
        "ROS_HOME": "/home/anzu/.ros",
        # Some tooling expects USER to be set; match the image's default user name.
        "USER": "anzu",
        # Allow Bazel/git to fetch remote repos without prompting for host key updates.
        "GIT_SSH_COMMAND": (
            "ssh -i /home/anzu/.ssh/github "
            "-o StrictHostKeyChecking=no "
            "-o UserKnownHostsFile=/home/anzu/.ssh/known_hosts"
        ),
        # Reduce CUDA fragmentation for inference.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Enable all NVIDIA driver capabilities (compute, graphics, video, etc.)
        # Required for EGL/VTK hardware-accelerated rendering in simulation.
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        # Skip Bazel build and use pre-built binary directly (much faster startup)
        "SKIP_BUILD": "1",
    }
    if job.launch_config_file:
        envs["LAUNCH_CONFIG_FILE"] = job.launch_config_file
    if args.launch_cuda_visible_devices:
        envs["LAUNCH_CUDA_VISIBLE_DEVICES"] = args.launch_cuda_visible_devices
    credential_env_keys = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
    ]
    has_static_credentials = False
    for key in credential_env_keys:
        value = os.environ.get(key)
        if value:
            envs[key] = value
            if key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
                has_static_credentials = True
    if args.aws_profile and not has_static_credentials:
        envs["AWS_PROFILE"] = args.aws_profile
    # Set LAUNCH_WORKDIR if needed (for launch_sim.sh)
    if launch_workdir:
        envs["LAUNCH_WORKDIR"] = launch_workdir
    # Add LAUNCH_TASK_NAME for the simulation launch script
    if job.launch_task_name:
        # Pre-calculate LAUNCH_T_MAX and LAUNCH_CONFIG_FILE from local code
        # to avoid stale code in container and handle truncated task names
        snake_task = pascal_to_snake(job.launch_task_name)

        # Lookup full skill name and station (handles truncated names)
        full_skill_name, station = lookup_skill_info(snake_task)

        # Use resolved full_skill_name (converted back to PascalCase) to handle truncated names
        if full_skill_name:
            envs["LAUNCH_TASK_NAME"] = snake_to_pascal(full_skill_name)
        else:
            envs["LAUNCH_TASK_NAME"] = job.launch_task_name

        if full_skill_name and station:
            config_relative = f"intuitive/visuomotor/config/{full_skill_name}_{station}.yaml"
            config_file = f"/opt/anzu/{config_relative}"
            # Note: Config files exist in Docker image, not needed locally
        else:
            # Fallback without station suffix if lookup fails
            print(f"Warning: Could not determine skill info for task '{snake_task}', using path without station suffix")
            config_file = f"/opt/anzu/intuitive/visuomotor/config/{snake_task}.yaml"
        envs["LAUNCH_CONFIG_FILE"] = config_file

        # Lookup t_max (also uses full_skill_name if available)
        t_max = lookup_t_max(full_skill_name if full_skill_name else snake_task)
        if t_max is not None:
            envs["LAUNCH_T_MAX"] = str(t_max)
        else:
            print(f"Warning: Could not find t_max for task '{snake_task}' (original: '{job.launch_task_name}')")
    if mount_src:
        envs["VLA_FOUNDRY_HOME"] = args.vla_foundry_mount_target
    # Always skip vla_foundry auto-update since deps are pre-installed in Docker image
    # This avoids slow git fetch and uv sync at startup
    envs["VLA_FOUNDRY_AUTO_UPDATE"] = "0"

    # Custom inference command configuration
    if getattr(args, "inference_script", None):
        envs["INFERENCE_SCRIPT"] = args.inference_script
    if getattr(args, "inference_script_args", None):
        envs["INFERENCE_SCRIPT_ARGS"] = args.inference_script_args
    if getattr(args, "inference_cmd_override", None):
        envs["INFERENCE_CMD_OVERRIDE"] = args.inference_cmd_override

    # S3 output path customization
    if getattr(args, "evaluation_subfolder", None):
        envs["LAUNCH_EVALUATION_SUBFOLDER"] = args.evaluation_subfolder

    # Use the image-provided run_inference_bundle.sh.
    entrypoint_path = "/usr/local/bin/run_inference_bundle.sh"
    docker_cmd = [
        "sudo",
        "docker",
        "run",
        "--rm",
        "--pull",
        "missing",  # Only pull if image not present (avoid "unknown server OS" errors)
        "--runtime=nvidia",
        "--device",
        "/dev/dri",
        "--group-add",
        "video",
        "__ENV_FILE_FLAG_PLACEHOLDER__",
        "__RENDER_GROUP_PLACEHOLDER__",
        "__GPU_FLAG_PLACEHOLDER__",
    ]
    docker_cmd.extend(
        [
            "-v",
            "/home/ubuntu/.aws:/home/anzu/.aws",
            "-v",
            "anzu_vla_foundry_logs:/tmp/lbm",
            "-v",
            "/mnt/local_storage/cache/huggingface:/home/anzu/.cache/huggingface",
        ]
    )
    # Only mount SSH container if it exists (for private repo access during model loading)
    if os.path.exists("/home/ubuntu/.ssh_container") and os.listdir("/home/ubuntu/.ssh_container"):
        docker_cmd.extend(["-v", "/home/ubuntu/.ssh_container:/home/anzu/.ssh:ro"])
    # Only mount SSH agent if socket exists
    if os.path.exists("/home/ubuntu/.ssh/ssh_auth_sock") and not os.path.isdir("/home/ubuntu/.ssh/ssh_auth_sock"):
        docker_cmd.extend(["-v", "/home/ubuntu/.ssh/ssh_auth_sock:/ssh-agent"])
        envs["SSH_AUTH_SOCK"] = "/ssh-agent"  # Add env var only if socket is available
    docker_cmd.extend(
        [
            "--label",
            f"{EVAL_DOCKER_LABEL}=1",
            "--label",
            f"{EVAL_DOCKER_JOB_LABEL}={job_suffix}",
        ]
    )
    if args.mount_local_run_bundle:
        repo_root_on_cluster = os.path.expandvars(os.path.expanduser(args.cluster_repo_root))
        bundle_host_path = os.path.join(
            repo_root_on_cluster,
            "run_inference_bundle.sh",
        )
        docker_cmd.extend(
            [
                "-v",
                f"{bundle_host_path}:/usr/local/bin/run_inference_bundle.sh:ro",
            ]
        )
        # Also mount launch_sim.sh to allow testing Bazel isolation fix without rebuilding image
        launch_sim_host_path = os.path.join(repo_root_on_cluster, "launch_sim.sh")
        # Mount to /tmp to avoid conflicts with existing paths in the image
        docker_cmd.extend(
            [
                "-v",
                f"{launch_sim_host_path}:/tmp/launch_sim_mounted.sh:ro",
            ]
        )
        # NOTE: Commenting out LAUNCH_SCRIPT update - Docker mount creates directory instead of file
        # Use the version baked into the Docker image instead
        if launch_script == "/opt/anzu/launch_sim.sh":
            envs["LAUNCH_SCRIPT"] = "/tmp/launch_sim_mounted.sh"
    if args.docker_run_extra_args:
        docker_cmd.extend(shlex.split(args.docker_run_extra_args))
    if mount_src:
        docker_cmd.extend(
            [
                "-v",
                f"{mount_src}:{args.vla_foundry_mount_target}",
            ]
        )
    # Build environment flags AFTER all env vars are finalized (including LAUNCH_SCRIPT update)
    env_flags = []
    for key, value in envs.items():
        env_flags.append("-e")
        env_flags.append(f"{key}={value}")
    docker_cmd.extend(env_flags)
    docker_cmd.append(args.image)
    docker_cmd.extend(["bash", entrypoint_path])
    docker_command = " ".join(shlex.quote(token) for token in docker_cmd)

    # Build the dynamic --gpus flag from CUDA_VISIBLE_DEVICES
    gpu_flag_logic = (
        '$( if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then '
        'echo "--gpus \\"device=${CUDA_VISIBLE_DEVICES}\\"" ; '
        "else "
        'echo "--gpus all"; '
        "fi )"
    )
    docker_command = docker_command.replace("__GPU_FLAG_PLACEHOLDER__", gpu_flag_logic)

    # Add render group for /dev/dri/renderD* access (needed for EGL hardware rendering)
    # Dynamically detect the group ID, fallback to 110 if detection fails
    render_group_logic = '--group-add $(stat -c "%g" /dev/dri/renderD128 2>/dev/null || echo 110)'
    docker_command = docker_command.replace("__RENDER_GROUP_PLACEHOLDER__", render_group_logic)

    # Optionally pass secrets from a standard env-file on the Ray node.
    # This is used for Hugging Face gated models (e.g., set HF_TOKEN or
    # HUGGINGFACE_HUB_TOKEN in /home/ubuntu/anzu_vla_foundry/secrets.env).
    env_file_logic = (
        '$( if [ -f "/home/ubuntu/anzu_vla_foundry/secrets.env" ]; then '
        'echo "--env-file /home/ubuntu/anzu_vla_foundry/secrets.env"; '
        "fi )"
    )
    docker_command = docker_command.replace("__ENV_FILE_FLAG_PLACEHOLDER__", env_file_logic)

    # Build pre-docker commands to copy cached venv into mount directory
    # This allows reusing the pre-built venv from the Docker image with custom code branches
    pre_docker_cmds = []

    # Ensure huggingface cache directory is writable by the container user (anzu, UID 1001)
    hf_cache_dir = "/mnt/local_storage/cache/huggingface"
    pre_docker_cmds.append(
        f'sudo mkdir -p "{hf_cache_dir}" && sudo chmod -R a+rwX "{hf_cache_dir}" 2>/dev/null || true'
    )

    if mount_src:
        # Make mount directory writable so uv can create venv inside container
        # The container user (anzu, UID 1001) needs write access
        pre_docker_cmds.append(
            f'echo "Making mount directory writable: {mount_src}"; '
            f'chmod -R a+w "{mount_src}" 2>/dev/null || sudo chmod -R a+w "{mount_src}"'
        )
        # Use file locking to prevent race conditions when multiple jobs run on same node
        # Using flock -c for portability (works with /bin/sh, not just bash)
        mount_venv = f"{mount_src}/.venv"
        lock_file = f"{mount_src}/.venv.lock"

        if not getattr(args, "inference_cmd_override", None):
            # Standard vla_foundry case: copy cached venv if not present (with lock)
            venv_cache = "/home/ubuntu/vla_foundry_venv_cache/.venv"
            pre_docker_cmds.append(
                f'flock -x "{lock_file}" -c \''
                f'if [ -d "{venv_cache}" ] && [ ! -d "{mount_venv}" ]; then '
                f'echo "Copying cached venv to mount directory..."; '
                f'cp -r "{venv_cache}" "{mount_venv}"; '
                f"fi'"
            )
        # For INFERENCE_CMD_OVERRIDE case: don't touch the .venv
        # The cmd_override should handle venv setup with its own locking (e.g., flock + uv sync)

    pre_docker = " && ".join(pre_docker_cmds) + " && " if pre_docker_cmds else ""
    return f"set -x; cd /tmp && {pre_docker}{docker_command}"


def validate_jobs(jobs):
    """Validate all jobs upfront before submitting to the cluster.

    Checks that:
    - Each task name can be resolved to a full skill name
    - Each skill has a valid station (cabot/riverway)
    - Each config file exists locally
    - Each task has a t_max value

    Raises an error immediately if any validation fails.
    """
    print("Validating all jobs before submission...")

    # Get repo root for checking config files
    repo_root = Path(__file__).resolve().parent.parent.parent
    workspace_root = Path.cwd()

    errors = []
    validated_tasks = set()  # Avoid re-validating same task

    for job in jobs:
        if not job.launch_task_name:
            continue

        # Skip if we already validated this task
        if job.launch_task_name in validated_tasks:
            continue
        validated_tasks.add(job.launch_task_name)

        snake_task = pascal_to_snake(job.launch_task_name)

        # Check skill info lookup
        full_skill_name, station = lookup_skill_info(snake_task)
        if not full_skill_name or not station:
            errors.append(
                f"Task '{job.launch_task_name}' -> '{snake_task}': Could not find matching skill in evaluate.py"
            )
            continue

        # Check config file exists locally (optional - files exist in Docker image)
        config_relative = f"intuitive/visuomotor/config/{full_skill_name}_{station}.yaml"
        local_config = repo_root / config_relative
        workspace_config = workspace_root / config_relative

        config_exists = local_config.exists() or workspace_config.exists()
        if not config_exists:
            # Just warn - config files exist in Docker image, not needed locally
            pass  # Config will be found in Docker image at runtime

        # Check t_max lookup
        t_max = lookup_t_max(full_skill_name)
        if t_max is None:
            errors.append(f"Task '{job.launch_task_name}' -> '{full_skill_name}': Could not find t_max in evaluate.py")
            continue

        print(f"  ✓ {job.launch_task_name} -> {full_skill_name}_{station}.yaml (t_max={t_max})")

    if errors:
        print("\n" + "=" * 60)
        print("VALIDATION FAILED - The following issues were found:")
        print("=" * 60)
        for error in errors:
            print(f"  ✗ {error}")
        print("=" * 60 + "\n")
        raise ValueError(f"Validation failed with {len(errors)} error(s). Fix the issues above before retrying.")

    print(f"✓ All {len(validated_tasks)} unique tasks validated successfully\n")


def submit_jobs(args, jobs):
    # Validate all jobs upfront before connecting to cluster
    validate_jobs(jobs)

    client = None
    if not args.dry_run:
        client = JobSubmissionClient(args.cluster_url)

    submission_ids = []
    submission_to_job = {}
    submission_to_retry_count = {}  # Track retry count per submission
    submission_kwargs = {}
    if args.entrypoint_num_cpus is not None:
        submission_kwargs["entrypoint_num_cpus"] = args.entrypoint_num_cpus
    if args.entrypoint_num_gpus is not None:
        submission_kwargs["entrypoint_num_gpus"] = args.entrypoint_num_gpus
    if args.entrypoint_memory is not None:
        submission_kwargs["entrypoint_memory"] = args.entrypoint_memory

    # In distribute mode, ignore repetitions
    repetitions = 1 if args.distribute else args.repetitions

    # Build queue of all jobs to submit
    # Queue items are (job, repetition, retry_count) tuples
    queue = []
    for job in jobs:
        for repetition in range(repetitions):
            queue.append((job, repetition, 0))  # retry_count=0 for initial submission

    if args.dry_run:
        for job, repetition, _retry_count in queue:
            entrypoint = build_docker_command(args, job, repetition)
            print(f"Would submit: {entrypoint}")
        return []

    # Smart Batching Loop
    active_submissions = set()
    total_submitted = 0
    total_to_submit = len(queue)
    failed_jobs = []  # Track permanently failed jobs
    retry_delay = 30  # Seconds to wait before retrying a failed job
    succeeded_count = 0  # Track successful jobs for periodic cleanup
    s3_verified_count = 0  # Track jobs that Ray marked as failed but S3 shows success
    status_check_failures = {}  # Track consecutive status check failures per job
    max_status_check_failures = 6  # After ~60s of failures, consider job dead

    print(f"Starting Smart Batching for {total_to_submit} jobs...")
    print("Jobs will be submitted as GPU resources become available.")
    if args.max_retries > 0:
        print(f"Job retry enabled: max_retries={args.max_retries}")

    # Initialize Ray connection for autoscaler resource requests
    # This is separate from JobSubmissionClient and needed for request_resources()
    ray_address = args.cluster_url.replace("http://", "ray://").replace(":8265", ":10001")
    try:
        ray.init(address=ray_address, ignore_reinit_error=True)
        print("Connected to Ray cluster for autoscaler communication")
    except Exception as e:
        print(f"Warning: Could not connect to Ray for autoscaler: {e}")
        print("  Autoscaling hints will be disabled")

    while queue or active_submissions:
        # 1. Update active submissions status and handle failures
        current_active_count = 0
        finished_ids = set()
        jobs_to_retry = []  # Jobs that failed and should be retried

        for sub_id in list(active_submissions):
            try:
                status = client.get_job_status(sub_id)
                if status == JobStatus.SUCCEEDED:
                    finished_ids.add(sub_id)
                    print(f"  Job {sub_id[:8]} SUCCEEDED")
                    succeeded_count += 1
                elif status in (JobStatus.FAILED, JobStatus.STOPPED):
                    finished_ids.add(sub_id)
                    job_info = submission_to_job.get(sub_id)
                    retry_count = submission_to_retry_count.get(sub_id, 0)

                    # ALWAYS check S3 first before deciding to retry
                    # Jobs may be killed by cleanup but have already succeeded and uploaded results
                    s3_success = False
                    found_count = 0
                    expected_count = 0
                    if job_info:
                        job, rep = job_info
                        s3_success, found_count, expected_count = verify_job_success_via_s3(
                            job.checkpoint,
                            job.launch_task_name or job.task,
                            job.demo_indices,
                            getattr(args, "evaluation_subfolder", None),
                        )

                    if s3_success:
                        # Job actually succeeded - results found in S3
                        print(
                            f"  Job {sub_id[:8]} {status} but VERIFIED SUCCESS via S3 "
                            f"({found_count}/{expected_count} rollouts found for {job.name})"
                        )
                        s3_verified_count += 1
                        succeeded_count += 1
                    elif job_info and retry_count < args.max_retries:
                        # S3 verification failed and retries remaining - schedule for retry
                        print(
                            f"  Job {sub_id[:8]} {status} (S3: {found_count}/{expected_count}) "
                            f"- retry {retry_count + 1}/{args.max_retries} scheduled"
                        )
                        jobs_to_retry.append((job_info, retry_count + 1))
                    elif job_info:
                        # Max retries exceeded
                        failed_jobs.append((sub_id, job_info))
                        print(
                            f"  Job {sub_id[:8]} {status} - max retries exceeded, giving up on {job.name} "
                            f"(S3 verification: {found_count}/{expected_count} rollouts)"
                        )
                    else:
                        print(f"  Job {sub_id[:8]} {status} - no retry info available")
                else:
                    current_active_count += 1
                    # Reset failure count on successful status check
                    status_check_failures.pop(sub_id, None)
            except Exception as e:
                # Track consecutive status check failures
                status_check_failures[sub_id] = status_check_failures.get(sub_id, 0) + 1
                fail_count = status_check_failures[sub_id]

                if fail_count >= max_status_check_failures:
                    # Job is likely dead/gone - mark as finished
                    finished_ids.add(sub_id)
                    job_info = submission_to_job.get(sub_id)
                    print(f"  Job {sub_id[:8]} status check failed {fail_count}x - marking as LOST: {e}")
                    if job_info:
                        failed_jobs.append((sub_id, job_info))
                else:
                    # Temporary failure, assume job still active
                    current_active_count += 1
                    if fail_count == 1:
                        print(f"  WARNING: Status check failed for {sub_id[:8]}: {e}")

        # Trigger container cleanup periodically when jobs finish to free memory
        # This helps nodes become idle so the autoscaler can terminate them
        if finished_ids and (succeeded_count % 5 == 0 or len(finished_ids) >= 3):
            try:
                cleanup_docker_containers(args)
            except Exception as e:
                print(f"  WARNING: Container cleanup failed: {e}")

        for sub_id in finished_ids:
            active_submissions.discard(sub_id)
            submission_to_job.pop(sub_id, None)
            submission_to_retry_count.pop(sub_id, None)
            status_check_failures.pop(sub_id, None)

        # Re-queue jobs for retry (add to front of queue for priority)
        for job_info, retry_count in reversed(jobs_to_retry):
            # job_info is (job, repetition) tuple
            queue.insert(0, (job_info[0], job_info[1], retry_count))

        # 2. Check Cluster Capacity via HTTP (for logging only)
        try:
            # Use HTTP API to check resources
            url = args.cluster_url.rstrip("/")
            resp = requests.get(f"{url}/nodes?view=summary", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                summary = data.get("data", {}).get("summary", [])
                total_gpus = 0
                for node in summary:
                    if node.get("state") == "ALIVE":
                        resources = node.get("raylet", {}).get("resourcesTotal", {})
                        total_gpus += int(resources.get("GPU", 0))
            else:
                total_gpus = 0
        except Exception:
            total_gpus = 0

        # 3. Calculate Free Slots based on Desired Parallelism (num_workers)
        # We want to maintain 'num_workers' active jobs (Running + Pending)
        # This ensures the autoscaler sees enough demand to provision nodes.

        target_parallelism = args.num_workers if args.num_workers else total_gpus
        # Fallback to total_gpus if num_workers is not set (legacy behavior)
        if target_parallelism == 0 and args.num_workers is None:
            # If no workers set and no GPUs, we can't submit anything unless we assume 1?
            # But legacy mode usually implies running on existing resources.
            target_parallelism = 0

        # If we are in distribute mode with num_workers, we force submission up to that count.
        if hasattr(args, "distribute") and args.distribute and args.num_workers:
            target_parallelism = args.num_workers

        # IMPORTANT: Scale down target parallelism when fewer jobs remain than workers.
        # This allows the autoscaler to terminate idle nodes as the campaign winds down.
        remaining_work = len(queue) + current_active_count
        if remaining_work < target_parallelism:
            target_parallelism = remaining_work

        # Signal the autoscaler about current resource demand.
        # This is critical for scaling down: when target_parallelism decreases,
        # the autoscaler will see reduced demand and terminate idle nodes.
        try:
            # Request one GPU bundle per target parallel job
            bundles = [{"GPU": 1}] * target_parallelism if target_parallelism > 0 else []
            request_resources(bundles=bundles)
        except Exception as e:
            # Non-fatal: autoscaler may not be available in all environments
            print(f"  Warning: Could not request resources from autoscaler: {e}")

        slots_available = target_parallelism - current_active_count

        # 4. Submit Jobs if slots available
        if slots_available > 0 and queue:
            to_submit = min(slots_available, len(queue))
            scaling_info = (
                f" (scaling down from {args.num_workers})" if remaining_work < (args.num_workers or 0) else ""
            )
            print(
                f"Status: {total_gpus} GPUs. {current_active_count}/{target_parallelism} jobs"
                f"{scaling_info}. Queue: {len(queue)}. Submitting {to_submit}..."
            )

            for _ in range(to_submit):
                job, repetition, retry_count = queue.pop(0)
                entrypoint = build_docker_command(args, job, repetition)

                retry_info = f" (retry {retry_count}/{args.max_retries})" if retry_count > 0 else ""
                print(f"Submitting {job.name} rep {repetition}{retry_info}...")

                # Add delay before retry to allow memory to be freed
                if retry_count > 0:
                    print(f"  Waiting {retry_delay}s before retry to allow resources to free up...")
                    time.sleep(retry_delay)

                try:
                    submission_id = client.submit_job(
                        entrypoint=entrypoint,
                        runtime_env={"env_vars": JOB_ENVVARS},
                        **submission_kwargs,
                    )
                    print(f"  -> submission id {submission_id}")
                    active_submissions.add(submission_id)
                    # Store (job, repetition) tuple for retry tracking
                    submission_to_job[submission_id] = (job, repetition)
                    submission_to_retry_count[submission_id] = retry_count
                    submission_ids.append(submission_id)
                    total_submitted += 1
                except Exception as e:
                    print(f"  Failed to submit job: {e}")
                    # Re-queue for retry if we haven't exceeded max retries
                    if retry_count < args.max_retries:
                        print(f"  Re-queuing for retry ({retry_count + 1}/{args.max_retries})")
                        queue.insert(0, (job, repetition, retry_count + 1))
        else:
            if queue:
                print(
                    f"Waiting... (GPUs: {total_gpus}, Active: {current_active_count}, "
                    f"Queue: {len(queue)}, Target: {target_parallelism})"
                )
            elif current_active_count > 0:
                # No more jobs to submit, just waiting for active jobs to complete
                print(f"Winding down... (Active: {current_active_count}, Target: {target_parallelism})")
            time.sleep(10)

    # Report final status
    print(f"\n{'=' * 60}")
    print("FINAL STATUS SUMMARY")
    print(f"{'=' * 60}")
    ray_succeeded = succeeded_count - s3_verified_count
    print(f"Jobs succeeded (Ray status):     {ray_succeeded}")
    if s3_verified_count > 0:
        print(f"Jobs verified via S3:            {s3_verified_count} (Ray showed FAILED/STOPPED but results found)")
    print(f"Total succeeded:                 {succeeded_count}")
    print(f"Jobs failed:                     {len(failed_jobs)}")

    if failed_jobs:
        print(f"\nWARNING: {len(failed_jobs)} job(s) failed after max retries (no results in S3):")
        for _sub_id, (job, rep) in failed_jobs:
            print(f"  - {job.name} rep {rep}")

    # Final cleanup to free resources on all nodes (safe to kill running containers now)
    print("\nRunning final container cleanup...")
    try:
        cleanup_docker_containers(args, final_cleanup=True)
    except Exception as e:
        print(f"  WARNING: Final cleanup failed: {e}")

    # Signal autoscaler that no more resources are needed (allow full scale down)
    try:
        request_resources(bundles=[])
        print("Signaled autoscaler to scale down (no resources requested)")
    except Exception as e:
        print(f"  Warning: Could not signal autoscaler for scale down: {e}")
    finally:
        with contextlib.suppress(Exception):
            ray.shutdown()

    return submission_ids


def wait_for_jobs(client, submission_ids):
    pending = set(submission_ids)
    while pending:
        finished = []
        for submission_id in pending:
            status = client.get_job_status(submission_id)
            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                print(f"{submission_id}: {status}")
                finished.append(submission_id)
        for submission_id in finished:
            pending.remove(submission_id)
        if pending:
            print(f"Waiting on {len(pending)} jobs ...")
            time.sleep(10)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Distribute multiple tasks from a file across workers
  python ray_policy_runner.py \\
    --cluster-url http://10.161.48.184:8265 \\
    --image 682769330988.dkr.ecr.us-east-1.amazonaws.com/anzu-vla-foundry:latest \\
    --distribute \\
    --tasks-file tasks.txt \\
    --num-samples 100 \\
    --num-workers 8 \\
    --start-index 100
    
  # Distribute 2 checkpoints for single task across 4 workers
  python ray_policy_runner.py \\
    --cluster-url http://10.161.48.184:8265 \\
    --image 682769330988.dkr.ecr.us-east-1.amazonaws.com/anzu-vla-foundry:latest \\
    --distribute \\
    --checkpoints s3://bucket/ckpt1 s3://bucket/ckpt2 \\
    --task BimanualPutRedBellPepperInBin \\
    --num-samples 100 \\
    --num-workers 4 \\
    --start-index 100
    
  # Use inline job specification (legacy mode)
  python ray_policy_runner.py \\q
    --cluster-url http://10.161.48.184:8265 \\
    --image 682769330988.dkr.ecr.us-east-1.amazonaws.com/anzu-vla-foundry:latest \\
    --job test-job,BimanualPutRedBellPepperInBin,s3://bucket/checkpoint

Tasks file format (tasks.txt):
  ("job_name", "TaskName", "s3://path/to/checkpoint"),
  ("job_name2", "TaskName2", "s3://path/to/checkpoint2"),
  
  Or CSV format:
  job_name,TaskName,s3://path/to/checkpoint
  job_name2,TaskName2,s3://path/to/checkpoint2
""",
    )
    parser.add_argument("--cluster-url", required=True, help="Ray cluster dashboard URL")
    parser.add_argument("--image", required=True, help="ECR image URI")
    parser.add_argument(
        "--aws-profile",
        default=None,
        help="Optional AWS profile name to pass through to docker (default: inherited from host).",
    )

    # Mode 1: Distribute mode (new, recommended)
    parser.add_argument(
        "--distribute",
        action="store_true",
        help="Enable intelligent work distribution mode",
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        help="File containing task specifications (job_name, task_name, checkpoint per line)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        help="List of checkpoint paths/URIs (for --distribute mode with single task)",
    )
    parser.add_argument(
        "--task",
        help="Task name (for --distribute mode with single task)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Total number of evaluation samples per checkpoint/task (for --distribute mode)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers to distribute work across (for --distribute mode)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=100,
        help="Starting demonstration index (for --distribute mode, default: 100)",
    )
    parser.add_argument(
        "--max-samples-per-job",
        type=int,
        default=None,
        help=(
            "If set, split each job's demo range so each submitted job evaluates at most this many demonstrations. "
            "Useful for keeping individual jobs bounded in runtime."
        ),
    )

    # Mode 2: Legacy job specification
    parser.add_argument(
        "--jobs-file",
        type=Path,
        help='JSON file describing jobs (list of {"name","task","checkpoint",...})',
    )
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        help="Inline job specified as name,task,checkpoint. May be repeated.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions per job (legacy mode)",
    )
    parser.add_argument(
        "--demo-indices",
        default="100:200",
        help="Default demonstration range start:end (legacy mode)",
    )

    # Launch configuration (shared)
    parser.add_argument(
        "--launch-config-file",
        default=None,
        help=("Path to launch configuration YAML. Omit when using launch_sim.sh to derive it from LAUNCH_TASK_NAME."),
    )
    parser.add_argument(
        "--launch-scenario",
        default="GrpcServerToSim",
        help="Launch scenario name",
    )
    parser.add_argument(
        "--launch-task-name",
        help="Task name passed to launch script (e.g., BimanualPutRedBellPepperInBin). "
        "If not provided, uses --task value in distribute mode",
    )
    parser.add_argument(
        "--launch-script",
        default="launch_sim.sh",
        help="Launch script to execute",
    )
    parser.add_argument(
        "--num-flow-steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--open-loop-steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--launch-cuda-visible-devices",
        help="Value forwarded as LAUNCH_CUDA_VISIBLE_DEVICES to the launch script",
    )
    parser.add_argument(
        "--docker-run-extra-args",
        help="Additional flags appended to docker run",
    )
    parser.add_argument(
        "--cluster-repo-root",
        default="/home/ubuntu/anzu",
        help=(
            "Absolute path on each Ray node where this repository is mounted. "
            "Used to volume-mount files (e.g., run_inference_bundle.sh) into containers."
        ),
    )
    parser.add_argument(
        "--mount-local-run-bundle",
        dest="mount_local_run_bundle",
        action="store_true",
        help=("Override /usr/local/bin/run_inference_bundle.sh with the version checked into this repository."),
    )
    parser.add_argument(
        "--no-mount-local-run-bundle",
        dest="mount_local_run_bundle",
        action="store_false",
        help=("Skip overriding /usr/local/bin/run_inference_bundle.sh with the version checked into this repository."),
    )
    parser.set_defaults(mount_local_run_bundle=False)
    parser.add_argument(
        "--vla-foundry-mount-src",
        help="Absolute path on each Ray node to bind-mount into the container as the vla_foundry workspace.",
    )
    parser.add_argument(
        "--vla-foundry-mount-target",
        default="/opt/vla_foundry",
        help="Container path for the vla_foundry bind mount (default: /opt/vla_foundry).",
    )
    parser.add_argument(
        "--disable-vla-foundry-auto-update",
        action="store_true",
        help="Set VLA_FOUNDRY_AUTO_UPDATE=0 to skip in-container git fetches.",
    )
    parser.add_argument(
        "--entrypoint-num-cpus",
        type=float,
        help="CPU cores to reserve for the Ray entrypoint process (default: 0).",
    )
    parser.add_argument(
        "--entrypoint-num-gpus",
        type=float,
        help="GPU resources to reserve for the Ray entrypoint process (default: 0).",
    )
    parser.add_argument(
        "--entrypoint-memory",
        type=int,
        help="Bytes of memory to reserve for the Ray entrypoint process (default: 0).",
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=float,
        help="Number of jobs to run per GPU (e.g. 4). Overrides --entrypoint-num-gpus.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without submitting jobs",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block until submitted jobs complete",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Maximum number of times to retry a failed job (default: 0, no retries)",
    )
    parser.add_argument(
        "--inference-script",
        help=(
            "Python script to run for inference (default: vla_foundry/inference/robotics/inference_policy.py). "
            "Use this to run a different inference script (e.g., grpc_workspace/diffusion_policy_server.py)."
        ),
    )
    parser.add_argument(
        "--inference-script-args",
        help=(
            "Additional arguments for the inference script. "
            "These are appended after the default args. Use this to pass custom arguments."
        ),
    )
    parser.add_argument(
        "--inference-cmd-override",
        help=(
            "Complete override for the inference command. "
            "If set, replaces the entire inference command (ignores --inference-script, etc). "
            "Supports placeholders: {checkpoint}, {num_flow_steps}, {open_loop_steps}, {device}."
        ),
    )
    parser.add_argument(
        "--evaluation-subfolder",
        help=(
            "Optional subfolder to insert in S3 output path: {checkpoint}/evaluation/{subfolder}/{task}/rollouts/. "
            "Useful for organizing results from different evaluation campaigns (e.g., 'oss', 'stage3')."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Determine launch_task_name
    launch_task_name = args.launch_task_name or (args.task if args.distribute else "")

    defaults = JobSpec(
        name="",
        task="",
        checkpoint="",
        demo_indices=args.demo_indices,
        launch_config_file=args.launch_config_file,
        launch_scenario=args.launch_scenario,
        launch_script=args.launch_script,
        launch_task_name=launch_task_name,
    )

    jobs = []

    # Mode 1: Distribute mode
    if args.distribute:
        if not all([args.num_samples, args.num_workers]):
            print("ERROR: --distribute mode requires --num-samples and --num-workers", file=sys.stderr)
            raise ValueError("ERROR: --distribute mode requires --num-samples and --num-workers")

        # Sub-mode A: Tasks file (each task has its own checkpoint and task name)
        if args.tasks_file:
            print(f"Loading tasks from: {args.tasks_file}")
            task_specs = load_tasks_from_file(args.tasks_file)
            print(f"Loaded {len(task_specs)} tasks")
            print("Distributing work:")
            print(f"  Tasks: {len(task_specs)}")
            print(f"  Samples per task: {args.num_samples}")
            print(f"  Workers: {args.num_workers}")
            print(f"  Starting index: {args.start_index}")
            print()

            jobs = distribute_tasks(
                tasks=task_specs,
                num_samples=args.num_samples,
                num_workers=args.num_workers,
                start_index=args.start_index,
                defaults=defaults,
            )

        # Sub-mode B: Single task with multiple checkpoints
        elif args.checkpoints and args.task:
            print("Distributing work:")
            print(f"  Task: {args.task}")
            print(f"  Checkpoints: {len(args.checkpoints)}")
            print(f"  Samples per checkpoint: {args.num_samples}")
            print(f"  Workers: {args.num_workers}")
            print(f"  Starting index: {args.start_index}")
            print()

            # Convert checkpoints to TaskSpec format
            task_specs = [
                TaskSpec(
                    name=f"{args.task}-ckpt{i}",
                    task_name=args.task,
                    checkpoint=ckpt,
                )
                for i, ckpt in enumerate(args.checkpoints)
            ]

            jobs = distribute_tasks(
                tasks=task_specs,
                num_samples=args.num_samples,
                num_workers=args.num_workers,
                start_index=args.start_index,
                defaults=defaults,
            )
        else:
            print(
                "ERROR: --distribute mode requires either --tasks-file OR (--checkpoints and --task)", file=sys.stderr
            )
            raise ValueError("ERROR: --distribute mode requires either --tasks-file OR (--checkpoints and --task)")

        print(f"Generated {len(jobs)} jobs:")
        for job in jobs:
            print(f"  - {job.name}: {job.task} | {job.checkpoint} | [{job.demo_indices}]")
        print()

    # Mode 2: Legacy job specification
    else:
        if args.jobs_file:
            jobs.extend(load_jobs_from_file(args.jobs_file, defaults))
        if args.job:
            jobs.extend(parse_inline_jobs(args.job, defaults))
        if not jobs:
            print(
                "ERROR: At least one job must be provided via --jobs-file, --job, or use --distribute mode",
                file=sys.stderr,
            )
            raise ValueError(
                "ERROR: At least one job must be provided via --jobs-file, --job, or use --distribute mode"
            )

    # Optionally split jobs so each submitted job runs at most N evaluations.
    if args.max_samples_per_job is not None:
        jobs = split_jobs_by_max_samples(jobs, args.max_samples_per_job)

    if args.distribute:
        print(f"Generated {len(jobs)} jobs:")
        for job in jobs:
            print(f"  - {job.name}: {job.task} | {job.checkpoint} | [{job.demo_indices}]")
        print()

    # Handle jobs-per-gpu logic
    if args.jobs_per_gpu:
        if args.entrypoint_num_gpus:
            print("WARNING: --jobs-per-gpu overrides --entrypoint-num-gpus", file=sys.stderr)
        args.entrypoint_num_gpus = 1.0 / args.jobs_per_gpu

    return submit_jobs(args, jobs)


if __name__ == "__main__":
    main()
