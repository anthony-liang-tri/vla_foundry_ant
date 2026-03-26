"""
Unified data preprocessing script for sim and real robotics data.

This script handles both simulation and real data preprocessing with Ray,
auto-detecting the data type from the source paths.

Preprocessing parameters (PreprocessParams) are loaded from a YAML config file
(--config, defaults to vla_foundry/tri/preprocessing_config.yaml). The YAML maps
directly to PreprocessParams fields (decoded via draccus), plus a few extra fields:
  - output_prefix: S3 prefix for output (overridable via --output-prefix)
  - stats_actor_memory_gb: memory for the statistics Ray actor (default: 8)
  - episode_worker_memory_gb: memory per episode worker (default: 16)

Some parameters can also be overridden from the command line:
  --output-prefix     overrides output_prefix from the config
  --camera-names      overrides camera_names (comma-separated)
  --skip-missing-cameras  overrides skip_episodes_missing_cameras

Usage:
    # Process sim tasks (local)
    python vla_foundry/tri/data_generation_ray.py \
        --tasks-file vla_foundry/tri/stage3_singletask_sim/stage3_sim_filenames.txt --local

    # Process real tasks on a Ray cluster
    python vla_foundry/tri/data_generation_ray.py \
        --tasks-file vla_foundry/tri/stage3_singletask_real/stage3_real_filenames.txt

    # Custom output prefix and config
    python vla_foundry/tri/data_generation_ray.py \
        --tasks-file vla_foundry/tri/stage3_singletask_sim/stage3_sim_filenames.txt \
        --output-prefix s3://my-bucket/output \
        --config path/to/my_config.yaml --local
"""

import argparse
import asyncio
import datetime
import getpass
import json
import logging
import os
import random
import re
import signal
import subprocess
import uuid
from contextlib import suppress

import draccus
import ray
import yaml
from ray.job_submission import JobSubmissionClient

from vla_foundry.data.preprocessing.metadata_utils import create_processing_metadata
from vla_foundry.data.preprocessing.robotics.converters import get_converter
from vla_foundry.data.preprocessing.robotics.preprocess_params import TYPE_MAPPER
from vla_foundry.data.preprocessing.robotics.preprocess_statistics import (
    LoggerActor,
    StreamingDatasetStatisticsRayActor,
)
from vla_foundry.data.preprocessing.utils import (
    create_episode_shard,
    create_shard,
    recursive_s3_copy,
    save_and_upload_config,
    save_and_upload_dict,
)
from vla_foundry.db_logger import get_git_env_vars, log_dataset_preprocessing
from vla_foundry.file_utils import check_directory_has_files_with_substring

# Reduce Ray logging verbosity
logging.getLogger("ray").setLevel(logging.WARNING)


def load_tasks_from_filenames(filenames_path: str, data_type: str | None = None) -> dict:
    """Load tasks from a filenames file. Returns dict mapping task_name -> list of episode paths."""
    tasks = {}
    if not os.path.exists(filenames_path):
        print(f"Warning: Filenames file not found: {filenames_path}")
        return tasks

    with open(filenames_path) as f:
        for line in f:
            if not line.strip():
                continue
            path = line.strip()

            if data_type:
                if data_type == "sim" and "/sim/" not in path:
                    continue
                if data_type == "real" and "/real/" not in path:
                    continue

            task_name = path.removeprefix("s3://robotics-manip-lbm/efs/data/tasks/").split("/")[0]
            if task_name not in tasks:
                tasks[task_name] = []
            tasks[task_name].append(path)

    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified data preprocessing for sim and real robotics data")
    parser.add_argument("--local", action="store_true", help="Run on local machine (starts local Ray instance)")
    parser.add_argument(
        "--on-head", action="store_true", help="Run on cluster head node (connects to existing cluster)"
    )
    parser.add_argument("--force", action="store_true", help="Force overwrite all existing data (including completed)")
    parser.add_argument(
        "--continue",
        dest="continue_mode",
        action="store_true",
        help="Skip completed tasks and retry partial/crashed ones (clean up partial data)",
    )
    parser.add_argument(
        "--tasks-file",
        type=str,
        required=True,
        help="File with S3 episode paths to process (one per line)",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="Max tasks to run in parallel (0=all)")
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["sim", "real", "all"],
        default="all",
        help="Type of data to process (default: all)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="S3 prefix for output directories (overrides config file)",
    )
    parser.add_argument(
        "--camera-names",
        type=str,
        default=None,
        help="Override camera names (comma-separated). If not set, auto-detects based on task/data type.",
    )
    parser.add_argument(
        "--skip-missing-cameras",
        action="store_true",
        help="Skip episodes that don't have ALL requested cameras",
    )
    parser.add_argument(
        "--no-episode-shards",
        action="store_true",
        help="Skip creating episode shards (only create random shards)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="vla_foundry/tri/preprocessing_config.yaml",
        help="Path to preprocessing config YAML file",
    )
    parser.add_argument(
        "--cluster-config",
        type=str,
        default="vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml",
        help="Path to Ray cluster config YAML template",
    )
    return parser.parse_args()


def preflight_check(tasks_file: str, data_type_filter: str | None, output_prefix: str):
    """Verify output dirs are clean before spinning up Ray/cluster.

    Runs locally using threads — one `aws s3 ls` per task.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    preflight_tasks = load_tasks_from_filenames(tasks_file, data_type_filter)
    preflight_items = []
    for task_name, episodes in preflight_tasks.items():
        dt = "sim" if any("/sim/" in ep for ep in episodes) else "real"
        preflight_items.append((task_name, dt, len(episodes)))

    def _check_target(task_name, data_type, num_source_episodes):
        output_dir = (
            f"{output_prefix.rstrip('/')}/{data_type}/{task_name}/"
            if data_type
            else f"{output_prefix.rstrip('/')}/{task_name}/"
        )
        result = subprocess.run(
            ["aws", "s3", "ls", "--recursive", "--summarize", output_dir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return task_name, None

        lines = result.stdout.strip().split("\n")
        total_objects = 0
        has_completed = False
        for line in lines:
            if "Total Objects:" in line:
                total_objects = int(line.split(":")[-1].strip())
            if "shards/COMPLETED" in line:
                has_completed = True

        if has_completed:
            return task_name, f"completed ({total_objects} objects, {num_source_episodes} source episodes)"
        return task_name, (
            f"partial ({total_objects} objects, no COMPLETED marker, {num_source_episodes} source episodes)"
        )

    print(f"Pre-flight check: verifying {len(preflight_items)} output directories are clean...")
    conflicts = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(_check_target, name, dt, n): name for name, dt, n in preflight_items}
        for future in as_completed(futures):
            task_name, status = future.result()
            if status is not None:
                conflicts.append((task_name, status))
    if conflicts:
        print(f"\n{len(conflicts)}/{len(preflight_items)} task(s) already have data in the output directory:")
        for name, status in sorted(conflicts)[:20]:
            print(f"  - {name}: {status}")
        if len(conflicts) > 20:
            print(f"  ... and {len(conflicts) - 20} more")
        print("\nUse --continue to skip completed and retry partial, or --force to overwrite everything.")
        exit(1)
    print(f"  All {len(preflight_items)} output directories are clean.")


def launch_on_cluster(args: argparse.Namespace, runtime_env_excludes: list[str]):
    """Start a Ray cluster, submit the job, stream logs, and tear down when done.

    This re-invokes the same script with --on-head on the cluster head node.
    Exits the process when done (never returns).
    """
    username = getpass.getuser()
    with open(args.cluster_config) as f:
        cluster_config_text = f.read()
    cluster_config_text, n_subs = re.subn(
        r"^(cluster_name:\s*)(.+)$",
        rf"\g<1>\2-{username}",
        cluster_config_text,
        count=1,
        flags=re.MULTILINE,
    )
    if n_subs == 0:
        raise ValueError(f"Could not find 'cluster_name:' in {args.cluster_config}")
    cluster_name = re.search(r"^cluster_name:\s*(.+)$", cluster_config_text, re.MULTILINE).group(1).strip()

    config_dir = os.path.dirname(args.cluster_config) or "."
    cluster_config = os.path.join(config_dir, f"ray_cluster_{username}.yaml")
    with open(cluster_config, "w") as f:
        f.write(cluster_config_text)
    print(f"Starting Ray cluster: {cluster_name}")
    print(f"   (expanded config: {cluster_config})")

    subprocess.run(["ray", "up", cluster_config, "-y", "--no-config-cache"], check=True, env=os.environ.copy())

    # Get head node IP
    result = subprocess.run(["ray", "get-head-ip", cluster_config], capture_output=True, text=True, check=True)
    head_ip = None
    for line in reversed(result.stdout.strip().split("\n")):
        if re.match(r"^\d+\.\d+\.\d+\.\d+$", line.strip()):
            head_ip = line.strip()
            break
    if not head_ip:
        raise RuntimeError(f"Could not parse head IP from output: {result.stdout}")
    print(f"Head node IP: {head_ip}")

    # Build entrypoint — re-invoke this script with --on-head
    config_rel = os.path.relpath(args.config)
    tasks_file_rel = os.path.relpath(args.tasks_file)
    entrypoint_parts = [
        f"python vla_foundry/tri/data_generation_ray.py --on-head --config {config_rel}",
        f"--tasks-file {tasks_file_rel}",
    ]
    if args.force:
        entrypoint_parts.append("--force")
    if args.continue_mode:
        entrypoint_parts.append("--continue")
    if args.batch_size:
        entrypoint_parts.append(f"--batch-size {args.batch_size}")
    if args.data_type != "all":
        entrypoint_parts.append(f"--data-type {args.data_type}")
    if args.output_prefix:
        entrypoint_parts.append(f"--output-prefix {args.output_prefix}")
    if args.camera_names:
        entrypoint_parts.append(f"--camera-names {args.camera_names}")
    if args.skip_missing_cameras:
        entrypoint_parts.append("--skip-missing-cameras")
    if args.no_episode_shards:
        entrypoint_parts.append("--no-episode-shards")
    entrypoint = " ".join(entrypoint_parts)

    # Clear local AWS credential env vars so workers use the EC2 instance profile
    env_vars = get_git_env_vars()
    for aws_key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN"]:
        env_vars[aws_key] = ""

    client = JobSubmissionClient(f"http://{head_ip}:8265")
    job_runtime_env = {"working_dir": ".", "excludes": runtime_env_excludes, "env_vars": env_vars}
    print(f"Submitting job with runtime_env excludes ({len(runtime_env_excludes)} patterns)...")
    job_id = client.submit_job(entrypoint=entrypoint, runtime_env=job_runtime_env)
    print(f"Submitted job: {job_id}")
    print(f"Monitor at: http://{head_ip}:8265/#/jobs/{job_id}")

    # Save job info for easy retrieval later
    with open("ray_job_info.json", "w") as f:
        json.dump(
            {
                "job_id": job_id,
                "head_ip": head_ip,
                "dashboard_url": f"http://{head_ip}:8265",
                "entrypoint": entrypoint,
                "cluster_config": cluster_config,
                "cluster_name": cluster_name,
            },
            f,
            indent=2,
        )

    def teardown_cluster():
        print(f"\nTearing down Ray cluster: {cluster_name}")
        subprocess.run(["ray", "down", cluster_config, "-y"], check=False)

    def _signal_handler(signum, frame):
        print(f"\nCaught signal {signal.Signals(signum).name}, stopping job and tearing down cluster...")
        with suppress(Exception):
            client.stop_job(job_id)
        teardown_cluster()
        exit(1)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:

        async def stream_logs():
            async for lines in client.tail_job_logs(job_id):
                print(lines, end="")

        asyncio.run(stream_logs())
    finally:
        teardown_cluster()
    exit(0)


def collect_results(task_items, futures, batch_label=""):
    """Collect Ray task results as they complete, reporting progress."""
    future_to_info = {f: (dt, name) for f, (name, _, _, dt) in zip(futures, batch_label, strict=True)}
    remaining = list(futures)
    results = []
    failed_count = 0
    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1)
        for future in done:
            dt, name = future_to_info[future]
            try:
                result = ray.get(future)
                results.append((dt, result))
                _, success, info = result
                status = "OK" if success else "FAILED"
                detail = f"{info['total_samples']} samples" if success and isinstance(info, dict) else str(info)
                if not success:
                    failed_count += 1
            except Exception as e:
                results.append((dt, (name, False, f"{type(e).__name__}: {e}")))
                failed_count += 1
                status = "CRASHED"
                detail = str(e)[:100]
            print(
                f"[Progress] {len(results)}/{len(task_items)} tasks done ({failed_count} failed in batch) | "
                f"{status}: {name} - {detail}"
            )
    return results


def print_summary(results, output_prefix, config):
    """Print run summary, upload run_summary.json to S3, and log to DynamoDB."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = 0
    total_samples = 0
    total_episodes = 0
    total_shards = 0
    total_frames = 0
    samples_per_shard = 0
    task_names = []
    total_failed_episodes = 0
    failed_tasks = []
    tasks_with_episode_failures = []

    for data_type, (task_name, success, info) in results:
        if success and isinstance(info, dict):
            ep_fails = info.get("failed_episodes", 0)
            warn = f" ({ep_fails} ep failed)" if ep_fails > 0 else ""
            print(f"  [{data_type}] {task_name}: {info['total_samples']} samples{warn}")
            total_samples += info["total_samples"]
            total_episodes += info["episode_count"]
            total_shards += info["num_shards"]
            total_frames += info["frame_count"]
            total_failed_episodes += ep_fails
            if ep_fails > 0:
                tasks_with_episode_failures.append(
                    {
                        "task": task_name,
                        "data_type": data_type,
                        "failed": ep_fails,
                        "total": info["episode_count"],
                        "details": info.get("failed_episode_details", []),
                    }
                )
            samples_per_shard = info["samples_per_shard"]
            task_names.append(task_name)
            success_count += 1
        elif success:
            print(f"  [{data_type}] {task_name}: {info}")
            success_count += 1
        else:
            print(f"  FAILED [{data_type}] {task_name}: {info}")
            failed_tasks.append((task_name, data_type, info))

    print(f"\nCompleted: {success_count}/{len(results)} tasks ({len(failed_tasks)} failed)")
    if total_failed_episodes > 0:
        print(f"  {total_failed_episodes} individual episodes failed across {len(tasks_with_episode_failures)} tasks:")
        for t in tasks_with_episode_failures:
            print(f"    {t['task']}: {t['failed']}/{t['total']} episodes failed")
            for d in t["details"][:3]:
                print(f"      - {d['episode']}: {d['error'][:120]}")
            if len(t["details"]) > 3:
                print(f"      ... and {len(t['details']) - 3} more")
    print(f"Total: {total_samples} samples, {total_episodes} episodes, {total_shards} shards, {total_frames} frames")
    if failed_tasks:
        print("\nFailed tasks:")
        for task_name, data_type, error in failed_tasks:
            print(f"  - [{data_type}] {task_name}: {error}")

    # Upload run summary JSON to S3
    run_summary = {
        "tasks_succeeded": success_count,
        "tasks_failed": len(results) - success_count,
        "total_failed_episodes": total_failed_episodes,
        "total_samples": total_samples,
        "total_episodes": total_episodes,
        "total_shards": total_shards,
        "succeeded": [
            {
                "task": tn,
                "data_type": dt,
                **{k: v for k, v in inf.items() if k != "failed_episode_details"},
            }
            for dt, (tn, ok, inf) in results
            if ok and isinstance(inf, dict)
        ],
        "failed": [{"task": tn, "data_type": dt, "error": str(inf)} for dt, (tn, ok, inf) in results if not ok],
        "episode_failures": tasks_with_episode_failures,
    }
    with open("/tmp/run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    subprocess.run(
        ["aws", "s3", "cp", "/tmp/run_summary.json", f"{output_prefix.rstrip('/')}/run_summary.json"],
        capture_output=True,
    )
    print(f"\nRun summary: {output_prefix.rstrip('/')}/run_summary.json")

    all_clean = len(failed_tasks) == 0 and total_failed_episodes == 0
    if all_clean:
        print("\nALL TASKS AND EPISODES PROCESSED SUCCESSFULLY")
    else:
        problems = []
        if failed_tasks:
            problems.append(f"{len(failed_tasks)} tasks failed")
        if total_failed_episodes > 0:
            problems.append(f"{total_failed_episodes} episodes failed")
        print(f"\nRUN COMPLETED WITH ISSUES: {', '.join(problems)}")
        print("   Check run_summary.json for details.")

    # Log group-level entry to DynamoDB
    if success_count > 0:
        db_logging = config.get("db_logging", True)
        output_dir_fixed_path = config.get("output_dir_fixed_path", "")
        dataset_uuid = str(uuid.uuid4())

        log_dataset_preprocessing(
            dataset_uuid=dataset_uuid,
            cfg=config,
            dataset_type=config.get("type", "spartan"),
            source_paths=list(task_names),
            target_path=output_prefix.rstrip("/"),
            fixed_path=output_dir_fixed_path.rstrip("/"),
            episode_count=total_episodes,
            frame_count=total_frames,
            samples_per_shard=samples_per_shard,
            num_shards=total_shards,
            total_samples=total_samples,
            enabled=db_logging,
        )


@ray.remote(num_cpus=1, max_retries=3, retry_exceptions=False)
def process_task(
    task_name: str,
    episodes: list[str],
    camera_names: list[str] | None,
    output_prefix: str,
    config_path: str,
    force: bool = False,
    continue_mode: bool = False,
    skip_missing_cameras: bool = False,
    create_episode_shards: bool = True,
    data_type: str = "",
):
    """Process a single task - runs the full preprocessing pipeline."""

    # Include data_type (sim/real) as subfolder when provided
    if data_type:
        output_dir = f"{output_prefix.rstrip('/')}/{data_type}/{task_name}/"
    else:
        output_dir = f"{output_prefix.rstrip('/')}/{task_name}/"

    # Safety checks for existing data:
    #   --force:    overwrite everything (skip all checks)
    #   --continue: skip completed tasks, clean up and retry partial data
    #   (default):  error on any existing data
    completion_marker = f"{output_dir.rstrip('/')}/shards/COMPLETED"
    if not force:
        marker_exists = subprocess.run(["aws", "s3", "ls", completion_marker], capture_output=True).returncode == 0
        if marker_exists:
            if continue_mode:
                print(f"[{task_name}] Already completed (COMPLETED marker found), skipping. Use --force to redo.")
                return (task_name, True, "already_completed")
            else:
                return (task_name, False, "already_completed — use --continue to skip or --force to redo")
        frames_dir = f"{output_dir}frames"
        existing_files = check_directory_has_files_with_substring(frames_dir, "_frame_")
        if existing_files:
            if continue_mode:
                print(
                    f"  [{task_name}] Found {len(existing_files)} existing files but no completion marker — "
                    f"likely a retry after crash. Cleaning up partial data..."
                )
                subprocess.run(["aws", "s3", "rm", "--recursive", output_dir], capture_output=True)
            else:
                return (
                    task_name,
                    False,
                    f"found {len(existing_files)} existing files — use --continue to retry or --force to redo",
                )

    print(f"{'=' * 50}\nStarting task: {task_name} ({len(episodes)} episodes)\n{'=' * 50}")

    # Load config from YAML and override task-specific values
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Remove non-draccus fields and override task-specific values
    config_dict.pop("output_prefix", None)  # Not a draccus field
    stats_memory_gb = int(config_dict.pop("stats_actor_memory_gb", 32))  # Not a draccus field
    episode_worker_memory_gb = int(config_dict.pop("episode_worker_memory_gb", 16))  # Not a draccus field
    config_dict.pop("runtime_env_excludes", None)  # Not a draccus field
    config_dict["source_episodes"] = episodes
    config_dict["output_dir"] = output_dir
    if camera_names is not None:
        config_dict["camera_names"] = camera_names
    config_dict["skip_episodes_missing_cameras"] = skip_missing_cameras

    # Create config using draccus
    cfg = draccus.decode(TYPE_MAPPER["spartan"], config_dict)

    # Create converter
    converter = get_converter(cfg)

    # Discover episodes
    discovered_episodes = converter.discover_episodes(cfg.source_episodes, cfg.max_episodes_to_process)
    print(f"[{task_name}] Found {len(discovered_episodes)} episodes")
    if len(discovered_episodes) == 0:
        print(f"[{task_name}] No episodes found!")
        return task_name, False, "No episodes found"

    # Create initial processing metadata
    metadata = create_processing_metadata(cfg, discovered_episodes)
    metadata["processing"]["timestamp_start"] = datetime.datetime.now().isoformat()

    # Process episodes - submit as nested Ray tasks
    print(f"[{task_name}] Processing {len(discovered_episodes)} episodes...")
    if cfg.compute_statistics:
        statistics_ray_actor = StreamingDatasetStatisticsRayActor.options(
            num_cpus=1,
            memory=stats_memory_gb * 1024 * 1024 * 1024,
            max_concurrency=1,
        ).remote(compute_stats=cfg.compute_statistics)
    else:
        statistics_ray_actor = None
    logger_actor = LoggerActor.options(num_cpus=0.1).remote()

    @ray.remote(num_cpus=1, max_retries=3, retry_exceptions=True)
    def episode_worker(episode_path, conv, stats_actor, log_actor):
        return conv.process_episode(episode_path, stats_actor, log_actor)

    episode_worker = episode_worker.options(memory=episode_worker_memory_gb * 1024 * 1024 * 1024)

    futures = [episode_worker.remote(ep, converter, statistics_ray_actor, logger_actor) for ep in discovered_episodes]
    results = []
    episode_errors = []
    total_eps = len(discovered_episodes)
    future_to_idx = {f: i for i, f in enumerate(futures)}
    remaining = list(futures)
    completed = 0
    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1)
        for future in done:
            completed += 1
            i = future_to_idx[future]
            ep_path = str(discovered_episodes[i]) if i < len(discovered_episodes) else f"episode_{i}"
            try:
                r = ray.get(future)
                if r is None:
                    episode_errors.append({"episode": ep_path, "error": "returned None (data error, not retried)"})
                elif len(r) == 0:
                    episode_errors.append({"episode": ep_path, "error": "returned 0 samples (no cameras or empty)"})
                else:
                    results.extend(r)
            except Exception as e:
                episode_errors.append({"episode": ep_path, "error": f"{type(e).__name__}: {e}"})
            if completed % max(1, total_eps // 10) == 0 or completed == total_eps:
                print(
                    f"[{task_name}] Episodes: {completed}/{total_eps} done, "
                    f"{len(episode_errors)} failed, "
                    f"{len(results)} samples so far"
                )

    # Upload error log to S3 for this task
    task_log = {
        "task_name": task_name,
        "total_episodes": len(discovered_episodes),
        "successful_episodes": len(discovered_episodes) - len(episode_errors),
        "failed_episodes": len(episode_errors),
        "total_samples": len(results),
        "errors": episode_errors,
    }
    save_and_upload_dict(task_log, f"{output_dir.rstrip('/')}/shards", "task_log.json")

    failed_episodes = len(episode_errors)
    if failed_episodes > 0:
        print(f"[{task_name}]  {failed_episodes}/{len(discovered_episodes)} episodes failed (see task_log.json)")
    if not results:
        return task_name, False, f"All {len(discovered_episodes)} episodes failed"
    print(f"[{task_name}] {len(discovered_episodes) - failed_episodes}/{len(discovered_episodes)} episodes succeeded")

    # Clear existing shards before re-sharding (in case of partial shard run)
    subprocess.run(["aws", "s3", "rm", "--recursive", f"{output_dir}shards/"], capture_output=True)

    # Sharding phase
    random.shuffle(results)
    shards = [results[i : i + cfg.samples_per_shard] for i in range(0, len(results), cfg.samples_per_shard)]
    print(f"[{task_name}] Creating {len(shards)} shards...")
    shard_futures = [create_shard.remote(shard_files, i, cfg.output_dir) for i, shard_files in enumerate(shards)]
    shard_results = ray.get(shard_futures)
    print(f"[{task_name}] Created {len(shard_results)} shards.")

    # Episode shards phase (if enabled)
    if create_episode_shards:
        subprocess.run(["aws", "s3", "rm", "--recursive", f"{output_dir}episodes/"], capture_output=True)
        episode_groups = {}
        for filename in results:
            episode_key = filename.rsplit("_frame_", 1)[0]
            episode_groups.setdefault(episode_key, []).append(filename)
        print(f"[{task_name}] Creating {len(episode_groups)} episode shards...")
        episode_shard_futures = [
            create_episode_shard.remote(files, episode_key, cfg.output_dir)
            for episode_key, files in episode_groups.items()
        ]
        episode_shard_results = ray.get(episode_shard_futures)
        print(f"[{task_name}] Created {len(episode_shard_results)} episode shards.")

        # Upload episode manifest
        episode_manifest_lines = [
            {"shard": shard_name, "num_sequences": num_seq} for shard_name, num_seq in episode_shard_results
        ]
        save_and_upload_dict(episode_manifest_lines, f"{cfg.output_dir.rstrip('/')}/episodes", "manifest.jsonl")

    # Upload manifest
    manifest_lines = [{"shard": shard_name, "num_sequences": num_seq} for shard_name, num_seq in shard_results]
    save_and_upload_dict(manifest_lines, f"{cfg.output_dir.rstrip('/')}/shards", "manifest.jsonl")

    # Upload statistics
    if cfg.compute_statistics:
        statistics_state = ray.get(statistics_ray_actor.get_statistics.remote())
        save_and_upload_dict(statistics_state, f"{cfg.output_dir.rstrip('/')}/shards", "stats.json")
        if create_episode_shards:
            save_and_upload_dict(statistics_state, f"{cfg.output_dir.rstrip('/')}/episodes", "stats.json")

    # Update metadata
    metadata["processing"]["total_samples_created"] = sum(num_seq for _, num_seq in shard_results)
    metadata["processing"]["timestamp_end"] = datetime.datetime.now().isoformat()
    metadata["processing"]["sample_counts"] = ray.get(logger_actor.get_values.remote())
    save_and_upload_dict(metadata, f"{cfg.output_dir.rstrip('/')}/shards", "processing_metadata.json")
    print("Sample counts:", metadata["processing"]["sample_counts"])

    # Save preprocessing config
    save_and_upload_config(cfg, f"{cfg.output_dir.rstrip('/')}/shards", "preprocessing_config.yaml")

    # Copy to fixed path (non-fatal — data is already written, don't block COMPLETED marker)
    dataset_uuid = str(uuid.uuid4())
    fixed_path = f"{cfg.output_dir_fixed_path.rstrip('/')}/{dataset_uuid}"
    try:
        recursive_s3_copy(cfg.output_dir, fixed_path)
    except Exception as e:
        print(f"[{task_name}] Warning: fixed-path copy failed (non-fatal): {e}")

    # Write completion marker so retries can distinguish partial vs completed data
    subprocess.run(
        ["aws", "s3", "cp", "-", f"{cfg.output_dir.rstrip('/')}/shards/COMPLETED"],
        input=datetime.datetime.now().isoformat().encode(),
        capture_output=True,
    )

    sample_counts = metadata["processing"]["sample_counts"]
    total_samples = metadata["processing"]["total_samples_created"]

    print(f"{'=' * 50}\nTask completed: {task_name} ({failed_episodes} episode failures)\n{'=' * 50}")
    return (
        task_name,
        True,
        {
            "total_samples": total_samples,
            "episode_count": len(discovered_episodes),
            "failed_episodes": failed_episodes,
            "failed_episode_details": episode_errors,
            "num_shards": len(shard_results),
            "samples_per_shard": cfg.samples_per_shard,
            "frame_count": sample_counts.get("total_frames", 0) if sample_counts else 0,
        },
    )


if __name__ == "__main__":
    args = parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_prefix = args.output_prefix or config.get("output_prefix")
    if not output_prefix:
        raise ValueError("output_prefix must be set in config file or via --output-prefix")
    print(f"Output prefix: {output_prefix}")

    camera_names = [c.strip() for c in args.camera_names.split(",")] if args.camera_names else None
    runtime_env_excludes = config.get("runtime_env_excludes", [])
    print(f"Tasks file: {args.tasks_file}")

    # Pre-flight check
    if not args.force and not args.continue_mode:
        preflight_check(args.tasks_file, args.data_type if args.data_type != "all" else None, output_prefix)

    # Initialize Ray based on mode
    if args.local:
        print("Starting local Ray instance...")
        ray.init()
    elif args.on_head:
        ray.init(address="auto")
    else:
        launch_on_cluster(args, runtime_env_excludes)  # exits the process

    print(f"Connected to Ray cluster: {ray.cluster_resources()}")

    # Quick health check
    @ray.remote(num_cpus=1)
    def _health_check():
        import platform
        import sys

        return {"python": sys.version, "node": platform.node(), "cwd": os.getcwd()}

    print("Running worker health check (waiting up to 30 min for workers to join)...")
    try:
        hc_result = ray.get(_health_check.remote(), timeout=1800)
        print(f"  Worker OK: node={hc_result['node']}, python={hc_result['python']}, cwd={hc_result['cwd']}")
    except Exception as e:
        print(f"  Worker health check FAILED: {e}")
        print("  This means workers cannot execute tasks. Check worker logs in the Ray dashboard.")
        ray.shutdown()
        exit(1)

    # Load tasks
    if not args.tasks_file:
        raise ValueError("--tasks-file is required. Provide a file with S3 episode paths (one per line).")
    data_type_filter = args.data_type if args.data_type != "all" else None
    file_tasks = load_tasks_from_filenames(args.tasks_file, data_type_filter)
    all_tasks = {}
    for task_name, episodes in file_tasks.items():
        data_type = "sim" if any("/sim/" in ep for ep in episodes) else "real"
        all_tasks[(task_name, data_type)] = episodes
    print(f"Loaded {len(file_tasks)} tasks from {args.tasks_file}")

    total_episodes = sum(len(eps) for eps in all_tasks.values())
    print(f"Total: {len(all_tasks)} task/data-type combinations ({total_episodes} episodes)")

    if not all_tasks:
        print("No tasks found! Check that the filenames file exists and has content.")
        ray.shutdown()
        exit(1)

    # Build and submit tasks
    task_items = [(name, eps, camera_names, dt) for (name, dt), eps in all_tasks.items()]
    task_items.sort(key=lambda x: (x[3], x[0]))

    batch_size = args.batch_size if args.batch_size > 0 else max(len(task_items), 1)
    print(f"\nLaunching {len(task_items)} tasks (batch size: {batch_size})...\n")

    results = []
    for i in range(0, len(task_items), batch_size):
        batch = task_items[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1} ({len(batch)} tasks)...")
        futures = [
            process_task.remote(
                name,
                eps,
                cams,
                output_prefix,
                args.config,
                args.force,
                args.continue_mode,
                args.skip_missing_cameras,
                not args.no_episode_shards,
                dt,
            )
            for name, eps, cams, dt in batch
        ]
        results.extend(collect_results(task_items, futures, batch))

    print_summary(results, output_prefix, config)
    ray.shutdown()
