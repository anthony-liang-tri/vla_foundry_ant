import logging
import os

import ray

# Reduce Ray logging verbosity
logging.getLogger("ray").setLevel(logging.WARNING)

# Default camera names for real data (6 cameras)
DEFAULT_CAMERA_NAMES = [
    "scene_right_0",
    "scene_left_0",
    "wrist_left_minus",
    "wrist_left_plus",
    "wrist_right_minus",
    "wrist_right_plus",
]

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILENAMES_PATH = os.path.join(SCRIPT_DIR, "stage3_real_filenames.txt")


def load_tasks_from_filenames(filenames_path: str) -> dict:
    """Load tasks from filenames file. Returns dict mapping task_name -> list of episode paths."""
    tasks = {}
    if not os.path.exists(filenames_path):
        print(f"⚠️  Warning: Filenames file not found: {filenames_path}")
        return tasks

    with open(filenames_path) as f:
        for line in f:
            if not line.strip():
                continue
            task_name = line.strip().removeprefix("s3://robotics-manip-lbm/efs/data/tasks/").split("/")[0]
            if task_name not in tasks:
                tasks[task_name] = []
            tasks[task_name].append(line.strip())
    return tasks


@ray.remote(num_cpus=4)  # Reserve 4 CPUs per task to limit concurrency (prevents memory overload)
def process_task(
    task_name: str,
    episodes: list[str],
    camera_names: list[str],
    force: bool = False,
    skip_missing_cameras: bool = False,
):
    """Process a single task - runs the full preprocessing pipeline."""
    import datetime
    import random
    import subprocess
    import uuid

    import draccus

    from vla_foundry.data.preprocessing.metadata_utils import create_processing_metadata
    from vla_foundry.data.preprocessing.robotics.converters import get_converter
    from vla_foundry.data.preprocessing.robotics.preprocess_params import TYPE_MAPPER
    from vla_foundry.data.preprocessing.robotics.preprocess_statistics import (
        LoggerActor,
        StreamingDatasetStatisticsRayActor,
    )
    from vla_foundry.data.preprocessing.utils import (
        create_shard,
        recursive_s3_copy,
        upload_config_to_s3,
        upload_dict_to_s3,
    )

    output_dir = f"s3://tri-ml-datasets-uw2/vla_foundry_datasets/v0.4.2/{task_name}/"

    # Safety check: fail if output directory already has data (unless --force)
    if not force:
        check_result = subprocess.run(
            ["aws", "s3", "ls", f"{output_dir}episodes/", "--recursive"],
            capture_output=True,
            text=True,
        )
        existing_files = [line for line in check_result.stdout.strip().split("\n") if line and "_frame_" in line]
        if existing_files:
            raise RuntimeError(
                f"❌ ERROR: Output directory already contains {len(existing_files)} episode files!\n"
                f"  Output directory: {output_dir}\n"
                f"  Use --force to overwrite existing data."
            )

    print(f"{'=' * 50}\n🚀 Starting task: {task_name} ({len(episodes)} episodes)\n{'=' * 50}")

    # Build config args - match original format exactly
    # source_episodes was: '"[{}]"'.format(", ".join([f"'{episode}'" for episode in episodes]))
    # camera_names was just: str(list) which gives ['a', 'b', 'c']
    source_episodes_str = "[{}]".format(", ".join([f"'{ep}'" for ep in episodes]))
    camera_names_str = str(camera_names)  # Gives ['scene_right_0', 'scene_left_0', ...]

    # Parse config using draccus
    import sys

    sys.argv = [
        "preprocess_robotics_to_tar.py",
        "--type",
        "spartan",
        "--source_episodes",
        source_episodes_str,
        "--output_dir",
        output_dir,
        "--past_lowdim_steps",
        "3",
        "--future_lowdim_steps",
        "19",
        "--max_padding_left",
        "5",
        "--max_padding_right",
        "16",
        "--data_discard_keys",
        "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml",
        "--camera_names",
        camera_names_str,
        "--skip_episodes_missing_cameras",
        str(skip_missing_cameras),
        "--language_annotations_path",
        "vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml",
        "--action_fields_config_path",
        "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml",
        "--samples_per_shard",
        "100",
        "--resize_images_size",
        "[342, 256]",
        "--config_path",
        "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml",
    ]

    cfg = draccus.parse(config_class=TYPE_MAPPER["spartan"])

    # Create converter
    converter = get_converter(cfg)

    # Discover episodes
    discovered_episodes = converter.discover_episodes(cfg.source_episodes, cfg.max_episodes_to_process)
    print(f"[{task_name}] Found {len(discovered_episodes)} episodes")
    if len(discovered_episodes) == 0:
        print(f"[{task_name}] ❌ No episodes found!")
        return task_name, False, "No episodes found"

    # Create initial processing metadata
    metadata = create_processing_metadata(cfg, discovered_episodes)
    metadata["processing"]["timestamp_start"] = datetime.datetime.now().isoformat()

    # Process episodes - submit as nested Ray tasks
    # Set memory constraints on actors to prevent OOM
    print(f"[{task_name}] 🚀 Processing {len(discovered_episodes)} episodes...")
    if cfg.compute_statistics:
        # Use max_concurrency to handle many concurrent updates without timeouts
        statistics_ray_actor = StreamingDatasetStatisticsRayActor.options(
            num_cpus=1,
            max_concurrency=1000,  # Allow large queue to prevent timeouts
        ).remote(compute_stats=cfg.compute_statistics)
    else:
        statistics_ray_actor = None
    logger_actor = LoggerActor.options(num_cpus=0.1).remote()  # Minimal CPU for logging actor

    # Use streaming_episode_worker pattern but inline
    @ray.remote(num_cpus=1)  # 1 CPU per episode worker to control concurrency
    def episode_worker(episode_path, conv, stats_actor, log_actor):
        return conv.process_episode(episode_path, stats_actor, log_actor)

    futures = [episode_worker.remote(ep, converter, statistics_ray_actor, logger_actor) for ep in discovered_episodes]
    results = ray.get(futures)
    results = [r for r in results if r is not None]
    results = [i for r in results for i in r]  # Flatten
    print(f"[{task_name}] ✅ Episode processing complete!")

    # Clear existing shards before re-sharding (in case of partial shard run)
    subprocess.run(["aws", "s3", "rm", "--recursive", f"{output_dir}shards/"], capture_output=True)

    # Sharding phase
    random.shuffle(results)
    shards = [results[i : i + cfg.samples_per_shard] for i in range(0, len(results), cfg.samples_per_shard)]
    print(f"[{task_name}] Creating {len(shards)} shards...")
    shard_futures = [create_shard.remote(shard_files, i, cfg.output_dir) for i, shard_files in enumerate(shards)]
    shard_results = ray.get(shard_futures)
    print(f"[{task_name}] ✅ Created {len(shard_results)} shards.")

    # Upload manifest
    manifest_lines = [{"shard": shard_name, "num_sequences": num_seq} for shard_name, num_seq in shard_results]
    upload_dict_to_s3(manifest_lines, f"{cfg.output_dir.rstrip('/')}/shards", "manifest.jsonl")

    # Upload statistics
    if cfg.compute_statistics:
        statistics_state = ray.get(statistics_ray_actor.get_statistics.remote())
        upload_dict_to_s3(statistics_state, f"{cfg.output_dir.rstrip('/')}/shards", "stats.json")

    # Update metadata
    metadata["processing"]["total_samples_created"] = sum(num_seq for _, num_seq in shard_results)
    metadata["processing"]["timestamp_end"] = datetime.datetime.now().isoformat()
    metadata["processing"]["sample_counts"] = ray.get(logger_actor.get_values.remote())
    upload_dict_to_s3(metadata, f"{cfg.output_dir.rstrip('/')}/shards", "processing_metadata.json")
    upload_config_to_s3(cfg, f"{cfg.output_dir.rstrip('/')}/shards", "preprocessing_config.yaml")

    # Copy to fixed path
    dataset_uuid = str(uuid.uuid4())
    recursive_s3_copy(cfg.output_dir, f"{cfg.output_dir_fixed_path.rstrip('/')}/{dataset_uuid}")

    print(f"{'=' * 50}\n✅ Task completed: {task_name}\n{'=' * 50}")
    return task_name, True, metadata["processing"]["total_samples_created"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run on local machine (starts local Ray instance)")
    parser.add_argument(
        "--on-head", action="store_true", help="Run on cluster head node (connects to existing cluster)"
    )
    parser.add_argument("--force", action="store_true", help="Force overwrite existing data")
    parser.add_argument("--tasks-file", type=str, help="File with task names to run (one per line)")
    parser.add_argument("--batch-size", type=int, default=0, help="Max tasks to run in parallel (0=all)")
    parser.add_argument(
        "--camera-names", type=str, default=None, help="Comma-separated camera names (default: 4 cameras)"
    )
    parser.add_argument(
        "--skip-missing-cameras",
        action="store_true",
        help="Skip episodes that don't have ALL requested cameras. Default: process with available cameras.",
    )
    args = parser.parse_args()

    # Parse camera names
    camera_names = [c.strip() for c in args.camera_names.split(",")] if args.camera_names else DEFAULT_CAMERA_NAMES

    # Load task filter from file if specified
    tasks_to_run = None
    if args.tasks_file:
        with open(args.tasks_file) as f:
            tasks_to_run = set(line.strip() for line in f if line.strip())
        print(f"Filtering to {len(tasks_to_run)} tasks from {args.tasks_file}")

    # Initialize Ray based on mode
    if args.local:
        # Run on local machine - start a local Ray instance
        print("🚀 Starting local Ray instance...")
        ray.init()
    elif args.on_head:
        # Running on the head node (via ray attach, ray submit, or job submission)
        # Connect to the existing cluster
        ray.init(address="auto")
    else:
        # Running locally - start cluster and submit job
        import subprocess

        from ray.job_submission import JobSubmissionClient

        cluster_config = "vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml"
        print("🚀 Starting Ray cluster...")
        # Use legacy autoscaler (v1) to avoid v2 reconciler killing nodes before tracking them
        env = os.environ.copy()
        # env["RAY_AUTOSCALER_V2"] = "0"
        # env["RAY_enable_autoscaler_v2"] = "0"
        subprocess.run(["ray", "up", cluster_config, "-y", "--no-config-cache"], check=True, env=env)

        # Get head node IP (filter out log lines, get just the IP)
        result = subprocess.run(["ray", "get-head-ip", cluster_config], capture_output=True, text=True, check=True)
        # Extract just the IP address (last line that looks like an IP)
        import re

        lines = result.stdout.strip().split("\n")
        head_ip = None
        for line in reversed(lines):
            line = line.strip()
            if re.match(r"^\d+\.\d+\.\d+\.\d+$", line):
                head_ip = line
                break
        if not head_ip:
            raise RuntimeError(f"Could not parse head IP from output: {result.stdout}")
        print(f"Head node IP: {head_ip}")

        # Submit job via Ray Jobs API
        client = JobSubmissionClient(f"http://{head_ip}:8265")
        entrypoint = "python vla_foundry/tri/stage3_singletask_real/data_generation_ray_v2.py --on-head"
        if args.force:
            entrypoint += " --force"
        if args.tasks_file:
            entrypoint += f" --tasks-file {args.tasks_file}"
        if args.batch_size:
            entrypoint += f" --batch-size {args.batch_size}"
        if args.camera_names:
            entrypoint += f" --camera-names {args.camera_names}"
        if args.skip_missing_cameras:
            entrypoint += " --skip-missing-cameras"
        job_id = client.submit_job(
            entrypoint=entrypoint,
            runtime_env={"working_dir": ".", "excludes": [".git", "*.pt", "*.pyc", "__pycache__", ".pytest_cache"]},
        )
        print(f"Submitted job: {job_id}")
        print(f"Monitor at: http://{head_ip}:8265/#/jobs/{job_id}")

        # Stream logs
        async def stream_logs():
            async for lines in client.tail_job_logs(job_id):
                print(lines, end="")

        import asyncio

        asyncio.run(stream_logs())
        exit(0)

    print(f"Connected to Ray cluster: {ray.cluster_resources()}")

    # Load tasks from filenames file
    tasks = load_tasks_from_filenames(FILENAMES_PATH)
    total_episodes = sum(len(eps) for eps in tasks.values())
    print(f"📁 Loaded {len(tasks)} tasks ({total_episodes} total episodes) from {FILENAMES_PATH}")

    # Debug: show first few tasks and their episode counts
    print(f"   Tasks found: {list(tasks.keys())[:10]}{'...' if len(tasks) > 10 else ''}")
    for task_name, episodes in list(tasks.items())[:5]:
        print(f"   - {task_name}: {len(episodes)} episodes")
    if len(tasks) > 5:
        print(f"   ... and {len(tasks) - 5} more tasks")

    if not tasks:
        print("❌ No tasks found! Check that the filenames file exists and has content.")
        ray.shutdown()
        exit(1)

    # Filter tasks if --tasks-file specified
    tasks_to_process = tasks
    if tasks_to_run:
        print(f"\n🔍 Filtering to {len(tasks_to_run)} tasks from --tasks-file")
        print(f"   Tasks to run: {list(tasks_to_run)[:10]}{'...' if len(tasks_to_run) > 10 else ''}")
        tasks_to_process = {k: v for k, v in tasks.items() if k in tasks_to_run}
        matched = set(tasks_to_process.keys())
        skipped = tasks_to_run - matched
        print(f"   Matched {len(matched)} tasks: {list(matched)[:10]}{'...' if len(matched) > 10 else ''}")
        if skipped:
            print(f"   ⚠️  {len(skipped)} tasks in filter file but not found in source data:")
            for s in list(skipped)[:10]:
                print(f"      - {s}")
            if len(skipped) > 10:
                print(f"      ... and {len(skipped) - 10} more")

    # Build list of tasks to submit - use same camera names for all tasks
    task_items = []
    for task_name, episodes in tasks_to_process.items():
        task_items.append((task_name, episodes, camera_names))

    # Submit tasks (in batches if --batch-size specified)
    batch_size = args.batch_size if args.batch_size > 0 else len(task_items)
    print(f"\n🚀 Launching {len(task_items)} tasks (batch size: {batch_size})...\n")

    results = []
    for i in range(0, len(task_items), batch_size):
        batch = task_items[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1} ({len(batch)} tasks)...")
        futures = [
            process_task.remote(name, eps, cams, args.force, args.skip_missing_cameras) for name, eps, cams in batch
        ]
        batch_results = ray.get(futures)
        results.extend(batch_results)

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    success_count = 0
    for task_name, success, info in results:
        status = "✅" if success else "❌"
        print(f"  {status} {task_name}: {info}")
        if success:
            success_count += 1
    print(f"\nCompleted: {success_count}/{len(results)} tasks")

    ray.shutdown()
