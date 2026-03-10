import argparse
import datetime
import os
import random
import uuid

import draccus
import ray

from vla_foundry.aws.s3_utils import get_aws_credentials_env
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


@ray.remote(memory=2 * 1024 * 1024 * 1024)  # 2GB per episode worker
def streaming_episode_worker(episode_path: str, converter, statistics_ray_actor, logger_actor):
    return converter.process_episode(episode_path, statistics_ray_actor, logger_actor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True)
    args, _ = parser.parse_known_args()
    cfg = draccus.parse(config_class=TYPE_MAPPER[args.type])

    # Safety check: ensure output directory doesn't have existing preprocessing outputs
    frames_dir = os.path.join(cfg.output_dir, "frames")
    existing_episode_files = check_directory_has_files_with_substring(frames_dir, "_frame_")
    if existing_episode_files:
        error_msg = (
            f"❌ ERROR: Output directory is not empty!\n"
            f"The output directory contains {len(existing_episode_files)} existing episode files':\n"
            f"  Output directory: {cfg.output_dir}\n"
            f"  Example files: {', '.join(existing_episode_files[:5])}"
            f"{'...' if len(existing_episode_files) > 5 else ''}\n"
            f"Pre-processing in a non-empty output directory is unsafe"
        )
        raise RuntimeError(error_msg)

    # Initialize Ray - forward AWS credentials to workers when needed for S3 I/O
    runtime_env = {"env_vars": {}}

    # Capture git info on head node and pass to workers (since .git is excluded)
    runtime_env["env_vars"].update(get_git_env_vars())
    # Capture the user who launched the job (head node user, not worker node user)
    if os.environ.get("USER"):
        runtime_env["env_vars"]["VLA_LAUNCHED_BY"] = os.environ["USER"]
    # Forward AWS credentials from head node to workers (avoids flaky IMDS on workers)
    runtime_env["env_vars"].update(get_aws_credentials_env())

    if cfg.ray_address:
        ray.init(address=cfg.ray_address, runtime_env=runtime_env)
        print(f"Connected to Ray cluster at {cfg.ray_address}")
    else:
        try:
            ray.init(
                address="auto",
                num_cpus=cfg.ray_num_cpus,
                runtime_env=runtime_env | {"excludes": [".git", "*.pt", "*.pyc", "__pycache__", ".pytest_cache"]},
            )
            print("Connected to existing Ray cluster (address='auto')")
        except ConnectionError:
            ray.init(
                num_cpus=cfg.ray_num_cpus,
                runtime_env=runtime_env | {"excludes": [".git", "*.pt", "*.pyc", "__pycache__", ".pytest_cache"]},
            )
            print(f"Started new local Ray cluster with num_cpus={cfg.ray_num_cpus}")

    # Create converter
    converter = get_converter(cfg)

    # Create the derived output subdirectory that may update the full output path
    output_subdir = cfg.output_dir.rstrip("/")
    if hasattr(converter, "get_output_subdir"):
        subdir = converter.get_output_subdir()
        if subdir:
            output_subdir = f"{output_subdir}/{subdir}"
            print(f"📂 Output subdirectory: {output_subdir}")

    # Point the converter's output to the subdirectory so process_episode
    # writes frames to the same location that create_shard reads from
    converter.output_dir = output_subdir

    # Discover episodes
    print("🔍 Discovering episodes...")
    episodes = converter.discover_episodes(cfg.source_episodes, cfg.max_episodes_to_process)
    print(f"Found {len(episodes)} episodes")
    if len(episodes) == 0:
        print("❌ No episodes found!")
        return

    # Create initial processing metadata
    metadata = create_processing_metadata(cfg, episodes)
    metadata["processing"]["timestamp_start"] = datetime.datetime.now().isoformat()

    # Ray Phase 1: Process frame individually and upload to S3
    print(f"🚀 Processing {len(episodes)} episodes and uploading to S3...")
    if cfg.compute_statistics:
        statistics_ray_actor = StreamingDatasetStatisticsRayActor.remote(compute_stats=cfg.compute_statistics)
    else:
        statistics_ray_actor = None
    logger_actor = LoggerActor.remote()

    # Conditionally request GPUs based on whether we're processing depth data for point clouds
    if cfg.use_depth_data:
        # Check if GPUs are available
        available_gpus = ray.cluster_resources().get("GPU", 0)
        if available_gpus == 0:
            raise RuntimeError(
                "❌ ERROR: use_depth_data=True requires GPUs for CUDA FPS point cloud processing, "
                "but no GPUs are available in the Ray cluster. "
                "Either provide GPUs or set use_depth_data=False."
            )

        # Calculate expected number of parallel workers
        expected_workers = int(available_gpus / cfg.ray_num_gpus_per_worker)
        print(
            f"🎮 GPU mode: Requesting {cfg.ray_num_gpus_per_worker} GPU per worker for CUDA FPS point cloud processing"
        )
        print(f"   📊 Cluster resources: {available_gpus} GPUs → {expected_workers} parallel workers")

        futures = [
            streaming_episode_worker.options(num_gpus=cfg.ray_num_gpus_per_worker).remote(
                episode, converter, statistics_ray_actor, logger_actor
            )
            for episode in episodes
        ]
    else:
        print("💻 CPU mode: Processing without point clouds")
        futures = [
            streaming_episode_worker.remote(episode, converter, statistics_ray_actor, logger_actor)
            for episode in episodes
        ]
    results = ray.get(futures)
    results = [result for result in results if result is not None]  # Remove None results
    results = [i for result in results for i in result]  # Result is a list of lists, flatten it
    print("✅ Upload phase complete! Starting sharding phase...")

    # Ray Phase 2: Shuffle and group files into shards in parallel
    random.shuffle(results)
    shards = [results[i : i + cfg.samples_per_shard] for i in range(0, len(results), cfg.samples_per_shard)]
    print(f"Creating {len(shards)} shards with up to {cfg.samples_per_shard} samples each")
    shard_futures = [create_shard.remote(shard_files, i, output_subdir) for i, shard_files in enumerate(shards)]
    shard_results = ray.get(shard_futures)
    print(f"✅ Created {len(shard_results)} shards.")

    # Ray Phase 3: Group files by episode and create episode-based shards
    episode_groups = {}
    for filename in results:
        # filename format: {unique_id}_{episode_id}_frame_{frame_idx}.tar
        episode_key = filename.rsplit("_frame_", 1)[0]
        episode_groups.setdefault(episode_key, []).append(filename)
    print(f"Creating {len(episode_groups)} episode shards")
    episode_shard_futures = [
        create_episode_shard.remote(files, episode_key, output_subdir) for episode_key, files in episode_groups.items()
    ]
    episode_shard_results = ray.get(episode_shard_futures)
    print(f"✅ Created {len(episode_shard_results)} episode shards.")

    # Upload episode manifest to S3 in the episodes/ directory
    episode_manifest_lines = []
    for shard_name, num_sequences in episode_shard_results:
        episode_manifest_lines.append({"shard": shard_name, "num_sequences": num_sequences})
    save_and_upload_dict(episode_manifest_lines, f"{output_subdir}/episodes", "manifest.jsonl")

    # Upload shards manifest to S3 in the shards/ directory
    manifest_lines = []
    for shard_name, num_sequences in shard_results:
        manifest_entry = {"shard": shard_name, "num_sequences": num_sequences}
        manifest_lines.append(manifest_entry)
    save_and_upload_dict(manifest_lines, f"{output_subdir}/shards", "manifest.jsonl")

    # Upload statistics to S3 in the shards/ directory and the episodes/ directory
    if cfg.compute_statistics:
        statistics_state = statistics_ray_actor.get_statistics.remote()
        statistics_state = ray.get(statistics_state)
        save_and_upload_dict(statistics_state, f"{output_subdir}/shards", "stats.json")
        save_and_upload_dict(statistics_state, f"{output_subdir}/episodes", "stats.json")

    # Update and save processing metadata with final statistics
    metadata["processing"]["total_samples_created"] = sum(num_sequences for _, num_sequences in shard_results)
    metadata["processing"]["timestamp_end"] = datetime.datetime.now().isoformat()
    metadata["processing"]["sample_counts"] = ray.get(logger_actor.get_values.remote())
    print("Sample counts:", metadata["processing"]["sample_counts"])
    save_and_upload_dict(metadata, f"{output_subdir}/shards", "processing_metadata.json")
    preprocessing_config_dict = vars(cfg).copy()
    save_and_upload_config(preprocessing_config_dict, f"{output_subdir}/shards", "preprocessing_config.yaml")

    # Make a copy of the output directory when source/destination backends match
    dataset_uuid = str(uuid.uuid4())
    fixed_path = f"{cfg.output_dir_fixed_path.rstrip('/')}/{dataset_uuid}"

    # Log to DynamoDB
    sample_counts = metadata["processing"]["sample_counts"]
    log_dataset_preprocessing(
        dataset_uuid=dataset_uuid,
        cfg=cfg,
        dataset_type=cfg.type,
        source_paths=cfg.source_episodes if isinstance(cfg.source_episodes, list) else [cfg.source_episodes],
        target_path=f"{output_subdir}/shards",
        fixed_path=f"{fixed_path}/shards",
        episode_count=len(episodes),
        frame_count=sample_counts.get("total_frames", 0) if sample_counts else 0,
        samples_per_shard=cfg.samples_per_shard,
        num_shards=len(shard_results),
        total_samples=metadata["processing"]["total_samples_created"],
        enabled=cfg.db_logging,
    )
    src_is_s3 = output_subdir.startswith("s3://")
    dst_is_s3 = fixed_path.startswith("s3://")
    if src_is_s3 and dst_is_s3:
        recursive_s3_copy(output_subdir, fixed_path)
    else:
        print(f"⚠️ Skipping fixed-path copy (only enabled for S3 -> S3): {output_subdir} -> {fixed_path}")

    ray.shutdown()
    print("🎉 Complete! All samples uploaded and sharded.")


if __name__ == "__main__":
    main()
