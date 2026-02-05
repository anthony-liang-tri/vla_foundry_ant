import argparse
import datetime
import os
import random
import uuid

import boto3
import draccus
import ray

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
from vla_foundry.file_utils import check_directory_has_files_with_substring


@ray.remote
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

    # Initialize Ray - forward AWS credentials to workers
    runtime_env = {"env_vars": {}}
    aws_profile = os.environ.get("AWS_PROFILE")
    if aws_profile:
        runtime_env["env_vars"]["AWS_PROFILE"] = aws_profile

    # Explicitly forward AWS credentials from head node to workers
    # This avoids reliance on IMDS on worker nodes, which can be flaky
    session = boto3.Session()
    credentials = session.get_credentials()
    if credentials:
        credentials = credentials.get_frozen_credentials()
        if credentials.access_key:
            runtime_env["env_vars"]["AWS_ACCESS_KEY_ID"] = credentials.access_key
        if credentials.secret_key:
            runtime_env["env_vars"]["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
        if credentials.token:
            runtime_env["env_vars"]["AWS_SESSION_TOKEN"] = credentials.token

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
    futures = [
        streaming_episode_worker.remote(episode, converter, statistics_ray_actor, logger_actor) for episode in episodes
    ]
    results = ray.get(futures)
    results = [result for result in results if result is not None]  # Remove None results
    results = [i for result in results for i in result]  # Result is a list of lists, flatten it
    print("✅ Upload phase complete! Starting sharding phase...")

    # Ray Phase 2: Shuffle and group files into shards in parallel
    random.shuffle(results)
    shards = [results[i : i + cfg.samples_per_shard] for i in range(0, len(results), cfg.samples_per_shard)]
    print(f"Creating {len(shards)} shards with up to {cfg.samples_per_shard} samples each")
    shard_futures = [create_shard.remote(shard_files, i, cfg.output_dir) for i, shard_files in enumerate(shards)]
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
        create_episode_shard.remote(files, episode_key, cfg.output_dir) for episode_key, files in episode_groups.items()
    ]
    episode_shard_results = ray.get(episode_shard_futures)
    print(f"✅ Created {len(episode_shard_results)} episode shards.")

    # Upload episode manifest to S3 in the episodes/ directory
    episode_manifest_lines = []
    for shard_name, num_sequences in episode_shard_results:
        episode_manifest_lines.append({"shard": shard_name, "num_sequences": num_sequences})
    save_and_upload_dict(episode_manifest_lines, f"{cfg.output_dir.rstrip('/')}/episodes", "manifest.jsonl")

    # Upload shards manifest to S3 in the shards/ directory
    manifest_lines = []
    for shard_name, num_sequences in shard_results:
        manifest_entry = {"shard": shard_name, "num_sequences": num_sequences}
        manifest_lines.append(manifest_entry)
    save_and_upload_dict(manifest_lines, f"{cfg.output_dir.rstrip('/')}/shards", "manifest.jsonl")

    # Upload statistics to S3 in the shards/ directory and the episodes/ directory
    if cfg.compute_statistics:
        statistics_state = statistics_ray_actor.get_statistics.remote()
        statistics_state = ray.get(statistics_state)
        save_and_upload_dict(statistics_state, f"{cfg.output_dir.rstrip('/')}/shards", "stats.json")
        save_and_upload_dict(statistics_state, f"{cfg.output_dir.rstrip('/')}/episodes", "stats.json")

    # Update and save processing metadata with final statistics
    metadata["processing"]["total_samples_created"] = sum(num_sequences for _, num_sequences in shard_results)
    metadata["processing"]["timestamp_end"] = datetime.datetime.now().isoformat()
    metadata["processing"]["sample_counts"] = ray.get(logger_actor.get_values.remote())
    print("Sample counts:", metadata["processing"]["sample_counts"])
    save_and_upload_dict(metadata, f"{cfg.output_dir.rstrip('/')}/shards", "processing_metadata.json")
    preprocessing_config_dict = vars(cfg).copy()
    save_and_upload_config(
        preprocessing_config_dict, f"{cfg.output_dir.rstrip('/')}/shards", "preprocessing_config.yaml"
    )

    # Make a copy of the ouput directory
    if cfg.output_dir.startswith("s3://"):
        dataset_uuid = str(uuid.uuid4())
        recursive_s3_copy(cfg.output_dir, f"{cfg.output_dir_fixed_path.rstrip('/')}/{dataset_uuid}")

    ray.shutdown()
    print("🎉 Complete! All samples uploaded and sharded.")


if __name__ == "__main__":
    main()
