import datetime
import os
import random

import draccus
import ray

from vla_foundry.data.scripts.preprocessing.metadata_utils import create_processing_metadata
from vla_foundry.data.scripts.preprocessing.robotics.converters import get_converter
from vla_foundry.data.scripts.preprocessing.robotics.preprocess_params import PreprocessParams
from vla_foundry.data.scripts.preprocessing.robotics.preprocess_statistics import (
    LoggerActor,
    StreamingDatasetStatisticsRayActor,
)
from vla_foundry.data.scripts.preprocessing.utils import create_shard, upload_config_to_s3, upload_dict_to_s3
from vla_foundry.file_utils import check_directory_has_files_with_substring


@ray.remote
def streaming_episode_worker(episode_path: str, converter, statistics_ray_actor, logger_actor):
    return converter.process_episode(episode_path, statistics_ray_actor, logger_actor)


def main():
    cfg = draccus.parse(config_class=PreprocessParams)

    # Safety check: ensure output directory doesn't have existing preprocessing outputs
    episodes_dir = os.path.join(cfg.output_dir, "episodes")
    existing_episode_files = check_directory_has_files_with_substring(episodes_dir, "_frame_")
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

    # Initialize Ray
    if cfg.ray_address:
        ray.init(address=cfg.ray_address)
        print(f"Connected to Ray cluster at {cfg.ray_address}")
    else:
        ray.init(
            address="auto",
            num_cpus=cfg.ray_num_cpus,
            runtime_env={"excludes": [".git", "*.pt", "*.pyc", "__pycache__", ".pytest_cache"]},
        )
        print(f"Started auto Ray cluster with num_cpus={cfg.ray_num_cpus}")

    # Create converter
    converter = get_converter(cfg.source_type, cfg)

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

    # Upload manifest to S3 in the same directory as the tar files
    manifest_lines = []
    for shard_name, num_sequences in shard_results:
        manifest_entry = {"shard": shard_name, "num_sequences": num_sequences}
        manifest_lines.append(manifest_entry)
    upload_dict_to_s3(manifest_lines, f"{cfg.output_dir.rstrip('/')}/shards", "manifest.jsonl")

    # Upload statistics to S3 in the same directory as the tar files
    if cfg.compute_statistics:
        statistics_state = statistics_ray_actor.get_statistics.remote()
        statistics_state = ray.get(statistics_state)
        upload_dict_to_s3(statistics_state, f"{cfg.output_dir.rstrip('/')}/shards", "stats.json")

    # Update and save processing metadata with final statistics
    metadata["processing"]["total_samples_created"] = sum(num_sequences for _, num_sequences in shard_results)
    metadata["processing"]["timestamp_end"] = datetime.datetime.now().isoformat()
    metadata["processing"]["sample_counts"] = ray.get(logger_actor.get_values.remote())
    print("Sample counts:", metadata["processing"]["sample_counts"])
    upload_dict_to_s3(metadata, f"{cfg.output_dir.rstrip('/')}/shards", "processing_metadata.json")
    upload_config_to_s3(cfg, f"{cfg.output_dir.rstrip('/')}/shards", "preprocessing_config.yaml")

    ray.shutdown()
    print("🎉 Complete! All samples uploaded and sharded.")


if __name__ == "__main__":
    main()
