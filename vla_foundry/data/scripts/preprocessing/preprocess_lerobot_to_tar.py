"""
We probably want this file as self-contained as possible, so we define most functions here.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import re
import subprocess
import tarfile
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import numpy as np
import pyarrow.parquet as pq
import ray
from botocore.config import Config
from PIL import Image

from vla_foundry.data.scripts.preprocessing.image_utils import image_to_bytes, init_jpeg_encoder
from vla_foundry.data.scripts.preprocessing.preprocess_lbm_to_tar import PaddingStrategy, create_shard
from vla_foundry.file_utils import (
    check_directory_has_files_with_prefix,
    copy_to_temp_file,
    file_exists,
    json_load,
    jsonl_load,
    list_directory,
)


def make_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


def create_metadata_and_lowdim_dicts(
    df, df_dict, frame_idx, num_past, num_future, pad_fn, lowdim_columns, camera_names, excluded_columns=None
):
    excluded_columns = excluded_columns or []
    metadata_dict, lowdim_dict = {}, {}

    past_nonpad = frame_idx if frame_idx - num_past < 0 else num_past
    past_pad = num_past - past_nonpad
    future_nonpad = len(df) - frame_idx - 1 if frame_idx + num_future >= len(df) else num_future
    future_pad = num_future - future_nonpad

    for col in df.columns:
        if col in lowdim_columns:
            series = df.iloc[frame_idx - past_nonpad : frame_idx + future_nonpad + 1][col]
            if len(series) > 0 and isinstance(series.iloc[0], np.ndarray):
                data = np.stack(series.values)
            else:
                data = series.values
            lowdim_dict[col] = pad_fn(data, past_pad, future_pad)
        elif col not in excluded_columns:
            metadata_dict[col] = df_dict[frame_idx][col]

    metadata_dict["camera_names"] = camera_names
    return metadata_dict, lowdim_dict


def create_past_and_future_masks(idx, num_past, num_future, episode_length):
    total_length = num_past + num_future + 1
    past_mask = np.ones(total_length, dtype=bool)
    future_mask = np.ones(total_length, dtype=bool)
    # Current time step is part of the future for low dim data
    past_mask[num_past:] = False
    future_mask[:num_past] = False

    # Check padding
    past_padding = max(0, -(idx - num_past))
    future_padding = max(0, (idx + num_future) - episode_length + 1)

    if past_padding > 0:
        past_mask[:past_padding] = False
        future_mask[:past_padding] = False
    if future_padding > 0:
        future_mask[-future_padding:] = False
        past_mask[-future_padding:] = False

    return past_mask, future_mask


def add_sample_metadata_to_tar(
    tar, file_prefix, df, df_dict, frame_idx, config, pad_fn, camera_names, row, original_image_sizes
):
    # Create metadata and lowdim dictionaries
    metadata_dict, lowdim_dict = create_metadata_and_lowdim_dicts(
        df=df,
        df_dict=df_dict,
        frame_idx=frame_idx,
        num_past=config["num_past"],
        num_future=config["num_future"],
        pad_fn=pad_fn,
        lowdim_columns=config["lowdim_columns"],
        camera_names=camera_names,
        excluded_columns=list(camera_names) + config["lowdim_columns"],
    )

    past_mask, future_mask = create_past_and_future_masks(frame_idx, config["num_past"], config["num_future"], len(df))

    # Create language_instructions dictionary
    language_instructions_dict = {"original": config["tasks_dict"][row[config["task_index_col"]]]["task"]}

    # Add original image sizes to metadata if available
    if original_image_sizes:
        metadata_dict["original_image_sizes"] = original_image_sizes

    # Add to tar
    add_metadata_json_to_tar(tar, file_prefix, metadata_dict)
    add_npz_to_tar(tar, file_prefix, lowdim_dict, "lowdim.npz")
    add_npz_to_tar(tar, file_prefix, {"data": past_mask}, "past_mask.npz")
    add_npz_to_tar(tar, file_prefix, {"data": future_mask}, "future_mask.npz")
    add_language_instructions_to_tar(tar, file_prefix, language_instructions_dict)


def get_image_bytes_from_parquet(df_dict, img_timestep, image_col):
    """Extract image bytes from parquet data at a specific timestep."""
    image_data = df_dict[img_timestep][image_col]
    if isinstance(image_data, dict) and "bytes" in image_data:
        return image_data["bytes"]  # LeRobot format: {'bytes': b'...', ...}
    elif isinstance(image_data, bytes):
        return image_data  # Raw bytes
    else:
        raise ValueError(f"Unsupported image data format in column {image_col}")


def add_to_tar(tar, filename, content_bytes):
    """Add a file to tar archive."""
    tarinfo = tarfile.TarInfo(name=filename)
    tarinfo.size = len(content_bytes)
    tar.addfile(tarinfo, io.BytesIO(content_bytes))


def add_metadata_json_to_tar(tar, file_prefix, metadata_dict):
    """Add JSON metadata file to tar archive."""
    json_bytes = json.dumps(make_json_serializable(metadata_dict), indent=2).encode("utf-8")
    add_to_tar(tar, f"{file_prefix}.metadata.json", json_bytes)


def add_npz_to_tar(tar, file_prefix, npz_dict, filename):
    """Add NPZ lowdim file to tar archive."""
    npz_buffer = io.BytesIO()
    np.savez_compressed(npz_buffer, **npz_dict)
    add_to_tar(tar, f"{file_prefix}.{filename}", npz_buffer.getvalue())


def add_language_instructions_to_tar(tar, file_prefix, language_instructions_dict):
    """Add language instruction file to tar archive."""
    json_bytes = json.dumps(make_json_serializable(language_instructions_dict), indent=2).encode("utf-8")
    add_to_tar(tar, f"{file_prefix}.language_instructions.json", json_bytes)


def resize_image_bytes(image_bytes, jpeg_quality, resize_size):
    if resize_size is None:
        return image_bytes, Image.open(io.BytesIO(image_bytes)).size

    # Decode JPEG to numpy array
    pil_image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(pil_image)

    # Use image_to_bytes to resize and encode (expects target_size as (width, height) for PIL)
    target_size = (resize_size[1], resize_size[0])  # Convert [height, width] to (width, height)
    resized_bytes, original_size = image_to_bytes(image_array, jpeg_quality, target_size)
    return resized_bytes, original_size


def add_image_to_tar(tar, file_prefix, camera_name, image_bytes):
    """Add image file to tar archive."""
    add_to_tar(tar, f"{file_prefix}.{camera_name}.jpg", image_bytes)


def parse_args():
    parser = argparse.ArgumentParser()

    # Input/Output paths
    parser.add_argument("--dataset_path", type=str, required=True, help="All other paths become relative to this.")
    parser.add_argument("--s3_output_path", type=str, required=True)
    parser.add_argument("--tmp_dir", type=str, default=None, help="Directory for tempfiles. If None, defaults to /tmp")

    # Processing parameters
    parser.add_argument("--shard_size", type=int, default=512, help="Number of rows per tar shard")
    parser.add_argument("--fps", type=float, default=None, help="Auto-detected from info_path if not specified")

    # Ray configuration
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address (default: auto)")
    parser.add_argument("--ray-num-cpus", type=int, default=None, help="Number of CPUs for Ray (default: auto-detect)")
    parser.add_argument(
        "--max-concurrent-shards", type=int, default=None, help="For batch processing (default: num_cpus)"
    )

    # File/column naming (probably no need to change any of these for standard LeRobot)
    parser.add_argument("--meta_episodes_path", type=str, default="meta/episodes.jsonl")
    parser.add_argument("--info_path", type=str, default="meta/info.json")
    parser.add_argument("--tasks_path", type=str, default="meta/tasks.jsonl")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--videos_path", type=str, default="videos")
    parser.add_argument(
        "--lowdim_columns", type=str, nargs="+", default=["action", "observation.state", "actions", "observations"]
    )
    parser.add_argument("--frame_index_col", type=str, default="frame_index")
    parser.add_argument("--episode_index_col", type=str, default="episode_index")
    parser.add_argument("--task_index_col", type=str, default="task_index")
    parser.add_argument("--episode_file_pattern", type=str, default="episode_{:06d}.parquet")
    parser.add_argument("--video_file_pattern", type=str, default="episode_{:06d}.mp4")
    parser.add_argument("--output_prefix_pattern", type=str, default="episode_{:06d}_frame_{:06d}")
    parser.add_argument("--num_past", type=int, default=1, help="Number of past frames to include")
    parser.add_argument("--num_future", type=int, default=14, help="Number of future frames to include")
    parser.add_argument("--padding_strategy", type=str, default="zero", help="Padding strategy")
    parser.add_argument(
        "--image_indices",
        type=int,
        nargs="+",
        default=[0],
        help="Image timestep offsets relative to current frame (e.g., -1 0 for previous and current)",
    )
    parser.add_argument(
        "--upload_threads", type=int, default=10, help="Number of parallel threads for S3 uploads per worker"
    )

    # Image processing mode
    parser.add_argument(
        "--no_video", action="store_true", help="Load images from parquet columns instead of video files"
    )
    parser.add_argument("--image_columns", type=str, nargs="+", default=None, help="auto-detected if not specified")

    # Image resizing and quality
    parser.add_argument(
        "--resize_images_size",
        type=int,
        nargs=2,
        default=None,
        help="Resize images to [height, width]",
    )
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality for image compression (1-100)")

    return parser.parse_args()


@ray.remote
def build_episode_lookup_chunk(chunk_dir: str) -> Dict[int, str]:
    """Build episode lookup for a single chunk directory."""
    episode_lookup = {}
    for file in list_directory(chunk_dir):
        if file.endswith(".parquet") and "episode_" in file:
            match = re.search(r"episode_(\d+)\.parquet", file)
            if match:
                ep_num = int(match.group(1))
                episode_lookup[ep_num] = f"{chunk_dir.rstrip('/')}/{file}"
    return episode_lookup


def build_episode_lookup(data_chunks: List[str]) -> Dict[int, str]:
    print(f"Building episode lookup dict from {len(data_chunks)} chunks...")
    # Process chunks in parallel
    chunk_futures = [build_episode_lookup_chunk.remote(chunk) for chunk in data_chunks]
    chunk_results = ray.get(chunk_futures)

    # Merge results
    episode_lookup = {}
    for chunk_result in chunk_results:
        episode_lookup.update(chunk_result)
    return episode_lookup


@ray.remote
def build_video_lookup_chunk(chunk_dir: str, cameras: Dict[str, str]) -> Dict[Tuple[int, str], str]:
    """Build video lookup for a single chunk directory."""
    video_lookup = {}
    for _camera_name, camera_path in cameras.items():
        camera_dir = f"{chunk_dir.rstrip('/')}/{camera_path}"
        files = list_directory(camera_dir)
        for file in files:
            if file.endswith(".mp4") and "episode_" in file:
                match = re.search(r"episode_(\d+)\.mp4", file)
                if match:
                    ep_num = int(match.group(1))
                    video_lookup[(ep_num, camera_path)] = f"{camera_dir}/{file}"
    return video_lookup


def build_video_lookup(video_chunks: List[str], cameras: Dict[str, str]) -> Dict[Tuple[int, str], str]:
    print(f"Building video lookup dict from {len(video_chunks)} chunks and {len(cameras)} cameras...")
    # Process chunks in parallel
    chunk_futures = [build_video_lookup_chunk.remote(chunk, cameras) for chunk in video_chunks]
    chunk_results = ray.get(chunk_futures)

    # Merge results
    video_lookup = {}
    for chunk_result in chunk_results:
        video_lookup.update(chunk_result)
    return video_lookup


def resolve_path(base_path: str, relative_path: str) -> str:
    # If relative_path is already absolute (starts with s3:// or /), return as-is
    if relative_path.startswith("s3://") or relative_path.startswith("/"):
        return relative_path
    return f"{base_path.rstrip('/')}/{relative_path.lstrip('/')}"


def detect_fps(info_path: str) -> float:
    """Automatically detect FPS from info.json file."""
    info = json_load(info_path)
    if "fps" in info:
        fps = float(info["fps"])
        print(f"Detected FPS from global setting: {fps}")
    else:
        fps = 30.0
        print(f"Warning: Could not detect FPS from info.json, using default FPS {fps}")
    return fps


def discover_chunks(base_path: str, pattern: str = "chunk-*") -> List[str]:
    """Discover all chunk directories in the base path."""
    chunk_dirs = []
    for item in list_directory(base_path):
        if item.startswith("chunk-"):
            chunk_path = f"{base_path.rstrip('/')}/{item}"
            chunk_dirs.append(chunk_path)
    chunk_dirs.sort()  # Sort to ensure consistent ordering

    print(f"Discovered {len(chunk_dirs)} chunks in {base_path}: {[os.path.basename(d) for d in chunk_dirs]}")
    return chunk_dirs


def discover_cameras(video_chunks: List[str]) -> Dict[str, str]:
    """Discover available cameras by scanning video chunk directories."""
    cameras = {}

    if not video_chunks:
        return cameras

    first_chunk = video_chunks[0]
    for item in list_directory(first_chunk):
        camera_name = item.split(".")[-1] if "." in item else item
        cameras[camera_name] = item

    print(f"Discovered cameras: {list(cameras.keys())}")
    return cameras


def discover_image_columns(data_chunks: List[str], episode_file_pattern: str) -> List[str]:
    """Discover image columns by checking first available episode."""
    for chunk_dir in data_chunks:
        for episode_idx in range(10):
            episode_path = f"{chunk_dir.rstrip('/')}/{episode_file_pattern.format(episode_idx)}"
            if not file_exists(episode_path):
                continue

            # Read parquet to examine columns
            if episode_path.startswith("s3"):
                with copy_to_temp_file(episode_path) as temp_parquet:
                    df = pq.read_table(temp_parquet).to_pandas()
            else:
                df = pq.read_table(episode_path).to_pandas()

            # Find columns containing image bytes
            image_columns = []
            for col in df.columns:
                if df[col].dtype == object and len(df[col]) > 0:
                    first_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    if isinstance(first_val, (dict, bytes)) and (isinstance(first_val, bytes) or "bytes" in first_val):
                        image_columns.append(col)

            if image_columns:
                print(f"Discovered image columns: {image_columns}")
                return image_columns

    print("Warning: Could not discover image columns")
    return []


def extract_multiple_frames_ffmpeg(
    video_path: str, frame_requests: List[Tuple[int, str]], fps: float = 10.0
) -> Dict[int, bool]:
    """Extract multiple frames from a video efficiently using ffmpeg filter. Much faster than individual extractions.

    Args:
        video_path: Path to the video file (should be local)
        frame_requests: List of (frame_index, output_path) tuples
        fps: Video FPS for timestamp calculation

    Returns:
        Dict mapping frame_index to success boolean
    """
    if not frame_requests:
        return {}

    results = {}

    try:
        # Sort requests by frame index for efficiency
        sorted_requests = sorted(frame_requests, key=lambda x: x[0])

        batch_size = 1000 if len(sorted_requests) <= 1000 else 500

        for i in range(0, len(sorted_requests), batch_size):
            batch = sorted_requests[i : i + batch_size]

            # Build filter for selecting multiple frames
            frame_numbers = [str(req[0]) for req in batch]
            select_filter = f"select='{'+'.join([f'eq(n,{num})' for num in frame_numbers])}'"

            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-vf",
                select_filter,
                "-vsync",
                "0",  # Don't duplicate frames
                "-y",
                str(Path(batch[0][1]).parent / f"batch_{i}_%d.jpg"),
                "-loglevel",
                "quiet",
            ]

            with open(os.devnull, "w") as devnull:
                result = subprocess.run(cmd, stdout=devnull, stderr=devnull, stdin=subprocess.DEVNULL, timeout=30)

            # Rename extracted files to target names and check success
            for j, (frame_index, target_path) in enumerate(batch):
                extracted_path = Path(batch[0][1]).parent / f"batch_{i}_{j + 1}.jpg"
                if result.returncode == 0 and extracted_path.exists():
                    try:
                        os.rename(str(extracted_path), target_path)
                        results[frame_index] = True
                    except Exception:
                        results[frame_index] = False
                        if extracted_path.exists():
                            extracted_path.unlink()
                else:
                    results[frame_index] = False

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Batch ffmpeg extraction failed: {e}.")
    return results


@ray.remote
def write_tar_and_upload(episode_entry: Dict, config: Dict):
    try:
        s3_config = Config(max_pool_connections=50, retries={"max_attempts": 3, "mode": "adaptive"})
        s3_client = boto3.client("s3", config=s3_config)
        pad_fn = PaddingStrategy.get_pad_fn(config["padding_strategy"])

        if config["tmp_dir"] and not os.path.exists(config["tmp_dir"]):
            os.makedirs(config["tmp_dir"], exist_ok=True)

        ep_idx = episode_entry[config["episode_index_col"]]

        with tempfile.TemporaryDirectory(dir=config["tmp_dir"]) as tmpdir:
            # Cache to avoid multiple downloads. Create inside tempdir.
            video_cache = {}

            def get_cached_video_path(video_path: str) -> str:
                if video_path.startswith("s3"):
                    if video_path not in video_cache:
                        hashed = hashlib.sha1(video_path.encode("utf-8")).hexdigest()
                        local_path = os.path.join(tmpdir, f"cached_{hashed}_{os.path.basename(video_path)}")
                        bucket, key = video_path.replace("s3://", "").split("/", 1)
                        s3_client.download_file(bucket, key, local_path)
                        video_cache[video_path] = local_path
                    return video_cache[video_path]
                else:
                    return video_path

            try:
                ep_path = config["episode_lookup"][ep_idx]

                # Read parquet file (download from S3 if needed)
                if ep_path.startswith("s3"):
                    with copy_to_temp_file(ep_path) as temp_parquet:
                        df = pq.read_table(temp_parquet).to_pandas()
                else:
                    df = pq.read_table(ep_path).to_pandas()
                df_dict = df.to_dict("index")
                print(f"Processing episode {ep_idx} with {len(df)} frames")

                uploaded_frames = []
                s3_path = config["s3_output_path"].removeprefix("s3://")
                bucket_name, s3_prefix = s3_path.split("/", 1)

                # Step 1: Prepare image data based on mode
                if config["no_video"]:
                    # No-video mode: images come from parquet columns
                    camera_names = config["image_columns"]

                    def get_image_bytes_for_frame(idx, img_offset, camera_name):
                        img_timestep = np.clip(idx + img_offset, 0, len(df) - 1)
                        return get_image_bytes_from_parquet(df_dict, img_timestep, camera_name)

                else:
                    # Video mode: Batch extract all needed frames upfront
                    camera_names = list(config["cameras"].keys())

                    # List out frames to extract per video
                    unique_frames_per_video = {}  # video_path -> set of frame_index_values
                    for idx, _ in df_dict.items():
                        for img_offset in config["image_indices"]:
                            img_timestep = np.clip(idx + img_offset, 0, len(df) - 1)
                            frame_index_value = df_dict[img_timestep][config["frame_index_col"]]

                            for _camera_name, camera_relative_path in config["cameras"].items():
                                video_path = config["video_lookup"][(ep_idx, camera_relative_path)]
                                local_video_path = get_cached_video_path(video_path)
                                unique_frames_per_video.setdefault(local_video_path, set()).add(frame_index_value)

                    # Batch extract all unique frames from videos
                    extracted_frames = {}  # (video_path, frame_index_value) -> jpg_bytes
                    for local_video_path, frame_indices in unique_frames_per_video.items():
                        frame_requests, temp_paths = [], {}

                        for frame_index_value in frame_indices:
                            temp_jpg_path = Path(tmpdir) / f"temp_{abs(hash(local_video_path))}_{frame_index_value}.jpg"
                            frame_requests.append((frame_index_value, str(temp_jpg_path)))
                            temp_paths[frame_index_value] = temp_jpg_path

                        extraction_results = extract_multiple_frames_ffmpeg(
                            local_video_path, frame_requests, config["fps"]
                        )

                        for frame_index_value, temp_jpg_path in temp_paths.items():
                            if extraction_results.get(frame_index_value, False) and temp_jpg_path.exists():
                                with open(temp_jpg_path, "rb") as f:
                                    extracted_frames[(local_video_path, frame_index_value)] = f.read()
                                temp_jpg_path.unlink()
                            else:
                                print(f"Failed to extract frame {frame_index_value} from {local_video_path}")

                    def get_image_bytes_for_frame(idx, img_offset, camera_name):
                        img_timestep = np.clip(idx + img_offset, 0, len(df) - 1)
                        frame_index_value = df_dict[img_timestep][config["frame_index_col"]]
                        camera_relative_path = config["cameras"][camera_name]
                        video_path = config["video_lookup"][(ep_idx, camera_relative_path)]
                        local_video_path = get_cached_video_path(video_path)
                        return extracted_frames.get((local_video_path, frame_index_value))

                # Step 2: Create tars (unified logic for both modes)
                tar_buffers = []
                for idx, row in df_dict.items():
                    frame_tar_filename = f"{config['output_prefix_pattern'].format(ep_idx, idx)}.tar"

                    # Create tar in memory
                    tar_buffer = io.BytesIO()
                    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
                        file_prefix = config["output_prefix_pattern"].format(ep_idx, idx)

                        # Track original image sizes per camera
                        original_image_sizes = {}

                        # Process and add images
                        for img_offset in config["image_indices"]:
                            for camera_name in camera_names:
                                image_bytes = get_image_bytes_for_frame(idx, img_offset, camera_name)

                                if image_bytes:
                                    # Resize image if requested
                                    resized_bytes, original_size = resize_image_bytes(
                                        image_bytes, config["jpeg_quality"], config["resize_images_size"]
                                    )

                                    # Track original size (store by camera name without timestep)
                                    if camera_name not in original_image_sizes:
                                        original_image_sizes[camera_name] = original_size

                                    camera_key = f"{camera_name}_t{img_offset}"
                                    add_image_to_tar(tar, file_prefix, camera_key, resized_bytes)

                        # Add metadata with original image sizes
                        add_sample_metadata_to_tar(
                            tar,
                            file_prefix,
                            df,
                            df_dict,
                            idx,
                            config,
                            pad_fn,
                            camera_names,
                            row,
                            original_image_sizes,
                        )

                    tar_buffer.seek(0)
                    tar_buffers.append((frame_tar_filename, tar_buffer))

                # Step 3: Upload in parallel (unified for both modes)
                def upload_tar(tar_data):
                    filename, buffer = tar_data
                    s3_key = f"{s3_prefix.rstrip('/')}/{filename}"
                    buffer.seek(0)
                    s3_client.upload_fileobj(buffer, bucket_name, s3_key)
                    return filename

                with ThreadPoolExecutor(max_workers=config["upload_threads"]) as executor:
                    futures = [executor.submit(upload_tar, tar_data) for tar_data in tar_buffers]
                    for future in as_completed(futures):
                        uploaded_frames.append(future.result())

                print(f"Uploaded episode {ep_idx} with {len(uploaded_frames)} frame tars")

            finally:
                # Clean up cached video files (only in video mode)
                if not config["no_video"]:
                    for local_path in video_cache.values():
                        with contextlib.suppress(Exception):
                            os.unlink(local_path)  # Temp files will be cleaned up by tempfile.TemporaryDirectory

        return {
            "episode_index": ep_idx,
            "num_sequences": len(uploaded_frames),
            "frame_tar_filenames": uploaded_frames,
        }

    except Exception as e:
        print(f"Failed on episode {ep_idx}: {e}")
        traceback.print_exc()
        return None


def upload_manifest(manifest_data: List[Dict], s3_output_path: str, subpath: str = ""):
    s3_client = boto3.client("s3")
    manifest_lines = [json.dumps(entry) for entry in manifest_data]
    manifest_content = "\n".join(manifest_lines)

    s3_path = s3_output_path.removeprefix("s3://")
    bucket_name, s3_prefix = s3_path.split("/", 1)

    if subpath:
        manifest_s3_key = f"{s3_prefix.rstrip('/')}/{subpath}/manifest.jsonl"
    else:
        manifest_s3_key = f"{s3_prefix.rstrip('/')}/manifest.jsonl"

    s3_client.put_object(
        Bucket=bucket_name,
        Key=manifest_s3_key,
        Body=manifest_content.encode("utf-8"),
        ContentType="application/json",
    )

    print(f"Uploaded manifest to s3://{bucket_name}/{manifest_s3_key}")
    return f"s3://{bucket_name}/{manifest_s3_key}"


def copy_stats_to_s3(dataset_path, s3_output_path):
    """Copy first available stats file to S3."""
    for stats_name in ["stats.json", "episode_stats.json", "stats.jsonl", "episode_stats.jsonl"]:
        stats_path = resolve_path(dataset_path, f"meta/{stats_name}")
        if file_exists(stats_path):
            destination_path = resolve_path(s3_output_path, stats_name)
            try:
                subprocess.run(
                    ["aws", "s3", "cp", stats_path, destination_path], capture_output=True, check=True, timeout=60
                )
                print(f"Uploaded {stats_name} to {destination_path}")
                return destination_path
            except subprocess.CalledProcessError:
                continue
    return None


def main():
    args = parse_args()

    # Safety check: ensure output directory doesn't have existing preprocessing outputs
    existing_shard_files = check_directory_has_files_with_prefix(args.s3_output_path, "shard_")
    if existing_shard_files:
        error_msg = (
            f"\n{'=' * 80}\n"
            f"❌ ERROR: Output directory is not empty!\n"
            f"\n"
            f"The output directory contains {len(existing_shard_files)} existing files starting with 'shard_':\n"
            f"  Output directory: {args.s3_output_path}\n"
            f"  Example files: {', '.join(existing_shard_files[:5])}"
            f"{'...' if len(existing_shard_files) > 5 else ''}\n"
            f"\n"
            f"Pre-processing in a non-empty output directory is unsafe because it may overwrite\n"
            f"existing shard files or mix data from different preprocessing runs.\n"
            f"\n"
            f"To fix this issue:\n"
            f"  1. Use a new, empty output directory, OR\n"
            f"  2. Delete/move the existing files from the output directory\n"
            f"\n"
            f"For S3 paths, you can clean the directory with:\n"
            f"  aws s3 rm --recursive {args.s3_output_path}\n"
            f"{'=' * 80}\n"
        )
        raise RuntimeError(error_msg)

    # Initialize Ray cluster
    if args.ray_address:
        ray.init(address=args.ray_address, runtime_env={"excludes": [".git/*", "tests/*"]})
        print(f"Connected to Ray cluster at {args.ray_address}")
    else:
        ray.init(address="auto", num_cpus=args.ray_num_cpus)
        print(f"Started auto Ray cluster with num_cpus={args.ray_num_cpus}")

    # Resolve all paths relative to base path if provided
    args.meta_episodes_path = resolve_path(args.dataset_path, args.meta_episodes_path)
    args.info_path = resolve_path(args.dataset_path, args.info_path)
    args.tasks_path = resolve_path(args.dataset_path, args.tasks_path)
    args.data_path = resolve_path(args.dataset_path, args.data_path)
    args.videos_path = resolve_path(args.dataset_path, args.videos_path)

    print(f"meta_episodes_path: {args.meta_episodes_path}")
    print(f"info_path: {args.info_path}")
    print(f"tasks_path: {args.tasks_path}")
    print(f"data_path: {args.data_path}")
    print(f"videos_path: {args.videos_path}")
    print(f"s3_output_path: {args.s3_output_path}")

    # Auto-detect FPS, data and video chunk paths
    fps = args.fps if args.fps is not None else detect_fps(args.info_path)
    data_chunks = discover_chunks(args.data_path, "chunk-*")
    assert data_chunks, "No data chunks found"
    if not args.no_video:
        video_chunks = discover_chunks(args.videos_path, "chunk-*")
        assert video_chunks, "No video chunks found"
    else:
        video_chunks = []

    # Read tasks metadata
    tasks_dict = jsonl_load(args.tasks_path)
    print(f"Loaded {len(tasks_dict)} tasks")

    # Pre-build episode lookup once and reuse it for all shards.
    episode_lookup = build_episode_lookup(data_chunks)
    print(f"Built episode lookup with {len(episode_lookup)} episodes")

    # Dummy values for video/image related variables
    image_columns, cameras, video_lookup = [], {}, {}
    if args.no_video:
        if not args.image_columns:
            image_columns = discover_image_columns(data_chunks, args.episode_file_pattern)
            if not image_columns:
                raise ValueError("No image columns found in parquet files and none specified with --image_columns")
        print(f"Using image columns: {image_columns}")
    else:
        # Discover cameras once and reuse it for all shards.
        cameras = discover_cameras(video_chunks)
        assert len(cameras) > 0, "No cameras found"
        print(f"Discovered cameras: {list(cameras.keys())}")

        # Pre-build video lookup once and reuse it for all shards.
        video_lookup = build_video_lookup(video_chunks, cameras)
        print(f"Built video lookup with {len(video_lookup)} video files")

    # Read episodes metadata
    entries = jsonl_load(args.meta_episodes_path)
    print(f"Processing {len(entries)} episodes (one episode per tar file)")

    # Initialize JPEG encoder with specified quality
    init_jpeg_encoder(args.jpeg_quality)

    config = {
        "data_chunks": data_chunks,
        "video_chunks": video_chunks,
        "episode_lookup": episode_lookup,
        "video_lookup": video_lookup,
        "cameras": cameras,
        "tasks_dict": tasks_dict,
        "tmp_dir": args.tmp_dir,
        "s3_output_path": args.s3_output_path,
        "lowdim_columns": args.lowdim_columns,
        "frame_index_col": args.frame_index_col,
        "task_index_col": args.task_index_col,
        "episode_file_pattern": args.episode_file_pattern,
        "video_file_pattern": args.video_file_pattern,
        "output_prefix_pattern": args.output_prefix_pattern,
        "fps": fps,
        "no_video": args.no_video,
        "image_columns": image_columns,
        "num_past": args.num_past,
        "num_future": args.num_future,
        "padding_strategy": args.padding_strategy,
        "episode_index_col": args.episode_index_col,
        "image_indices": sorted(args.image_indices),  # Sort for consistent ordering
        "upload_threads": args.upload_threads,
        "resize_images_size": tuple(args.resize_images_size) if args.resize_images_size else None,
        "jpeg_quality": args.jpeg_quality,
    }

    # Process episodes with controlled concurrency to avoid overwhelming the system
    max_concurrent = int(args.max_concurrent_shards or ray.cluster_resources().get("CPU", 1))
    print(f"Processing {len(entries)} episodes with max concurrency: {max_concurrent}")

    # Process in batches to control memory usage and resource consumption
    results = []
    for i in range(0, len(entries), max_concurrent):
        batch = entries[i : i + max_concurrent]
        batch_futures = [write_tar_and_upload.remote(episode_entry, config) for episode_entry in batch]
        batch_results = ray.get(batch_futures)
        results.extend(batch_results)
        print(f"Completed batch {i // max_concurrent + 1}/{(len(entries) + max_concurrent - 1) // max_concurrent}")

    # Collect all frame tar filenames
    all_frame_tars = []
    for result in results:
        if result is not None:
            all_frame_tars.extend(result["frame_tar_filenames"])

    print(f"Found {len(all_frame_tars)} total frames across {len(results)} episodes")

    # Shuffle and group into shards
    random.shuffle(all_frame_tars)
    shards = [all_frame_tars[i : i + args.shard_size] for i in range(0, len(all_frame_tars), args.shard_size)]
    print(f"Creating {len(shards)} shards with up to {args.shard_size} frames each")
    shard_futures = [create_shard.remote(frame_tars, i, args.s3_output_path) for i, frame_tars in enumerate(shards)]
    shard_results = ray.get(shard_futures)

    # Upload shard manifest and copy stats to s3
    shard_manifest = [
        {"shard": shard_name, "num_sequences": num_sequences} for shard_name, num_sequences in shard_results
    ]
    upload_manifest(shard_manifest, args.s3_output_path, subpath="shards")
    copy_stats_to_s3(args.dataset_path, os.path.join(args.s3_output_path, "shards"))

    # Print summary
    total_sequences = sum(num_seq for _, num_seq in shard_results)
    avg_sequences_per_shard = total_sequences / len(shard_results) if shard_results else 0
    print("\nSharding complete:")
    print(f"  Total shards: {len(shard_results)}")
    print(f"  Total sequences: {total_sequences}")
    print(f"  Target frames per shard: {args.shard_size}")
    print(f"  Avg frames per shard: {avg_sequences_per_shard:.1f}")
    ray.shutdown()


if __name__ == "__main__":
    main()
