"""
We probably want this file as self-contained as possible, so we define all functions here.
Except the file_utils. We rely on them for the s3 stuff.
"""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import numpy as np
import pyarrow.parquet as pq
import ray

from lbm2.file_utils import copy_to_temp_file, file_exists, json_load, jsonl_load, list_directory


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


def create_metadata_and_lowdim_dicts(row, lowdim_columns, camera_names, excluded_columns=None):
    """Create metadata and lowdim dictionaries from a DataFrame row."""
    excluded_columns = excluded_columns or []
    metadata_dict, lowdim_dict = {}, {}

    for col, value in row.items():
        if col in lowdim_columns:
            lowdim_dict[col] = value
        elif col not in excluded_columns:
            metadata_dict[col] = value

    metadata_dict["camera_names"] = camera_names
    return metadata_dict, lowdim_dict


def add_metadata_json_to_tar(tar, file_prefix, metadata_dict):
    """Add JSON metadata file to tar archive."""
    json_bytes = json.dumps(make_json_serializable(metadata_dict), indent=2).encode("utf-8")
    tarinfo_json = tarfile.TarInfo(name=f"{file_prefix}.metadata.json")
    tarinfo_json.size = len(json_bytes)
    tar.addfile(tarinfo_json, io.BytesIO(json_bytes))


def add_npz_to_tar(tar, file_prefix, lowdim_dict):
    """Add NPZ lowdim file to tar archive."""
    npz_buffer = io.BytesIO()
    np.savez_compressed(npz_buffer, **lowdim_dict)
    npz_bytes = npz_buffer.getvalue()
    npz_buffer.close()
    tarinfo_npz = tarfile.TarInfo(name=f"{file_prefix}.lowdim.npz")
    tarinfo_npz.size = len(npz_bytes)
    tar.addfile(tarinfo_npz, io.BytesIO(npz_bytes))


def add_image_to_tar(tar, file_prefix, camera_name, image_bytes):
    """Add image file to tar archive."""
    tarinfo_img = tarfile.TarInfo(name=f"{file_prefix}.{camera_name}.jpg")
    tarinfo_img.size = len(image_bytes)
    tar.addfile(tarinfo_img, io.BytesIO(image_bytes))


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
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--videos_path", type=str, default="videos")
    parser.add_argument("--lowdim_columns", type=str, nargs="+", default=["action", "observation.state", "actions", "observations"])
    parser.add_argument("--frame_index_col", type=str, default="frame_index")
    parser.add_argument("--episode_index_col", type=str, default="episode_index")
    parser.add_argument("--episode_file_pattern", type=str, default="episode_{:06d}.parquet")
    parser.add_argument("--video_file_pattern", type=str, default="episode_{:06d}.mp4")
    parser.add_argument("--output_prefix_pattern", type=str, default="episode_{:06d}_frame_{:06d}")

    # Image processing mode
    parser.add_argument(
        "--no_video", action="store_true", help="Load images from parquet columns instead of video files"
    )
    parser.add_argument("--image_columns", type=str, nargs="+", default=None, help="auto-detected if not specified")

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
    image_columns = []

    for chunk_dir in data_chunks:
        for episode_idx in range(10):  # Check first 10 episodes
            episode_file = episode_file_pattern.format(episode_idx)
            episode_path = f"{chunk_dir.rstrip('/')}/{episode_file}"

            if file_exists(episode_path):
                # Read parquet file to examine columns
                if episode_path.startswith("s3"):
                    with copy_to_temp_file(episode_path) as temp_parquet:
                        df = pq.read_table(temp_parquet).to_pandas()
                else:
                    df = pq.read_table(episode_path).to_pandas()

                # Look for columns that contain bytes data (likely images)
                for col in df.columns:
                    if df[col].dtype == object and len(df[col]) > 0:
                        # Check if first non-null value is bytes
                        first_value = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                        if isinstance(first_value, dict) and "bytes" in first_value or isinstance(first_value, bytes):
                            image_columns.append(col)

                print(f"Discovered image columns from {episode_path}: {image_columns}")
                return image_columns

    print("Warning: Could not discover image columns from parquet files")
    return image_columns


def find_episode_file(episode_index: int, data_chunks: List[str], file_pattern: str) -> str:
    """Find the correct episode path across multiple chunks-* folders."""
    episode_file = file_pattern.format(episode_index)

    for chunk_dir in data_chunks:
        episode_path = f"{chunk_dir.rstrip('/')}/{episode_file}"
        if file_exists(episode_path):
            return episode_path

    raise FileNotFoundError(f"Episode file {episode_file} not found in any chunk: {data_chunks}")


def find_video_file(episode_index: int, camera_path: str, video_chunks: List[str], file_pattern: str) -> str:
    """Find the correct video path for a specific camera across multiple chunks-* folders."""
    video_file = file_pattern.format(episode_index)
    camera_name = os.path.basename(camera_path)

    for chunk_dir in video_chunks:
        video_path = f"{chunk_dir.rstrip('/')}/{camera_name}/{video_file}"
        if file_exists(video_path):
            return video_path

    raise FileNotFoundError(f"Video file {video_file} for camera {camera_name} not found in any chunk: {video_chunks}")


def episodes_to_shard_slices(
    entries: List[Dict], max_shard_length: int, episode_index_key: str = "episode_index"
) -> List[List[Tuple[int, int, int]]]:
    """
    Slices episodes into shards of specified size.

    Sample output for max_shard_length = 512:
    [
        [(0, 0, 512)],
        [(0, 512, 586), (1, 0, 381), (2, 0, 57)],
        [(2, 57, 369), (3, 0, 200)],
    ]
    """
    shards = []
    current_shard = []
    remaining_space = max_shard_length

    for entry in entries:
        ep_idx = entry[episode_index_key]
        ep_len = entry["length"]
        start = 0

        while ep_len > 0:
            take = min(remaining_space, ep_len)
            current_shard.append((ep_idx, start, start + take))
            ep_len -= take
            start += take
            remaining_space -= take

            if remaining_space == 0:
                shards.append(current_shard)
                current_shard = []
                remaining_space = max_shard_length

    # Add the last shard
    if current_shard:
        shards.append(current_shard)

    return shards


def extract_single_frame_ffmpeg(video_path: str, frame_index: int, output_path: str, fps: float = 30.0) -> bool:
    """Extract a specific frame from video using ffmpeg. Video_path should be local."""
    try:
        timestamp = frame_index / fps

        cmd = [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            video_path,
            "-vframes",
            "1",
            "-y",
            output_path,
            "-loglevel",
            "quiet",
        ]

        with open(os.devnull, "w") as devnull:
            result = subprocess.run(cmd, stdout=devnull, stderr=devnull, stdin=subprocess.DEVNULL, timeout=10)

        return result.returncode == 0 and file_exists(output_path)

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        return False


def extract_multiple_frames_ffmpeg(
    video_path: str, frame_requests: List[Tuple[int, str]], fps: float = 30.0
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

        # Use ffmpeg select filter to extract frames in batches of 10 to avoid command line length limits
        batch_size = 10

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
        print(f"Batch ffmpeg extraction failed: {e}. Falling back to single frame extractions.")
        # Fall back to individual extractions
        for frame_index, output_path in frame_requests:
            if frame_index not in results:
                results[frame_index] = extract_single_frame_ffmpeg(video_path, frame_index, output_path, fps)

    return results


@ray.remote
def write_tar_and_upload(shard_idx: int, slices: List[Tuple[int, int, int]], config: Dict):
    try:
        s3_client = boto3.client("s3")

        # Use pre-built episode, video, and camera lookups from config
        episode_lookup = config["episode_lookup"]
        video_lookup = config["video_lookup"]
        cameras = config["cameras"]
        no_video_mode = config["no_video"]
        image_columns = config["image_columns"]

        if config["tmp_dir"] and not os.path.exists(config["tmp_dir"]):
            os.makedirs(config["tmp_dir"], exist_ok=True)

        # Write files to temporary directory, then upload to S3, then delete tmpdir.
        with tempfile.TemporaryDirectory(dir=config["tmp_dir"]) as tmpdir:
            tar_filename = f"shard_{shard_idx:05d}.tar"
            tar_path = Path(tmpdir) / tar_filename
            num_sequences = 0

            # Cache to avoid multiple downloads. Create inside tempdir.
            video_cache = {}

            def get_cached_video_path(video_path: str) -> str:
                if video_path.startswith("s3"):
                    # Use video_cache to avoid downloading the same episode video multiple times
                    if video_path not in video_cache:
                        # Create a unique local filename to avoid collisions between different cameras
                        hashed = hashlib.sha1(video_path.encode("utf-8")).hexdigest()
                        local_path = os.path.join(tmpdir, f"cached_{hashed}_{os.path.basename(video_path)}")

                        cmd = f"aws s3 cp {video_path} {local_path}"
                        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        stdout, stderr = proc.communicate()

                        if proc.returncode != 0:
                            raise RuntimeError(f"Failed to download video {video_path}: {stderr.decode().strip()}")

                        video_cache[video_path] = local_path
                    return video_cache[video_path]
                else:
                    return video_path

            try:
                with tarfile.open(tar_path, "w") as tar:
                    if no_video_mode:
                        # No-video mode: process images from parquet columns
                        for ep_idx, start, end in slices:
                            try:
                                ep_path = episode_lookup[ep_idx]

                                # Read parquet file (download from S3 if needed)
                                if ep_path.startswith("s3"):
                                    with copy_to_temp_file(ep_path) as temp_parquet:
                                        df = pq.read_table(temp_parquet).to_pandas().iloc[start:end]
                                else:
                                    df = pq.read_table(ep_path).to_pandas().iloc[start:end]

                            except Exception as e:
                                print(f"Error reading episode {ep_idx}: {e}")
                                continue

                            print(f"Shard {shard_idx}: Processing episode {ep_idx}, frames {start}-{end}")

                            # Process each frame
                            for idx, (_, row) in enumerate(df.iterrows()):
                                frame_idx = start + idx
                                file_prefix = config["output_prefix_pattern"].format(ep_idx, frame_idx)

                                # Create metadata and lowdim dictionaries
                                metadata_dict, lowdim_dict = create_metadata_and_lowdim_dicts(
                                    row=row,
                                    lowdim_columns=config["lowdim_columns"],
                                    camera_names=image_columns,
                                    excluded_columns=image_columns + config["lowdim_columns"],
                                )

                                # Add JSON and NPZ files to tar
                                add_metadata_json_to_tar(tar, file_prefix, metadata_dict)
                                add_npz_to_tar(tar, file_prefix, lowdim_dict)

                                # Process image columns
                                for image_col in image_columns:
                                    image_data = row[image_col]

                                    # Handle different image data formats
                                    if isinstance(image_data, dict) and "bytes" in image_data:
                                        # LeRobot format: {'bytes': b'...', ...}
                                        image_bytes = image_data["bytes"]
                                    elif isinstance(image_data, bytes):
                                        # Raw bytes
                                        image_bytes = image_data
                                    else:
                                        print(f"Warning: Unsupported image data format in column {image_col}")
                                        continue

                                    # Add image file to tar
                                    add_image_to_tar(tar, file_prefix, image_col, image_bytes)

                                num_sequences += 1

                    else:
                        # Video mode: extract frames from video files
                        # Group frames by video to enable batch processing
                        video_frame_groups = {}  # key=video_path, value=list of frames to extract from video

                        # First pass: collect all data and group by video
                        for ep_idx, start, end in slices:
                            try:
                                ep_path = episode_lookup[ep_idx]

                                # Read parquet file (download from S3 if needed)
                                if ep_path.startswith("s3"):
                                    with copy_to_temp_file(ep_path) as temp_parquet:
                                        df = pq.read_table(temp_parquet).to_pandas().iloc[start:end]
                                else:
                                    df = pq.read_table(ep_path).to_pandas().iloc[start:end]

                            except Exception as e:
                                print(f"Error reading episode {ep_idx}: {e}")
                                continue

                            print(f"Shard {shard_idx}: Processing episode {ep_idx}, frames {start}-{end}")

                            # Process each frame and group by video for batched extraction
                            for idx, (_, row) in enumerate(df.iterrows()):
                                frame_idx = start + idx
                                file_prefix = config["output_prefix_pattern"].format(ep_idx, frame_idx)
                                frame_index_value = row[config["frame_index_col"]]

                                # Group frames by video for batch processing
                                for camera_name, camera_relative_path in cameras.items():
                                    video_path = video_lookup[(ep_idx, camera_relative_path)]
                                    local_video_path = get_cached_video_path(video_path)

                                    video_frame_groups.setdefault(local_video_path, []).append(
                                        {
                                            "frame_index_value": frame_index_value,
                                            "camera_name": camera_name,
                                            "file_prefix": file_prefix,
                                            "row": row,
                                            "ep_idx": ep_idx,
                                            "frame_idx": frame_idx,
                                        }
                                    )

                        # Second pass: batch extract frames and create tar entries
                        for local_video_path, frames_to_extract in video_frame_groups.items():
                            # Prepare batch frame extraction requests
                            frame_requests = []
                            frame_data_map = {}

                            for frame_data in frames_to_extract:
                                temp_jpg_path = (
                                    Path(tmpdir)
                                    / f"temp_{shard_idx}_{frame_data['camera_name']}_{frame_data['frame_idx']}.jpg"
                                )
                                frame_requests.append((frame_data["frame_index_value"], str(temp_jpg_path)))
                                frame_data_map[frame_data["frame_index_value"]] = {
                                    **frame_data,
                                    "temp_jpg_path": temp_jpg_path,
                                }

                            # Batch extract all frames from this video
                            extraction_results = extract_multiple_frames_ffmpeg(
                                local_video_path, frame_requests, config["fps"]
                            )

                            # Process each frame's data and add to tar
                            for frame_index_value, success in extraction_results.items():
                                frame_data = frame_data_map[frame_index_value]
                                row = frame_data["row"]
                                file_prefix = frame_data["file_prefix"]
                                temp_jpg_path = frame_data["temp_jpg_path"]
                                camera_name = frame_data["camera_name"]

                                # Create metadata and lowdim dictionaries
                                metadata_dict, lowdim_dict = create_metadata_and_lowdim_dicts(
                                    row, config["lowdim_columns"], list(cameras.keys())
                                )

                                # Create and add json file (only once per frame, not per camera)
                                if camera_name == list(cameras.keys())[0]:  # Only for first camera to avoid duplicates
                                    add_metadata_json_to_tar(tar, file_prefix, metadata_dict)
                                    add_npz_to_tar(tar, file_prefix, lowdim_dict)
                                    num_sequences += 1

                                # Add jpg file if extraction was successful
                                if success and temp_jpg_path.exists():
                                    with open(temp_jpg_path, "rb") as jpg_file:
                                        jpg_bytes = jpg_file.read()
                                    add_image_to_tar(tar, file_prefix, camera_name, jpg_bytes)
                                    temp_jpg_path.unlink()
                                else:
                                    print(f"Failed to extract frame {frame_index_value} from {local_video_path}")

                # Upload to S3 with multipart upload for better performance
                s3_path = config["s3_output_path"].removeprefix("s3://")
                bucket_name, s3_prefix = s3_path.split("/", 1)
                s3_key = f"{s3_prefix.rstrip('/')}/{tar_filename}"

                # Use multipart upload for files larger than 100MB for better performance
                file_size = tar_path.stat().st_size
                if file_size > 100 * 1024 * 1024:  # 100MB threshold
                    # Use multipart upload
                    transfer_config = boto3.s3.transfer.TransferConfig(
                        multipart_threshold=1024 * 25,  # 25MB
                        max_concurrency=10,
                        multipart_chunksize=1024 * 25,
                        use_threads=True,
                    )
                    s3_client.upload_file(str(tar_path), bucket_name, s3_key, Config=transfer_config)
                else:
                    s3_client.upload_file(str(tar_path), bucket_name, s3_key)
                print(f"Uploaded {tar_filename} ({file_size / (1024 * 1024):.1f}MB) to s3://{bucket_name}/{s3_key}")

            finally:
                # Explicitly clean up cached video files (only in video mode)
                if not no_video_mode:
                    for _cache_key, local_path in video_cache.items():
                        try:
                            if os.path.exists(local_path):
                                os.unlink(local_path)
                                print(f"Cleaned up cached video: {os.path.basename(local_path)}")
                        except Exception as e:
                            print(f"Warning: Failed to clean up cached video {local_path}: {e}")
                    video_cache.clear()

        return {
            "s3_path": f"s3://{bucket_name}/{s3_key}",
            "shard_name": f"shard_{shard_idx:05d}",
            "num_sequences": num_sequences,
        }

    except Exception as e:
        print(f"Failed on shard {shard_idx}: {e}")
        traceback.print_exc()
        return None


def create_and_upload_manifest(results: List[Dict], s3_output_path: str):
    try:
        s3_client = boto3.client("s3")

        successful_results = [r for r in results if r is not None]
        manifest_lines = []
        for result in successful_results:
            manifest_entry = {"shard": result["shard_name"], "num_sequences": result["num_sequences"]}
            manifest_lines.append(json.dumps(manifest_entry))

        manifest_content = "\n".join(manifest_lines)

        # Upload manifest to S3 in the same directory as the tar files
        s3_path = s3_output_path.removeprefix("s3://")
        bucket_name, s3_prefix = s3_path.split("/", 1)
        manifest_s3_key = f"{s3_prefix.rstrip('/')}/manifest.jsonl"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=manifest_s3_key,
            Body=manifest_content.encode("utf-8"),
            ContentType="application/json",
        )

        print(f"Uploaded manifest.jsonl to s3://{bucket_name}/{manifest_s3_key}")
        return f"s3://{bucket_name}/{manifest_s3_key}"

    except Exception as e:
        print(f"Failed to create/upload manifest: {e}")
        traceback.print_exc()
        return None


def copy_stats_to_s3(dataset_path, s3_output_path):
    stats_names = ["stats.json", "episode_stats.json", "stats.jsonl", "episode_stats.jsonl"]
    for stats_name in stats_names:
        stats_path = resolve_path(dataset_path, f"meta/{stats_name}")
        destination_path = resolve_path(s3_output_path, f"{stats_name}")
        if file_exists(stats_path):
            cmd = f"aws s3 cp {stats_path} {destination_path}"
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = proc.communicate()
            if proc.returncode != 0:
                print(f"Failed to copy {stats_name} to {destination_path}: {stderr.decode().strip()}")
                continue
            print(f"Uploaded {stats_name} to {destination_path}")
            return destination_path
    return None


def main():
    args = parse_args()

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
    args.data_path = resolve_path(args.dataset_path, args.data_path)
    args.videos_path = resolve_path(args.dataset_path, args.videos_path)

    print(f"meta_episodes_path: {args.meta_episodes_path}")
    print(f"info_path: {args.info_path}")
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

    # Read episodes metadata and split into shards
    entries = jsonl_load(args.meta_episodes_path)
    shards = episodes_to_shard_slices(entries, args.shard_size, args.episode_index_col)
    print(f"Created {len(shards)} shards from {len(entries)} episodes")

    config = {
        "data_chunks": data_chunks,
        "video_chunks": video_chunks,
        "episode_lookup": episode_lookup,
        "video_lookup": video_lookup,
        "cameras": cameras,
        "tmp_dir": args.tmp_dir,
        "s3_output_path": args.s3_output_path,
        "lowdim_columns": args.lowdim_columns,
        "frame_index_col": args.frame_index_col,
        "episode_file_pattern": args.episode_file_pattern,
        "video_file_pattern": args.video_file_pattern,
        "output_prefix_pattern": args.output_prefix_pattern,
        "fps": fps,
        "no_video": args.no_video,
        "image_columns": image_columns,
    }

    # Process shards with controlled concurrency to avoid overwhelming the system
    max_concurrent = int(args.max_concurrent_shards or ray.cluster_resources().get("CPU", 1))
    print(f"Processing {len(shards)} shards with max concurrency: {max_concurrent}")

    # Process in batches to control memory usage and resource consumption
    results = []
    for i in range(0, len(shards), max_concurrent):
        batch = shards[i : i + max_concurrent]
        batch_futures = [write_tar_and_upload.remote(i + j, slices, config) for j, slices in enumerate(batch)]
        batch_results = ray.get(batch_futures)
        results.extend(batch_results)
        print(f"Completed batch {i // max_concurrent + 1}/{(len(shards) + max_concurrent - 1) // max_concurrent}")

    # Create manifest and copy stats to s3
    manifest_path = create_and_upload_manifest(results, args.s3_output_path)
    stats_path = copy_stats_to_s3(args.dataset_path, args.s3_output_path)

    # Print results
    successful = [r for r in results if r is not None]
    failed = len(results) - len(successful)
    total_sequences = sum(r["num_sequences"] for r in successful)

    print("\nProcessing complete:")
    print(f"  Successful shards: {len(successful)}")
    print(f"  Failed shards: {failed}")
    print(f"  Total sequences processed: {total_sequences}")
    if manifest_path:
        print(f"  Manifest created: {manifest_path}")
    if stats_path:
        print(f"  Stats copied to: {stats_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
