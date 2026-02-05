import io
import json
import random
import tarfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import boto3
import numpy as np
import ray
from botocore.config import Config
from PIL import Image

from vla_foundry.data.preprocessing.image_utils import depth_image_to_bytes, image_to_bytes


def upload_sample_to_s3(
    sample_data: Dict[str, Any],
    output_dir: str,
    episode_path: str,
    episode_id: str,
    frame_idx: int,
    jpeg_quality: int = 95,
    resize_images_size: List[int] = None,
) -> None:
    """Upload sample data to S3 as tar file. (or save locally)"""
    s3_client = (
        boto3.client("s3", config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))
        if output_dir.startswith("s3://")
        else None
    )
    tar_buffer = io.BytesIO()
    uuid_prefix = str(uuid.uuid4())

    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        original_image_sizes = {}
        # Convert images to bytes (JPEG for RGB, PNG for depth)
        for img_key, img_data in sample_data["images"].items():
            # Check if this is a depth image
            is_depth = "depth" in img_key

            if not isinstance(img_data, bytes):
                assert resize_images_size is not None
                if is_depth:
                    # Depth images: PNG with uint16
                    image_bytes, original_image_size = depth_image_to_bytes(img_data, resize_images_size)
                    file_extension = "png"
                else:
                    # RGB images: JPEG
                    image_bytes, original_image_size = image_to_bytes(img_data, jpeg_quality, resize_images_size)
                    file_extension = "jpg"
            else:
                # Bytes passed directly - resize cannot be applied
                if resize_images_size is not None:
                    raise ValueError(
                        f"Image '{img_key}' is already encoded as bytes but resize_images_size={resize_images_size} "
                        "is configured. Converters must return numpy arrays for resizing to work. "
                        "Either return numpy arrays from the converter or set resize_images_size=null."
                    )
                image_bytes = img_data
                original_image_size = Image.open(io.BytesIO(img_data)).size
                file_extension = "jpg"  # Assume pre-encoded bytes are JPEG

            # Log original image sizes
            camera_name_without_timestep = img_key.rsplit("_t", 1)[0]
            if isinstance(sample_data["metadata"], dict):
                assert camera_name_without_timestep in sample_data["metadata"].get("camera_names")
            else:
                assert camera_name_without_timestep in sample_data["metadata"].camera_names
            original_image_sizes[camera_name_without_timestep] = original_image_size

            tarinfo = tarfile.TarInfo(name=f"{uuid_prefix}.{img_key}.{file_extension}")
            tarinfo.size = len(image_bytes)
            tar.addfile(tarinfo, io.BytesIO(image_bytes))

        if isinstance(sample_data["metadata"], dict):
            sample_data["metadata"]["original_image_sizes"] = original_image_sizes
        else:
            sample_data["metadata"].original_image_sizes = original_image_sizes

        for key, value in sample_data.items():
            data_buffer = io.BytesIO()
            if key == "images":  # Already added
                continue
            elif key in ["metadata", "language_instructions"]:
                # Save as JSON
                if isinstance(value, dict):
                    json_str = json.dumps(value, indent=2, default=str)
                elif value is None:  # e.g. language_instructions can be None
                    continue
                else:
                    json_str = json.dumps(asdict(value), indent=2, default=str)
                data_buffer.write(json_str.encode("utf-8"))
                data_buffer.seek(0)
                tarinfo = tarfile.TarInfo(name=f"{uuid_prefix}.{key}.json")
                tarinfo.size = len(data_buffer.getvalue())
                tar.addfile(tarinfo, data_buffer)
            else:
                # Everything else as NPZ
                if isinstance(value, dict):
                    np.savez_compressed(data_buffer, **value)
                else:
                    np.savez_compressed(data_buffer, data=value)
                data_buffer.seek(0)
                tarinfo = tarfile.TarInfo(name=f"{uuid_prefix}.{key}.npz")
                tarinfo.size = len(data_buffer.getvalue())
                tar.addfile(tarinfo, data_buffer)

    tar_buffer.seek(0)
    unique_id = extract_unique_id(episode_path)
    tar_filename = f"{unique_id}_{episode_id}_frame_{frame_idx}.tar"

    if output_dir.startswith("s3://"):
        # Upload to S3
        bucket_name, s3_prefix = output_dir.removeprefix("s3://").split("/", 1)
        s3_key = f"{s3_prefix.rstrip('/')}/frames/{tar_filename}"
        s3_client.upload_fileobj(tar_buffer, bucket_name, s3_key)
        print(f"Uploaded s3://{bucket_name}/{s3_key}", flush=True)
    else:
        # Save to local filesystem
        local_path = Path(output_dir) / "frames" / tar_filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(tar_buffer.getvalue())
        print(f"Saved {local_path}", flush=True)

    return tar_filename


def extract_unique_id(episode_path: str) -> str:
    """Extract a deterministic unique ID from the episode path."""
    if "diffusion_spartan" in episode_path:
        # For diffusion_spartan, use the datetime as unique id
        return episode_path.split("/")[-3]
    else:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, episode_path))


def save_and_upload_dict(dict_data: Dict, output_path: str, file_name: str):
    # Used to upload manifest.jsonl and stats.json (or save locally)
    bucket_name, s3_prefix = output_path.removeprefix("s3://").split("/", 1)
    body = "\n".join(json.dumps(record) for record in dict_data) if "jsonl" in file_name else json.dumps(dict_data)

    if output_path.startswith("s3://"):
        # Upload to S3
        bucket_name, s3_prefix = output_path.removeprefix("s3://").split("/", 1)
        s3_key = f"{s3_prefix.rstrip('/')}/{file_name}"
        boto3.client("s3").put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        print(f"Uploaded {file_name} to s3://{bucket_name}/{s3_key}")
    else:
        # Save to local filesystem
        local_path = Path(output_path) / file_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(body)
        print(f"Saved {file_name} to {local_path}")


def save_and_upload_config(config, output_path: str, file_name: str):
    # Draccus dump to temp file then upload to s3 (or save locally)
    import shutil
    import tempfile

    import draccus

    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w") as temp_file:
        draccus.dump(config, temp_file)
        temp_path = temp_file.name

    if output_path.startswith("s3://"):
        # Upload to S3
        bucket_name, s3_prefix = output_path.removeprefix("s3://").split("/", 1)
        s3_key = f"{s3_prefix.rstrip('/')}/{file_name}"
        boto3.client("s3").upload_file(temp_path, bucket_name, s3_key)
        print(f"Uploaded {file_name} to s3://{bucket_name}/{s3_key}")
    else:
        # Save to local filesystem
        local_path = Path(output_path) / file_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(temp_path, local_path)
        print(f"Saved {file_name} to {local_path}")


def _download_tar_from_s3(s3_key: str, s3_client, bucket_name: str, s3_prefix: str):
    """Download a single tar file from S3 with retry logic."""
    full_key = f"{s3_prefix.rstrip('/')}/frames/{s3_key}"
    max_retries = 10
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            obj_buffer = io.BytesIO()
            s3_client.download_fileobj(bucket_name, full_key, obj_buffer)
            obj_buffer.seek(0)
            return (s3_key, obj_buffer)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff with jitter
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            print(f"S3 download failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s: {e}")
            time.sleep(delay)


@ray.remote
def create_episode_shard(shard_files: List[str], episode_key: str, output_dir: str) -> str:
    """Download tar files from S3 and create an episode-based shard."""
    s3_config = Config(max_pool_connections=50, retries={"max_attempts": 10, "mode": "adaptive"})
    s3_client = boto3.client("s3", config=s3_config)

    bucket_name, s3_prefix = output_dir.removeprefix("s3://").split("/", 1)

    download_tar = partial(_download_tar_from_s3, s3_client=s3_client, bucket_name=bucket_name, s3_prefix=s3_prefix)

    # Download all tars in parallel
    downloaded_tars = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_tar, s3_key) for s3_key in shard_files]
        for future in as_completed(futures):
            s3_key, obj_buffer = future.result()
            downloaded_tars[s3_key] = obj_buffer

    # Sort files by frame index to maintain temporal order within episode
    def get_frame_idx(filename):
        # filename format: {unique_id}_{episode_id}_frame_{frame_idx}.tar
        return int(filename.rsplit("_frame_", 1)[1].replace(".tar", ""))

    sorted_files = sorted(shard_files, key=get_frame_idx)

    # Create shard by combining all downloaded tars
    shard_buffer = io.BytesIO()
    with tarfile.open(fileobj=shard_buffer, mode="w") as shard_tar:
        for s3_key in sorted_files:
            obj_buffer = downloaded_tars[s3_key]
            obj_buffer.seek(0)

            with tarfile.open(fileobj=obj_buffer, mode="r") as tar:
                for member in tar.getmembers():
                    shard_tar.addfile(member, tar.extractfile(member))

    # Upload shard back to S3
    shard_buffer.seek(0)
    shard_key = f"episode_{episode_key}.tar"
    s3_client.upload_fileobj(shard_buffer, bucket_name, f"{s3_prefix.rstrip('/')}/episodes/{shard_key}")
    print(f"Uploaded episode shard {shard_key} to s3://{bucket_name}/{s3_prefix.rstrip('/')}/episodes/{shard_key}")
    return (shard_key.rstrip(".tar"), len(shard_files))


@ray.remote
def create_shard(shard_files: List[str], shard_idx: int, output_dir: str) -> str:
    """Download tar files from S3 and create a shard. OPTIMIZED with parallel downloads."""
    is_s3 = output_dir.startswith("s3://")

    if is_s3:
        s3_config = Config(max_pool_connections=50, retries={"max_attempts": 10, "mode": "adaptive"})
        s3_client = boto3.client("s3", config=s3_config)
        bucket_name, s3_prefix = output_dir.removeprefix("s3://").split("/", 1)

        def read_tar(tar_key):
            """Download a single tar file from S3."""
            obj_buffer = io.BytesIO()
            full_key = f"{s3_prefix.rstrip('/')}/frames/{tar_key}"
            s3_client.download_fileobj(bucket_name, full_key, obj_buffer)
            obj_buffer.seek(0)
            return (tar_key, obj_buffer)
    else:
        episodes_dir = Path(output_dir) / "episodes"

        def read_tar(tar_key):
            """Read a single tar file from local filesystem."""
            tar_path = episodes_dir / tar_key
            with open(tar_path, "rb") as f:
                obj_buffer = io.BytesIO(f.read())
            return (tar_key, obj_buffer)

    # Read all tars in parallel (use 5 threads. reduced concurrency to avoid S3 throttling)
    downloaded_tars = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(read_tar, tar_key) for tar_key in shard_files]
        for future in as_completed(futures):
            tar_key, obj_buffer = future.result()
            downloaded_tars[tar_key] = obj_buffer

    # Create shard by combining all tars
    shard_buffer = io.BytesIO()
    with tarfile.open(fileobj=shard_buffer, mode="w") as shard_tar:
        # Process in original order for consistency
        for tar_key in shard_files:
            obj_buffer = downloaded_tars[tar_key]
            obj_buffer.seek(0)

            # Extract contents and add to shard
            with tarfile.open(fileobj=obj_buffer, mode="r") as tar:
                for member in tar.getmembers():
                    shard_tar.addfile(member, tar.extractfile(member))

    # Save shard
    shard_buffer.seek(0)
    shard_name = f"shard_{shard_idx:06d}.tar"

    max_retries = 10
    base_delay = 1.0
    for attempt in range(max_retries):
        try:
            shard_buffer.seek(0)
            if is_s3:
                s3_client.upload_fileobj(shard_buffer, bucket_name, f"{s3_prefix.rstrip('/')}/shards/{shard_name}")
                print(f"Uploaded shard {shard_name} to s3://{bucket_name}/{s3_prefix.rstrip('/')}/shards/{shard_name}")
            else:
                shard_path = Path(output_dir) / "shards" / shard_name
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                with open(shard_path, "wb") as f:
                    f.write(shard_buffer.getvalue())
                print(f"Saved shard {shard_name} to {shard_path}")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            print(f"S3 upload failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s: {e}")
            time.sleep(delay)

    return (shard_name.rstrip(".tar"), len(shard_files))


def is_still_sample(lowdim_data: Dict[str, np.ndarray], start_idx: int, end_idx: int, still_threshold: float) -> bool:
    """Check if sample is still."""
    movement_keys = [k for k in lowdim_data if any(x in k.lower() for x in ["joint_position", "poses", "xyz"])]

    # If no movement keys found, don't filter the sample
    if not movement_keys:
        return False

    for key in movement_keys:
        data = lowdim_data[key][start_idx : end_idx + 1]
        if len(data) > 1:
            movement = np.std(data, axis=0).max()
            if movement > still_threshold:
                return False
    return True


def transform_points_to_world(points: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """
    Transform points from camera space to world space.

    Args:
        points: Camera-space points (N, 3) in meters
        extrinsics: Camera extrinsic matrix (4, 4) - camera-to-world transform (translation in meters)

    Returns:
        world_points: World-space points (N, 3) in meters
    """
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    points_homog = np.concatenate([points, ones], axis=-1)  # (N, 4)

    # Apply transformation
    world_points_homog = points_homog @ extrinsics.T  # (N, 4)
    world_points = world_points_homog[:, :3]  # (N, 3)

    return world_points


def voxel_downsample(points: np.ndarray, voxel_size: float, return_indices: bool = False):
    """
    Downsample point cloud using voxel grid filtering.
    Much faster than FPS for initial downsampling.

    Args:
        points: Point cloud (N, 3)
        voxel_size: Size of voxel grid in meters
        return_indices: If True, return (downsampled_points, indices) instead of just points

    Returns:
        downsampled_points: Downsampled point cloud (M, 3) where M <= N
        indices: (optional) Indices of kept points in original array
    """
    if len(points) == 0:
        return (points, np.array([], dtype=np.int32)) if return_indices else points

    # Compute voxel indices for each point
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # Shift to avoid negative indices
    min_indices = voxel_indices.min(axis=0)
    voxel_indices = voxel_indices - min_indices

    # Use lexsort + unique trick (much faster than dict for large arrays)
    # Sort by z, then y, then x
    sorted_indices = np.lexsort((voxel_indices[:, 2], voxel_indices[:, 1], voxel_indices[:, 0]))
    sorted_voxels = voxel_indices[sorted_indices]

    # Find unique consecutive voxels (much faster than full unique)
    unique_mask = np.ones(len(sorted_voxels), dtype=bool)
    unique_mask[1:] = np.any(sorted_voxels[1:] != sorted_voxels[:-1], axis=1)

    unique_indices = sorted_indices[unique_mask]

    if return_indices:
        return points[unique_indices], unique_indices
    return points[unique_indices]


@ray.remote
def copy_s3_object(source_bucket: str, source_key: str, dest_bucket: str, dest_key: str) -> str:
    """Copy a single S3 object from source to destination."""
    s3_client = boto3.client("s3")
    copy_source = {"Bucket": source_bucket, "Key": source_key}
    s3_client.copy_object(CopySource=copy_source, Bucket=dest_bucket, Key=dest_key)
    return dest_key


def recursive_s3_copy(path1: str, path2: str) -> None:
    """
    Recursively copy all objects from path1 to path2 using Ray for parallelization.
    """
    from vla_foundry.file_utils import list_s3_directory_recursive, parse_s3_path

    # Parse source and destination paths
    source_bucket, source_prefix = parse_s3_path(path1)
    dest_bucket, dest_prefix = parse_s3_path(path2)
    if source_prefix and not source_prefix.endswith("/"):
        source_prefix += "/"
    if dest_prefix and not dest_prefix.endswith("/"):
        dest_prefix += "/"

    relative_paths = list(list_s3_directory_recursive(path1))
    print(f"Found {len(relative_paths)} objects to copy")
    print(f"Starting parallel copy from {path1} to {path2}")

    # Build copy tasks: (source_bucket, source_key, dest_bucket, dest_key)
    copy_tasks = []
    for relative_path in relative_paths:
        source_key = source_prefix + relative_path
        dest_key = dest_prefix + relative_path
        copy_tasks.append((source_bucket, source_key, dest_bucket, dest_key))

    # Launch Ray tasks in parallel for copying
    futures = [
        copy_s3_object.remote(src_bucket, src_key, dst_bucket, dst_key)
        for src_bucket, src_key, dst_bucket, dst_key in copy_tasks
    ]
    copied_keys = ray.get(futures)
    print(f"✅ Successfully copied {len(copied_keys)} objects from {path1} to {path2}")


def depth_images_to_point_cloud(
    depth_images: dict,
    rgb_images: dict,
    intrinsics: dict,
    extrinsics: dict,
    num_points: int = 50000,
    voxel_size: float = 0.0025,
    depth_scale: float = 1000.0,
    filter_ground_plane: bool = False,
    depth_subsample_factor: int = 2,
    normalize_colors: bool = True,
) -> np.ndarray | None:
    """
    Convert multi-view depth images to a single downsampled colored point cloud.

    Args:
        depth_images: Dict of depth images {camera_name: (H, W) array in units specified by depth_scale}
        rgb_images: Dict of RGB images {camera_name: (H, W, 3) uint8 array}
        intrinsics: Dict of intrinsic matrices {camera_name: (3, 3) array}
        extrinsics: Dict of extrinsic matrices {camera_name: (4, 4) array}
        num_points: Number of points to downsample to
        voxel_size: Voxel size in meters for initial downsampling (default: 2.5mm)
        depth_scale: Scale factor to convert depth values to meters (default: 1000.0 for mm→m).
                     Similar to Open3D's depth_scale parameter.
        filter_ground_plane: Whether to filter out points below z=0 (default: False).
                             Set to True for datasets where z=0 represents ground plane.
        depth_subsample_factor: Subsample depth images by this factor before processing (default: 2).
                               Higher values = faster but lower quality. Set to 1 to disable.

    Returns:
        point_cloud: Downsampled world-space colored point cloud (num_points, 6) with [x,y,z,r,g,b],
                     or None if no valid points are available
    """
    all_points = []
    all_colors = []

    # Process each camera view
    for camera_name, depth_img in depth_images.items():
        K = intrinsics[camera_name]
        Rt = extrinsics[camera_name]
        rgb_img = rgb_images[camera_name]

        # Estimate max possible points after subsampling
        h, w = depth_img.shape
        estimated_points_per_cam = (h * w) // (depth_subsample_factor**2) // 4  # Rough estimate after filtering

        # Only subsample if we'll still have enough points
        should_subsample = (
            depth_subsample_factor > 1 and (estimated_points_per_cam * len(depth_images)) > num_points * 2
        )

        if should_subsample:
            depth_img = depth_img[::depth_subsample_factor, ::depth_subsample_factor]
            rgb_img = rgb_img[::depth_subsample_factor, ::depth_subsample_factor]
            # Adjust intrinsics for subsampled image
            K = K.copy()
            K[0, 0] /= depth_subsample_factor  # fx
            K[1, 1] /= depth_subsample_factor  # fy
            K[0, 2] /= depth_subsample_factor  # cx
            K[1, 2] /= depth_subsample_factor  # cy

        # Generate points efficiently
        h, w = depth_img.shape
        depth_flat = depth_img.flatten()

        # Generate all points
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        y = y.flatten()
        x = x.flatten()
        depths = depth_flat.astype(np.float32) / depth_scale

        colors = rgb_img.reshape(-1, 3).astype(np.float32)
        if normalize_colors:
            colors = colors / 255.0

        # Generate 3D points from depth
        ones = np.ones_like(x, dtype=np.float32)
        pixels = np.stack([x, y, ones], axis=-1)

        # Apply inverse intrinsics (compute once per camera)
        K_inv = np.linalg.inv(K)
        cam_coords = pixels @ K_inv.T

        # Scale by depth
        cam_points = cam_coords * depths[:, None]
        cam_colors = colors

        if len(cam_points) == 0:
            continue

        # Transform to world space
        world_points = transform_points_to_world(cam_points, Rt)

        # Aggressive per-camera voxel downsampling (reduces data before concatenation)
        # This is the main bottleneck reduction - voxel per camera instead of after concatenation
        world_points, voxel_indices = voxel_downsample(world_points, voxel_size, return_indices=True)
        cam_colors = cam_colors[voxel_indices]

        all_points.append(world_points)
        all_colors.append(cam_colors)

    if len(all_points) == 0:
        # No valid points available
        return None

    # Combine all views
    combined_points = np.concatenate(all_points, axis=0)
    combined_colors = np.concatenate(all_colors, axis=0) if all_colors else None

    # Filter out points below ground plane (z < 0) if requested
    if filter_ground_plane:
        valid_z_mask = combined_points[:, 2] >= 0
        combined_points = combined_points[valid_z_mask]
        if combined_colors is not None:
            combined_colors = combined_colors[valid_z_mask]

    # Check if we have enough points
    if len(combined_points) < num_points:
        # Not enough valid points
        return None

    # Random downsampling to exact num_points
    # Use simpler random permutation + slicing (faster than np.random.choice without replacement)
    # Skip final voxel pass since we already voxeled per-camera
    random_indices = np.random.permutation(len(combined_points))[:num_points]
    point_cloud_xyz = combined_points[random_indices]
    point_cloud_rgb = combined_colors[random_indices] if combined_colors is not None else None

    # Concatenate XYZ and RGB into single array (N, 6)
    point_cloud = np.concatenate([point_cloud_xyz, point_cloud_rgb], axis=1)  # (N, 6)

    return point_cloud.astype(np.float16)
