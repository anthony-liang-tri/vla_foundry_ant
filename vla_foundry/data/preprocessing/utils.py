import io
import json
import re
import tarfile
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from PIL import Image

from vla_foundry.aws.s3_io import download_fileobj_from_s3, upload_fileobj_to_s3
from vla_foundry.aws.s3_path import S3Path
from vla_foundry.aws.s3_utils import create_s3_client, is_s3_path
from vla_foundry.data.constants import (
    CAMERA_FISHEYE_DISTORTION,
    POINT_MAP_MAX_MM,
    POINT_MAP_MIN_MM,
    POINT_MAP_UINT16_OFFSET,
)
from vla_foundry.data.preprocessing.image_utils import (
    ImageResizingMethod,
    depth_image_to_bytes,
    image_to_bytes,
    point_map_to_bytes,
)
from vla_foundry.data.robotics.cv_utils import scale_intrinsics_3x3_for_resize_and_crop
from vla_foundry.file_utils import list_s3_directory_recursive


def is_cabot_fisheye_camera(camera_name: str, intrinsics: np.ndarray) -> bool:
    """
    Detect if a camera is a cabot fisheye camera by matching intrinsics.

    Due to a simulator bug, we need to distinguish cabot (fisheye) from riverway (pinhole)
    cameras that share the same camera names. We do this by matching the intrinsics matrix
    against known cabot fisheye intrinsics (up to 3 decimal places).

    Args:
        camera_name: Camera semantic name (e.g., "wrist_left_minus")
        intrinsics: Camera intrinsic matrix (3, 3)

    Returns:
        True if this is a cabot fisheye camera that needs distortion correction
    """
    if camera_name not in CAMERA_FISHEYE_DISTORTION:
        return False

    fisheye_params = CAMERA_FISHEYE_DISTORTION[camera_name]
    K_fisheye_expected = np.array(fisheye_params["K_fisheye"], dtype=np.float32)

    # Match up to 3 decimal places to handle floating point precision
    return np.allclose(intrinsics, K_fisheye_expected, atol=1e-3, rtol=0)


def apply_fisheye_distortion_to_depth_images(depth_images: dict, intrinsics: dict, depth_scale: float = 1000.0) -> dict:
    """
    Apply fisheye distortion to depth images for cabot cameras.

    This converts pinhole depth images to fisheye-distorted depth images that align with RGB.

    Args:
        depth_images: Dict of depth images {camera_name: (H, W) array in units specified by depth_scale}
        intrinsics: Dict of intrinsic matrices {camera_name: (3, 3) array}
        depth_scale: Scale factor to convert depth values to meters (default: 1000.0 for mm→m)

    Returns:
        Dict {camera_name: (H, W) depth array in meters} (fisheye-distorted for cabot, unchanged for others)
    """
    depth_fisheye_dict = {}

    for camera_name, depth_img in depth_images.items():
        K = intrinsics[camera_name]

        if is_cabot_fisheye_camera(camera_name, K):
            # Cabot fisheye camera: convert pinhole depth to fisheye coordinates
            fisheye_params = CAMERA_FISHEYE_DISTORTION[camera_name]
            K_pinhole = np.array(fisheye_params["K_pinhole"], dtype=np.float32)
            K_fisheye = K  # The stored intrinsics are fisheye intrinsics
            distortion_coeffs = fisheye_params["d"]

            # Convert to fisheye camera coordinates
            cam_coords = convert_pinhole_depth_to_fisheye_coords(
                depth_img, K_pinhole, K_fisheye, distortion_coeffs, depth_scale
            )

            # Extract Z as distorted depth in meters
            depth_fisheye_dict[camera_name] = cam_coords[:, :, 2]
        else:
            # Pinhole camera: just convert to meters
            depth_fisheye_dict[camera_name] = depth_img.astype(np.float32) / depth_scale

    return depth_fisheye_dict


def upload_sample_to_s3(
    sample_data: dict[str, Any],
    output_dir: str,
    episode_path: str,
    episode_id: str,
    frame_idx: int,
    jpeg_quality: int = 95,
    resize_images_size: list[int] | None = None,
    image_resizing_method: ImageResizingMethod = ImageResizingMethod.CENTER_CROP,
) -> None:
    """Upload sample data to S3 as tar file. (or save locally)"""
    s3_client = create_s3_client() if is_s3_path(output_dir) else None
    tar_buffer = io.BytesIO()
    uuid_prefix = str(uuid.uuid4())

    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        original_image_sizes = {}
        # Convert images to bytes (JPEG for RGB, PNG for depth)
        for img_key, img_data in sample_data["images"].items():
            # Check if this is a depth image
            is_depth = "depth" in img_key

            if not isinstance(img_data, bytes):
                if resize_images_size is None:
                    raise ValueError(
                        f"Image '{img_key}' is a numpy array but resize_images_size is not configured. "
                        "resize_images_size must be specified in preprocessing config when using numpy array images. "
                        "Set resize_images_size to your desired [width, height], e.g., [384, 384]."
                    )
                if is_depth:
                    # Depth images: PNG with uint16
                    image_bytes, original_image_size = depth_image_to_bytes(img_data, target_size=resize_images_size)
                    file_extension = "png"
                else:
                    # RGB images: JPEG
                    image_bytes, original_image_size = image_to_bytes(
                        img_data,
                        quality=jpeg_quality,
                        target_size=resize_images_size,
                        resize_method=image_resizing_method,
                    )
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

        # Handle point maps as TIFF files (3-channel 16-bit images)
        if "point_maps" in sample_data and sample_data["point_maps"] is not None:
            for pm_key, pm_data in sample_data["point_maps"].items():
                # pm_data is (H, W, 3) uint16 array with XYZ coordinates
                # (offset by +POINT_MAP_UINT16_OFFSET to handle negatives)
                assert pm_data.dtype == np.uint16, f"Point map must be uint16, got {pm_data.dtype}"
                assert pm_data.ndim == 3 and pm_data.shape[2] == 3, f"Point map must be (H, W, 3), got {pm_data.shape}"

                # Convert to TIFF bytes with resizing to match image size
                image_bytes, _ = point_map_to_bytes(pm_data, target_size=resize_images_size)

                # Insert "_point_map" before the timestep suffix
                # pm_key format: "camera_t0" -> "camera_point_map_t0"
                # e.g., "scene_right_0_t0" -> "scene_right_0_point_map_t0"
                if "_t" in pm_key:
                    camera_part, t_part = pm_key.rsplit("_t", 1)
                    filename = f"{uuid_prefix}.{camera_part}_point_map_t{t_part}.tiff"
                else:
                    filename = f"{uuid_prefix}.{pm_key}_point_map.tiff"

                tarinfo = tarfile.TarInfo(name=filename)
                tarinfo.size = len(image_bytes)
                tar.addfile(tarinfo, io.BytesIO(image_bytes))
        # Add the rescaled intrinsics into the sample_data
        sample_lowdim_data_with_rescaled_intrinsics = dict()
        if "lowdim" in sample_data and sample_data["lowdim"] is not None:
            for lowdim_key in sample_data["lowdim"]:
                # Check if the lowdim_key is of the form "original_intrinsics.{camera_name}"
                if bool(re.fullmatch(r"original_intrinsics\.[A-Za-z0-9_-]+", lowdim_key)):
                    camera_name = lowdim_key.split(".", 1)[1]
                    original_intrinsics = sample_data["lowdim"][lowdim_key]
                    assert camera_name in original_image_sizes, (
                        f"Camera name {camera_name} not found in original_image_sizes"
                    )

                    original_image_size = original_image_sizes[camera_name]
                    scaled_intrinsics = scale_intrinsics_3x3_for_resize_and_crop(
                        original_intrinsics, original_image_size, resize_images_size, image_resizing_method
                    )

                    # Add rescaled intrinsics.
                    sample_lowdim_data_with_rescaled_intrinsics[f"rescaled_intrinsics.{camera_name}"] = (
                        scaled_intrinsics
                    )
            sample_data["lowdim"].update(sample_lowdim_data_with_rescaled_intrinsics)

        for key, value in sample_data.items():
            data_buffer = io.BytesIO()
            if key == "images" or key == "point_maps":  # Already added
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

    if is_s3_path(output_dir):
        # Upload to S3
        parsed = S3Path(s3_path=output_dir)
        s3_key = f"{parsed.key.rstrip('/')}/frames/{tar_filename}"
        s3_client.upload_fileobj(tar_buffer, parsed.bucket, s3_key)
        print(f"Uploaded s3://{parsed.bucket}/{s3_key}", flush=True)
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


def save_and_upload_dict(dict_data: dict, output_path: str, file_name: str):
    # Used to upload manifest.jsonl and stats.json (or save locally)
    body = "\n".join(json.dumps(record) for record in dict_data) if "jsonl" in file_name else json.dumps(dict_data)

    if is_s3_path(output_path):
        # Upload to S3
        parsed = S3Path(s3_path=output_path)
        s3_key = f"{parsed.key.rstrip('/')}/{file_name}"
        create_s3_client().put_object(
            Bucket=parsed.bucket,
            Key=s3_key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        print(f"Uploaded {file_name} to s3://{parsed.bucket}/{s3_key}")
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

    import draccus

    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w") as temp_file:
        draccus.dump(config, temp_file)
        temp_path = temp_file.name

    if is_s3_path(output_path):
        # Upload to S3
        parsed = S3Path(s3_path=output_path)
        s3_key = f"{parsed.key.rstrip('/')}/{file_name}"
        create_s3_client().upload_file(temp_path, parsed.bucket, s3_key)
        print(f"Uploaded {file_name} to s3://{parsed.bucket}/{s3_key}")
    else:
        # Save to local filesystem
        local_path = Path(output_path) / file_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(temp_path, local_path)
        print(f"Saved {file_name} to {local_path}")


def _download_tar_from_s3(s3_key: str, s3_client, bucket_name: str, s3_prefix: str):
    """Download a single tar file from S3 with retry logic."""
    full_key = f"{s3_prefix.rstrip('/')}/frames/{s3_key}"
    obj_buffer = download_fileobj_from_s3(bucket_name, full_key, s3_client=s3_client)
    return (s3_key, obj_buffer)


@ray.remote
def create_episode_shard(shard_files: list[str], episode_key: str, output_dir: str) -> str:
    """Download/read tar files and create an episode-based shard. Supports both S3 and local filesystem."""
    is_s3 = is_s3_path(output_dir)

    if is_s3:
        s3_client = create_s3_client()
        parsed = S3Path(s3_path=output_dir)
        bucket_name, s3_prefix = parsed.bucket, parsed.key
        download_tar = partial(_download_tar_from_s3, s3_client=s3_client, bucket_name=bucket_name, s3_prefix=s3_prefix)
    else:
        frames_dir = Path(output_dir) / "frames"

        def download_tar(tar_key):
            """Read a single tar file from local filesystem."""
            tar_path = frames_dir / tar_key
            with open(tar_path, "rb") as f:
                obj_buffer = io.BytesIO(f.read())
            return (tar_key, obj_buffer)

    # Download/read all tars in parallel
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

    # Save shard
    shard_buffer.seek(0)
    shard_key = f"episode_{episode_key}.tar"

    if is_s3:
        s3_client.upload_fileobj(shard_buffer, bucket_name, f"{s3_prefix.rstrip('/')}/episodes/{shard_key}")
        print(f"Uploaded episode shard {shard_key} to s3://{bucket_name}/{s3_prefix.rstrip('/')}/episodes/{shard_key}")
    else:
        episodes_dir = Path(output_dir) / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        shard_path = episodes_dir / shard_key
        with open(shard_path, "wb") as f:
            f.write(shard_buffer.getvalue())
        print(f"Saved episode shard {shard_key} to {shard_path}")

    return (shard_key.rstrip(".tar"), len(shard_files))


@ray.remote
def create_shard(shard_files: list[str], shard_idx: int, output_dir: str) -> str:
    """Download tar files from S3 and create a shard. OPTIMIZED with parallel downloads."""
    is_s3 = is_s3_path(output_dir)

    if is_s3:
        s3_client = create_s3_client()
        parsed = S3Path(s3_path=output_dir)
        bucket_name, s3_prefix = parsed.bucket, parsed.key

        def read_tar(tar_key):
            """Download a single tar file from S3."""
            obj_buffer = io.BytesIO()
            full_key = f"{s3_prefix.rstrip('/')}/frames/{tar_key}"
            s3_client.download_fileobj(bucket_name, full_key, obj_buffer)
            obj_buffer.seek(0)
            return (tar_key, obj_buffer)
    else:
        frames_dir = Path(output_dir) / "frames"

        def read_tar(tar_key):
            """Read a single tar file from local filesystem."""
            tar_path = frames_dir / tar_key
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

    if is_s3:
        upload_fileobj_to_s3(
            shard_buffer, bucket_name, f"{s3_prefix.rstrip('/')}/shards/{shard_name}", s3_client=s3_client
        )
        print(f"Uploaded shard {shard_name} to s3://{bucket_name}/{s3_prefix.rstrip('/')}/shards/{shard_name}")
    else:
        shard_path = Path(output_dir) / "shards" / shard_name
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        with open(shard_path, "wb") as f:
            f.write(shard_buffer.getvalue())
        print(f"Saved shard {shard_name} to {shard_path}")

    return (shard_name.rstrip(".tar"), len(shard_files))


def is_still_sample(lowdim_data: dict[str, np.ndarray], start_idx: int, end_idx: int, still_threshold: float) -> bool:
    """Check if sample is still by looking at action/position/pose keys."""
    recognized_patterns = ["action", "joint", "poses", "xyz", "actual"]
    movement_keys = [k for k in lowdim_data if any(x in k.lower() for x in recognized_patterns)]

    # If no recognized keys found, we can't determine stillness; don't filter
    if not movement_keys:
        return False

    for key in movement_keys:
        data = lowdim_data[key][start_idx : end_idx + 1]
        if len(data) > 1:
            # max() handles multi-dimensional arrays (e.g. 7-DoF joints)
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


@ray.remote
def copy_s3_object(source_bucket: str, source_key: str, dest_bucket: str, dest_key: str) -> str:
    """Copy a single S3 object from source to destination."""
    s3_client = create_s3_client()
    copy_source = {"Bucket": source_bucket, "Key": source_key}
    s3_client.copy_object(CopySource=copy_source, Bucket=dest_bucket, Key=dest_key)
    return dest_key


def recursive_s3_copy(path1: str, path2: str) -> None:
    """
    Recursively copy all objects from path1 to path2 using Ray for parallelization.
    """
    from vla_foundry.aws.s3_path import S3Path

    # Parse source and destination paths
    source = S3Path(s3_path=path1)
    dest = S3Path(s3_path=path2)
    source_bucket, source_prefix = source.bucket, source.key.rstrip("/") + "/"
    dest_bucket, dest_prefix = dest.bucket, dest.key.rstrip("/") + "/"

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


def apply_inverse_fisheye_distortion(
    cam_coords: np.ndarray, distortion_params: list[float] | None, max_valid_radius: float = None
) -> np.ndarray:
    """
    Apply inverse fisheye distortion to convert fisheye coordinates to pinhole coordinates.

    Args:
        cam_coords: Normalized camera coordinates (N, 3) where each row is [x, y, 1]
        distortion_params: Fisheye distortion parameters [d0, d1, d2, d3] or None
        max_valid_radius: Maximum valid radius in normalized coordinates. Pixels beyond this
                         radius will not have distortion correction applied. If None, no masking.

    Returns:
        corrected_coords: Distortion-corrected normalized camera coordinates (N, 3)
    """
    if distortion_params is None:
        return cam_coords  # No distortion correction needed

    d = np.array(distortion_params, dtype=np.float32)

    # Extract x, y coordinates (third component is 1)
    x = cam_coords[:, 0]
    y = cam_coords[:, 1]

    # Compute r = sqrt(x^2 + y^2)
    r = np.sqrt(x**2 + y**2 + 1e-8)  # Add epsilon to avoid division by zero

    # Inverse fisheye distortion model (map fisheye to pinhole)
    theta_d = r
    theta = theta_d / (1 + d[0] * theta_d**2 + d[1] * theta_d**4 + d[2] * theta_d**6 + d[3] * theta_d**8)

    # Compute scale factor: tan(theta) / r
    # Use np.where to handle r ≈ 0 case
    scale = np.where(r > 1e-8, np.tan(theta) / r, np.ones_like(r))

    # Apply masking if max_valid_radius is specified
    # Pixels beyond this radius should not have distortion correction
    if max_valid_radius is not None:
        scale = np.where(r <= max_valid_radius, scale, np.ones_like(scale))

    # Apply scaling to get pinhole coordinates
    x_corrected = x * scale
    y_corrected = y * scale

    # Return corrected coordinates with 1 in the third component
    return np.stack([x_corrected, y_corrected, np.ones_like(x)], axis=-1)


def convert_pinhole_depth_to_fisheye_coords(
    depth_pinhole: np.ndarray,
    K_pinhole: np.ndarray,
    K_fisheye: np.ndarray,
    distortion_params: list[float],
    depth_scale: float = 1000.0,
) -> np.ndarray:
    """
    Convert pinhole depth image to 3D camera coordinates in fisheye image space.

    This function addresses the simulator bug where depth is pinhole but RGB is fisheye.
    Instead of transferring depth values (which are ray-dependent), we convert to 3D
    camera coordinates which are invariant to the camera model.

    Args:
        depth_pinhole: Pinhole depth image (H, W) - undistorted, in units specified by depth_scale
        K_pinhole: Pinhole camera intrinsic matrix (3, 3)
        K_fisheye: Fisheye camera intrinsic matrix (3, 3)
        distortion_params: Fisheye distortion parameters [d0, d1, d2, d3]
        depth_scale: Scale factor to convert depth values to meters (default: 1000.0 for mm→m)

    Returns:
        cam_coords_fisheye: (H, W, 3) array of XYZ camera coordinates in meters in fisheye image space
    """
    h, w = depth_pinhole.shape

    # Step 1: Convert pinhole depth image to 3D camera coordinates
    y_pinhole, x_pinhole = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    ones = np.ones_like(x_pinhole, dtype=np.float32)
    pixels_pinhole = np.stack([x_pinhole, y_pinhole, ones], axis=-1).astype(np.float32)

    # Convert to normalized pinhole coordinates
    K_pinhole_inv = np.linalg.inv(K_pinhole)
    pixels_pinhole_flat = pixels_pinhole.reshape(-1, 3)
    cam_coords_normalized = pixels_pinhole_flat @ K_pinhole_inv.T  # (H*W, 3)

    # Scale by depth to get 3D camera coordinates (convert to meters)
    depth_flat = depth_pinhole.flatten().astype(np.float32) / depth_scale
    X = cam_coords_normalized[:, 0] * depth_flat
    Y = cam_coords_normalized[:, 1] * depth_flat
    Z = depth_flat
    cam_coords_3d = np.stack([X, Y, Z], axis=-1)  # (H*W, 3) in meters

    # Step 2: For each fisheye pixel, find corresponding 3D point
    # Create fisheye pixel grid
    y_fisheye, x_fisheye = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pixels_fisheye = np.stack([x_fisheye, y_fisheye, ones], axis=-1).astype(np.float32)

    # Convert fisheye pixels to normalized fisheye coordinates
    K_fisheye_inv = np.linalg.inv(K_fisheye)
    pixels_fisheye_flat = pixels_fisheye.reshape(-1, 3)
    cam_coords_fisheye_norm = pixels_fisheye_flat @ K_fisheye_inv.T

    # Apply inverse fisheye distortion to get pinhole normalized coordinates
    cam_coords_pinhole_norm = apply_inverse_fisheye_distortion(cam_coords_fisheye_norm, distortion_params)

    # Convert to pinhole pixel coordinates to find which 3D point to sample
    pixels_pinhole_lookup = cam_coords_pinhole_norm @ K_pinhole.T
    x_lookup = pixels_pinhole_lookup[:, 0]
    y_lookup = pixels_pinhole_lookup[:, 1]

    # Sample 3D coordinates using nearest neighbor
    x_lookup_int = np.round(x_lookup).astype(np.int32)
    y_lookup_int = np.round(y_lookup).astype(np.int32)

    # Create output array
    cam_coords_fisheye = np.zeros((h * w, 3), dtype=np.float32)

    # Create valid mask
    valid_mask = (x_lookup_int >= 0) & (x_lookup_int < w) & (y_lookup_int >= 0) & (y_lookup_int < h)

    # Sample 3D coordinates
    valid_indices = y_lookup_int[valid_mask] * w + x_lookup_int[valid_mask]
    cam_coords_fisheye[valid_mask] = cam_coords_3d[valid_indices]

    # Reshape to image dimensions
    return cam_coords_fisheye.reshape(h, w, 3)


def depth_images_to_point_cloud(
    depth_images: dict,
    rgb_images: dict,
    intrinsics: dict,
    extrinsics: dict,
    num_points: int = 4096,
    depth_scale: float = 1000.0,
    filter_ground_plane: bool = False,
    depth_subsample_factor: int = 2,
    normalize_colors: bool = True,
    min_depth: float = 0.001,
    max_depth: float = 3.0,
) -> np.ndarray | None:
    """
    Convert multi-view depth images to a single downsampled colored point cloud using CUDA FPS.

    Args:
        depth_images: Dict of depth images {camera_name: (H, W) array in units specified by depth_scale}
        rgb_images: Dict of RGB images {camera_name: (H, W, 3) uint8 array}
        intrinsics: Dict of intrinsic matrices {camera_name: (3, 3) array}
        extrinsics: Dict of extrinsic matrices {camera_name: (4, 4) array}
        num_points: Total number of points in output (default: 4096).
                    Divided evenly across views: each view contributes num_points // num_views points via FPS.
                    If a view has insufficient valid points, the output is zero-padded to maintain num_points.
        depth_scale: Scale factor to convert depth values to meters (default: 1000.0 for mm→m).
                     Similar to Open3D's depth_scale parameter.
        filter_ground_plane: Whether to filter out points below z=0 (default: False).
                             Set to True for datasets where z=0 represents ground plane.
        depth_subsample_factor: Subsample depth images by this factor before processing (default: 2).
                               Higher values = faster but lower quality. Set to 1 to disable.
        normalize_colors: Whether to normalize RGB values to [0, 1] range (default: True).
        min_depth: Minimum valid depth in meters (default: 0.001m = 1mm). Filters out invalid/too-close points.
        max_depth: Maximum valid depth in meters (default: 3.0m). Filters out too-far/unreliable points.

    Returns:
        point_cloud: World-space colored point cloud (num_points, 6) with [x,y,z,r,g,b],
                     or None if no valid points are available
    """
    import cuda_fps

    all_point_clouds = []

    # Compute points per view (divide total evenly across views)
    num_views = len(depth_images)
    points_per_view = num_points // num_views

    # Process each camera view
    for camera_name, depth_img in depth_images.items():
        K = intrinsics[camera_name]
        Rt = extrinsics[camera_name]
        rgb_img = rgb_images[camera_name]

        # Convert depth to camera coordinates (apply fisheye distortion for cabot)
        if is_cabot_fisheye_camera(camera_name, K):
            fisheye_params = CAMERA_FISHEYE_DISTORTION[camera_name]
            K_pinhole = np.array(fisheye_params["K_pinhole"], dtype=np.float32)
            K_fisheye = K
            distortion_coeffs = fisheye_params["d"]
            cam_coords = convert_pinhole_depth_to_fisheye_coords(
                depth_img, K_pinhole, K_fisheye, distortion_coeffs, depth_scale
            )
            h, w, _ = cam_coords.shape
        else:
            h, w = depth_img.shape
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            ones = np.ones_like(x, dtype=np.float32)
            pixels = np.stack([x, y, ones], axis=-1).astype(np.float32)

            K_inv = np.linalg.inv(K)
            pixels_flat = pixels.reshape(-1, 3)
            cam_coords_normalized = pixels_flat @ K_inv.T

            depths = depth_img.flatten().astype(np.float32) / depth_scale
            X = cam_coords_normalized[:, 0] * depths
            Y = cam_coords_normalized[:, 1] * depths
            Z = depths
            cam_coords = np.stack([X, Y, Z], axis=-1).reshape(h, w, 3)

        # Estimate max possible points after subsampling
        h, w = depth_img.shape
        # Rough estimate after filtering
        estimated_points_per_cam = (h * w) // (depth_subsample_factor**2) // 4

        # Only subsample if we'll still have enough points
        should_subsample = (
            depth_subsample_factor > 1 and (estimated_points_per_cam * len(depth_images)) > num_points * 2
        )

        if should_subsample:
            cam_coords = cam_coords[::depth_subsample_factor, ::depth_subsample_factor]
            rgb_img = rgb_img[::depth_subsample_factor, ::depth_subsample_factor]
            h, w, _ = cam_coords.shape

        # Extract XYZ and colors
        cam_coords_flat = cam_coords.reshape(-1, 3)
        X = cam_coords_flat[:, 0]
        Y = cam_coords_flat[:, 1]
        Z = cam_coords_flat[:, 2]

        colors = rgb_img.reshape(-1, 3).astype(np.float32)
        if normalize_colors:
            colors = colors / 255.0

        # Filter invalid depths (0 values, too close, or too far)
        valid_depth_mask = (min_depth < Z) & (max_depth > Z)
        X = X[valid_depth_mask]
        Y = Y[valid_depth_mask]
        Z = Z[valid_depth_mask]
        colors = colors[valid_depth_mask]

        # Stack into camera points
        cam_points = np.stack([X, Y, Z], axis=-1)
        cam_colors = colors

        if len(cam_points) == 0:
            # No valid points for this camera, add zero-padded placeholder
            sampled_points = np.zeros((points_per_view, 6), dtype=np.float32)
            all_point_clouds.append(sampled_points)
            continue

        # Transform to world space
        world_points = transform_points_to_world(cam_points, Rt)

        # Filter ground plane if requested (before FPS)
        if filter_ground_plane:
            valid_z_mask = world_points[:, 2] >= 0
            world_points = world_points[valid_z_mask]
            cam_colors = cam_colors[valid_z_mask]

        if len(world_points) == 0:
            # No valid points after filtering, add zero-padded placeholder
            sampled_points = np.zeros((points_per_view, 6), dtype=np.float32)
            all_point_clouds.append(sampled_points)
            continue

        # Apply FPS to sample points_per_view points from this camera view
        # Use CUDA if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points_tensor = torch.from_numpy(world_points).float().to(device)
        colors_tensor = torch.from_numpy(cam_colors).float().to(device)
        points_with_colors = torch.cat([points_tensor, colors_tensor], dim=1)  # (N, 6)

        if len(points_with_colors) >= points_per_view:
            # Enough points, apply FPS to downsample
            offsets = torch.tensor([0, len(points_with_colors)], dtype=torch.int32, device=points_with_colors.device)
            # distance_dim=3: compute FPS distances using only XYZ coordinates (first 3 dims), ignoring RGB channels
            sampled = cuda_fps.fps(points_with_colors, offsets, points_per_view, distance_dim=3)
            sampled_points = sampled.cpu().numpy()  # (points_per_view, 6)
        else:
            # Not enough points, keep all and pad with zeros
            sampled_points = points_with_colors.cpu().numpy()
            padding = np.zeros((points_per_view - len(sampled_points), 6), dtype=np.float32)
            sampled_points = np.concatenate([sampled_points, padding], axis=0)

        all_point_clouds.append(sampled_points)

    point_cloud = np.concatenate(all_point_clouds, axis=0)  # (num_views * points_per_view, 6)

    # Pad with points at origin if we're short due to integer division
    actual_num_points = len(point_cloud)
    if actual_num_points < num_points:
        num_padding = num_points - actual_num_points
        padding = np.zeros((num_padding, 6), dtype=np.float32)
        point_cloud = np.concatenate([point_cloud, padding], axis=0)

    return point_cloud.astype(np.float16)


def depth_images_to_point_maps(
    depth_images: dict,
    intrinsics: dict,
    depth_scale: float = 1000.0,
    min_depth: float = 0.001,
    max_depth: float = 3.0,
) -> dict:
    """
    Generate point maps (camera coordinates in image space) from depth images.

    Point maps preserve the full spatial resolution (H, W) of the depth images and store
    3D camera coordinates (X, Y, Z) for each pixel in uint16 millimeter format with offset.

    Args:
        depth_images: Dict of depth images {camera_name: (H, W) uint16 array in mm}
        intrinsics: Dict of intrinsic matrices {camera_name: (3, 3) array}
        depth_scale: Scale factor to convert depth values to meters (default: 1000.0 for mm→m)
        min_depth: Minimum valid depth in meters (default: 0.001m = 1mm). Invalid pixels set to 0.
        max_depth: Maximum valid depth in meters (default: 3.0m). Invalid pixels set to 0.

    Returns:
        Dict mapping camera_name to (H, W, 3) uint16 array with XYZ camera coordinates in mm.
        Coordinates are offset by +POINT_MAP_UINT16_OFFSET to handle negative values.
        Invalid depth pixels (out of range) are set to [POINT_MAP_UINT16_OFFSET] (representing [0, 0, 0]).
    """
    point_maps = {}

    for camera_name, depth_img in depth_images.items():
        # Compute camera coordinates
        K = intrinsics[camera_name]
        if is_cabot_fisheye_camera(camera_name, K):
            fisheye_params = CAMERA_FISHEYE_DISTORTION[camera_name]
            K_pinhole = np.array(fisheye_params["K_pinhole"], dtype=np.float32)
            K_fisheye = K
            distortion_coeffs = fisheye_params["d"]
            cam_points = convert_pinhole_depth_to_fisheye_coords(
                depth_img, K_pinhole, K_fisheye, distortion_coeffs, depth_scale
            )
        else:
            h, w = depth_img.shape
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            ones = np.ones_like(x, dtype=np.float32)
            pixels = np.stack([x, y, ones], axis=-1).astype(np.float32)

            K_inv = np.linalg.inv(K)
            pixels_flat = pixels.reshape(-1, 3)
            cam_coords_normalized = pixels_flat @ K_inv.T

            depths = depth_img.astype(np.float32) / depth_scale
            depths_flat = depths.flatten()
            X = cam_coords_normalized[:, 0] * depths_flat
            Y = cam_coords_normalized[:, 1] * depths_flat
            Z = depths_flat
            cam_points = np.stack([X, Y, Z], axis=-1).reshape(h, w, 3)

        h, w, _ = cam_points.shape
        depths = cam_points[:, :, 2]  # Extract Z for depth filtering

        # Convert from meters to millimeters
        cam_points_mm = cam_points * depth_scale
        cam_points_mm = np.clip(cam_points_mm, POINT_MAP_MIN_MM, POINT_MAP_MAX_MM)  # Clip to valid range

        # Add offset to handle negative coordinates (shifts range to [0, 65535])
        cam_points_offset = cam_points_mm + POINT_MAP_UINT16_OFFSET

        # Filter invalid depths: set to POINT_MAP_UINT16_OFFSET (representing 0) for out-of-range values (after offset)
        valid_depth_mask = (depths > min_depth) & (depths < max_depth)
        cam_points_offset[~valid_depth_mask] = (
            POINT_MAP_UINT16_OFFSET  # Invalid pixels = POINT_MAP_UINT16_OFFSET (represents 0 mm)
        )

        cam_points_uint16 = cam_points_offset.astype(np.uint16)

        point_maps[camera_name] = cam_points_uint16

    return point_maps
