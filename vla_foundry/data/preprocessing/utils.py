import io
import json
import os
import tarfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Dict, List

import boto3
import numpy as np
import ray
from botocore.config import Config
from PIL import Image

from vla_foundry.data.preprocessing.image_utils import image_to_bytes


def upload_sample_to_s3(
    sample_data: Dict[str, Any],
    output_dir: str,
    episode_path: str,
    frame_idx: int,
    jpeg_quality: int = 95,
    resize_images_size: List[int] = None,
) -> None:
    """Upload sample data to S3 as tar file."""
    if resize_images_size is None:
        resize_images_size = [224, 224]
    episode_id = os.path.basename(episode_path.rstrip("/"))
    s3_client = boto3.client("s3")
    tar_buffer = io.BytesIO()
    uuid_prefix = str(uuid.uuid4())

    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        original_image_sizes = {}
        # Use image_to_bytes to convert numpy arrays to JPEG bytes
        for img_key, img_data in sample_data["images"].items():
            if not isinstance(img_data, bytes):
                jpeg_bytes, original_image_size = image_to_bytes(img_data, jpeg_quality, resize_images_size)
            else:
                jpeg_bytes = img_data
                original_image_size = Image.open(io.BytesIO(img_data)).size

            # Log original image sizes
            camera_name_without_timestep = img_key.rsplit("_t", 1)[0]
            if isinstance(sample_data["metadata"], dict):
                assert camera_name_without_timestep in sample_data["metadata"].get("camera_names")
            else:
                assert camera_name_without_timestep in sample_data["metadata"].camera_names
            original_image_sizes[camera_name_without_timestep] = original_image_size

            tarinfo = tarfile.TarInfo(name=f"{uuid_prefix}.{img_key}.jpg")
            tarinfo.size = len(jpeg_bytes)
            tar.addfile(tarinfo, io.BytesIO(jpeg_bytes))

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
    bucket_name, s3_prefix = output_dir.removeprefix("s3://").split("/", 1)
    unique_id = extract_unique_id(episode_path)
    s3_key = f"{s3_prefix.rstrip('/')}/episodes/{unique_id}_{episode_id}_frame_{frame_idx}.tar"
    s3_client.upload_fileobj(tar_buffer, bucket_name, s3_key)
    print(f"Uploaded {bucket_name.rstrip('/')}/{s3_key}", flush=True)
    return s3_key.split("/")[-1]


def extract_unique_id(episode_path: str) -> str:
    """Extract a unique ID from the episode path."""
    if "diffusion_spartan" in episode_path:
        # For diffusion_spartan, use the datetime as unique id
        return episode_path.split("/")[-3]
    else:
        return str(uuid.uuid4())


def upload_dict_to_s3(dict_data: Dict, s3_path: str, file_name: str):
    # Used to upload manifest.jsonl and stats.json
    bucket_name, s3_prefix = s3_path.removeprefix("s3://").split("/", 1)
    body = "\n".join(json.dumps(record) for record in dict_data) if "jsonl" in file_name else json.dumps(dict_data)
    s3_key = f"{s3_prefix.rstrip('/')}/{file_name}"
    boto3.client("s3").put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )
    print(f"Uploaded {file_name} to s3://{bucket_name}/{s3_key}")


def upload_config_to_s3(config, s3_path: str, file_name: str):
    # Draccus dump to temp file then upload to s3
    import tempfile

    import draccus

    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w") as temp_file:
        draccus.dump(config, temp_file)
        temp_path = temp_file.name
    bucket_name, s3_prefix = s3_path.removeprefix("s3://").split("/", 1)
    s3_key = f"{s3_prefix.rstrip('/')}/{file_name}"
    boto3.client("s3").upload_file(temp_path, bucket_name, s3_key)
    print(f"Uploaded {file_name} to s3://{bucket_name}/{s3_key}")


@ray.remote
def create_shard(shard_files: List[str], shard_idx: int, output_dir: str) -> str:
    """Download tar files from S3 and create a shard. OPTIMIZED with parallel downloads."""
    s3_config = Config(max_pool_connections=50, retries={"max_attempts": 3, "mode": "adaptive"})
    s3_client = boto3.client("s3", config=s3_config)

    bucket_name, s3_prefix = output_dir.removeprefix("s3://").split("/", 1)

    def download_tar(s3_key):
        """Download a single tar file from S3."""
        obj_buffer = io.BytesIO()
        full_key = f"{s3_prefix.rstrip('/')}/episodes/{s3_key}"
        s3_client.download_fileobj(bucket_name, full_key, obj_buffer)
        obj_buffer.seek(0)
        return (s3_key, obj_buffer)

    # Download all tars in parallel (use 20 threads for download phase)
    downloaded_tars = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_tar, s3_key) for s3_key in shard_files]
        for future in as_completed(futures):
            s3_key, obj_buffer = future.result()
            downloaded_tars[s3_key] = obj_buffer

    # Create shard by combining all downloaded tars
    shard_buffer = io.BytesIO()
    with tarfile.open(fileobj=shard_buffer, mode="w") as shard_tar:
        # Process in original order for consistency
        for s3_key in shard_files:
            obj_buffer = downloaded_tars[s3_key]
            obj_buffer.seek(0)

            # Extract contents and add to shard
            with tarfile.open(fileobj=obj_buffer, mode="r") as tar:
                for member in tar.getmembers():
                    shard_tar.addfile(member, tar.extractfile(member))

    # Upload shard back to S3
    shard_buffer.seek(0)
    shard_key = f"shard_{shard_idx:06d}.tar"
    s3_client.upload_fileobj(shard_buffer, bucket_name, f"{s3_prefix.rstrip('/')}/shards/{shard_key}")
    print(f"Uploaded shard {shard_key} to s3://{bucket_name}/{s3_prefix.rstrip('/')}/shards/{shard_key}")
    return (shard_key.rstrip(".tar"), len(shard_files))


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
