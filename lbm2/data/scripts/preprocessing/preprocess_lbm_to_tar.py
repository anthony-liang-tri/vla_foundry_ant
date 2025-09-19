#!/usr/bin/env python3

import datetime
import io
import json
import os
import random
import tarfile
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import boto3
import draccus
import fsspec
import numpy as np
import ray
import yaml

from lbm2.data.scripts.preprocessing.image_utils import image_to_bytes, init_jpeg_encoder

# Params base class
from lbm2.data.scripts.preprocessing.params import PreprocessParams, SampleMetadata
from lbm2.data.scripts.preprocessing.preprocess_statistics import StreamingDatasetStatisticsRayActor
from lbm2.data.scripts.preprocessing.utils import create_processing_metadata, discover_episodes_targeted
from lbm2.file_utils import list_directory


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


class PaddingStrategy:
    """Optimized padding strategies with vectorized operations."""

    @staticmethod
    def copy_edge(data: np.ndarray, pad_before: int, pad_after: int) -> np.ndarray:
        """Vectorized edge padding."""
        if pad_before == 0 and pad_after == 0:
            return data

        pad_width = [(pad_before, pad_after)] + [(0, 0)] * (data.ndim - 1)
        return np.pad(data, pad_width, mode="edge")

    @staticmethod
    def zero_pad(data: np.ndarray, pad_before: int, pad_after: int) -> np.ndarray:
        """Vectorized zero padding."""
        if pad_before == 0 and pad_after == 0:
            return data

        pad_width = [(pad_before, pad_after)] + [(0, 0)] * (data.ndim - 1)
        return np.pad(data, pad_width, mode="constant", constant_values=0)

    @staticmethod
    def reflect_pad(data: np.ndarray, pad_before: int, pad_after: int) -> np.ndarray:
        """Vectorized reflect padding."""
        if pad_before == 0 and pad_after == 0:
            return data

        pad_width = [(pad_before, pad_after)] + [(0, 0)] * (data.ndim - 1)
        return np.pad(data, pad_width, mode="reflect")


def load_language_annotations(yaml_path: str) -> Dict[str, Dict[str, List[str]]]:
    """Load language annotations from YAML file."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("language_dict", {})


def extract_task_name_from_path(episode_path: str) -> str:
    """Extract task name from episode path (e.g., 'BimanualLayCerealBoxOnCuttingBoardFromUnderShelf')."""
    # Parse: s3://robotics-manip-lbm/efs/data/tasks/<task_name>/...
    path_parts = episode_path.split("/")
    tasks_index = path_parts.index("tasks")
    return path_parts[tasks_index + 1]


def get_language_instructions(
    task_name: str, annotations: Dict, instruction_types: List[str] = None
) -> Dict[str, List[str]]:
    """Get language instructions for a given task, organized by type.

    Args:
        task_name: Name of the task
        annotations: Language annotations dictionary
        instruction_types: List of instruction types to include. If None, includes all types.
                          Valid types: "original", "randomized", "verbose", "alternative"

    Returns:
        Dictionary mapping instruction type to list of instructions
    """
    if task_name not in annotations:
        return {}

    task_annotations = annotations[task_name]
    instructions_by_type = {}

    # Default to all types if none specified
    if instruction_types is None:
        instruction_types = ["original", "randomized", "verbose", "alternative"]

    # Add instructions for each requested type
    for instruction_type in instruction_types:
        if instruction_type in task_annotations:
            instructions_by_type[instruction_type] = task_annotations[instruction_type]

    return instructions_by_type


def upload_sample_to_s3(
    sample_data: Dict[str, Any],
    output_dir: str,
    episode_id: str,
    frame_idx: int,
    jpeg_quality: int = 95,
    resize_images_size: List[int] = None,
    s3_client=None,
) -> None:
    """Upload sample data to S3 as tar file."""
    if resize_images_size is None:
        resize_images_size = [224, 224]
    if s3_client is None:
        s3_client = boto3.client("s3")
    tar_buffer = io.BytesIO()
    uuid_prefix = str(uuid.uuid4())

    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        for key, value in sample_data.items():
            data_buffer = io.BytesIO()

            if key == "images":
                # Use image_to_bytes to convert numpy arrays to JPEG bytes
                for img_key, img_data in value.items():
                    jpeg_bytes, _ = image_to_bytes(img_data, jpeg_quality, resize_images_size)
                    tarinfo = tarfile.TarInfo(name=f"{uuid_prefix}.{img_key}.jpg")
                    tarinfo.size = len(jpeg_bytes)
                    tar.addfile(tarinfo, io.BytesIO(jpeg_bytes))
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
    s3_key = f"{s3_prefix.rstrip('/')}/{episode_id}_frame_{frame_idx}.tar"
    s3_client.upload_fileobj(tar_buffer, bucket_name, s3_key)
    print(f"Uploaded {bucket_name.rstrip('/')}/{s3_key}")


@ray.remote
def create_shard(shard_files: List[str], shard_idx: int, output_dir: str) -> str:
    """Download tar files from S3 and create a shard."""
    s3_client = boto3.client("s3")
    shard_buffer = io.BytesIO()

    with tarfile.open(fileobj=shard_buffer, mode="w") as shard_tar:
        for s3_key in shard_files:
            # Download tar file from S3
            obj_buffer = io.BytesIO()
            bucket_name, s3_prefix = output_dir.removeprefix("s3://").split("/", 1)
            s3_key = f"{s3_prefix.rstrip('/')}/{s3_key}"
            s3_client.download_fileobj(bucket_name, s3_key, obj_buffer)
            obj_buffer.seek(0)

            # Extract contents and add to shard
            with tarfile.open(fileobj=obj_buffer, mode="r") as tar:
                for member in tar.getmembers():
                    shard_tar.addfile(member, tar.extractfile(member))

    # Upload shard back to S3
    shard_buffer.seek(0)
    shard_key = f"shard_{shard_idx:06d}.tar"
    s3_client.upload_fileobj(shard_buffer, bucket_name, f"{s3_prefix.rstrip('/')}/shards/{shard_key}")
    return (shard_key.rstrip(".tar"), len(shard_files))


class EpisodeProcessor:
    """Episode processor with streaming."""

    def __init__(
        self,
        output_dir: str,
        past_lowdim_steps: int = 1,
        future_lowdim_steps: int = 14,
        image_indices: List[int] = None,
        max_padding_left: int = 1,
        max_padding_right: int = 7,
        padding_strategy: str = "copy",
        filter_still_samples: bool = False,
        still_threshold: float = 0.01,
        stride: int = 1,
        jpeg_quality: int = 95,
        fail_on_nan: bool = True,
        camera_names: Optional[List[str]] = None,
        camera_discard_keys: Optional[List[str]] = None,
        compute_statistics: bool = True,
        resize_images_size: List[int] = None,  # From LBM1
        language_annotations: Optional[Dict] = None,
    ):
        if resize_images_size is None:
            resize_images_size = [224, 224]
        if image_indices is None:
            image_indices = [-1, 0]
        self.output_dir = output_dir
        self.past_lowdim_steps = past_lowdim_steps
        self.future_lowdim_steps = future_lowdim_steps
        self.image_indices = sorted(image_indices)
        self.max_padding_left = max_padding_left
        self.max_padding_right = max_padding_right
        self.stride = stride
        self.jpeg_quality = jpeg_quality
        self.fail_on_nan = fail_on_nan
        self.camera_names = camera_names
        self.filter_still_samples = filter_still_samples
        self.still_threshold = still_threshold
        self.camera_discard_keys = camera_discard_keys or []
        self.compute_statistics = compute_statistics
        self.resize_images_size = resize_images_size
        self.language_annotations = language_annotations or {}

        # Statistics tracking
        self.total_potential_samples = 0
        self.still_samples_filtered = 0
        self.padding_samples_filtered = 0

        # Initialize JPEG encoder
        init_jpeg_encoder(jpeg_quality)

        # Create S3 client once for reuse
        self.s3_client = boto3.client("s3")

        # Set padding function
        self.pad_functions = {
            "copy": PaddingStrategy.copy_edge,
            "zero": PaddingStrategy.zero_pad,
            "reflect": PaddingStrategy.reflect_pad,
        }
        self.pad_fn = self.pad_functions[padding_strategy]

    def load_episode_data(self, episode_path: str) -> Dict[str, Any]:
        """Load episode data with optimization and retry logic."""
        processed_path = os.path.join(episode_path, "processed")

        # Load metadata with retry
        metadata_path = os.path.join(processed_path, "metadata.yaml")
        for attempt in range(3):
            try:
                with fsspec.open(metadata_path, "r") as f:
                    metadata = yaml.safe_load(f)
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                print(f"Retry {attempt + 1} loading metadata for {episode_path}")
                time.sleep(1)

        # Load observations with retry
        obs_path = os.path.join(processed_path, "observations.npz")
        for attempt in range(3):
            try:
                with fsspec.open(obs_path, "rb") as f:
                    observations = np.load(f, allow_pickle=True)
                    observations = {k: v for k, v in observations.items() if k not in self.camera_discard_keys}
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                print(f"Retry {attempt + 1} loading observations for {episode_path}")
                time.sleep(1)

        # Load actions with retry
        actions = {}
        actions_path = os.path.join(processed_path, "actions.npz")
        for attempt in range(3):
            try:
                with fsspec.open(actions_path, "rb") as f:
                    actions_archive = np.load(f, allow_pickle=True)
                    # Extract the 'actions' key specifically
                    if "actions" in actions_archive:
                        actions = {"actions": actions_archive["actions"]}
                    else:
                        print(f"Warning: 'actions' key not found in {actions_path}")
                        print(f"Available keys: {list(actions_archive.keys())}")
                        actions = {}
                break
            except Exception:
                if attempt == 2:
                    # If actions.npz doesn't exist, continue with empty actions
                    print(f"Warning: No actions.npz found for {episode_path}, continuing with empty actions")
                    break
                print(f"Retry {attempt + 1} loading actions for {episode_path}")
                time.sleep(1)

        # Load camera params (optional)
        intrinsics, extrinsics = {}, {}
        try:
            intrinsics_path = os.path.join(processed_path, "intrinsics.npz")
            extrinsics_path = os.path.join(processed_path, "extrinsics.npz")
            with fsspec.open(intrinsics_path, "rb") as f:
                intrinsics_archive = np.load(f)
                intrinsics = {key: intrinsics_archive[key] for key in intrinsics_archive.files}
            with fsspec.open(extrinsics_path, "rb") as f:
                extrinsics_archive = np.load(f)
                extrinsics = {key: extrinsics_archive[key] for key in extrinsics_archive.files}
        except Exception:
            pass

        # Transform camera calibration keys from camera IDs to semantic names
        intrinsics, extrinsics = self.transform_camera_calibration_keys(intrinsics, extrinsics, metadata)

        return {
            "metadata": metadata,
            "observations": observations,
            "actions": actions,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }

    def extract_camera_data(
        self, observations: Dict[str, np.ndarray], metadata: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Extract camera data with filtering."""
        camera_mapping = metadata.get("camera_id_to_semantic_name", {})

        if self.camera_names:
            filtered_mapping = {
                cid: sname
                for cid, sname in camera_mapping.items()
                if sname in self.camera_names and cid in observations
            }
        else:
            filtered_mapping = {cid: sname for cid, sname in camera_mapping.items() if cid in observations}

        return {sname: observations[cid] for cid, sname in filtered_mapping.items()}

    def extract_lowdim_data(
        self, observations: Dict[str, np.ndarray], actions: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Extract low-dimensional observation data and handle 'actions' key if present."""
        result = {}

        # Extract low-dimensional observations
        if observations:
            result.update(
                {
                    key: value
                    for key, value in observations.items()
                    if len(value.shape) <= 2 or key.startswith(("robot__", "language_"))
                }
            )

        # Extract 'actions' if available
        if actions and "actions" in actions:
            result["actions"] = actions["actions"]

        return result

    def is_still_sample(self, lowdim_data: Dict[str, np.ndarray], start_idx: int, end_idx: int) -> bool:
        """Check if sample is still."""
        if not self.filter_still_samples:
            return False

        movement_keys = [k for k in lowdim_data if any(x in k.lower() for x in ["joint_position", "poses", "xyz"])]

        # If no movement keys found, don't filter the sample
        if not movement_keys:
            # Debug: Print available keys once to help troubleshoot
            if not hasattr(self, "_debug_keys_printed"):
                print(f"🔍 Debug: No movement keys found. Available keys: {list(lowdim_data.keys())[:10]}")
                self._debug_keys_printed = True
            return False

        for key in movement_keys:
            data = lowdim_data[key][start_idx : end_idx + 1]
            if len(data) > 1:
                movement = np.std(data, axis=0).max()
                if movement > self.still_threshold:
                    return False

        return True

    def transform_camera_calibration_keys(
        self, intrinsics: Dict[str, Any], extrinsics: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Transform camera calibration keys from camera IDs to semantic names."""
        # Get camera mapping from metadata
        camera_mapping = metadata.get("camera_id_to_semantic_name", {})
        if not camera_mapping:
            # If no mapping available, return original data
            return intrinsics, extrinsics

        # Transform intrinsics keys
        transformed_intrinsics = {}
        for camera_id, semantic_name in camera_mapping.items():
            if camera_id in intrinsics:
                transformed_intrinsics[semantic_name] = intrinsics[camera_id]

        # Transform extrinsics keys
        transformed_extrinsics = {}
        for camera_id, semantic_name in camera_mapping.items():
            if camera_id in extrinsics:
                transformed_extrinsics[semantic_name] = extrinsics[camera_id]

        return transformed_intrinsics, transformed_extrinsics

    def extract_sample_camera_calibration(
        self,
        episode_intrinsics: Dict[str, Any],
        episode_extrinsics: Dict[str, Any],
        valid_start: int,
        valid_end: int,
        past_padding: int,
        future_padding: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract camera calibration data for the lowdim sequence timespan."""
        sample_intrinsics = {}
        sample_extrinsics = {}

        # Extract intrinsics for the sequence timespan
        for camera_name, intrinsics_data in episode_intrinsics.items():
            if isinstance(intrinsics_data, np.ndarray):
                if intrinsics_data.ndim == 3:  # Time-varying intrinsics (timesteps, 3, 3)
                    # Extract the sequence from valid_start to valid_end (inclusive)
                    valid_intrinsics = intrinsics_data[valid_start : valid_end + 1]

                    # Apply padding to match lowdim data
                    if past_padding > 0 or future_padding > 0:
                        # Use edge padding for camera calibration
                        pad_width = [(past_padding, future_padding)] + [(0, 0)] * (valid_intrinsics.ndim - 1)
                        valid_intrinsics = np.pad(valid_intrinsics, pad_width, mode="edge")

                    sample_intrinsics[camera_name] = valid_intrinsics
                elif intrinsics_data.ndim == 2:  # Static intrinsics (3, 3)
                    # For static calibration, repeat for the sequence length
                    sequence_length = (valid_end - valid_start + 1) + past_padding + future_padding
                    sample_intrinsics[camera_name] = np.tile(intrinsics_data[np.newaxis, :, :], (sequence_length, 1, 1))

        # Extract extrinsics for the sequence timespan
        for camera_name, extrinsics_data in episode_extrinsics.items():
            if isinstance(extrinsics_data, np.ndarray):
                if extrinsics_data.ndim == 3:  # Time-varying extrinsics (timesteps, 4, 4)
                    # Extract the sequence from valid_start to valid_end (inclusive)
                    valid_extrinsics = extrinsics_data[valid_start : valid_end + 1]

                    # Apply padding to match lowdim data
                    if past_padding > 0 or future_padding > 0:
                        # Use edge padding for camera calibration
                        pad_width = [(past_padding, future_padding)] + [(0, 0)] * (valid_extrinsics.ndim - 1)
                        valid_extrinsics = np.pad(valid_extrinsics, pad_width, mode="edge")

                    sample_extrinsics[camera_name] = valid_extrinsics
                elif extrinsics_data.ndim == 2:  # Static extrinsics (4, 4)
                    # For static calibration, repeat for the sequence length
                    sequence_length = (valid_end - valid_start + 1) + past_padding + future_padding
                    sample_extrinsics[camera_name] = np.tile(extrinsics_data[np.newaxis, :, :], (sequence_length, 1, 1))

        return sample_intrinsics, sample_extrinsics

    def process_episode(
        self, episode_path: str, statistics_ray_actor: StreamingDatasetStatisticsRayActor
    ) -> Iterator[Dict[str, Any]]:
        """Process episode with streaming output."""
        try:
            episode_data = self.load_episode_data(episode_path)
            episode_id = os.path.basename(episode_path.rstrip("/"))

            # Extract task name and get language instructions
            task_name = extract_task_name_from_path(episode_path)
            language_instructions = get_language_instructions(task_name, self.language_annotations)

            observations = episode_data["observations"]
            first_obs_key = next(iter(observations.keys()))
            episode_length = observations[first_obs_key].shape[0]

            # Pre-extract data
            camera_data = self.extract_camera_data(observations, episode_data["metadata"])
            lowdim_data = self.extract_lowdim_data(observations, episode_data["actions"])

            print(f"Processing episode {episode_id} with length {episode_length}")
            # Generate samples
            for anchor_timestep in range(0, episode_length, self.stride):
                self.total_potential_samples += 1

                # Calculate windows
                lowdim_start = anchor_timestep - self.past_lowdim_steps
                lowdim_end = anchor_timestep + self.future_lowdim_steps

                # Check padding
                past_padding = max(0, -lowdim_start)
                future_padding = max(0, lowdim_end - episode_length + 1)

                if past_padding > self.max_padding_left or future_padding > self.max_padding_right:
                    self.padding_samples_filtered += 1
                    continue

                valid_start = max(0, lowdim_start)
                valid_end = min(episode_length - 1, lowdim_end)

                # Check if robot is stationary (e.g. to filter pauses)
                if self.is_still_sample(lowdim_data, valid_start, valid_end):
                    self.still_samples_filtered += 1
                    continue

                # Extract images
                sample_images = {}
                actual_image_timesteps = []

                for img_offset in self.image_indices:
                    img_timestep = np.clip(anchor_timestep + img_offset, 0, episode_length - 1)
                    actual_image_timesteps.append(int(img_timestep))

                    for camera_name, camera_images in camera_data.items():
                        key = f"{camera_name}_t{img_offset}"
                        sample_images[key] = camera_images[img_timestep]

                # Process lowdim data (which includes actions)
                sample_lowdim = {}
                for key, data in lowdim_data.items():
                    valid_data = data[valid_start : valid_end + 1]
                    if past_padding > 0 or future_padding > 0:
                        valid_data = self.pad_fn(valid_data, past_padding, future_padding)
                    sample_lowdim[key] = valid_data

                # Create masks
                total_length = self.past_lowdim_steps + self.future_lowdim_steps + 1
                past_mask = np.ones(total_length, dtype=bool)
                future_mask = np.ones(total_length, dtype=bool)
                # Current time step is part of the future for low dim data
                past_mask[self.past_lowdim_steps :] = False
                future_mask[: self.past_lowdim_steps] = False

                if past_padding > 0:
                    past_mask[:past_padding] = False
                    future_mask[:past_padding] = False
                if future_padding > 0:
                    future_mask[-future_padding:] = False
                    past_mask[-future_padding:] = False

                # Extract sequence-specific camera calibration
                sample_intrinsics, sample_extrinsics = self.extract_sample_camera_calibration(
                    episode_data.get("intrinsics", {}),
                    episode_data.get("extrinsics", {}),
                    valid_start,
                    valid_end,
                    past_padding,
                    future_padding,
                )

                # Create metadata
                sample_metadata = SampleMetadata(
                    episode_id=episode_id,
                    sample_id=f"{uuid.uuid4()}_{episode_id}_t{anchor_timestep:04d}",
                    anchor_timestep=None,
                    anchor_relative_idx=None,
                    image_timesteps=actual_image_timesteps,
                    lowdim_start_timestep=int(lowdim_start),
                    lowdim_end_timestep=int(lowdim_end),
                    past_padding=int(past_padding),
                    future_padding=int(future_padding),
                    camera_names=list(camera_data.keys()),
                    original_episode_length=int(episode_length),
                    original_image_sizes={},
                    is_padded=bool(past_padding > 0 or future_padding > 0),
                )

                # Upload to S3 instead of yielding
                sample_data = {
                    "images": sample_images,
                    "lowdim": sample_lowdim,
                    "past_mask": past_mask,
                    "future_mask": future_mask,
                    "metadata": sample_metadata,
                    "intrinsics": sample_intrinsics,
                    "extrinsics": sample_extrinsics,
                    "language_instructions": language_instructions,
                }
                statistics_ray_actor.merge_from_samples.remote([sample_data])
                upload_sample_to_s3(
                    sample_data,
                    self.output_dir,
                    episode_id,
                    anchor_timestep,
                    self.jpeg_quality,
                    self.resize_images_size,
                    self.s3_client,
                )
            return True

        except Exception as e:
            if self.fail_on_nan:
                raise e
            print(f"Warning: Failed to process episode {episode_path}: {e}")
            return False


@ray.remote
def streaming_episode_worker(
    episode_path: str, processor_config: Dict[str, Any], statistics_ray_actor: StreamingDatasetStatisticsRayActor
) -> Tuple[str, int, int, int]:
    processor = EpisodeProcessor(**processor_config)
    return processor.process_episode(episode_path, statistics_ray_actor)


def main():
    """Optimized robotics preprocessing using draccus-configured params."""
    # Parse CLI into dataclass
    cfg = draccus.parse(config_class=PreprocessParams)

    # Validate required paths
    assert cfg.source_episodes is not None, "--source_episodes is required (or set in config_path)"
    assert cfg.output_dir is not None, "--output_dir is required (or set in config_path)"

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

    camera_names = cfg.camera_names

    # Load language annotations
    print("📚 Loading language annotations...")
    language_annotations = load_language_annotations(cfg.language_annotations_path)
    print(f"Loaded language annotations for {len(language_annotations)} tasks")

    processor_config = {
        "output_dir": cfg.output_dir,
        "past_lowdim_steps": cfg.past_lowdim_steps,
        "future_lowdim_steps": cfg.future_lowdim_steps,
        "image_indices": cfg.image_indices,
        "max_padding_left": cfg.max_padding_left,
        "max_padding_right": cfg.max_padding_right,
        "padding_strategy": cfg.padding_strategy,
        "filter_still_samples": cfg.filter_still_samples,
        "still_threshold": cfg.still_threshold,
        "stride": cfg.stride,
        "jpeg_quality": cfg.jpeg_quality,
        "fail_on_nan": cfg.fail_on_nan,
        "camera_names": camera_names,
        "camera_discard_keys": cfg.camera_discard_keys,
        "compute_statistics": not cfg.no_statistics,
        "resize_images_size": cfg.resize_images_size,
        "language_annotations": language_annotations,
    }

    print("🚀 Starting optimized preprocessing")
    print(f"Statistics: {'disabled' if cfg.no_statistics else 'enabled'}")
    if cfg.resize_images_size and len(cfg.resize_images_size) == 2:
        print(f"Image resize to {cfg.resize_images_size[0]}x{cfg.resize_images_size[1]}")
    else:
        print("Image resize disabled")

    # Create initial metadata with git info (before any code changes during processing)
    print("📋 Capturing initial metadata...")
    start_time = datetime.datetime.now()

    # Discover episodes
    print("🔍 Discovering episodes...")
    episodes = discover_episodes_targeted(cfg.source_episodes, cfg.max_episodes_to_process)
    print(f"Found {len(episodes)} episodes")

    # Create initial processing metadata with placeholder stats
    initial_processing_stats = {
        "total_potential_samples": 0,
        "samples_created": 0,
        "still_samples_filtered": 0,
        "padding_samples_filtered": 0,
        "episodes_processed": len(episodes),
    }

    # Create metadata with git info captured at start
    metadata = create_processing_metadata(cfg, episodes, 0, initial_processing_stats)
    metadata["processing"]["timestamp_start"] = start_time.isoformat()

    if len(episodes) == 0:
        print("❌ No episodes found!")
        return

    # Ray Phase 1: Process frame individually and upload to S3
    print(f"🚀 Processing {len(episodes)} episodes and uploading to S3...")
    statistics_ray_actor = StreamingDatasetStatisticsRayActor.remote(compute_stats=cfg.no_statistics)
    futures = [streaming_episode_worker.remote(episode, processor_config, statistics_ray_actor) for episode in episodes]
    ray.get(futures)
    print("✅ Upload phase complete! Starting sharding phase...")

    # Get and save statistics
    statistics_state = statistics_ray_actor.get_statistics.remote()
    statistics_state = ray.get(statistics_state)
    upload_dict_to_s3(statistics_state, f"{cfg.output_dir.rstrip('/')}/shards", "stats.json")

    # Ray Phase 2: List files and group them together to create shards
    all_files = list_directory(cfg.output_dir)
    all_files = [f for f in all_files if f.startswith("episode_")]
    print(f"Found {len(all_files)} files to shard")

    # Shuffle and create shard assignments
    random.shuffle(all_files)
    shards = [all_files[i : i + cfg.samples_per_shard] for i in range(0, len(all_files), cfg.samples_per_shard)]
    print(f"Creating {len(shards)} shards with up to {cfg.samples_per_shard} samples each")

    # Create shards in parallel
    shard_futures = [create_shard.remote(shard_files, i, cfg.output_dir) for i, shard_files in enumerate(shards)]
    shard_results = ray.get(shard_futures)

    print(f"✅ Created {len(shard_results)} shards.")

    # Upload manifest to S3 in the same directory as the tar files
    manifest_lines = []
    for shard_name, num_sequences in shard_results:
        manifest_entry = {"shard": shard_name, "num_sequences": num_sequences}
        manifest_lines.append(manifest_entry)
    upload_dict_to_s3(manifest_lines, f"{cfg.output_dir.rstrip('/')}/shards", "manifest.jsonl")

    ray.shutdown()
    print("🎉 Complete! All samples uploaded and sharded.")


if __name__ == "__main__":
    main()
