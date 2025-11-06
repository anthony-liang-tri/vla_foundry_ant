#!/usr/bin/env python3
"""
LBM to LeRobot Data Preprocessing Pipeline

Converts LBM robotics data format to LeRobot format while preserving all original data.
Dataset gets organized at task-level, and task name is appened to repo_id when pushing to HuggingFace Hub.

LeRobot Format Structure:
- data/chunk-000/episode_XXXXXX.parquet: Main episode data with observation.state, action, timestamps, etc.
- videos/chunk-000/{camera_name}/episode_XXXXXX.mp4: Video files for each camera
- meta/episodes.jsonl: Episode metadata with tasks and lengths
- meta/stats.jsonl: Statistics for each field
- meta/tasks.jsonl: Task definitions
- meta/info.json: Dataset metadata and feature definitions
- additional_data/depth/chunk-000/{camera_name}/episode_XXXXXX.npz: Optional depth data
- additional_data/label/chunk-000/{camera_name}/episode_XXXXXX.npz: Optional label data

Usage:
    # Create separate datasets for each task and push to HF
    uv run --group preprocessing python vla_foundry/data/preprocessing/preprocess_lbm_to_lerobot.py \
        --source_eps_csv_path examples/preprocessing/s3_episodes_list.csv \
        --output_dir lerobot/task_datasets/ \
        --dataset_name "lbm_eval" \
        --robot_type "lbm_bimanual_panda" \
        --fps 10 \
        --chunk_size 1000 \
        --num_workers 16 \
        --preserve_depth True \
        --preserve_segmentation True \
        --preserve_calibration True \
        --private True 

"""

import datetime
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import av
import datasets
import draccus
import fsspec
import numpy as np
import pandas as pd
import ray
import torch
import yaml
from huggingface_hub import HfApi
from PIL import Image
from tqdm import tqdm

from vla_foundry.data.preprocessing.robotics.converters.spartan import discover_episodes
from vla_foundry.params.base_params import BaseParams

HF_AVAILABLE = True


def encode_video_frames_from_array(
    frames: np.ndarray,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = None,
) -> None:
    """Encode video frames from numpy array using PyAV (adapted from LeRobot)."""

    # Check encoder availability
    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.")

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Encoders/pixel formats incompatibility check
    if (vcodec == "libsvtav1" or vcodec == "hevc") and pix_fmt == "yuv444p":
        logging.warning(f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'")
        pix_fmt = "yuv420p"

    # Check frames shape
    if len(frames.shape) != 4 or frames.shape[3] != 3:
        raise ValueError(f"Expected frames shape (N, H, W, 3), got {frames.shape}")

    num_frames, height, width = frames.shape[:3]
    print(f"Encoding video at {fps} FPS, {width}x{height}, codec={vcodec}, pix_fmt={pix_fmt}")

    if num_frames == 0:
        raise ValueError("No frames to encode")

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options["g"] = str(g)

    if crf is not None:
        video_options["crf"] = str(crf)

    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    # Set logging level
    if log_level is not None:
        logging.getLogger("libav").setLevel(log_level)

    # Create and open output file
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through frames and encode them
        for frame_idx in range(num_frames):
            frame_data = frames[frame_idx]

            # Ensure frame is uint8
            if frame_data.dtype != np.uint8:
                if frame_data.max() <= 1.0:
                    frame_data = (frame_data * 255).astype(np.uint8)
                else:
                    frame_data = frame_data.astype(np.uint8)

            # Convert to PIL Image and then to av.VideoFrame
            pil_image = Image.fromarray(frame_data)
            input_frame = av.VideoFrame.from_image(pil_image)

            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


def concatenate_episodes(ep_dicts):
    """Concatenate episode dictionaries into a single dataset dictionary."""
    data_dict = {}

    keys = ep_dicts[0].keys()
    for key in keys:
        if torch.is_tensor(ep_dicts[0][key][0]):
            data_dict[key] = torch.cat([ep_dict[key] for ep_dict in ep_dicts])
        else:
            if key not in data_dict:
                data_dict[key] = []
            for ep_dict in ep_dicts:
                for x in ep_dict[key]:
                    data_dict[key].append(x)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def calculate_episode_data_index(hf_dataset) -> dict:
    """Calculate episode data index for the provided HuggingFace Dataset."""
    episode_data_index = {"from": [], "to": []}

    current_episode = None
    if len(hf_dataset) == 0:
        episode_data_index = {
            "from": torch.tensor([]),
            "to": torch.tensor([]),
        }
        return episode_data_index

    for idx, episode_idx in enumerate(hf_dataset["episode_index"]):
        if episode_idx != current_episode:
            episode_data_index["from"].append(idx)
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            current_episode = episode_idx

    episode_data_index["to"].append(idx + 1)

    for k in ["from", "to"]:
        episode_data_index[k] = torch.tensor(episode_data_index[k])

    return episode_data_index


def check_repo_id(repo_id: str) -> None:
    """Check if repo_id has correct format."""
    if len(repo_id.split("/")) != 2:
        raise ValueError(
            f"""`repo_id` is expected to contain a community or user id `/` the name of the dataset
            (e.g. 'lerobot/pusht'), but contains '{repo_id}'."""
        )


@dataclass(frozen=True)
class LeRobotPreprocessParams(BaseParams):
    """Parameters for LeRobot format preprocessing."""

    # Core I/O
    source_episodes: Optional[List[str]] = field(default=None)
    source_eps_csv_path: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default=None)

    # Dataset metadata
    dataset_name: str = field(default="lbm_dataset")
    robot_type: str = field(default="lbm_bimanual_panda")
    fps: int = field(default=10)

    # Processing options
    chunk_size: int = field(default=1000)  # Episodes per chunk
    num_workers: int = field(default=16)
    max_episodes_to_process: int = field(default=-1)

    # Ray configuration
    ray_address: Optional[str] = field(default=None)  # Ray cluster address
    ray_num_cpus: Optional[int] = field(default=None)  # Number of CPUs for Ray

    # Video encoding
    video_codec: str = field(default="mp4v")
    video_quality: int = field(default=95)
    skip_videos: bool = field(default=False)  # Skip video creation if codecs fail
    sync_cameras: str = field(default="truncate")  # Camera sync method: "truncate", "pad", or "none"

    # Camera selection
    camera_names: Optional[List[str]] = field(default=None)

    # Additional data preservation
    preserve_depth: bool = field(default=True)
    preserve_segmentation: bool = field(default=True)
    preserve_calibration: bool = field(default=True)
    preserve_relative_coords: bool = field(default=True)

    # Dataset pushing
    push_to_hub: bool = field(default=False)
    repo_id: Optional[str] = field(default=None)
    token: Optional[str] = field(default=None)
    private: bool = field(default=True)


@ray.remote
class LeRobotDatasetWriter:
    """Writes data in LeRobot format."""

    def __init__(
        self,
        output_dir: str,
        dataset_name: str,
        robot_type: str,
        fps: int = 30,
        video_codec: str = "libsvtav1",
        preserve_depth: bool = True,
        preserve_segmentation: bool = True,
        preserve_calibration: bool = True,
        skip_videos: bool = False,
        chunk_size: int = 1000,
        sync_cameras: str = "truncate",
    ):
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.robot_type = robot_type
        self.fps = fps
        self.video_codec = video_codec
        self.skip_videos = skip_videos
        self.chunk_size = chunk_size
        self.sync_cameras = sync_cameras
        self.preserve_depth = preserve_depth
        self.preserve_segmentation = preserve_segmentation
        self.preserve_calibration = preserve_calibration

        # Create directory structure
        self.data_dir = self.output_dir / "data"
        self.videos_dir = self.output_dir / "videos"
        self.meta_dir = self.output_dir / "meta"

        for dir_path in [self.data_dir, self.videos_dir, self.meta_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Track dataset statistics
        self.episodes_metadata = []
        self.field_stats = {}
        self.tasks = {}
        self.features = {}
        self.total_frames = 0

        # Camera setup
        self.camera_keys = set()
        self.video_dimensions = {}  # Track actual video dimensions for each camera

    def create_chunk_dirs(self, chunk_index: int = 0):
        """Create chunk directories."""
        chunk_name = f"chunk-{chunk_index:03d}"
        data_chunk_dir = self.data_dir / chunk_name
        data_chunk_dir.mkdir(exist_ok=True)

        # Create video chunk dirs for each camera (will be created as needed)
        return data_chunk_dir, chunk_name

    def process_episode(self, episode_path: str, episode_index: int, chunk_index: int = 0) -> Dict[str, Any]:
        """Process a single episode and convert to LeRobot format."""
        try:
            # Load episode data
            episode_data = self.load_lbm_episode(episode_path)
            os.path.basename(episode_path.rstrip("/"))

            # Extract basic info
            observations = episode_data["observations"]
            metadata = episode_data["metadata"]
            actions = episode_data.get("actions", {})

            # Get episode length from first observation
            first_obs_key = next(iter(observations.keys()))
            episode_length = observations[first_obs_key].shape[0]

            # Extract camera data and create videos
            camera_data = self.extract_camera_data(observations, metadata)
            self.camera_keys.update(camera_data.keys())

            # Create videos for each camera
            chunk_name = f"chunk-{chunk_index:03d}"
            video_paths = {}
            for camera_name, images in camera_data.items():
                video_path = self.create_video(images, camera_name, episode_index, chunk_name)
                if video_path is not None:  # Only add if video was successfully created
                    video_paths[camera_name] = video_path

            # Optionally preserve depth and segmentation data as separate files
            additional_data_paths = {}
            if hasattr(self, "preserve_depth") and self.preserve_depth:
                additional_data_paths.update(
                    self.save_additional_camera_data(observations, metadata, episode_index, chunk_name, "_depth")
                )

            if hasattr(self, "preserve_segmentation") and self.preserve_segmentation:
                additional_data_paths.update(
                    self.save_additional_camera_data(observations, metadata, episode_index, chunk_name, "_label")
                )

            # Extract low-dimensional observation data (robot state)
            observation_state = self.extract_observation_state(observations)

            # Extract actions
            action_data = self.extract_actions(observations, actions)

            # Create timestamps
            timestamps = self.create_timestamps(episode_length)

            # Extract task information
            task_info = self.extract_task_info(observations, episode_path)
            task_index = self.register_task(task_info)

            # Create DataFrame for this episode
            episode_df = self.create_episode_dataframe(
                observation_state=observation_state,
                action_data=action_data,
                timestamps=timestamps,
                episode_index=episode_index,
                task_index=task_index,
                episode_length=episode_length,
            )

            # Update statistics
            self.update_statistics(observation_state, action_data)

            # Store episode metadata
            episode_metadata = {
                "episode_index": episode_index,
                "tasks": [task_info["task"]],
                "length": episode_length,
                "video_paths": video_paths,
                "additional_data_paths": additional_data_paths,
                "source_path": episode_path,
                "created_at": datetime.datetime.now().isoformat(),
            }

            # Add any additional LBM-specific metadata
            if "language_instruction" in observations:
                episode_metadata["language_instruction"] = (
                    observations["language_instruction"][0] if len(observations["language_instruction"]) > 0 else ""
                )

            # Add success information if available
            if "summary" in episode_data:
                summary = episode_data["summary"]
                if "episode_success" in summary:
                    episode_metadata["success"] = bool(summary["episode_success"])
                if "episode_message" in summary:
                    episode_metadata["message"] = str(summary["episode_message"])
                if "reward" in summary:
                    episode_metadata["reward_trajectory"] = (
                        summary["reward"].tolist() if hasattr(summary["reward"], "tolist") else summary["reward"]
                    )

            # Preserve camera calibration information
            if episode_data.get("intrinsics"):
                episode_metadata["camera_intrinsics"] = {
                    k: v.tolist() if hasattr(v, "tolist") else v for k, v in episode_data["intrinsics"].items()
                }

            if episode_data.get("extrinsics"):
                # For extrinsics, we might want to store just a representative sample since it's time-varying
                episode_metadata["camera_extrinsics_sample"] = {}
                for k, v in episode_data["extrinsics"].items():
                    if hasattr(v, "shape") and len(v.shape) > 2:
                        # Store first and last frames for time-varying extrinsics
                        episode_metadata["camera_extrinsics_sample"][k] = {
                            "first_frame": v[0].tolist(),
                            "last_frame": v[-1].tolist(),
                            "shape": list(v.shape),
                        }
                    else:
                        episode_metadata["camera_extrinsics_sample"][k] = v.tolist() if hasattr(v, "tolist") else v

            # Store depth and segmentation availability
            depth_cameras = []
            segmentation_cameras = []
            for key in observations:
                if key.endswith("_depth"):
                    depth_cameras.append(key.replace("_depth", ""))
                elif key.endswith("_label"):
                    segmentation_cameras.append(key.replace("_label", ""))

            if depth_cameras:
                episode_metadata["depth_cameras"] = depth_cameras
            if segmentation_cameras:
                episode_metadata["segmentation_cameras"] = segmentation_cameras

            self.episodes_metadata.append(episode_metadata)
            self.total_frames += episode_length

            return {"episode_df": episode_df, "episode_metadata": episode_metadata, "success": True}

        except Exception as e:
            print(f"Error processing episode {episode_path}: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def load_lbm_episode(self, episode_path: str) -> Dict[str, Any]:
        """Load LBM episode data."""
        processed_path = os.path.join(episode_path, "processed")

        # Load metadata
        metadata_path = os.path.join(processed_path, "metadata.yaml")
        with fsspec.open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

        # Load observations
        obs_path = os.path.join(processed_path, "observations.npz")
        with fsspec.open(obs_path, "rb") as f:
            observations = dict(np.load(f, allow_pickle=True))

        # Load actions (optional)
        actions = {}
        try:
            actions_path = os.path.join(processed_path, "actions.npz")
            with fsspec.open(actions_path, "rb") as f:
                actions_archive = np.load(f, allow_pickle=True)
                # Extract the 'actions' key specifically
                if "actions" in actions_archive:
                    actions = {"actions": actions_archive["actions"]}
                    print(f"Loaded actions with shape: {actions['actions'].shape}")
                else:
                    print(f"Warning: 'actions' key not found in {actions_path}")
                    print(f"Available keys: {list(actions_archive.keys())}")
                    actions = {}
        except Exception as e:
            print(f"Warning: Could not load actions from {actions_path}: {e}")
            pass

        # Load summary (optional)
        summary = {}
        try:
            summary_path = os.path.join(processed_path, "summary.npz")
            with fsspec.open(summary_path, "rb") as f:
                summary = dict(np.load(f, allow_pickle=True))
        except Exception:
            pass

        # Load camera calibration (optional)
        intrinsics, extrinsics = {}, {}
        try:
            intrinsics_path = os.path.join(processed_path, "intrinsics.npz")
            with fsspec.open(intrinsics_path, "rb") as f:
                intrinsics = dict(np.load(f))
        except Exception:
            pass

        try:
            extrinsics_path = os.path.join(processed_path, "extrinsics.npz")
            with fsspec.open(extrinsics_path, "rb") as f:
                extrinsics = dict(np.load(f))
        except Exception:
            pass

        return {
            "metadata": metadata,
            "observations": observations,
            "actions": actions,
            "summary": summary,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }

    def extract_camera_data(
        self,
        observations: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Extract camera images with semantic names and synchronization diagnostics."""
        camera_mapping = metadata.get("camera_id_to_semantic_name", {})
        camera_data = {}
        frame_counts = {}

        for camera_id, semantic_name in camera_mapping.items():
            if camera_id in observations and not camera_id.endswith(("_depth", "_label")):
                camera_data[semantic_name] = observations[camera_id]
                frame_counts[semantic_name] = observations[camera_id].shape[0]

        # Check for frame count mismatches (synchronization issues)
        if len(set(frame_counts.values())) > 1:
            print("⚠️  Camera synchronization issue detected:")
            for camera, count in frame_counts.items():
                print(f"   {camera}: {count} frames")

            if self.sync_cameras == "none":
                print("   Keeping original frame counts (no synchronization)")
                return camera_data
            elif self.sync_cameras == "truncate":
                # Truncate to minimum frame count
                min_frames = min(frame_counts.values())
                print(f"   Synchronizing to {min_frames} frames (truncate method)")

                synchronized_data = {}
                for camera_name, frames in camera_data.items():
                    synchronized_data[camera_name] = frames[:min_frames]

                return synchronized_data
            elif self.sync_cameras == "pad":
                # Pad to maximum frame count
                max_frames = max(frame_counts.values())
                print(f"   Synchronizing to {max_frames} frames (pad method)")

                synchronized_data = {}
                for camera_name, frames in camera_data.items():
                    if frames.shape[0] < max_frames:
                        # Pad with the last frame
                        padding_needed = max_frames - frames.shape[0]
                        last_frame = frames[-1:] if len(frames) > 0 else np.zeros_like(frames[:1])
                        padded_frames = np.concatenate([frames] + [last_frame] * padding_needed, axis=0)
                        synchronized_data[camera_name] = padded_frames
                    else:
                        synchronized_data[camera_name] = frames

                return synchronized_data

        return camera_data

    def extract_observation_state(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract robot state observations."""
        state_data = {}

        # Define what constitutes robot state (low-dimensional observations)
        state_keys = [
            # Joint positions
            "robot__actual__joint_position__left::panda",
            "robot__actual__joint_position__right::panda",
            # End effector poses
            "robot__actual__poses__left::panda__xyz",
            "robot__actual__poses__right::panda__xyz",
            "robot__actual__poses__left::panda__rot_6d",
            "robot__actual__poses__right::panda__rot_6d",
            # Gripper states
            "robot__actual__grippers__left::panda_hand",
            "robot__actual__grippers__right::panda_hand",
            # Joint velocities
            "robot__actual__joint_velocity__left::panda",
            "robot__actual__joint_velocity__right::panda",
            # External forces/torques
            "robot__actual__external_wrench__left::panda",
            "robot__actual__external_wrench__right::panda",
        ]

        for key in state_keys:
            if key in observations:
                state_data[key] = observations[key]

        # Also include any other low-dimensional data
        for key, value in observations.items():
            if (
                len(value.shape) <= 2
                and key not in state_data
                and not any(suffix in key for suffix in ["_depth", "_label"])
                and key not in [k for k in observations if len(observations[k].shape) > 2]
            ):
                state_data[key] = value

        return state_data

    def extract_actions(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Extract action data."""
        action_data = {}

        # Check if we have the 'actions' key from actions.npz
        if "actions" in actions:
            # Use the actions directly from the actions.npz file
            action_data["action"] = actions["actions"]
            print(f"Using actions from actions.npz with shape: {actions['actions'].shape}")
            print(f"Action data type: {actions['actions'].dtype}")
            print(f"Action range: [{actions['actions'].min():.3f}, {actions['actions'].max():.3f}]")
            return action_data
        else:
            print("No 'actions' key found in actions.npz, returning empty action data.")

        return action_data

    def create_timestamps(self, episode_length: int) -> np.ndarray:
        """Create timestamps for the episode."""
        if episode_length <= 0:
            raise ValueError(f"Episode length must be positive, got {episode_length}")

        # Create timestamps at the specified FPS
        timestamps = np.arange(episode_length, dtype=np.float32) / self.fps

        if len(timestamps) == 0:
            raise ValueError("Generated timestamps array is empty")

        return timestamps

    def extract_task_info(
        self,
        observations: Dict[str, np.ndarray],
        episode_path: str,
    ) -> Dict[str, str]:
        """Extract task information."""
        # Try to get language instruction
        task_description = "unknown_task"

        if "language_instruction" in observations:
            instructions = observations["language_instruction"]
            if len(instructions) > 0:
                task_description = str(instructions[0])

        else:
            print(
                f"Warning: No language instruction found in episode {episode_path}. Using task description from path."
            )
        # Extract task from path if possible
        task_from_path = extract_task_from_path(episode_path)
        return {
            "task": task_description if task_description != "unknown_task" else task_from_path,
            "task_from_path": task_from_path,
        }

    def register_task(self, task_info: Dict[str, str]) -> int:
        """Register a task and return its index."""
        task = task_info["task"]
        if task not in self.tasks:
            task_index = len(self.tasks)
            self.tasks[task] = {"task_index": task_index, "task": task}
            return task_index
        else:
            return self.tasks[task]["task_index"]

    def create_video(self, images: np.ndarray, camera_name: str, episode_index: int, chunk_name: str) -> str:
        """Create MP4 video from image sequence."""

        # Create camera video directory following LeRobot format: videos/chunk-{episode_chunk:03d}/{video_key}/
        camera_video_dir = self.videos_dir / chunk_name / f"observation.images.{camera_name}"
        camera_video_dir.mkdir(parents=True, exist_ok=True)

        # Use .mp4 extension regardless of codec for compatibility
        video_path = camera_video_dir / f"episode_{episode_index:06d}.mp4"

        # Get video properties
        height, width = images.shape[1:3]

        # Store video dimensions for this camera
        self.video_dimensions[camera_name] = (height, width)
        if not hasattr(self, "video_codecs"):
            self.video_codecs = {}

        # Use PyAV with LeRobot-style encoding
        encode_video_frames_from_array(
            frames=images,
            video_path=video_path,
            fps=self.fps,
            vcodec=self.video_codec if self.video_codec in ["h264", "hevc", "libsvtav1"] else "libsvtav1",
            crf=23,  # Good quality default
            log_level=av.logging.ERROR,
        )
        print(f"Created video using PyAV for {camera_name}")
        # Return relative path for consistency with LeRobot format
        return str(video_path.relative_to(self.output_dir))

    def save_additional_camera_data(
        self,
        observations: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
        episode_index: int,
        chunk_name: str,
        suffix: str,
    ) -> Dict[str, str]:
        """Save additional camera data (depth, segmentation) as NPZ files."""
        camera_mapping = metadata.get("camera_id_to_semantic_name", {})
        saved_paths = {}

        # Create additional data directory
        additional_dir = self.output_dir / "additional_data" / f"{suffix.strip('_')}" / chunk_name
        additional_dir.mkdir(parents=True, exist_ok=True)

        for camera_id, semantic_name in camera_mapping.items():
            data_key = f"{camera_id}{suffix}"
            if data_key in observations:
                inner_additional_dir = additional_dir / f"observation.images.{semantic_name}"
                inner_additional_dir.mkdir(parents=True, exist_ok=True)
                data_path = inner_additional_dir / f"episode_{episode_index:06d}.npz"
                np.savez_compressed(data_path, data=observations[data_key])
                saved_paths[f"{semantic_name}{suffix}"] = str(data_path.relative_to(self.output_dir))

        return saved_paths

    def create_episode_dataframe(
        self,
        observation_state: Dict[str, np.ndarray],
        action_data: Dict[str, np.ndarray],
        timestamps: np.ndarray,
        episode_index: int,
        task_index: int,
        episode_length: int,
    ) -> pd.DataFrame:
        """Create DataFrame for episode in LeRobot format."""

        # Combine observation state data into arrays
        if observation_state:
            # Stack all observation state values into a single array
            obs_state_arrays = []
            obs_state_names = []
            for key, values in observation_state.items():
                if values.ndim == 1:
                    obs_state_arrays.append(values.reshape(-1, 1))
                    obs_state_names.append(key)
                else:
                    obs_state_arrays.append(values)
                    for i in range(values.shape[1]):
                        obs_state_names.append(f"{key}.{i}")

            if obs_state_arrays:
                obs_state_combined = np.concatenate(obs_state_arrays, axis=1)
            else:
                obs_state_combined = np.array([]).reshape(episode_length, 0)
        else:
            obs_state_combined = np.array([]).reshape(episode_length, 0)
            obs_state_names = []

        print(f"Observation state combined shape: {obs_state_combined.shape}")
        # Combine action data into arrays
        if action_data:
            # Stack all action values into a single array
            action_arrays = []
            action_names = []
            for key, values in action_data.items():
                if values.ndim == 1:
                    action_arrays.append(values.reshape(-1, 1))
                    action_names.append(key)
                else:
                    action_arrays.append(values)
                    for i in range(values.shape[1]):
                        action_names.append(f"{key}.{i}")

            if action_arrays:
                action_combined = np.concatenate(action_arrays, axis=1)
            else:
                action_combined = np.array([]).reshape(episode_length, 0)
        else:
            action_combined = np.array([]).reshape(episode_length, 0)
            action_names = []

        print(f"Action combined shape: {action_combined.shape}")
        # Create base DataFrame
        df_data = {
            "timestamp": timestamps,
            "frame_index": np.arange(episode_length),
            "episode_index": np.full(episode_length, episode_index),
            "index": np.arange(episode_length),  # Global index will be set later
            "task_index": np.full(episode_length, task_index),
        }

        # Add observation state array (if any)
        if obs_state_combined.shape[1] > 0:
            df_data["observation.state"] = [list(row) for row in obs_state_combined]
            # Store names for features dict (only do this once)
            if not hasattr(self, "_obs_state_names"):
                self._obs_state_names = obs_state_names

        # Add action array (if any)
        if action_combined.shape[1] > 0:
            df_data["action"] = [list(row) for row in action_combined]
            # Store names for features dict (only do this once)
            if not hasattr(self, "_action_names"):
                self._action_names = action_names

        # Create DataFrame
        df = pd.DataFrame(df_data)

        # Debug: Print column information
        print(f"Created episode dataframe with {len(df)} rows and columns: {list(df.columns)}")
        print(f"Timestamp column type: {df['timestamp'].dtype}, shape: {df['timestamp'].shape}")

        # Debug: Check what's actually in observation.state and action
        if "observation.state" in df.columns:
            print(f"observation.state sample: {df['observation.state'].iloc[0]}")
            print(f"observation.state type: {type(df['observation.state'].iloc[0])}")
        if "action" in df.columns:
            print(f"action sample: {df['action'].iloc[0]}")
            print(f"action type: {type(df['action'].iloc[0])}")

        # Validate that essential columns are present
        required_columns = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in episode dataframe: {missing_columns}")

        # Check for observation.state and action columns
        if "observation.state" in df.columns:
            print(f"observation.state column found with shape: {np.array(df['observation.state'].iloc[0]).shape}")
        if "action" in df.columns:
            print(f"action column found with shape: {np.array(df['action'].iloc[0]).shape}")

        # Validate timestamp column
        if df["timestamp"].isnull().any():
            raise ValueError("Timestamp column contains null values")

        # Ensure timestamp is float32 (as expected by LeRobot)
        df["timestamp"] = df["timestamp"].astype(np.float32)

        return df

    def update_statistics(self, observation_state: Dict[str, np.ndarray], action_data: Dict[str, np.ndarray]):
        """Update field statistics with real computed values."""

        # Compute statistics for observation state fields
        for field_name, field_data in observation_state.items():
            # Skip non-numeric fields (strings, etc.)
            if not np.issubdtype(field_data.dtype, np.number):
                continue

            # Skip fields with all NaN values (try to convert to check)
            try:
                test_data = field_data.astype(float)
                if np.all(np.isnan(test_data)):
                    continue
            except (ValueError, TypeError):
                # Can't convert to float, skip this field
                continue

            if field_name not in self.field_stats:
                self.field_stats[field_name] = {"min": [], "max": [], "mean": [], "std": []}

            try:
                # Convert to float and handle NaN values
                numeric_data = field_data.astype(float)

                # Compute statistics for this episode
                if len(numeric_data.shape) > 1:
                    # Multi-dimensional data - compute stats across time dimension
                    field_min = np.nanmin(numeric_data, axis=0)
                    field_max = np.nanmax(numeric_data, axis=0)
                    field_mean = np.nanmean(numeric_data, axis=0)
                    field_std = np.nanstd(numeric_data, axis=0)
                else:
                    # 1D data
                    field_min = np.array([np.nanmin(numeric_data)])
                    field_max = np.array([np.nanmax(numeric_data)])
                    field_mean = np.array([np.nanmean(numeric_data)])
                    field_std = np.array([np.nanstd(numeric_data)])

                # Only add if we got valid statistics (not all NaN)
                if not np.all(np.isnan(field_min)):
                    self.field_stats[field_name]["min"].append(field_min)
                    self.field_stats[field_name]["max"].append(field_max)
                    self.field_stats[field_name]["mean"].append(field_mean)
                    self.field_stats[field_name]["std"].append(field_std)

            except (ValueError, TypeError) as e:
                # Skip fields that can't be converted to numeric
                print(f"Skipping statistics for non-numeric field {field_name}: {e}")
                continue

        # Compute statistics for action fields
        for field_name, field_data in action_data.items():
            # Skip non-numeric fields (strings, etc.)
            if not np.issubdtype(field_data.dtype, np.number):
                continue

            # Skip fields with all NaN values (try to convert to check)
            try:
                test_data = field_data.astype(float)
                if np.all(np.isnan(test_data)):
                    continue
            except (ValueError, TypeError):
                # Can't convert to float, skip this field
                continue

            if field_name not in self.field_stats:
                self.field_stats[field_name] = {"min": [], "max": [], "mean": [], "std": []}

            try:
                # Convert to float and handle NaN values
                numeric_data = field_data.astype(float)

                # Compute statistics for this episode
                if len(numeric_data.shape) > 1:
                    # Multi-dimensional data - compute stats across time dimension
                    field_min = np.nanmin(numeric_data, axis=0)
                    field_max = np.nanmax(numeric_data, axis=0)
                    field_mean = np.nanmean(numeric_data, axis=0)
                    field_std = np.nanstd(numeric_data, axis=0)
                else:
                    # 1D data
                    field_min = np.array([np.nanmin(numeric_data)])
                    field_max = np.array([np.nanmax(numeric_data)])
                    field_mean = np.array([np.nanmean(numeric_data)])
                    field_std = np.array([np.nanstd(numeric_data)])

                # Only add if we got valid statistics (not all NaN)
                if not np.all(np.isnan(field_min)):
                    self.field_stats[field_name]["min"].append(field_min)
                    self.field_stats[field_name]["max"].append(field_max)
                    self.field_stats[field_name]["mean"].append(field_mean)
                    self.field_stats[field_name]["std"].append(field_std)

            except (ValueError, TypeError) as e:
                # Skip fields that can't be converted to numeric
                print(f"Skipping statistics for non-numeric action field {field_name}: {e}")
                continue

    def save_chunk(self, episode_dfs: List[pd.DataFrame], chunk_index: int = 0):
        """Save a chunk of episodes to parquet."""
        if not episode_dfs:
            return

        # Combine all episode DataFrames
        combined_df = pd.concat(episode_dfs, ignore_index=True)

        # Update global index
        combined_df["index"] = range(len(combined_df))

        # Save to parquet
        chunk_name = f"chunk-{chunk_index:03d}"
        data_chunk_dir = self.data_dir / chunk_name
        data_chunk_dir.mkdir(exist_ok=True)

        for episode_idx, episode_df in enumerate(episode_dfs):
            # Use the actual episode_index from the DataFrame, not the local chunk index
            actual_episode_index = (
                episode_df["episode_index"].iloc[0] if "episode_index" in episode_df.columns else episode_idx
            )
            parquet_path = data_chunk_dir / f"episode_{actual_episode_index:06d}.parquet"

            # Debug: Check columns before saving
            print(f"Saving episode {episode_idx} with columns: {list(episode_df.columns)}")
            if "timestamp" in episode_df.columns:
                print(f"Timestamp column present with type: {episode_df['timestamp'].dtype}")
                print(f"First few timestamp values: {episode_df['timestamp'].head()}")
            else:
                print("WARNING: Timestamp column missing before saving!")

            episode_df.to_parquet(parquet_path, index=False)

            # Debug: Verify saved file
            saved_df = pd.read_parquet(parquet_path)
            print(f"Saved episode {episode_idx} has columns: {list(saved_df.columns)}")
            if "timestamp" not in saved_df.columns:
                print(f"ERROR: Timestamp column lost during save for episode {episode_idx}!")
            else:
                print(f"Timestamp column preserved with type: {saved_df['timestamp'].dtype}")
                print(f"First few saved timestamp values: {saved_df['timestamp'].head()}")

    def finalize_dataset(self):
        """Create final metadata files."""
        # Create episodes.jsonl
        episodes_path = self.meta_dir / "episodes.jsonl"
        # Sort episodes by episode_index to ensure correct order
        sorted_episodes = sorted(self.episodes_metadata, key=lambda x: x["episode_index"])
        with open(episodes_path, "w") as f:
            for episode_meta in sorted_episodes:
                f.write(json.dumps(episode_meta) + "\n")
        # Validate that episodes.jsonl is non-empty and matches metadata count
        if not episodes_path.exists() or episodes_path.stat().st_size == 0:
            raise RuntimeError("episodes.jsonl is empty; no episodes were written. Aborting.")
        with open(episodes_path, "r") as f:
            num_lines = sum(1 for _ in f)
        if num_lines != len(self.episodes_metadata):
            raise RuntimeError(
                f"episodes.jsonl line count ({num_lines}) does not "
                f"match episodes_metadata length ({len(self.episodes_metadata)})."
            )

        # Create stats.jsonl
        stats_path = self.meta_dir / "stats.jsonl"
        with open(stats_path, "w") as f:
            for field, stats in self.field_stats.items():
                # Skip if no statistics were collected or if arrays are empty
                if not stats["min"] or len(stats["min"]) == 0:
                    continue

                # Concatenate all episode statistics
                all_mins = np.concatenate(stats["min"]) if len(stats["min"]) > 1 else stats["min"][0]
                all_maxs = np.concatenate(stats["max"]) if len(stats["max"]) > 1 else stats["max"][0]
                all_means = np.concatenate(stats["mean"]) if len(stats["mean"]) > 1 else stats["mean"][0]
                all_stds = np.concatenate(stats["std"]) if len(stats["std"]) > 1 else stats["std"][0]

                # Compute global statistics across all episodes and dimensions
                global_stats = {
                    "field": field,
                    "min": float(np.min(all_mins)),
                    "max": float(np.max(all_maxs)),
                    "mean": float(np.mean(all_means)),
                    "std": float(np.mean(all_stds)),  # Average of all episode stds
                }
                f.write(json.dumps(global_stats) + "\n")

        # Create tasks.jsonl
        tasks_path = self.meta_dir / "tasks.jsonl"
        with open(tasks_path, "w") as f:
            for task_info in self.tasks.values():
                f.write(json.dumps(task_info) + "\n")

        # Create info.json
        info = {
            "codebase_version": "v2.1",  # Mandatory for LeRobot tools
            "dataset_name": self.dataset_name,
            "robot_type": self.robot_type,
            "total_episodes": len(self.episodes_metadata),
            "total_frames": self.total_frames,
            "total_tasks": len(self.tasks),
            "chunks_size": self.chunk_size,  # Episodes per chunk - use actual chunk_size parameter
            "fps": self.fps,
            "splits": {"train": f"0:{len(self.episodes_metadata)}"},
            "created_at": datetime.datetime.now().isoformat(),
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": self.create_features_dict(),
            "camera_names": list(self.camera_keys),
        }

        # Calculate file sizes
        data_size_mb = self.calculate_data_files_size()
        video_size_mb = self.calculate_video_files_size()

        info["data_files_size_in_mb"] = data_size_mb
        info["video_files_size_in_mb"] = video_size_mb

        info_path = self.meta_dir / "info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def create_features_dict(self) -> Dict[str, Any]:
        """Create features dictionary for info.json."""
        features = {}

        # Add observation state feature (as array) - only if we have observation state data
        if hasattr(self, "_obs_state_names") and self._obs_state_names:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": [len(self._obs_state_names)],
                "names": self._obs_state_names,
            }

        # Add action feature (as array) - only if we have action data
        if hasattr(self, "_action_names") and self._action_names:
            features["action"] = {"dtype": "float32", "shape": [len(self._action_names)], "names": self._action_names}

        # Note: We don't add empty observation.state or action fields since we're using flattened field names

        # Add standard LeRobot fields (match official format)
        features.update(
            {
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            }
        )

        # Add video features for each camera
        for camera_name in self.camera_keys:
            # Use actual video dimensions if available, otherwise default
            if camera_name in self.video_dimensions:
                height, width = self.video_dimensions[camera_name]
            else:
                height, width = 256, 256  # Default fallback

            features[f"observation.images.{camera_name}"] = {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": float(self.fps),
                    "video.height": height,
                    "video.width": width,
                    "video.channels": 3,
                    "video.codec": self.video_codecs.get(camera_name, "libsvtav1")
                    if hasattr(self, "video_codecs")
                    else "libsvtav1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }

        return features

    def calculate_data_files_size(self) -> float:
        """Calculate total size of data files in MB."""
        total_size = 0
        for chunk_dir in self.data_dir.glob("chunk-*"):
            for parquet_file in chunk_dir.glob("episode_*.parquet"):
                if parquet_file.exists():
                    total_size += parquet_file.stat().st_size
        return round(total_size / (1024 * 1024), 2)

    def calculate_video_files_size(self) -> float:
        """Calculate total size of video files in MB."""
        total_size = 0
        for chunk_dir in self.videos_dir.glob("chunk-*"):
            for camera_dir in chunk_dir.glob("observation.images.*"):
                for video_file in camera_dir.glob("episode_*.mp4"):
                    if video_file.exists():
                        file_size = video_file.stat().st_size
                        total_size += file_size
                        print(f"Video file: {video_file.name} = {file_size} bytes")

        print(f"Total video size: {round(total_size / (1024 * 1024), 2)} MB")
        return round(total_size / (1024 * 1024), 2)

    def create_huggingface_dataset(self) -> Optional[Any]:
        """Create a Hugging Face Dataset from the processed episodes."""
        if not HF_AVAILABLE:
            print("Warning: Hugging Face libraries not available. Cannot create dataset.")
            return None

        if not self.episodes_metadata:
            print("Warning: No episodes processed. Cannot create dataset.")
            return None

        print("📦 Creating Hugging Face Dataset...")

        # Load all parquet files and combine them
        all_data = []

        for chunk_dir in self.data_dir.glob("chunk-*"):
            for parquet_file in chunk_dir.glob("episode_*.parquet"):
                df = pd.read_parquet(parquet_file)
                all_data.append(df)

        if not all_data:
            print("Warning: No parquet files found. Cannot create dataset.")
            return None

        # Combine all episode data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Convert to Hugging Face Dataset
        dataset = datasets.Dataset.from_pandas(combined_df)

        # Add episode data index
        episode_data_index = calculate_episode_data_index(dataset)

        # Create dataset info
        dataset_info = {
            "dataset_name": self.dataset_name,
            "robot_type": self.robot_type,
            "total_episodes": len(self.episodes_metadata),
            "total_frames": self.total_frames,
            "fps": self.fps,
            "episode_data_index": episode_data_index,
            "camera_names": list(self.camera_keys),
        }

        # Add custom attributes to dataset (excluding reserved attributes)
        for key, value in dataset_info.items():
            setattr(dataset, key, value)

        return dataset

    def push_dataset_to_hub(self, repo_id: str, token: Optional[str] = None, private: bool = False) -> None:
        """Push the dataset to Hugging Face Hub."""
        if not HF_AVAILABLE:
            raise RuntimeError("Hugging Face libraries are required for pushing datasets.")

        check_repo_id(repo_id)

        print(f"🚀 Pushing dataset to {repo_id}...")

        # Initialize the API
        api = HfApi(token=token)

        # Create the repository
        try:
            api.create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="dataset")
        except Exception as e:
            print(f"Repository creation failed or already exists: {e}")

        # Upload all files to maintain the directory structure
        print(f"📁 Uploading data files to {self.data_dir}...")

        # Upload data directory (parquet files)
        if self.data_dir.exists():
            api.upload_folder(
                folder_path=str(self.data_dir),
                path_in_repo="data",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )

        # Upload videos directory
        print(f"📹 Uploading videos to {self.videos_dir}...")
        if self.videos_dir.exists():
            api.upload_folder(
                folder_path=str(self.videos_dir),
                path_in_repo="videos",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )

        # Upload metadata files
        print(f"Uploading metadata files to {self.meta_dir}...")
        if self.meta_dir.exists():
            for meta_file in self.meta_dir.glob("*.json*"):
                # Refuse to upload an empty episodes.jsonl
                if meta_file.name == "episodes.jsonl" and (not meta_file.exists() or meta_file.stat().st_size == 0):
                    raise RuntimeError("Refusing to upload empty meta/episodes.jsonl to Hub.")
                api.upload_file(
                    path_or_fileobj=str(meta_file),
                    path_in_repo=f"meta/{meta_file.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )

        # Upload additional data if it exists
        additional_data_dir = self.output_dir / "additional_data"
        if additional_data_dir.exists():
            print("💾 Uploading additional data...")
            api.upload_folder(
                folder_path=str(additional_data_dir),
                path_in_repo="additional_data",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )

        # Create and upload README
        readme_content = self.generate_dataset_readme()
        readme_path = self.output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

        print(f"✅ Dataset successfully pushed to https://huggingface.co/datasets/{repo_id}")

    def generate_dataset_readme(self) -> str:
        """Generate a README for the dataset."""
        readme = f"""# {self.dataset_name}

This dataset was converted from LBM format to LeRobot format.

## Dataset Information

- **Robot Type**: {self.robot_type}
- **Total Episodes**: {len(self.episodes_metadata)}
- **Total Frames**: {self.total_frames}
- **FPS**: {self.fps}
- **Cameras**: {", ".join(self.camera_keys)}

## Dataset Structure

This dataset follows the LeRobot format:

- `data/`: Contains episode data in Parquet format
- `videos/`: Contains video files for each camera
- `meta/`: Contains metadata files (episodes.jsonl, stats.jsonl, tasks.jsonl, info.json)
- `additional_data/` (optional): Contains preserved LBM-specific data (depth, segmentation)

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{self.dataset_name}")

# Access episode data
episode_data = dataset["train"]

# Get video paths from metadata
import json
with open("meta/episodes.jsonl", "r") as f:
    episodes_meta = [json.loads(line) for line in f]

print(f"First episode videos: {{episodes_meta[0]['video_paths']}}")
```

## Original Data

This dataset preserves all original LBM data:
- Robot state observations mapped to `observation.state.*`
- Actions mapped to `action.*`  
- Camera images converted to MP4 videos
- Additional data (depth, segmentation, calibration) preserved in metadata and additional files

## Citation

If you use this dataset, please cite the original LBM work and this conversion.
"""
        return readme


def extract_task_from_path(episode_path: str) -> str:
    """Extract task name from episode path.

    Assumes path format: .../TaskName/...
    """
    path_parts = Path(episode_path).parts
    # Look for task pattern - typically uppercase task names
    for part in path_parts:
        if part and part[0].isupper() and not part.startswith("episode"):
            return part
    print(f"No task found, using parent directory: {Path(episode_path).parent.name}")
    # Fallback: use parent directory name
    return Path(episode_path).parent.name


def group_episodes_by_task(episode_paths: List[str]) -> Dict[str, List[str]]:
    """Group episode paths by task name."""
    task_groups = {}
    for episode_path in episode_paths:
        task_name = extract_task_from_path(episode_path)
        if task_name not in task_groups:
            task_groups[task_name] = []
        task_groups[task_name].append(episode_path)
    return task_groups


def load_episodes_from_csv(csv_path: str) -> List[str]:
    """Load episode paths from a CSV file."""
    episodes = []
    with open(csv_path, "r") as f:
        for line in f:
            episode_path = line.strip()
            if episode_path:
                episodes.append(episode_path)
    return episodes


def main():
    """Main preprocessing function."""
    # Parse CLI arguments
    cfg = draccus.parse(config_class=LeRobotPreprocessParams)

    # Validate required arguments
    assert cfg.source_episodes or cfg.source_eps_csv_path, (
        "Either --source_episodes or --source_eps_csv_path is required"
    )
    assert cfg.output_dir is not None, "--output_dir is required"
    object.__setattr__(cfg, "output_dir", str(Path(cfg.output_dir).resolve()))

    print("🚀 Starting LBM to LeRobot conversion")
    print(f"Dataset name: {cfg.dataset_name}")
    print(f"Robot type: {cfg.robot_type}")
    print(f"Output directory: {cfg.output_dir}")
    print(f"Workers: {cfg.num_workers}")

    # Discover episodes
    print("🔍 Discovering episodes...")
    if cfg.source_eps_csv_path:
        print(f"Using episode list from CSV: {cfg.source_eps_csv_path}")
        episodes = load_episodes_from_csv(cfg.source_eps_csv_path)
        print(f"Loaded {len(episodes)} episodes from CSV")
    elif cfg.source_episodes:
        print(f"Using source episodes from: {cfg.source_episodes}")
        episodes = discover_episodes(cfg.source_episodes, cfg.max_episodes_to_process)

    print(f"Found {len(episodes)} episodes")
    # Useful debug prints
    if len(episodes) > 0:
        print(f"First episode: {episodes[0]}")
        print(f"Last episode: {episodes[-1]}")

    if len(episodes) == 0:
        print("❌ No episodes found!")
        return

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

    # Group episodes by task and create separate datasets
    print("📋 Organizing episodes by task...")

    task_groups = group_episodes_by_task(episodes)

    print(f"Found {len(task_groups)} tasks: {list(task_groups.keys())}")

    total_processed = 0
    for task_name, task_episodes in task_groups.items():
        print(f"\n🎯 Processing task: {task_name} ({len(task_episodes)} episodes)")

        # Create task-specific output directory
        task_output_dir = Path(cfg.output_dir) / task_name
        task_dataset_name = f"{cfg.dataset_name}_{task_name}"

        # Create the dataset writer
        writer_args = dict(
            output_dir=str(task_output_dir),
            dataset_name=task_dataset_name,
            robot_type=cfg.robot_type,
            fps=cfg.fps,
            video_codec=cfg.video_codec,
            preserve_depth=cfg.preserve_depth,
            preserve_segmentation=cfg.preserve_segmentation,
            preserve_calibration=cfg.preserve_calibration,
            skip_videos=cfg.skip_videos,
            chunk_size=cfg.chunk_size,
            sync_cameras=cfg.sync_cameras,
        )
        writer = LeRobotDatasetWriter.remote(**writer_args)

        total_episodes = len(task_episodes)
        num_processed = 0
        chunk_size = min(cfg.chunk_size, total_episodes)

        episode_chunks = [task_episodes[i : i + chunk_size] for i in range(0, len(task_episodes), chunk_size)]

        for chunk_idx, episode_chunk in enumerate(tqdm(episode_chunks, desc="Processing chunks")):
            chunk_results = []

            ray_futures = []
            for episode_idx, episode_path in enumerate(episode_chunk):
                global_episode_idx = chunk_idx * chunk_size + episode_idx
                ray_futures.append(writer.process_episode.remote(episode_path, global_episode_idx, chunk_idx))
            # Gather results
            results = ray.get(ray_futures)

            for result in tqdm(results, total=len(ray_futures), desc=f"Chunk {chunk_idx}"):
                chunk_results.append(result)

            chunk_dfs = []
            for result in chunk_results:
                if result["success"]:
                    chunk_dfs.append(result["episode_df"])

            if chunk_dfs:
                ray.get(writer.save_chunk.remote(chunk_dfs, chunk_idx))
                num_processed += len(chunk_dfs)

        # Finalize dataset
        print("📋 Creating metadata files...")
        ray.get(writer.finalize_dataset.remote())

        # Verify output directory has content
        task_output_dir = Path(cfg.output_dir) / task_name
        if task_output_dir.exists():
            print(f"✅ Task output directory exists: {task_output_dir}")
            if any(task_output_dir.iterdir()):
                print("✅ Task output directory is not empty")
            else:
                print("❌ Task output directory is empty!")
        else:
            print(f"❌ Task output directory missing: {task_output_dir}")

        total_processed += num_processed

        # Push task dataset to hub if requested
        if cfg.push_to_hub and cfg.repo_id:
            task_repo_id = f"{cfg.repo_id}_{task_name}"
            try:
                ray.get(writer.push_dataset_to_hub.remote(repo_id=task_repo_id, private=cfg.private))
                print(f"🚀 Task dataset pushed to: https://huggingface.co/datasets/{task_repo_id}")
                # Remove local copy to save space
                shutil.rmtree(task_output_dir)
            except Exception as e:
                print(f"❌ Error pushing task {task_name} to hub: {e}")

    ray.shutdown()

    print("✅ Conversion complete!")
    print(f"📊 Processed {total_processed} episodes successfully")
    print(f"📁 Dataset saved to: {cfg.output_dir}")
    if task_groups:
        print("📋 Task-level datasets created:")
        for task_name in task_groups:
            task_output_dir = Path(cfg.output_dir) / task_name
            print(f"  - {task_name}: {task_output_dir}")

    print("\n📋 LeRobot dataset structure:")
    print("  - data/chunk-000/episode_*.parquet: Episode data")
    print("  - videos/observation.images.*/chunk-000/episode_*.mp4: Video files")
    print("  - meta/episodes.jsonl: Episode metadata")
    print("  - meta/stats.jsonl: Field statistics")
    print("  - meta/tasks.jsonl: Task definitions")
    print("  - meta/info.json: Dataset info")
    print("  - additional_data/ (optional): Preserved LBM-specific data")
    if cfg.push_to_hub:
        print("  - README.md: Dataset documentation")


if __name__ == "__main__":
    main()
