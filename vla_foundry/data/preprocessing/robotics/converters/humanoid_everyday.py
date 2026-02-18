import json
import lzma
import os
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from vla_foundry.data.preprocessing.robotics.converters.base import BaseRoboticsConverter
from vla_foundry.data.preprocessing.robotics.preprocess_masks import create_past_and_future_masks


@dataclass
class SampleMetadata:
    """Metadata for each preprocessed sample from HumanoidEveryday dataset."""

    episode_id: str
    sample_id: str
    anchor_timestep: int | None
    anchor_relative_idx: int | None
    image_timesteps: list[int]
    lowdim_start_timestep: int
    lowdim_end_timestep: int
    past_padding: int
    future_padding: int
    camera_names: list[str]
    original_episode_length: int
    original_image_sizes: dict[str, tuple[int, int]]
    is_padded: bool
    task_name: str | None = None
    task_category: str | None = None


class HumanoidEverydayConverter(BaseRoboticsConverter):
    """
    Converter for the HumanoidEveryday dataset (Unitree G1 humanoid demonstrations).

    Source format:
        {task_name}/
            metadata/metadata.json          # Task-level language annotation
            episode_N/
                data.json                   # Per-frame records (states, actions, file refs)
                color/frame_XXXXXX.jpg      # RGB images
                depth/frame_XXXXXX.npy.lzma # LZMA-compressed depth numpy arrays

    Target format: WebDataset tar shards compatible with VLA Foundry training pipeline.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # Build language annotations lookup from metadata.json files
        self.language_annotations = {}
        self.task_categories = {}
        self._load_language_annotations(cfg.source_episodes)

    def _load_language_annotations(self, source_paths: list[str]) -> None:
        """Load language annotations from metadata.json files across all task directories.
           Saves the language annotations to self.language_annotations and the task categories to self.task_categories.
        Args:
            source_paths: list of source paths to load language annotations from.
        """
        for source_path in source_paths:
            if not os.path.isdir(source_path):
                print(f"Warning: source path '{source_path}' is not a directory, skipping language annotation loading")
                continue
            for task_name in sorted(os.listdir(source_path)):
                metadata_path = os.path.join(source_path, task_name, "metadata", "metadata.json")
                if not os.path.isfile(metadata_path):
                    print(f"Warning: no metadata.json found at '{metadata_path}', skipping task '{task_name}'")
                    continue
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                self.language_annotations[task_name] = {
                    "original": [metadata.get("description", task_name)],
                    "alternative": [metadata.get("title", task_name)],
                    "randomized": None,
                    "verbose": None,
                }
                self.task_categories[task_name] = metadata.get("category", "unknown")

        print(f"Loaded language annotations for {len(self.language_annotations)} tasks")

    def _get_episode_robot_type(self, episode_path: str) -> str:
        """
        Read the robot_type from an episode's data.json.

        The robot_type is expected to be a top-level key in data.json (which is a
        list of per-frame records).  We inspect the first frame to retrieve it.
        If the key is absent — as is the case for older H1 recordings that
        pre-date the field — we default to ``"h1"``.

        Args:
            episode_path: Path to the episode directory containing data.json.

        Returns:
            The robot type string (e.g. ``"h1"``, ``"g1"``).
        """
        data_json_path = os.path.join(episode_path, "data.json")
        with open(data_json_path, "r") as f:
            frames = json.load(f)

        if not frames:
            return "h1"

        return frames[0].get("robot_type", "h1")

    def discover_episodes(self, source_paths: list[str], max_episodes_to_process: int = -1) -> list[str]:
        """
        Discover all episode directories across all task directories.

        Walks {source_path}/{task_name}/episode_* and returns a flat sorted list of episode paths.
        Skips .zip files and directories without data.json.

        Args:
            source_paths: list of source paths to discover episodes from.
            max_episodes_to_process: Maximum number of episodes to return. If -1, all episodes are returned.

        Returns:
            list of episode paths sorted for deterministic ordering.
        """
        # TODO: Add support for s3 paths for the dataset on s3
        task_filter = getattr(self.cfg, "task_filter", None)
        episode_filter = getattr(self.cfg, "episode_filter", None)
        embodiment_filter = getattr(self.cfg, "embodiment_filter", None)

        if task_filter:
            print(f"Task filter: {task_filter}")
        if episode_filter:
            print(f"Episode filter: {episode_filter}")
        if embodiment_filter:
            print(f"Embodiment filter: {embodiment_filter}")

        all_episodes = []
        for source_path in source_paths:
            if not os.path.isdir(source_path):
                print(f"Warning: source path '{source_path}' is not a directory, skipping")
                continue

            for task_name in sorted(os.listdir(source_path)):
                # Apply task filter
                if task_filter and task_name not in task_filter:
                    continue

                task_dir = os.path.join(source_path, task_name)
                if not os.path.isdir(task_dir) or task_name.endswith(".zip"):
                    continue

                for item in sorted(os.listdir(task_dir)):
                    if not item.startswith("episode_"):
                        continue
                    # Apply episode filter
                    if episode_filter and item not in episode_filter:
                        continue

                    episode_path = os.path.join(task_dir, item)
                    if not os.path.isdir(episode_path):
                        continue
                    # Validate episode has data.json
                    data_json_path = os.path.join(episode_path, "data.json")
                    if not os.path.isfile(data_json_path):
                        continue

                    # Apply embodiment filter
                    if embodiment_filter is not None:
                        robot_type = self._get_episode_robot_type(episode_path)
                        if robot_type not in embodiment_filter:
                            continue

                    all_episodes.append(episode_path)

        # Sorted for deterministic processing order across runs. Lexicographic order is intentional —
        # exact numeric episode ordering is not required, only reproducibility.
        all_episodes = sorted(all_episodes)

        # Apply max_episodes_to_process limit if specified
        if max_episodes_to_process > 0 and len(all_episodes) > max_episodes_to_process:
            all_episodes = all_episodes[:max_episodes_to_process]

        print(f"Total episodes discovered: {len(all_episodes)}")
        return all_episodes

    def _extract_state_arrays(self, frames: list[dict], robot_type: str) -> dict[str, np.ndarray]:
        """Extract all state fields from per-frame records into named numpy arrays.

        Handles both G1 and H1 embodiments. G1-only fields (hand pressure, odometry)
        are included only when ``robot_type == "g1"``.

        Dimension differences across embodiments:
            - ``leg_state``:  G1=15, H1=13
            - ``hand_state``: G1=14, H1=12

        Args:
            frames: list of per-frame dicts from data.json.
            robot_type: ``"g1"`` or ``"h1"``.

        Returns:
            dict mapping ``humanoid__state__*`` keys to ``(T, D)`` arrays.
        """
        # TODO: Add a centralized embodiment key mapping layer.
        # We currently assume HumanoidEveryday keys and write them directly.
        states: dict[str, np.ndarray] = {
            "humanoid__state__arm": np.array([f["states"]["arm_state"] for f in frames], dtype=np.float32),
            "humanoid__state__leg": np.array([f["states"]["leg_state"] for f in frames], dtype=np.float32),
            "humanoid__state__hand": np.array([f["states"]["hand_state"] for f in frames], dtype=np.float32),
            # IMU — present for both embodiments
            "humanoid__state__imu_quaternion": np.array(
                [f["states"]["imu"]["quaternion"] for f in frames], dtype=np.float32
            ),
            "humanoid__state__imu_accelerometer": np.array(
                [f["states"]["imu"]["accelerometer"] for f in frames], dtype=np.float32
            ),
            "humanoid__state__imu_gyroscope": np.array(
                [f["states"]["imu"]["gyroscope"] for f in frames], dtype=np.float32
            ),
            "humanoid__state__imu_rpy": np.array([f["states"]["imu"]["rpy"] for f in frames], dtype=np.float32),
        }

        # Hand pressure — G1 only (18 sensors: 12 type-A × 4 readings, 6 type-B × 3 readings)
        if robot_type == "g1":
            states["humanoid__state__hand_pressure_a"] = np.array(
                [
                    [
                        r
                        for s in f["states"]["hand_pressure_state"]
                        if s["sensor_type"] == "A"
                        for r in s["usable_readings"]
                    ]
                    for f in frames
                ],
                dtype=np.float32,
            )  # (T, 48)
            states["humanoid__state__hand_pressure_b"] = np.array(
                [
                    [
                        r
                        for s in f["states"]["hand_pressure_state"]
                        if s["sensor_type"] == "B"
                        for r in s["usable_readings"]
                    ]
                    for f in frames
                ],
                dtype=np.float32,
            )  # (T, 18)

            # Odometry — G1 only
            states["humanoid__state__odometry_position"] = np.array(
                [f["states"]["odometry"]["position"] for f in frames], dtype=np.float32
            )
            states["humanoid__state__odometry_velocity"] = np.array(
                [f["states"]["odometry"]["velocity"] for f in frames], dtype=np.float32
            )
            states["humanoid__state__odometry_rpy"] = np.array(
                [f["states"]["odometry"]["rpy"] for f in frames], dtype=np.float32
            )
            states["humanoid__state__odometry_quat"] = np.array(
                [f["states"]["odometry"]["quat"] for f in frames], dtype=np.float32
            )

        return states

    def _extract_action_arrays(self, frames: list[dict], episode_length: int) -> dict[str, np.ndarray]:
        """Extract all action fields from per-frame records into named numpy arrays.

        Both G1 and H1 share the same action keys, though ``left_angles`` / ``right_angles``
        have different dimensionality (G1=7, H1=12).

        Args:
            frames: list of per-frame dicts from data.json.
            episode_length: number of frames (used for reshape).

        Returns:
            dict mapping ``humanoid__action__*`` keys to ``(T, D)`` arrays.
        """
        return {
            "humanoid__action__left_angles": np.array([f["actions"]["left_angles"] for f in frames], dtype=np.float32),
            "humanoid__action__right_angles": np.array(
                [f["actions"]["right_angles"] for f in frames], dtype=np.float32
            ),
            "humanoid__action__sol_q": np.array([f["actions"]["sol_q"] for f in frames], dtype=np.float32),
            "humanoid__action__tau_ff": np.array([f["actions"]["tau_ff"] for f in frames], dtype=np.float32),
            # head_rmat: 3x3 rotation matrix, flattened to 9
            "humanoid__action__head_rmat": np.array(
                [f["actions"]["head_rmat"] for f in frames], dtype=np.float32
            ).reshape(episode_length, -1),
            # left_pose / right_pose: 4x4 homogeneous pose matrices, flattened to 16
            "humanoid__action__left_pose": np.array(
                [f["actions"]["left_pose"] for f in frames], dtype=np.float32
            ).reshape(episode_length, -1),
            "humanoid__action__right_pose": np.array(
                [f["actions"]["right_pose"] for f in frames], dtype=np.float32
            ).reshape(episode_length, -1),
        }

    def _load_rgb_images(self, frames: list[dict], episode_path: str) -> np.ndarray:
        """Load all RGB images for an episode from disk.

        Args:
            frames: list of per-frame dicts from data.json (each must contain an ``"image"`` key).
            episode_path: path to the episode directory.

        Returns:
            ``(T, H, W, 3)`` uint8 array of RGB images.
        """
        rgb_images = []
        for frame in frames:
            img_path = os.path.join(episode_path, frame["image"])
            img = np.array(Image.open(img_path).convert("RGB"))
            rgb_images.append(img)
        return np.stack(rgb_images, axis=0)

    def _load_depth_images(self, frames: list[dict], episode_path: str) -> np.ndarray | None:
        """Load all depth images for an episode from disk if depth data is enabled.

        Depth files are LZMA-compressed raw uint16 binary data (millimeters).

        Args:
            frames: list of per-frame dicts from data.json (each must contain a ``"depth"`` key).
            episode_path: path to the episode directory.

        Returns:
            ``(T, H, W)`` uint16 array of depth images, or ``None`` if depth is disabled.
        """
        if not self.cfg.use_depth_data:
            return None

        depth_list = []
        for frame in frames:
            depth_path = os.path.join(episode_path, frame["depth"])
            with lzma.open(depth_path, "rb") as lz_f:
                raw_bytes = lz_f.read()
            depth_arr = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(self.cfg.depth_resolution)
            depth_list.append(depth_arr)
        return np.stack(depth_list, axis=0)

    def load_episode_data(self, episode_path: str) -> dict[str, Any]:
        """
        Load all episode data: parse data.json, pre-extract states/actions into numpy arrays,
        and read all RGB (and optionally depth) images into memory.

        Supports both G1 and H1 embodiments.  Fields that only exist for a
        specific robot (e.g. ``hand_pressure_state`` and ``odometry`` are G1-only)
        are included conditionally — their lowdim keys are simply omitted when
        the source data does not contain them.

        Dimension differences across embodiments (all are handled automatically):
            - ``leg_state``:  G1=15, H1=13
            - ``hand_state``: G1=14, H1=12
            - ``left_angles`` / ``right_angles``: G1=7, H1=12

        Args:
            episode_path: Path to the episode directory containing data.json.

        Returns:
            dict of episode data.
            - frames: list of frames in the episode.
            - task_name: name of the task.
            - robot_type: type of the robot.
            - lowdim: dict of lowdim data.
            - rgb_images: array of RGB images.
            - depth_images: array of depth images, or None.
        """
        data_json_path = os.path.join(episode_path, "data.json")
        with open(data_json_path, "r") as f:
            frames = json.load(f)

        episode_length = len(frames)

        # Detect robot type (H1 episodes pre-date the field and lack it)
        robot_type = frames[0].get("robot_type", "h1") if frames else "h1"

        # Extract task name from path: .../task_name/episode_N/
        task_name = os.path.basename(os.path.dirname(episode_path))

        # Extract lowdim arrays by category
        state_arrays = self._extract_state_arrays(frames, robot_type)
        action_arrays = self._extract_action_arrays(frames, episode_length)
        lowdim = {**state_arrays, **action_arrays}

        # Load images
        rgb_images = self._load_rgb_images(frames, episode_path)
        depth_images = self._load_depth_images(frames, episode_path)

        return {
            "frames": frames,
            "task_name": task_name,
            "robot_type": robot_type,
            "lowdim": lowdim,
            "rgb_images": rgb_images,
            "depth_images": depth_images,
        }

    def get_episode_length(self, episode_data: dict[str, Any]) -> int:
        """Get episode length from the number of frames in data.json.
        Args:
            episode_data: dict of episode data.

        Returns:
            int of episode length.
        """
        return len(episode_data["frames"])

    def extract_camera_data(self, episode_data: dict[str, Any]) -> dict[str, np.ndarray]:
        """Extract camera data: head RGB and optionally depth.
        Args:
            episode_data: dict of episode data.

        Returns:
            dict of camera data.
            - head_rgb: array of RGB images.
            - head_rgb_depth: array of depth images.
        """
        result = {"head_rgb": episode_data["rgb_images"]}
        if self.cfg.use_depth_data and episode_data["depth_images"] is not None:
            result["head_rgb_depth"] = episode_data["depth_images"]
        return result

    def extract_lowdim_data(self, episode_data: dict[str, Any]) -> dict[str, np.ndarray]:
        """Extract all low-dimensional state and action data.
        Args:
            episode_data: dict of episode data.

        Returns:
            dict of lowdim data.
        """
        return episode_data["lowdim"]

    def extract_intrinsics_extrinsics_data(self, episode_data: dict[str, Any]) -> tuple[None, None]:
        """No camera calibration available in HumanoidEveryday dataset.
        Args:
            episode_data: dict of episode data.

        Returns:
            tuple of (None, None).
        """
        return None, None

    def extract_metadata_data(self, episode_data: dict[str, Any]) -> dict[str, Any]:
        """Extract task-level metadata.
        Args:
            episode_data: dict of episode data.

        Returns:
            dict of metadata data.
            - task_name: name of the task.
            - task_category: category of the task.
        """
        task_name = episode_data["task_name"]
        return {
            "task_name": task_name,
            "task_category": self.task_categories.get(task_name, "unknown"),
        }

    def get_language_instructions(self, episode_path: str) -> dict[str, Any]:
        """Get language instructions for the task this episode belongs to.
        Args:
            episode_path: Path to the episode directory containing data.json.

        Returns:
            dict of language instructions.
            - original: list of original language instructions.
            - alternative: list of alternative language instructions.
            - randomized: list of randomized language instructions.
            - verbose: list of verbose language instructions.
        """
        task_name = os.path.basename(os.path.dirname(episode_path))
        if task_name in self.language_annotations:
            return self.language_annotations[task_name]
        # Fallback: use the task name itself as the instruction
        return {
            "original": [task_name.replace("_", " ")],
            "alternative": None,
            "randomized": None,
            "verbose": None,
        }

    def extract_sample_data(
        self,
        anchor_timestep: int,
        episode_path: str,
        episode_length: int,
        camera_data: dict[str, np.ndarray],
        lowdim_data: dict[str, np.ndarray],
        intrinsics_data,
        extrinsics_data,
        metadata_data: dict[str, Any],
        statistics_ray_actor,
        logger_actor,
    ) -> tuple:
        """
        Extract a single training sample centered at anchor_timestep.

        Args:
            anchor_timestep: The timestep to extract the sample data for.
            episode_path: Path to the episode directory containing data.json.
            episode_length: The number of timesteps in the episode.
            camera_data: dict of camera data.
            lowdim_data: dict of lowdim data.
            intrinsics_data: dict of intrinsics data.
            extrinsics_data: dict of extrinsics data.
            metadata_data: dict of metadata data.
            statistics_ray_actor: Ray actor for statistics.
            logger_actor: Logger actor.

        Returns:
            tuple of (sample_images, sample_lowdim, sample_metadata, language_instructions,
             sample_point_clouds, stats_sample).
            - sample_images: dict of sample images.
            - sample_lowdim: dict of sample lowdim data.
            - sample_metadata: dict of sample metadata.
            - language_instructions: dict of language instructions.
            - sample_point_clouds: dict of sample point clouds.
            - stats_sample: dict of stats sample.
        """
        logger_actor.increment_total_potential_samples.remote()

        # Calculate temporal windows
        lowdim_start = anchor_timestep - self.cfg.past_lowdim_steps
        lowdim_end = anchor_timestep + self.cfg.future_lowdim_steps

        # Check padding requirements
        past_padding = max(0, -lowdim_start)
        future_padding = max(0, lowdim_end - episode_length + 1)

        if past_padding > self.cfg.max_padding_left or future_padding > self.cfg.max_padding_right:
            logger_actor.increment_padding_samples_filtered.remote()
            return None, None, None, None

        valid_start = max(0, lowdim_start)
        valid_end = min(episode_length - 1, lowdim_end)

        # --- Extract images at configured offsets ---
        sample_images = {}
        actual_image_timesteps = []

        for img_offset in self.image_indices:
            img_timestep = int(np.clip(anchor_timestep + img_offset, 0, episode_length - 1))
            actual_image_timesteps.append(img_timestep)

            for camera_name, camera_images in camera_data.items():
                key = f"{camera_name}_t{img_offset}"
                sample_images[key] = camera_images[img_timestep]

        # --- Extract lowdim window with padding ---
        sample_lowdim = {}
        for key, data in lowdim_data.items():
            valid_data = data[valid_start : valid_end + 1]
            if past_padding > 0 or future_padding > 0:
                valid_data = self.pad_fn(valid_data, past_padding, future_padding)
            sample_lowdim[key] = valid_data

        # --- Create temporal masks ---
        past_mask, future_mask = create_past_and_future_masks(
            anchor_timestep, self.cfg.past_lowdim_steps, self.cfg.future_lowdim_steps, episode_length
        )

        # --- Build stats_sample for batched statistics update ---
        stats_sample = None
        if statistics_ray_actor is not None:
            stats_sample = {
                "lowdim": {k: v.copy() for k, v in sample_lowdim.items()},
                "past_mask": past_mask,
                "future_mask": future_mask,
            }

        # Add masks to lowdim (after building stats_sample to avoid including masks in statistics)
        sample_lowdim["past_mask"] = past_mask
        sample_lowdim["future_mask"] = future_mask

        # --- Create sample metadata ---
        episode_id = self.get_episode_id(episode_path)
        sample_metadata = SampleMetadata(
            episode_id=episode_id,
            sample_id=f"{uuid.uuid4()}_{episode_id}_t{anchor_timestep:04d}",
            anchor_timestep=int(anchor_timestep),
            anchor_relative_idx=int(self.cfg.past_lowdim_steps),
            image_timesteps=actual_image_timesteps,
            lowdim_start_timestep=int(lowdim_start),
            lowdim_end_timestep=int(lowdim_end),
            past_padding=int(past_padding),
            future_padding=int(future_padding),
            camera_names=list(camera_data.keys()),
            original_episode_length=int(episode_length),
            original_image_sizes={},  # Filled by upload_sample_to_s3
            is_padded=bool(past_padding > 0 or future_padding > 0),
            task_name=metadata_data.get("task_name"),
            task_category=metadata_data.get("task_category"),
        )

        # --- Get language instructions ---
        language_instructions = self.get_language_instructions(episode_path)

        # TODO: Add lidar/point-cloud conversion (.pcd) and expose it via sample_point_clouds
        sample_point_clouds = None

        return sample_images, sample_lowdim, sample_metadata, language_instructions, sample_point_clouds, stats_sample
