"""
MCAP Converter for VLA Foundry

Converts MCAP ROS 2 recordings to preprocessed samples. Uses rosbags for
automatic deserialization with structured extraction for standard messages
(JointState, PoseStamped) producing composite keys compatible with LBM
training (e.g. "action_ee_left__xyz", "action_ee_left__rot_6d").

Supported ROS 2 Message Types
-----------------------------

Message type detection uses attribute inspection rather than importing
ROS 2 message definitions. This avoids ROS 2 dependencies but relies on
assumptions that may break for custom messages with similar attribute
signatures.

TODO (naveen): Consider importing message types from rosbags.typesys or adding
explicit msgtype string checks (connection.msgtype) for robustness.
Custom messages like PolicyKeyframe will need explicit handling.

STRUCTURED EXTRACTION (extract_structured_msg -> composite keys):
    sensor_msgs/JointState      -> __<joint_name> per joint
    geometry_msgs/PoseStamped   -> __xyz, __rot_6d
    geometry_msgs/Pose          -> __xyz, __rot_6d

FLAT EXTRACTION (extract_array_from_msg -> single array):
    sensor_msgs/Imu             -> [quat(4), angular_vel(3), linear_accel(3)]
    geometry_msgs/WrenchStamped -> [force(3), torque(3)]
    geometry_msgs/Wrench        -> [force(3), torque(3)]
    Custom messages             -> all numeric fields (recursive)

FIELD PATH EXTRACTION (extract_field_path -> config-driven):
    Any message with field_extraction config in topics YAML config.
    Used for: Dex3 tactile, lowstate, PolicyKeyframe, etc.

IMAGE EXTRACTION (extract_image_from_msg -> numpy):
    sensor_msgs/CompressedImage (jpeg, png)
    sensor_msgs/Image (rgb8, bgr8, mono8)
    Depth images (16UC1, 32FC1) skipped

Usage
-----
uv run --group preprocessing vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
    --type mcap \
    --source_episodes <s3 or local/path/to/episodes> \
    --output_dir <s3 or local/path/to/output> \
    --output_dir_fixed_path <s3 path to fixed dataset bucket> \
    --config_path <local path to preprocessing_config.yaml> \
    --action_fields_config_path <local path to action_fields_config.yaml> \
    --topics_to_fields_path <local path to topics_to_fields_config.yaml> \
    --camera_names "include <local path to camera_names_config.yaml>" \
    --task_filter '["<task_name>"]' \
    --domain_filter '["<sim_or_real>"]' \
    --source_filter '["<teleop_or_filtered>"]'
"""

import os
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import fsspec
import numpy as np
import ray
import yaml
from numpy.typing import NDArray
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as R

from vla_foundry.data.preprocessing.robotics.converters.base import BaseRoboticsConverter
from vla_foundry.data.preprocessing.robotics.preprocess_masks import create_past_and_future_masks
from vla_foundry.data.preprocessing.utils import is_still_sample, nearest_indices, validate_pose_groups
from vla_foundry.data.robotics.utils import load_action_field_config, matrix_to_rot_6d
from vla_foundry.file_utils import copy_to_temp_file, file_exists, yaml_load


@dataclass
class SampleMetadata:
    """Metadata for each preprocessed sample extracted from an MCAP episode.

    Attributes:
        episode_id: Unique identifier for the source episode (derived from the
            episode directory name).
        sample_id: Globally unique identifier for this sample, combining a UUID
            with the episode_id and zero-padded anchor timestep.
        anchor_timestep: The absolute timestep index within the episode that
            this sample is centred on. None if the sample has no anchor.
        anchor_episode_timestamp: The resampled wall-clock timestamp (relative
            to episode start) at the anchor timestep. None if unavailable.
        anchor_relative_idx: Index of the anchor within the extracted lowdim
            window (i.e. ``past_lowdim_steps``). None if the sample has no anchor.
        image_timesteps: Absolute episode timestep indices for each image frame
            in the sample, one per entry in ``cfg.image_indices``.
        lowdim_start_timestep: Absolute episode index of the first lowdim
            timestep in the temporal window (may be negative before padding).
        lowdim_end_timestep: Absolute episode index of the last lowdim timestep
            in the temporal window (may exceed episode length before padding).
        past_padding: Number of timesteps that were left-padded to fill the
            window when the anchor is near the start of the episode.
        future_padding: Number of timesteps that were right-padded to fill the
            window when the anchor is near the end of the episode.
        camera_names: Ordered list of camera field names present in this sample
            (keys into the camera data dictionary).
        original_episode_length: Total number of resampled timesteps in the
            source episode before any windowing or padding.
        original_episode_duration_s: Wall-clock duration of the source episode
            in seconds, as recorded in the episode metadata. None if unavailable.
        original_image_sizes: Mapping from camera name to native (height, width)
            resolution. Populated downstream by ``upload_sample_to_s3``.
        is_padded: True if the sample required any temporal padding (past or
            future) to fill the lowdim window.
        task_name: Semantic task label(s) for the episode (e.g. skill name from
            the episode metadata). None if not specified.
        teleop_or_rollout: Whether the episode was collected via teleoperation
            or autonomous rollout. None if not specified.
        robot: Robot platform identifier(s) from the episode metadata. None if
            not specified.
        station_name: Name of the physical recording station or workstation.
            None if not specified.
        domain: ``"sim"`` or ``"real"``, indicating whether the episode was
            recorded in simulation or on a physical robot. None if not specified.
    """

    episode_id: str
    sample_id: str
    anchor_timestep: int | None
    anchor_episode_timestamp: float | None
    anchor_relative_idx: int | None
    image_timesteps: list[int]
    lowdim_start_timestep: int
    lowdim_end_timestep: int
    past_padding: int
    future_padding: int
    camera_names: list[str]
    original_episode_length: int
    original_episode_duration_s: float | None
    original_image_sizes: dict[str, tuple[int, int]]
    is_padded: bool
    task_name: str | None
    teleop_or_rollout: str | None
    robot: str | None
    station_name: str | None
    domain: str | None


def extract_image_from_msg(msg: Any, return_numpy: bool = True) -> bytes | np.ndarray:
    """
    Extract image data from ROS 2 image types. Raises exceptions for unsupported formats.

    Args:
        msg: ROS 2 image message (CompressedImage or Image).
        return_numpy: If True, return RGB numpy array. If False, return JPEG bytes.

    Returns:
        RGB uint8 numpy array (H, W, 3) or JPEG bytes.
    """
    # 1. Handle CompressedImage
    if hasattr(msg, "format") and hasattr(msg, "data") and not hasattr(msg, "height"):
        fmt = msg.format.lower()
        if not any(ext in fmt for ext in ("jpeg", "jpg", "png")):
            raise ValueError(f"Unsupported CompressedImage format: {fmt}")

        img = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode {fmt.upper()} CompressedImage")

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 2. Handle Raw Image
    else:
        if not all(hasattr(msg, a) for a in ("height", "width", "encoding", "data")):
            raise TypeError(f"Message is not a valid Image or CompressedImage: {type(msg)}")

        encoding = msg.encoding.lower()
        h, w = msg.height, msg.width
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if encoding == "rgb8":
            rgb_img = raw.reshape(h, w, 3).copy()
        elif encoding == "bgr8":
            rgb_img = cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_BGR2RGB)
        elif encoding in ("mono8", "8uc1"):
            rgb_img = cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_GRAY2RGB)
        elif "16uc1" in encoding or "32fc1" in encoding:
            raise ValueError(f"Depth images ({encoding}) are not supported in the RGB pipeline")
        else:
            raise ValueError(f"Unsupported raw image encoding: {encoding}")

    if return_numpy:
        return rgb_img

    success, jpeg_data = cv2.imencode(".jpg", cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise ValueError("Failed to encode raw image to JPEG")

    return jpeg_data.tobytes()


def extract_structured_msg(msg: Any) -> dict[str, np.ndarray] | None:
    """
    Extract structured key-value data. Raises an exception for errors in extraction.

    Handles:
    - sensor_msgs/JointState: Returns dict with __<joint_name> keys
    - geometry_msgs/PoseStamped, Pose: Returns __xyz and __rot_6d keys

    Returns:
        Dictionary mapping field names to float32 arrays, or None if not a structured message.
    """
    msg_type = type(msg).__name__

    try:
        # sensor_msgs/JointState
        if hasattr(msg, "name") and hasattr(msg, "position") and hasattr(msg, "velocity") and hasattr(msg, "effort"):
            # Extract only position data for now
            return {f"__{n}": np.asarray([p], dtype=np.float32) for n, p in zip(msg.name, msg.position, strict=True)}

        # Unwrap geometry_msgs/PoseStamped -> geometry_msgs/Pose
        if hasattr(msg, "pose") and hasattr(msg.pose, "position"):
            msg = msg.pose

        # Handling geometry_msgs/Pose (`velocity` guard against Odometry-like messages)
        if hasattr(msg, "position") and hasattr(msg, "orientation") and not hasattr(msg, "velocity"):
            xyz = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float32)
            quat = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
            rot_matrix = R.from_quat(quat).as_matrix()
            rot_6d = matrix_to_rot_6d(rot_matrix).astype(np.float32)
            return {"__xyz": xyz, "__rot_6d": rot_6d}

        # Will fallback to extracting an array
        return None
    except Exception as e:
        print(f"⚠️  Structured message extraction failed for message type {msg_type}: {e}")
        return None


def extract_array_from_msg(msg: Any) -> np.ndarray | None:
    """
    Fallback: extract flat numeric array from ROS 2 message using attribute inspection.

    Handles common message types:
    - geometry_msgs/WrenchStamped, Wrench: [force(3), torque(3)]
    - sensor_msgs/Imu, JointState, etc.
    - Custom messages: recursively extracts all numeric fields

    Returns:
        Flattened float32 array or None if extraction fails
    """
    msg_type = type(msg).__name__
    try:
        arrays = []
        # Unwrap geometry_msgs/WrenchStamped -> geometry_msgs/`Wrench
        if hasattr(msg, "wrench"):
            msg = msg.wrench
        if hasattr(msg, "force") and hasattr(msg, "torque"):
            f, t = msg.force, msg.torque
            arrays.extend([np.array([f.x, f.y, f.z]), np.array([t.x, t.y, t.z])])
        else:
            # Generic collection of all numeric attributes
            for name in dir(msg):
                if name.startswith("_"):
                    continue
                val = getattr(msg, name)
                if isinstance(val, (int, float, np.number)):
                    arrays.append(np.array([val]))
                elif isinstance(val, (list, tuple)):
                    try:
                        arr = np.array(val, dtype=np.float32)
                        if arr.size > 0:
                            arrays.append(arr.flatten())
                    except (ValueError, TypeError):
                        # Non-numeric list so skip
                        pass

        if not arrays:
            print(f"🐛 No numeric fields in {msg_type}")
            return None

        return np.concatenate(arrays).astype(np.float32)
    except Exception as e:
        print(f"⚠️  Failed to extract array for message type {msg_type}: {e}")
        return None


def extract_field_path(msg: Any, path: str) -> np.ndarray | None:
    """
    Extract nested values using dot-notation path with fail-fast error handling.

    Supports:
    - Simple paths: "temperature"
    - Nested paths: "imu.x"
    - Array iteration: "motor_state[*].q"

    Args:
        msg: Any ROS message with field_extraction config in topics YAML.
        path: Dot-notation field path

    Returns:
        Float32 array or None if path is invalid
    """
    try:
        parts = path.split(".")
        obj = msg

        for i, part in enumerate(parts):
            if "[*]" in part:
                field = part.replace("[*]", "")
                items = getattr(obj, field)

                # Traverse remaining path for each item, e.g. "motor_state[*].q"
                remaining = parts[i + 1 :]
                if remaining:
                    extracted = []
                    for item in items:
                        val = item
                        for sub_part in remaining:
                            val = getattr(val, sub_part)
                        extracted.append(val)
                    return np.array(extracted, dtype=np.float32).flatten()

                # No remaining path, could be an array of primitives, e.g. "positions[*]"
                return np.array(list(items), dtype=np.float32).flatten()

            obj = getattr(obj, part)

        if np.isscalar(obj):
            return np.array([obj], dtype=np.float32)
        return np.array(obj, dtype=np.float32).flatten()
    except Exception as e:
        print(f"⚠️  Failed to extract path {path} from {type(msg).__name__}: {e}")
        return None


def parse_episode_path(episode_path: str) -> dict[str, Any] | None:
    """
    Parse episode path components assuming .../task/domain/source/episode_id suffix convention.
    Extracts the numeric episode index from the directory name prefix (e.g. "0026_20260129_161531" -> 26).

    Returns:
        Dict with keys {task, domain, source, episode}, or None if the path has too few components.
    """
    parts = episode_path.rstrip("/").split("/")
    # Need at least 4 components: task/domain/source/episode_id
    if len(parts) < 4:
        return None

    episode_dir = parts[-1]
    # Extract leading numeric index from directory name (e.g. "0026_..." -> 26)
    prefix = episode_dir.split("_")[0]
    try:
        episode_index = int(prefix)
    except ValueError:
        print(f"⚠️  Could not parse episode index from directory name: {episode_dir}")
        return None

    return {
        "task": parts[-4],
        "domain": parts[-3],
        "source": parts[-2],
        "episode_dir": episode_dir,
        "episode_index": episode_index,
    }


@ray.remote
def check_mcap_episode_validity_ray(episode_path: str) -> str | None:
    """Check if an episode directory has valid processed data. Returns episode path if valid, None otherwise."""
    fs, fs_path = fsspec.core.url_to_fs(episode_path)
    fs_path = fs_path.rstrip("/")

    try:
        # Check for required files
        dir_name = os.path.basename(fs_path)
        mcap_path = f"{fs_path}/{dir_name}_0.mcap"
        metadata_path = f"{fs_path}/metadata.yaml"
        info_path = f"{fs_path}/info.yaml"
        for required_file in [mcap_path, metadata_path, info_path]:
            if not fs.exists(required_file):
                return None
        return episode_path
    except Exception as e:
        print(f"⚠️  Could not validate {episode_path}: {e}")
        return None


@ray.remote
def discover_and_validate_mcap_episodes_in_directory(mcap_dirs_path: str) -> list[str]:
    """Discover and validate episodes from a directory of MCAP files in parallel."""

    fs, fs_path = fsspec.core.url_to_fs(mcap_dirs_path)
    fs_path = fs_path.rstrip("/")

    # Determine protocol prefix (e.g. "s3://")
    protocol = fs.protocol
    if isinstance(protocol, (list, tuple)):
        protocol = protocol[0]
    protocol_prefix = f"{protocol}://"

    # Recursively list all objects under the path
    all_files = fs.find(fs_path)

    # Collect parent paths of .mcap files
    # HACK(mark.zolotas): skipping anything under "needs_post_processing"
    # Should remove this filter once these directories become irrelevant to
    # separate post-processed logs from the original raw logs
    episode_paths = {
        f"{protocol_prefix}{os.path.dirname(p)}"
        for p in all_files
        if p.lower().endswith(".mcap") and "/needs_post_processing/" not in p
    }

    if not episode_paths:
        return []

    # Validate episode
    validation_futures = [check_mcap_episode_validity_ray.remote(ep_path) for ep_path in episode_paths]
    validation_results = ray.get(validation_futures)

    # Filter out None results (invalid episodes)
    valid_episodes = [ep for ep in validation_results if ep is not None]

    return valid_episodes


class MCAPConverter(BaseRoboticsConverter):
    """
    Converter to .tar for robot logs stored in the MCAP file format.

    The converter handles:
      - Multi-camera RGB images
      - Low-dimensional state and actions
      - Temporal alignment of topics (nearest-neighbor to pivot source timeline)
      - Episode metadata and language instructions
    """

    VALID_DOMAINS = {"sim", "real"}
    # TODO(mark.zolotas): Processing a "filtered" folder is a temporary solution
    # to handle our QAed data, but this will be deprecated once we have a better
    # solution of querying QAed data based on the updated episode metadata
    VALID_SOURCES = {"teleop", "filtered"}

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        # Load language annotations
        print("📚 Loading language annotations...")
        with open(cfg.language_annotations_path) as f:
            data = yaml.safe_load(f)
        self.language_annotations = data.get("language_dict", {})
        print(f"Loaded language annotations for {len(self.language_annotations)} tasks")

        # Load MCAP-specific settings from topics config
        if not getattr(cfg, "topics_to_fields_path", None):
            raise ValueError("No `topics_to_fields_path` specified in cfg")
        if not file_exists(cfg.topics_to_fields_path):
            raise FileNotFoundError(f"Topics config not found: {cfg.topics_to_fields_path}")

        topics_cfg = yaml_load(cfg.topics_to_fields_path)

        self._init_topics(topics_cfg)
        self._init_field_mappings(topics_cfg)
        self._init_action_fields(cfg)
        self._validate_filters(cfg)

        # Image indices for temporal window
        self.image_indices = getattr(cfg, "image_indices", [0])

        print(f"🧭 MCAP Converter initialized: {self.output_mode} mode, pivot_source_field={self.pivot_source_field}")
        print(f"🧭 Action topics: {self.action_topics}")
        print(f"🧭 State topics: {self.state_topics}")
        print(f"🧭 Camera topics: {self.camera_topics}")

    def _init_topics(self, topics_cfg):
        # Only two supported output modes:
        # - 'separate': Returns dict with individual field keys
        # - 'concatenated': Returns dict with 'state' and 'actions' keys
        self.output_mode = topics_cfg.get("output_mode", "separate")
        if self.output_mode not in ("separate", "concatenated"):
            raise ValueError(f"output_mode must be 'separate' or 'concatenated', got: {self.output_mode}")

        # Pivot source for temporal alignment; must be a field name from
        # camera_topics_field_map, state_field_map, or action_field_map.
        self.pivot_source_field = topics_cfg.get("pivot_source_field", None)
        if not self.pivot_source_field:
            raise ValueError("pivot_source_field must be specified in topics config")
        # Temporal alignment thresholds (seconds)
        self.gap_threshold_dt = topics_cfg.get("gap_threshold_dt", 0.1)
        self.snap_threshold_dt = topics_cfg.get("snap_threshold_dt", 0.1)

        self.action_topics = topics_cfg.get("action_topics", [])

        self.action_topics = topics_cfg.get("action_topics", [])
        self.state_topics = topics_cfg.get("state_topics", [])
        self.camera_topics = topics_cfg.get("camera_topics", {})

        # Validate topic configurations
        for name, topics in [
            ("action_topics", self.action_topics),
            ("state_topics", self.state_topics),
            ("camera_topics", self.camera_topics),
        ]:
            if topics and not all(isinstance(t, str) and t for t in topics):
                raise ValueError(f"{name} must be a list of non-empty strings")

    def _init_field_mappings(self, topics_cfg):
        self.state_topic_subfield_extraction = topics_cfg.get("state_topic_subfield_extraction", {})
        self.reference_field_prefixes = topics_cfg.get("reference_field_prefixes", {})

        self.action_field_map = topics_cfg.get("action_field_map", {})
        self.state_field_map = topics_cfg.get("state_field_map", {})
        self.state_topic_subfield_map = topics_cfg.get("state_topic_subfield_map", {})
        self.camera_topics_field_map = topics_cfg.get("camera_topics_field_map", {})

        self.state_key_fields = topics_cfg.get("state_key_fields", [])
        if self.output_mode == "concatenated" and not self.state_key_fields:
            raise ValueError("state_key_fields must be provided when output_mode is 'concatenated'")

    def _init_action_fields(self, cfg):
        print("📘 Loading action field configuration...")
        action_field_config = load_action_field_config(cfg.action_fields_config_path)
        self.action_key_fields = action_field_config["action_key_fields"]
        self.action_index_fields = action_field_config["action_index_fields"]

        # Load and validate pose groups for relative coordinate computation
        if "pose_groups" not in action_field_config:
            raise ValueError(
                "pose_groups not found in action field config. "
                "Please add pose_groups to enable relative coordinate computation."
            )
        self.pose_groups = action_field_config["pose_groups"]
        validate_pose_groups(self.pose_groups)

        # Compute per-field sizes from cumulative index boundaries
        self.action_field_sizes = []
        prev_idx = 0
        for key, cumulative_idx in zip(self.action_key_fields, self.action_index_fields, strict=True):
            size = cumulative_idx - prev_idx
            if size <= 0:
                raise ValueError(
                    f"Action field indices must be strictly increasing. Field '{key}' produced size {size}."
                )
            self.action_field_sizes.append(size)
            prev_idx = cumulative_idx

        print(f"Loaded {len(self.action_key_fields)} action fields, {len(self.pose_groups)} pose groups")
        if self.action_field_sizes:
            slices = [
                f"{name} (dim={size})"
                for name, size in zip(self.action_key_fields, self.action_field_sizes, strict=True)
            ]
            print(f"🧭 Action field slices: {slices}")

    def _validate_filters(self, cfg):
        """Validate episode filter values at init time."""
        if getattr(cfg, "domain_filter", None):
            invalid = set(cfg.domain_filter) - self.VALID_DOMAINS
            if invalid:
                raise ValueError(f"Invalid domain_filter values: {invalid}. Must be one of {self.VALID_DOMAINS}")

        if getattr(cfg, "source_filter", None):
            invalid = set(cfg.source_filter) - self.VALID_SOURCES
            if invalid:
                raise ValueError(f"Invalid source_filter values: {invalid}. Must be one of {self.VALID_SOURCES}")

    def _filter_episodes(self, episodes: list[str]) -> list[str]:
        """Apply filters to discovered episode paths."""
        source_filter = getattr(self.cfg, "source_filter", None)
        task_filter = getattr(self.cfg, "task_filter", None)
        domain_filter = getattr(self.cfg, "domain_filter", None)

        # No filters active, return as-is
        if not any([source_filter, task_filter, domain_filter]):
            return episodes

        filtered = []
        for ep in episodes:
            components = parse_episode_path(ep)
            if components is None:
                print(f"⚠️  Could not parse path components for filtering, skipping: {ep}")
                continue
            if source_filter and components["source"] not in source_filter:
                continue
            if task_filter and components["task"] not in task_filter:
                continue
            if domain_filter and components["domain"] not in domain_filter:
                continue
            filtered.append(ep)

        print(
            f"🔍 Filtered from {len(episodes)} to {len(filtered)} episodes "
            f"(source={source_filter}, task={task_filter}, domain={domain_filter})"
        )
        return filtered

    def get_output_subdir(self) -> str:
        """
        Build output subdirectories from active filters.

        Naming conventions:
        - task: single value used directly, multiple/no filter becomes "multitask"
        - domain: single value used directly, multiple/no filter joins all valid domains
        - source: single value used directly, multiple/no filter joins all valid sources
        """
        parts = []
        task_filter = getattr(self.cfg, "task_filter", None)
        domain_filter = getattr(self.cfg, "domain_filter", None)
        source_filter = getattr(self.cfg, "source_filter", None)

        # Task
        if task_filter and len(task_filter) == 1:
            parts.append(task_filter[0])
        else:
            parts.append("multitask")

        # Domain (sorting to ensure deterministic paths)
        if domain_filter and len(domain_filter) == 1:
            parts.append(domain_filter[0])
        else:
            parts.append("_and_".join(sorted(domain_filter or self.VALID_DOMAINS)))

        # Source (sorting to ensure deterministic paths)
        if source_filter and len(source_filter) == 1:
            parts.append(source_filter[0])
        else:
            parts.append("_and_".join(sorted(source_filter or self.VALID_SOURCES)))

        return "/".join(parts)

    def discover_episodes(self, source_paths: list[str], max_episodes_to_process: int = -1) -> list[str]:
        """
        Discover MCAP episodes from source paths.

        Args:
            source_paths: List of paths to MCAP files or directories
            max_episodes_to_process: Maximum number of episodes to return (-1 for all)
        Returns:
            List of episode paths (MCAP files or directories containing MCAP files)
        """
        if isinstance(source_paths, str):
            source_paths = [source_paths]

        # Discover and validate episodes in parallel using Ray
        discover_futures = [
            discover_and_validate_mcap_episodes_in_directory.remote(dir_path) for dir_path in source_paths
        ]
        discover_results = ray.get(discover_futures)

        # Merge results from all directories
        all_episodes = []
        for result in discover_results:
            all_episodes.extend(result)

        # Apply path-based filters (task, domain, source, episode)
        all_episodes = self._filter_episodes(all_episodes)

        # Sorted for deterministic processing order across runs
        all_episodes = sorted(all_episodes)

        # Apply max_episodes_to_process limit if specified
        if max_episodes_to_process > 0 and len(all_episodes) > max_episodes_to_process:
            all_episodes = all_episodes[:max_episodes_to_process]

        print(f"Total episodes discovered: {len(all_episodes)}")
        return sorted(all_episodes)

    def load_episode_data(self, episode_path: str) -> dict[str, Any]:
        """
        Load and align MCAP episode to a common frequency across channels.

        Args:
            episode_path: Path to MCAP file

        Returns:
            Dictionary containing:
            - Episode metadata
            - Time-aligned state & camera images (JPEG bytes or numpy arrays)
            - Time-aligned action data
            - Resampled timestamps
        """
        dir_name = os.path.basename(episode_path.rstrip("/"))
        # TODO(mark.zolotas): Assumption of single `_0`.mcap might need to change
        # in the near future, especially if we version mcaps
        mcap_file = os.path.join(episode_path, f"{dir_name}_0.mcap")
        print(f"Loading MCAP: {mcap_file}")

        # Extract raw topic data from MCAP (handles S3 or local path)
        if episode_path.startswith("s3://"):
            # copy_to_temp_file is a context manager that auto-cleans up
            with copy_to_temp_file(mcap_file) as local_mcap_path:
                raw = self._extract_raw_data(local_mcap_path)
        else:
            raw = self._extract_raw_data(mcap_file)

        if raw is None:
            return {}

        time_aligned = self._align_to_pivot_source(raw)
        metadata = self._load_episode_metadata(episode_path)

        return {
            "metadata": metadata,
            "observations": time_aligned["observations"],
            "actions": time_aligned["actions"],
            "timestamps": time_aligned["timestamps"],
        }

    def _extract_raw_data(self, mcap_path: str) -> dict[str, Any] | None:
        """
        Read an MCAP file and extract raw per-topic data and timestamps.

        Returns:
            Dict with keys {state_data, action_data, camera_data, per_topic_log_timestamps,
            per_topic_sensor_timestamps, t_min, t_max}, or None if extraction fails.
        """
        state_data = defaultdict(list)
        action_data = defaultdict(list)
        camera_data = defaultdict(list)
        per_topic_log_timestamps = defaultdict(list)
        per_topic_sensor_timestamps = defaultdict(list)
        # To later be used to set the episode relative target timeline
        t_min, t_max = float("inf"), float("-inf")

        with AnyReader([Path(mcap_path)]) as reader:
            for connection, t, rawdata in reader.messages():
                t_sec = t * 1e-9
                t_min = min(t_min, t_sec)
                t_max = max(t_max, t_sec)
                topic = connection.topic

                try:
                    msg = reader.deserialize(rawdata, connection.msgtype)

                    # Extract message header timestamp if available
                    msg_t_sec = None
                    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                        stamp = msg.header.stamp
                        msg_t_sec = stamp.sec + stamp.nanosec * 1e-9

                    is_camera = topic in self.camera_topics_field_map
                    is_state = topic in self.state_field_map
                    is_action = topic in self.action_field_map

                    if is_camera:
                        img = extract_image_from_msg(msg, return_numpy=True)
                        field = self.camera_topics_field_map[topic]
                        camera_data[field].append(img)
                        per_topic_log_timestamps[field].append(t_sec)
                        if msg_t_sec is not None:
                            per_topic_sensor_timestamps[field].append(msg_t_sec)

                    if is_state or is_action:
                        # Config-driven sub-field extraction (e.g., /dex3/* topics)
                        if topic in self.state_topic_subfield_extraction:
                            for sub_key, path in self.state_topic_subfield_extraction[topic].items():
                                val = extract_field_path(msg, path)
                                if val is not None:
                                    virtual_topic = f"{topic}.{sub_key}"
                                    final_name = self.state_topic_subfield_map.get(virtual_topic, virtual_topic)
                                    state_data[final_name].append(val)
                                    per_topic_log_timestamps[final_name].append(t_sec)
                                    if msg_t_sec is not None:
                                        per_topic_sensor_timestamps[final_name].append(msg_t_sec)
                        # Structured messages (Pose/JointState -> __xyz, __rot_6d)
                        else:
                            base_name = self.action_field_map.get(topic, self.state_field_map.get(topic, topic))
                            target = action_data if is_action else state_data
                            structured_data = extract_structured_msg(msg)
                            if structured_data:
                                for suffix, val in structured_data.items():
                                    final_name = f"{base_name}{suffix}"
                                    target[final_name].append(val)
                                    per_topic_log_timestamps[final_name].append(t_sec)
                                    if msg_t_sec is not None:
                                        per_topic_sensor_timestamps[final_name].append(msg_t_sec)
                            else:
                                arr = extract_array_from_msg(msg)
                                if arr is not None:
                                    target[base_name].append(arr)
                                    per_topic_log_timestamps[base_name].append(t_sec)
                                    if msg_t_sec is not None:
                                        per_topic_sensor_timestamps[base_name].append(msg_t_sec)
                except Exception as e:
                    print(f"⚠️  Skipping message on {topic}: {e}")
                    continue

        if not per_topic_log_timestamps:
            print(f"❌ No data extracted from: {mcap_path}")
            return None

        # Sanity check: ensure we actually converted some messages, not just
        # recorded timestamps while every extraction silently failed/was skipped.
        has_any_data = any(len(v) > 0 for v in (*state_data.values(), *action_data.values(), *camera_data.values()))
        if not has_any_data:
            print(
                f"❌ Messages were read from {mcap_path} but all extractions failed. "
                f"Topics with timestamps: {list(per_topic_log_timestamps.keys())}"
            )
            return None

        return {
            "state_data": state_data,
            "action_data": action_data,
            "camera_data": camera_data,
            "per_topic_log_timestamps": per_topic_log_timestamps,
            "per_topic_sensor_timestamps": per_topic_sensor_timestamps,
            "t_min": t_min,
            "t_max": t_max,
        }

    def _align_to_pivot_source(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Align all channels to a pivot source's timestamps via nearest-neighbor.

        Uses the pivot source's message header timestamps as the ground-truth
        timeline. Every other channel (cameras, state, actions) is snapped to
        the nearest sample.

        Args:
            raw: Output from _extract_raw_data

        Returns:
            Dict with keys {observations, actions, timestamps}
        """
        state_data = raw["state_data"]
        action_data = raw["action_data"]
        camera_data = raw["camera_data"]
        per_topic_log_timestamps = raw["per_topic_log_timestamps"]
        per_topic_sensor_timestamps = raw["per_topic_sensor_timestamps"]

        # Convert to episode-relative time
        log_t0 = raw["t_min"]
        per_topic_log_relative_time = {k: (np.asarray(v) - log_t0) for k, v in per_topic_log_timestamps.items()}
        per_topic_sensor_relative_time = {k: (np.asarray(v) - log_t0) for k, v in per_topic_sensor_timestamps.items()}

        # Build per-field best-available timestamps, preferring sensor (header) stamps
        best_timestamps = {}
        for field in per_topic_log_relative_time:
            log_ts = per_topic_log_relative_time[field]
            sensor_ts = per_topic_sensor_relative_time.get(field)

            if sensor_ts is not None and len(sensor_ts) == len(log_ts):
                best_timestamps[field] = sensor_ts
            else:
                # Fall back to log (MCAP connection) timestamps when headers are missing or incomplete
                if sensor_ts is not None and len(sensor_ts) != len(log_ts):
                    raise ValueError(
                        f"❌  [{field}] sensor timestamp count mismatch "
                        f"(log={len(log_ts)}, sensor={len(sensor_ts)}), "
                        f"falling back to log timestamps"
                    )
                best_timestamps[field] = log_ts

        # Ensure all timestamp arrays are sorted (MCAP iteration order should
        # guarantee this, but multi-publisher topics could interleave)
        for field, ts in best_timestamps.items():
            if len(ts) > 1 and not np.all(np.diff(ts) >= 0):
                sort_idx = np.argsort(ts)
                best_timestamps[field] = ts[sort_idx]
                print(f"⚠️  [{field}] timestamps were not sorted, reordered {len(ts)} entries")
                # Reorder corresponding data to match
                if field in camera_data:
                    camera_data[field] = [camera_data[field][i] for i in sort_idx]
                elif field in state_data:
                    state_data[field] = [state_data[field][i] for i in sort_idx]
                elif field in action_data:
                    action_data[field] = [action_data[field][i] for i in sort_idx]

        # Build target timeline from pivot source
        all_fields = set(camera_data.keys()) | set(state_data.keys()) | set(action_data.keys())
        if self.pivot_source_field not in best_timestamps:
            raise ValueError(
                f"❌ pivot_source_field '{self.pivot_source_field}' not found in extracted fields: {sorted(all_fields)}"
            )

        target_timeline = best_timestamps[self.pivot_source_field].astype(np.float32).copy()
        print(f"{len(target_timeline)} frames over {target_timeline[-1] - target_timeline[0]:.2f}s")

        # Sanity check: flag gaps in pivot timeline (e.g., dropped camera frames)
        if len(target_timeline) > 1:
            dts = np.diff(target_timeline)
            expected_dt = np.median(dts)
            gap_mask = dts > self.gap_threshold_dt
            if gap_mask.any():
                raise ValueError(
                    f"❌  {gap_mask.sum()} gaps in pivot timeline exceed "
                    f"{self.gap_threshold_dt:.4f}s. "
                    f"Largest: {dts.max():.4f}s at index {dts.argmax()}. "
                    f"Median dt: {expected_dt:.4f}s"
                )

        # Assume sensor header stamps and .mcap log stamps share the same
        # time base (both episode-relative). Sanity check large drift based on
        # whether a field's time range has no overlap with the pivot.
        pivot_ts = best_timestamps[self.pivot_source_field]
        pivot_start, pivot_end = float(pivot_ts[0]), float(pivot_ts[-1])
        margin = max(pivot_end - pivot_start, 1.0) * 0.1
        for field, ts in best_timestamps.items():
            if field == self.pivot_source_field or len(ts) == 0:
                continue
            field_start, field_end = float(ts[0]), float(ts[-1])
            if field_end < pivot_start - margin or field_start > pivot_end + margin:
                raise ValueError(
                    f"❌  Likely clock mismatch: '{field}' time range "
                    f"[{field_start:.2f}s, {field_end:.2f}s] has no overlap with "
                    f"pivot source '{self.pivot_source_field}' range "
                    f"[{pivot_start:.2f}s, {pivot_end:.2f}s]. "
                    f"All channels are assumed to share the same time base."
                )

        # Stack low-dim data
        state_data = {k: np.stack(v) for k, v in state_data.items()}
        action_data = {k: np.stack(v) for k, v in action_data.items()}

        # Validate all action key fields are present before concatenation
        missing_action_fields = [k for k in self.action_key_fields if k not in action_data]
        if missing_action_fields:
            raise ValueError(
                f"Missing action fields after extraction: {missing_action_fields}. "
                f"Available: {list(action_data.keys())}"
            )

        # Nearest-neighbor align low-dim data with snap-distance gating
        for data in (state_data, action_data):
            for k, v in data.items():
                source_ts = best_timestamps[k]
                idx = nearest_indices(source_ts, target_timeline, max_snap_distance=self.snap_threshold_dt)
                data[k] = v[idx].astype(np.float32)

        # Nearest-neighbor align images
        for k, imgs in camera_data.items():
            idx = nearest_indices(best_timestamps[k], target_timeline, max_snap_distance=self.snap_threshold_dt)
            camera_data[k] = np.stack([imgs[i] for i in idx], axis=0)

        # Merge state and camera into observations
        observations = {**state_data, **camera_data}

        # Reorder actions to match action_key_fields and concatenate
        actions = {
            "actions": np.concatenate([action_data[k] for k in self.action_key_fields], axis=1).astype(np.float32)
        }

        duration = target_timeline[-1] - target_timeline[0]
        effective_hz = (len(target_timeline) - 1) / duration if duration > 0 else 0.0
        print(f"Episode duration: {duration:.2f}s, {len(target_timeline)} timesteps at ~{effective_hz:.1f}Hz")

        return {
            "observations": observations,
            "actions": actions,
            "timestamps": target_timeline,
        }

    def _load_episode_metadata(self, episode_path: str) -> dict[str, Any]:
        """Load episode metadata from info.yaml in the episode directory."""
        # TODO(mark.zolotas): account for versioning of metadata files
        metadata_path = os.path.join(episode_path.rstrip("/"), "info.yaml")
        try:
            with fsspec.open(metadata_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  Could not load metadata from {metadata_path}: {e}")
            return {"episode_id": os.path.basename(episode_path.rstrip("/"))}

    def get_episode_length(self, episode_data: dict[str, Any]) -> int:
        """Get episode length from episode data."""
        return len(episode_data["timestamps"])

    def extract_camera_data(self, episode_data: dict[str, Any]) -> dict[str, np.ndarray]:
        """Extract camera data and return a dict of camera names to RGB images."""
        result = {}
        camera_names = self.cfg.camera_names
        if not camera_names:
            # Attempt to automatically extract camera names from topic-camera mapping
            if not self.camera_topics_field_map:
                raise ValueError(
                    "No camera names can be extracted without either specifying "
                    "then in the cfg or populating camera_topics_to_fields "
                )

            camera_names = list(self.camera_topics_field_map.values())
            if len(camera_names) != len(set(camera_names)):
                raise ValueError("camera_topics_fields_map must be one-to-one: duplicate values found")

        for cname in camera_names:
            # Add RGB image
            if cname in episode_data["observations"]:
                result[cname] = episode_data["observations"][cname]

        return result

    def extract_lowdim_data(self, episode_data: dict[str, Any]) -> dict[str, np.ndarray]:
        """
        Extract lowdim data for ALL timesteps.

        Two output modes:
        - 'concatenated': Returns dict with 'state' and 'actions' keys
        - 'separate': Returns dict with individual field keys

        Args:
            episode_data: Episode data dictionary

        Returns:
            Dictionary with lowdim data arrays of shape [T, dim]
        """
        result = {}
        if self.output_mode == "concatenated":
            if episode_data["observations"]:
                state_components = []
                # Ensure we match the keys exactly as they exist in episode_data
                for key in self.state_key_fields:
                    if key in episode_data["observations"]:
                        data = episode_data["observations"][key]
                        state_components.append(data if data.ndim > 1 else data[:, np.newaxis])
                if state_components:
                    result["state"] = np.concatenate(state_components, axis=1).astype(np.float32)

            # Actions already concatenated
            if episode_data["actions"] and "actions" in episode_data["actions"]:
                result["actions"] = next(iter(episode_data["actions"].values()))
        else:
            # Extract low-dimensional observations
            if episode_data["observations"]:
                # When state_key_fields is set, use it to filter to a consistent
                # subset across heterogeneous sources (e.g., sim and real)
                allowed_state_keys = set(self.state_key_fields) if self.state_key_fields else None
                result.update(
                    {
                        key: value
                        for key, value in episode_data["observations"].items()
                        if key.startswith("language_")
                        or (len(value.shape) <= 2 and (allowed_state_keys is None or key in allowed_state_keys))
                    }
                )

            # Extract 'actions' if available
            if episode_data["actions"] and "actions" in episode_data["actions"] and self.action_key_fields:
                total_action_dim = episode_data["actions"]["actions"].shape[1]
                expected_action_dim = self.action_index_fields[-1]
                if total_action_dim < expected_action_dim:
                    raise ValueError(
                        "Action tensor has insufficient dimension.\n "
                        f"Expected at least {expected_action_dim}, got {total_action_dim}."
                    )

                prev_index = 0
                for key, index in zip(self.action_key_fields, self.action_index_fields, strict=False):
                    result[key] = episode_data["actions"]["actions"][:, prev_index:index]
                    prev_index = index

                if prev_index != expected_action_dim:
                    raise ValueError(
                        "Action slicing did not consume expected dimensions. "
                        f"Expected {expected_action_dim}, consumed {prev_index}."
                    )
            elif self.action_key_fields:
                raise ValueError(
                    "Configured action fields but no actions were found in episode data. "
                    "Ensure actions.npz is present for each episode."
                )

        return result

    def extract_intrinsics_extrinsics_data(self, episode_data: dict[str, Any]) -> tuple[dict | None, dict | None]:
        """
        Extract camera intrinsics/extrinsics if available.

        Args:
            episode_data: Episode data dictionary

        Returns:
            Tuple of (intrinsics_dict, extrinsics_dict) or (None, None)
        """
        return None, None

    def extract_metadata_data(self, episode_data: dict[str, Any]) -> dict:
        """
        Extract metadata.

        Args:
            episode_data: Episode data dictionary

        Returns:
            Metadata dictionary including timestamps
        """
        return {
            **episode_data.get("metadata", {}),
            "timestamps": episode_data["timestamps"],
        }

    def _any_field_to_actual_key(self, field: str) -> str:
        """
        Generic mapping of target fields to observation fields using config-driven prefixes.
        """
        for action_prefix, obs_prefix in self.reference_field_prefixes.items():
            if field.startswith(action_prefix):
                return obs_prefix + field[len(action_prefix) :]

        return field

    def get_language_instructions(self, episode_path: str, instruction_types: list[str] = None) -> dict[str, list[str]]:
        """Get language instructions for a given task, organized by type.

        Args:
            episode_path: Path to the episode
            instruction_types: List of instruction types to include. If None, includes all types.
                            Valid types: "original", "randomized", "verbose", "alternative"

        Returns:
            Dictionary mapping instruction type to list of instructions
        """
        components = parse_episode_path(episode_path)
        if components is None:
            print(f"⚠️  Could not parse task name from episode path: {episode_path}")
            return {}

        task_name = components.get("task")
        if task_name not in self.language_annotations:
            return {}

        task_annotations = self.language_annotations[task_name]
        instructions_by_type = {}

        # Default to all types if none specified
        if instruction_types is None:
            instruction_types = ["original", "randomized", "verbose", "alternative"]

        # Add instructions for each requested type
        for instruction_type in instruction_types:
            if instruction_type in task_annotations:
                instructions_by_type[instruction_type] = task_annotations[instruction_type]

        return instructions_by_type

    def extract_sample_data(
        self,
        anchor_timestep: int,
        episode_path: str,
        episode_length: int,
        camera_data: dict[str, list[bytes]],
        lowdim_data: dict[str, np.ndarray],
        intrinsics_data: dict[str, Any],
        extrinsics_data: dict[str, Any],
        metadata_data: dict[str, Any],
        statistics_ray_actor,
        logger_actor,
    ) -> tuple[dict | None, dict | None, SampleMetadata | None, dict | None, NDArray | None, dict | None, dict | None]:
        """
        Extract sample data for a single timestep with temporal windowing and stillness filtering.

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
            Tuple of (sample_images, sample_lowdim, sample_metadata, language_instructions, *extra).
            We don't yet process camera intrinsics/extrinsics so the sample_point_clouds (point clouds
            from depth & RGB images) and sample_point_maps (camera coordinates in image space) in *extra
            are returned as None. However, de do return sample_stats if there is a statistics ray actor.
        """
        logger_actor.increment_total_potential_samples.remote()

        # Calculate windows
        lowdim_start = anchor_timestep - self.cfg.past_lowdim_steps
        lowdim_end = anchor_timestep + self.cfg.future_lowdim_steps

        # Check padding
        past_padding = max(0, -lowdim_start)
        future_padding = max(0, lowdim_end - episode_length + 1)

        if past_padding > self.cfg.max_padding_left or future_padding > self.cfg.max_padding_right:
            logger_actor.increment_padding_samples_filtered.remote()
            return None, None, None, None, None, None, None

        valid_start = max(0, lowdim_start)
        valid_end = min(episode_length - 1, lowdim_end)

        # Check if robot is stationary (e.g. to filter pauses)
        if self.cfg.filter_still_samples and is_still_sample(
            lowdim_data, valid_start, valid_end, self.cfg.still_threshold
        ):
            logger_actor.increment_still_samples_filtered.remote()
            return None, None, None, None, None, None, None

        # Extract images
        sample_images = {}
        actual_image_timesteps = []
        for img_offset in self.cfg.image_indices:
            img_timestep = np.clip(anchor_timestep + img_offset, 0, episode_length - 1)
            actual_image_timesteps.append(int(img_timestep))

            for camera_name, camera_images in camera_data.items():
                key = f"{camera_name}_t{img_offset}"
                sample_images[key] = camera_images[img_timestep]

        # Process lowdim data (which includes actions)
        sample_lowdim = {}
        reference_data = {}
        for key, data in lowdim_data.items():
            valid_data = data[valid_start : valid_end + 1]
            if past_padding > 0 or future_padding > 0:
                valid_data = self.pad_fn(valid_data, past_padding, future_padding)
            sample_lowdim[key] = valid_data
            actual_key = self._any_field_to_actual_key(key)
            if actual_key is not None and actual_key in lowdim_data:
                reference_data[key] = lowdim_data[actual_key][anchor_timestep]

        # Add relative lowdim data wrt the actual position at the current timestep
        # Doesn't work for concatenated mode yet as needs action_index_fields organization
        if self.output_mode != "concatenated":
            sample_lowdim_relative = self.create_relative_lowdim_data(sample_lowdim, reference_data)
            sample_lowdim.update(sample_lowdim_relative)

        # Create masks
        past_mask, future_mask = create_past_and_future_masks(
            anchor_timestep, self.cfg.past_lowdim_steps, self.cfg.future_lowdim_steps, episode_length
        )

        # Create metadata
        episode_id = self.get_episode_id(episode_path)
        sample_metadata = SampleMetadata(
            episode_id=episode_id,
            sample_id=f"{uuid.uuid4()}_{episode_id}_t{anchor_timestep:04d}",
            anchor_timestep=int(anchor_timestep),
            anchor_episode_timestamp=metadata_data["timestamps"][anchor_timestep],
            anchor_relative_idx=int(self.cfg.past_lowdim_steps),
            image_timesteps=actual_image_timesteps,
            lowdim_start_timestep=int(lowdim_start),
            lowdim_end_timestep=int(lowdim_end),
            past_padding=int(past_padding),
            future_padding=int(future_padding),
            camera_names=list(camera_data.keys()),
            original_episode_length=int(episode_length),
            original_episode_duration_s=metadata_data.get("episode_duration"),
            original_image_sizes={},  # Filled by upload_sample_to_s3
            is_padded=bool(past_padding > 0 or future_padding > 0),
            task_name=metadata_data.get("skill"),
            teleop_or_rollout=metadata_data.get("teleop_or_rollout"),
            robot=metadata_data.get("robot"),
            station_name=metadata_data.get("station_name"),
            domain=metadata_data.get("domain"),
        )

        # Build stats_sample for batched statistics update (don't send immediately)
        stats_sample = None
        if statistics_ray_actor is not None:
            configured_keys = set(self.action_key_fields) | set(self.state_key_fields or [])
            # Also include relative variants of any configured key that were
            # produced by create_relative_lowdim_data and are present in sample_lowdim
            relative_keys = {f"{k}_relative" for k in configured_keys}
            stats_keys = configured_keys | (relative_keys & sample_lowdim.keys())
            # Construct stats from sample only for configured data fields
            # Important when e.g., handling different domains (sim / real),
            # that may have a different number of lowdim data fields
            stats_sample = {
                "lowdim": {k: v.copy() for k, v in sample_lowdim.items() if k in stats_keys},
                "past_mask": past_mask,
                "future_mask": future_mask,
            }

        # Add past_mask, future_mask to lowdim (after building stats_sample)
        sample_lowdim["past_mask"] = past_mask
        sample_lowdim["future_mask"] = future_mask

        # Language instructions with only `original` instructions for now
        language_instructions = self.get_language_instructions(episode_path)

        return sample_images, sample_lowdim, sample_metadata, language_instructions, None, None, stats_sample
