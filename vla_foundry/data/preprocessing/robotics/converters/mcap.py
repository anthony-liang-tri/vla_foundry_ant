"""
MCAP Converter for VLA Foundry

Converts MCAP ROS2 recordings to preprocessed samples. Uses rosbags for
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
    Any message with field_extraction config in topics YAML.
    Used for: Dex3 tactile, lowstate, PolicyKeyframe, etc.

IMAGE EXTRACTION (extract_image_from_msg -> numpy):
    sensor_msgs/CompressedImage (jpeg, png)
    sensor_msgs/Image (rgb8, bgr8, mono8)
    Depth images (16UC1, 32FC1) skipped

Usage
-----
uv run python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \\
    --type mcap \\
    --source_episodes "['<path>/']" \\
    --output_dir s3://bucket/path \\
    --camera_names "include config.yaml" \\
    --action_fields_config_path action_topics.yaml \\
    --config_path config_path.yaml
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

from vla_foundry.data.preprocessing.robotics.converters.base import BaseRoboticsConverter
from vla_foundry.data.preprocessing.robotics.preprocess_masks import create_past_and_future_masks
from vla_foundry.data.preprocessing.utils import is_still_sample
from vla_foundry.data.robotics.utils import matrix_to_rot_6d
from vla_foundry.file_utils import file_exists, is_dir, list_directory_recursive, yaml_load

logger = logging.getLogger(__name__)

# Minimum number of samples required to compute reliable median for anti-aliasing filter
MIN_SAMPLES_FOR_ANTIALIASING = 8
MIN_EPISODE_DURATION = 0.001


def extract_image_from_msg(msg: Any, return_numpy: bool = True) -> Union[bytes, np.ndarray]:
    """
    Extract image data from ROS2 image message. Raises exceptions for unsupported formats.

    Args:
        msg: ROS2 image message (CompressedImage or Image)
        return_numpy: If True, return RGB numpy array. If False, return JPEG bytes.

    Returns:
        RGB uint8 numpy array (H, W, 3) or JPEG bytes.
    """
    # 1. Handle CompressedImage
    if hasattr(msg, "format") and hasattr(msg, "data") and not hasattr(msg, "height"):
        fmt = msg.format.lower()
        if "jpeg" in fmt or "jpg" in fmt or "png" in fmt:
            img = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode {fmt.upper()} CompressedImage")

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if return_numpy:
                return rgb_img

            success, jpeg_data = cv2.imencode(".jpg", rgb_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise ValueError("Failed to re-encode image to JPEG")
            return jpeg_data.tobytes()

        raise ValueError(f"Unsupported CompressedImage format: {fmt}")

    # 2. Handle Raw Image
    if not all(hasattr(msg, a) for a in ("height", "width", "encoding", "data")):
        raise TypeError(f"Message is not a valid Image or CompressedImage: {type(msg)}")

    encoding = msg.encoding.lower()
    if "uc1" in encoding or "fc1" in encoding:
        raise ValueError(f"Depth images ({encoding}) are not supported in the RGB pipeline.")

    if "rgb8" in encoding:
        img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    elif "bgr8" in encoding:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif "mono8" in encoding or "gray" in encoding or encoding == "8uc1":
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        img_array = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unsupported raw image encoding: {encoding}")

    if return_numpy:
        return img_array

    success, jpeg_data = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise ValueError("Failed to encode raw image to JPEG")
    return jpeg_data.tobytes()


def extract_structured_msg(msg: Any) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract structured key-value data with 6D rotation.

    Handles:
    - sensor_msgs/JointState: Returns dict with __<joint_name> keys
    - geometry_msgs/PoseStamped, Pose: Returns __xyz and __rot_6d keys

    Returns:
        Dictionary mapping field names to float32 arrays, or None if not a structured message
    """
    if hasattr(msg, "name") and hasattr(msg, "position") and hasattr(msg, "velocity"):
        if not msg.name:
            return None
        return {f"__{n}": np.asarray([p], dtype=np.float32) for n, p in zip(msg.name, msg.position, strict=True)}

    pose = None
    if hasattr(msg, "pose") and hasattr(msg.pose, "position"):
        pose = msg.pose
    elif hasattr(msg, "position") and hasattr(msg, "orientation") and not hasattr(msg, "velocity"):
        pose = msg

    if pose is not None:
        xyz = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
        quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        rot_matrix = R.from_quat(quat).as_matrix()
        rot_6d = matrix_to_rot_6d(rot_matrix).astype(np.float32)
        return {"__xyz": xyz, "__rot_6d": rot_6d}

    return None


def extract_array_from_msg(msg: Any) -> Optional[np.ndarray]:
    """
    Fallback: extract flat numeric array from ROS2 message using attribute inspection.

    Handles common message types:
    - geometry_msgs/WrenchStamped, Wrench: [force(3), torque(3)]
    - sensor_msgs/Imu, JointState, etc.
    - Custom messages: recursively extracts all numeric fields

    Returns:
        Flattened float32 array or None if extraction fails
    """
    arrays = []
    if hasattr(msg, "wrench") and hasattr(msg.wrench, "force"):
        w = msg.wrench
        arrays.extend([np.array([w.force.x, w.force.y, w.force.z]), np.array([w.torque.x, w.torque.y, w.torque.z])])
    elif hasattr(msg, "force") and hasattr(msg, "torque") and hasattr(msg.force, "x"):
        arrays.extend(
            [np.array([msg.force.x, msg.force.y, msg.force.z]), np.array([msg.torque.x, msg.torque.y, msg.torque.z])]
        )
    else:
        try:
            attrs = [a for a in dir(msg) if not a.startswith("_")]
            for a in attrs:
                val = getattr(msg, a)
                if isinstance(val, (int, float, np.number)):
                    arrays.append(np.array([val]))
                elif isinstance(val, (list, tuple)):
                    try:
                        arr = np.array(val, dtype=np.float32)
                        if arr.size > 0:
                            arrays.append(arr.flatten())
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass
    return np.concatenate(arrays).astype(np.float32) if arrays else None


def extract_field_path(msg: Any, path: str) -> Optional[np.ndarray]:
    """
    Extract nested values using dot-notation path with fail-fast error handling.

    Supports:
    - Simple paths: "temperature"
    - Nested paths: "imu.x"
    - Array iteration: "motor_state[*].q"

    Args:
        msg: ROS2 message
        path: Dot-notation field path

    Returns:
        Float32 array or None if path is invalid

    Raises:
        AttributeError: If field path is invalid (fail-fast pattern for debugging)
    """
    parts = path.split(".")
    obj = msg

    for i, part in enumerate(parts):
        if "[*]" in part:
            field = part.replace("[*]", "")
            items = getattr(obj, field)

            # If there's more path after [*], e.g. "press_sensor_state[*].pressure"
            if i + 1 < len(parts):
                remaining_path = parts[i + 1 :]
                extracted = []
                for item in items:
                    val = item
                    for sub_part in remaining_path:
                        val = getattr(val, sub_part)
                    extracted.append(val)
                return np.array(extracted, dtype=np.float32).flatten()

            # Default behavior for simple iterators (e.g. joints with .q)
            return np.array([getattr(item, "q", item) for item in items], dtype=np.float32).flatten()

        obj = getattr(obj, part)

    return np.array([obj]).astype(np.float32) if np.isscalar(obj) else np.array(obj).astype(np.float32)


class TemporalResampler:
    """
    Resamples multi-rate signals to uniform target frequency with anti-aliasing.

    Supports:
    - Continuous signals (linear interpolation with anti-aliasing filter)
    - Images (nearest-neighbor selection)
    """

    def __init__(self, target_hz: float):
        """
        Initialize resampler.

        Args:
            target_hz: Target sampling frequency in Hz
        """
        self.target_hz = target_hz

    def create_target_timeline(self, start_time: float, end_time: float) -> np.ndarray:
        """
        Create target timeline with truly uniform frequency spacing.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Array of target timestamps
        """
        return np.arange(start_time, end_time, 1.0 / self.target_hz)

    def _apply_antialiasing_filter(self, values: np.ndarray, source_hz: float) -> np.ndarray:
        """
        Apply 4th-order Butterworth low-pass filter to prevent aliasing when downsampling.

        Args:
            values: Source values to filter
            source_hz: Source sampling frequency

        Returns:
            Filtered values
        """
        nyquist = source_hz / 2.0
        cutoff = self.target_hz / 2.0
        norm_cutoff = cutoff / nyquist
        if 0 < norm_cutoff < 1:
            b, a = butter(4, norm_cutoff, btype="low")
            if values.ndim == 1:
                return filtfilt(b, a, values)
            filtered = values.copy()
            for i in range(values.shape[1]):
                filtered[:, i] = filtfilt(b, a, values[:, i])
            return filtered
        return values

    def resample_continuous(
        self, source_times: np.ndarray, source_values: np.ndarray, target_times: np.ndarray, method: str = "linear"
    ) -> np.ndarray:
        """
        Resample continuous signals with Nyquist-aware anti-aliasing.

        Applies anti-aliasing filter when source frequency > 2x target frequency
        to prevent aliasing artifacts during downsampling.

        Args:
            source_times: Source timestamps
            source_values: Source values (1D or 2D)
            target_times: Target timestamps
            method: Interpolation method ('linear', 'cubic', etc.)

        Returns:
            Resampled values at target times
        """
        values = source_values

        # Apply anti-aliasing if downsampling significantly
        if len(source_times) > MIN_SAMPLES_FOR_ANTIALIASING:
            dt_median = np.median(np.diff(source_times))
            if dt_median > 0 and (1.0 / dt_median) > self.target_hz * 2.0:
                values = self._apply_antialiasing_filter(values, 1.0 / dt_median)

        # Interpolate
        if values.ndim == 1:
            return interp1d(source_times, values, kind=method, fill_value="extrapolate")(target_times)

        resampled = np.zeros((len(target_times), values.shape[1]))
        for i in range(values.shape[1]):
            resampled[:, i] = interp1d(source_times, values[:, i], kind=method, fill_value="extrapolate")(target_times)
        return resampled

    def resample_discrete(
        self, source_times: np.ndarray, source_values: np.ndarray, target_times: np.ndarray
    ) -> np.ndarray:
        """
        Resample discrete signal using zero-order hold.

        Args:
            source_times: Source timestamps
            source_values: Source values
            target_times: Target timestamps

        Returns:
            Resampled values (held from previous source value)
        """
        resampled = np.zeros((len(target_times),) + source_values.shape[1:], dtype=source_values.dtype)
        source_idx = 0

        for target_idx, target_time in enumerate(target_times):
            while source_idx < len(source_times) - 1 and source_times[source_idx + 1] <= target_time:
                source_idx += 1
            resampled[target_idx] = source_values[source_idx]

        return resampled

    def resample_images(
        self, source_times: np.ndarray, source_images: List[Any], target_times: np.ndarray
    ) -> List[Any]:
        """
        Temporal resampling for images using nearest-neighbor lookup.

        Args:
            source_times: Source timestamps
            source_images: List of images (bytes or arrays)
            target_times: Target timestamps

        Returns:
            List of resampled images
        """
        return [source_images[np.argmin(np.abs(source_times - t))] for t in target_times]


class MCAPConverter(BaseRoboticsConverter):
    """
    MCAP converter for ROS2 bag files.

    Converts MCAP recordings to VLA Foundry format with:
    - Multi-camera RGB images
    - Low-dimensional state and actions
    - Temporal resampling to uniform frequency
    - Episode metadata and language instructions
    """

    def __init__(self, cfg):
        """
        Initialize MCAP converter.

        Args:
            cfg: Configuration object with preprocessing parameters
        """
        super().__init__(cfg)
        self.cfg = cfg

        # Load topics configuration
        topics_config_path = cfg.action_fields_config_path
        if not file_exists(topics_config_path):
            raise FileNotFoundError(f"Topics config not found: {topics_config_path}")

        topics_cfg = yaml_load(topics_config_path)

        # Parse topics configuration
        self.target_hz = float(topics_cfg.get("target_hz", 30.0))
        self.output_mode = topics_cfg.get("output_mode", "separate")

        # Only two supported output modes:
        # - 'separate': Returns dict with individual field keys
        # - 'concatenated': Returns dict with 'state' and 'actions' keys
        if self.output_mode not in ["separate", "concatenated"]:
            raise ValueError(f"output_mode must be 'separate' or 'concatenated', got: {self.output_mode}")

        self.action_topics = topics_cfg.get("action_topics", [])
        self.state_topics = topics_cfg.get("state_topics", [])
        self.camera_topics = topics_cfg.get("camera_topics", {})

        self.state_field_extraction = topics_cfg.get("state_field_extraction", {})
        self.state_subfield_names = topics_cfg.get("state_subfield_names", {})
        self.reference_field_prefixes = topics_cfg.get("reference_field_prefixes", {})

        # Field name mappings
        self.action_field_names = topics_cfg.get("action_field_names", {})
        self.state_field_names = topics_cfg.get("state_field_names", {})

        # For concatenated mode
        self.action_key_fields = topics_cfg.get("action_key_fields", [])
        self.state_key_fields = topics_cfg.get("state_key_fields", [])

        # Setup resampler
        self.resampler = TemporalResampler(self.target_hz)

        # Image indices for temporal window
        self.image_indices = getattr(cfg, "image_indices", [0])

        # Load pose groups for relative coordinate computation
        self.pose_groups = topics_cfg.get("pose_groups", [])
        if self.pose_groups:
            logger.info(f"Loaded {len(self.pose_groups)} pose groups for relative coordinates")
        logger.info(f"MCAP Converter initialized: {self.target_hz}Hz, {self.output_mode} mode")
        logger.info(f"Action topics: {len(self.action_topics)}, State topics: {len(self.state_topics)}")
        logger.info(f"Camera topics: {len(self.camera_topics)}")

    def discover_episodes(self, source_episode_paths: List[str], max_episodes_to_process: int = -1) -> List[str]:
        """
        Discover MCAP episode files.

        Args:
            source_episode_paths: List of directories or S3 paths containing MCAP files
            max_episodes_to_process: Maximum number of episodes to process (-1 for all)

        Returns:
            List of MCAP file paths
        """
        episode_paths = []

        for source_path in source_episode_paths:
            # 1. Handle case where a single MCAP file is provided directly
            if source_path.endswith(".mcap") and file_exists(source_path):
                episode_paths.append(source_path)
                continue

            # 2. Handle directory (Local or S3)
            if is_dir(source_path):
                # list_directory_recursive handles S3 pagination and local os.walk
                # It returns paths relative to the source_path
                relative_files = list_directory_recursive(source_path)

                # Reconstruct full URIs for any .mcap files found
                base = source_path if source_path.endswith("/") else source_path + "/"
                for rel_f in relative_files:
                    if rel_f.endswith(".mcap"):
                        episode_paths.append(base + rel_f)
            else:
                logger.error(f"Path does not exist or is not a directory: {source_path}")

        # Apply max episodes limit
        if max_episodes_to_process > 0:
            episode_paths = episode_paths[:max_episodes_to_process]

        logger.info(f"Discovered {len(episode_paths)} MCAP episodes")
        return episode_paths

    def load_episode_data(self, episode_path: str) -> Dict[str, Any]:
        """
        Load and resample MCAP episode to target frequency.

        Args:
            episode_path: Path to MCAP file

        Returns:
            Dictionary containing:
            - Resampled camera images (JPEG bytes or numpy arrays)
            - Resampled state/action data
            - Timestamps
            - Episode metadata
        """
        logger.info(f"Loading MCAP: {episode_path}")

        # Handle S3 vs local paths
        if episode_path.startswith("s3://"):
            from vla_foundry.file_utils import copy_to_temp_file

            # copy_to_temp_file is a context manager that auto-cleans up
            with copy_to_temp_file(episode_path) as local_mcap_path:
                return self._process_mcap_file(local_mcap_path, episode_path)
        else:
            return self._process_mcap_file(episode_path, episode_path)

    def _process_mcap_file(self, mcap_path: str, original_path: str) -> Dict[str, Any]:
        """
        Process a local MCAP file and resample to target frequency.

        Args:
            mcap_path: Local path to MCAP file
            original_path: Original path (for metadata)

        Returns:
            Dictionary containing resampled episode data
        """
        # Read MCAP file
        with AnyReader([Path(mcap_path)]) as reader:
            raw_data = defaultdict(lambda: {"timestamps": [], "data": []})

            for connection, timestamp, rawdata in reader.messages():
                msg = reader.deserialize(rawdata, connection.msgtype)
                time_sec = timestamp / 1e9

                topic = connection.topic

                # Process based on topic type
                if topic in self.camera_topics.values():
                    img = extract_image_from_msg(msg, return_numpy=False)
                    if img is not None:
                        raw_data[topic]["timestamps"].append(time_sec)
                        raw_data[topic]["data"].append(img)

                # 2. Process State and Action Topics
                elif topic in self.action_topics or topic in self.state_topics:
                    # Get the base alias (e.g., /ee_target_left -> action_ee_pose_left)
                    base_name = self.action_field_names.get(topic, self.state_field_names.get(topic, topic))

                    # A. Handle Config-Driven Sub-field Extraction (e.g. Dex3 hands)
                    if topic in self.state_field_extraction:
                        for sub_key, path in self.state_field_extraction[topic].items():
                            val = extract_field_path(msg, path)
                            virtual_topic = f"{topic}.{sub_key}"
                            final_name = self.state_subfield_names.get(virtual_topic, virtual_topic)
                            raw_data[final_name]["timestamps"].append(time_sec)
                            raw_data[final_name]["data"].append(val)

                    # B. Handle Structured Messages (Pose/JointState -> __xyz, __rot_6d)
                    else:
                        structured_data = extract_structured_msg(msg)
                        if structured_data:
                            for suffix, val in structured_data.items():
                                final_name = f"{base_name}{suffix}"
                                raw_data[final_name]["timestamps"].append(time_sec)
                                raw_data[final_name]["data"].append(val)
                        else:
                            # C. Fallback to flat array
                            arr = extract_array_from_msg(msg)
                            if arr is not None:
                                raw_data[base_name]["timestamps"].append(time_sec)
                                raw_data[base_name]["data"].append(arr)

        if not raw_data:
            logger.error(f"No data extracted from: {original_path}")
            return {}

        # Determine episode time range
        all_times = []
        for topic_data in raw_data.values():
            all_times.extend(topic_data["timestamps"])

        if not all_times:
            logger.error(f"No timestamps found in: {original_path}")
            return {}

        start_time = min(all_times)
        end_time = max(all_times)
        target_timeline = self.resampler.create_target_timeline(start_time, end_time)

        logger.info(f"Episode duration: {end_time - start_time:.2f}s, Target samples: {len(target_timeline)}")

        # Resample all data to target frequency
        episode_data = {}

        # Resample camera data
        for _camera_name, camera_topic in self.camera_topics.items():
            if camera_topic in raw_data:
                source_times = np.array(raw_data[camera_topic]["timestamps"])
                source_images = raw_data[camera_topic]["data"]

                resampled_images = self.resampler.resample_images(source_times, source_images, target_timeline)

                # Map to camera name
                episode_data[camera_topic] = resampled_images

        # Resample all extracted state/action data
        episode_data = {}
        for field_name, data_info in raw_data.items():
            # Skip cameras (handled via image resampling logic if needed elsewhere)
            if field_name in self.camera_topics.values():
                continue

            src_ts = np.array(data_info["timestamps"])
            src_val = data_info["data"]

            # Sanity check: need at least 2 points to interpolate
            if len(src_ts) < 2:
                continue

            episode_data[field_name] = self.resampler.resample_continuous(src_ts, np.array(src_val), target_timeline)

        # Store timestamps and metadata
        episode_data["timestamps"] = target_timeline
        episode_data["_episode_metadata"] = self._load_episode_metadata(original_path)

        logger.info(f"Resampled to {len(target_timeline)} timesteps at {self.target_hz}Hz")
        return episode_data

    def _load_episode_metadata(self, episode_path: Union[str, Path]) -> Dict:
        """
        Load episode metadata from accompanying files.

        Looks for metadata.yaml, info.yaml, metadata.json, info.json
        in the same directory as the MCAP file.

        Args:
            episode_path: Path to MCAP file

        Returns:
            Metadata dictionary (contains at minimum episode_id)
        """
        episode_path = Path(episode_path)

        if episode_path.is_file():
            episode_path = episode_path.parent

        # Try various metadata file names
        for metadata_file in ["metadata.yaml", "info.yaml", "metadata.json", "info.json"]:
            metadata_path = episode_path / metadata_file
            if metadata_path.exists():
                if metadata_path.suffix == ".yaml":
                    with open(metadata_path, "r") as f:
                        return yaml.safe_load(f)
                else:
                    with open(metadata_path, "r") as f:
                        return json.load(f)

        # Default metadata
        return {"episode_id": episode_path.name}

    def get_episode_length(self, episode_data: Dict[str, Any]) -> int:
        """
        Get number of timesteps in episode.

        Args:
            episode_data: Episode data dictionary

        Returns:
            Number of timesteps
        """
        return len(episode_data["timestamps"])

    def extract_camera_data(self, episode_data: Dict[str, Any]) -> Dict[str, List[bytes]]:
        """
        Extract camera data for ALL timesteps.

        Args:
            episode_data: Episode data dictionary

        Returns:
            Dictionary mapping camera names to lists of JPEG bytes [T]
        """
        camera_data = {}

        if self.camera_topics:
            for camera_name, camera_topic in self.camera_topics.items():
                if camera_topic in episode_data:
                    camera_data[camera_name] = episode_data[camera_topic]
                else:
                    logger.debug(f"Camera topic not found: {camera_topic}")

        return camera_data

    def extract_lowdim_data(self, episode_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract lowdim data for ALL timesteps.

        Two output modes:
        - 'separate': Returns dict with individual field keys
        - 'concatenated': Returns dict with 'state' and 'actions' keys

        Args:
            episode_data: Episode data dictionary

        Returns:
            Dictionary with lowdim data arrays of shape [T, dim]
        """
        if self.output_mode == "concatenated":
            state_components = []
            # Ensure we match the keys exactly as they exist in episode_data
            for key in self.state_key_fields:
                if key in episode_data:
                    data = episode_data[key]
                    state_components.append(data if data.ndim > 1 else data[:, np.newaxis])

            action_components = []
            for key in self.action_key_fields:
                if key in episode_data:
                    data = episode_data[key]
                    action_components.append(data if data.ndim > 1 else data[:, np.newaxis])

            return {
                "state": np.concatenate(state_components, axis=1) if state_components else np.array([]),
                "actions": np.concatenate(action_components, axis=1) if action_components else np.array([]),
            }

        else:  # separate mode
            # Return all fields individually
            lowdim_data = {}

            for field_name in list(episode_data.keys()):
                # Skip non-lowdim fields
                if field_name in ["timestamps", "_episode_metadata"] or field_name in self.camera_topics.values():
                    continue

                data = episode_data[field_name]
                if isinstance(data, np.ndarray):
                    if len(data.shape) == 1:
                        data = data[:, np.newaxis]
                    lowdim_data[field_name] = data

            return lowdim_data

    def extract_intrinsics_extrinsics_data(self, episode_data: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Extract camera intrinsics/extrinsics if available.

        Args:
            episode_data: Episode data dictionary

        Returns:
            Tuple of (intrinsics_dict, extrinsics_dict) or (None, None)
        """
        return None, None

    def extract_metadata_data(self, episode_data: Dict[str, Any]) -> Dict:
        """
        Extract metadata.

        Args:
            episode_data: Episode data dictionary

        Returns:
            Metadata dictionary including timestamps
        """
        metadata = episode_data.get("_episode_metadata", {})
        metadata["timestamps"] = episode_data["timestamps"]
        return metadata

    def _any_field_to_actual_key(self, field: str) -> str:
        """
        Generic mapping of target fields to observation fields using config-driven prefixes.
        """
        for action_prefix, obs_prefix in self.reference_field_prefixes.items():
            if field.startswith(action_prefix):
                return obs_prefix + field[len(action_prefix) :]

        return field

    def extract_sample_data(
        self,
        anchor_timestep: int,
        episode_path: str,
        episode_length: int,
        camera_data: Dict[str, List[bytes]],
        lowdim_data: Dict[str, np.ndarray],
        intrinsics_data: Dict[str, Any],
        extrinsics_data: Dict[str, Any],
        metadata_data: Dict[str, Any],
        statistics_ray_actor,
        logger_actor,
    ) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict], Optional[int], Optional[Dict]]:
        """
        Extract sample data for a single timestep with temporal windowing and stillness filtering.

        Args:
            anchor_timestep: Central timestep index for the sample.
            episode_path: Source file path for metadata logging.
            episode_length: Total timesteps in episode.
            camera_data: Map of camera names to lists of JPEG bytes.
            lowdim_data: Map of field names to resampled [T, D] numpy arrays.
            metadata_data: Dictionary containing task/language instructions and timestamps.

        Returns:
            A tuple containing:
            - sample_images: Dict mapping camera names to image bytes (or list of bytes for history).
            - sample_lowdim: Dict containing windowed state, actions, and masks.
            - sample_metadata: Dict with episode/step identifiers and timestamps.
            - language_instructions: Dict with "original" key containing the task instruction.
            - anchor_timestep: The index of the central frame.
            - episode_info: Dict containing high-level episode stats (e.g., episode_length).
            Returns (None,)*6 if the sample is filtered (e.g., stillness or boundary conditions).
        """
        if logger_actor is not None:
            try:  # noqa: SIM105
                logger_actor.increment_total_potential_samples.remote()
            except Exception:
                pass

        # Define temporal window: [anchor - past, anchor + future]
        past_steps = self.cfg.past_lowdim_steps
        future_steps = self.cfg.future_lowdim_steps
        window_size = past_steps + 1 + future_steps

        start_idx = anchor_timestep - past_steps
        end_idx = anchor_timestep + future_steps + 1

        sample_lowdim = {}

        if self.output_mode == "concatenated":
            states_window = self._extract_window(lowdim_data["state"], start_idx, end_idx, window_size)
            actions_window = self._extract_window(lowdim_data["actions"], start_idx, end_idx, window_size)

            # Stillness filtering: uses standard LBM thresholding logic
            if self.cfg.filter_still_samples:
                # Wrap in dict to match vla_foundry utility signature
                temp_dict = {"actions": actions_window}
                if is_still_sample(
                    temp_dict, start_idx=0, end_idx=window_size - 1, still_threshold=self.cfg.still_threshold
                ):
                    if logger_actor is not None:
                        try:  # noqa: SIM105
                            logger_actor.increment_still_samples_filtered.remote()
                        except Exception:
                            pass
                    return (None,) * 6

            sample_lowdim["state"] = states_window
            sample_lowdim["actions"] = actions_window

        else:  # separate mode
            for field_name, field_data in lowdim_data.items():
                sample_lowdim[field_name] = self._extract_window(field_data, start_idx, end_idx, window_size)

            # Build reference data from current observations at anchor timestep for all applicable keys
            # Build reference data using matching keys for deltas
            reference_data = {}
            for key in sample_lowdim:
                if key.startswith("action_"):
                    # Simply swap the prefix. YAML naming now ensures they match.
                    obs_key = key.replace("action_", "obs_")
                    if obs_key in lowdim_data:
                        reference_data[key] = lowdim_data[obs_key][anchor_timestep]

            # Compute relative coordinates (deltas) using the base class method
            if reference_data:
                sample_lowdim.update(self.create_relative_lowdim_data(sample_lowdim, reference_data))

            if self.cfg.filter_still_samples:
                action_fields = [k for k in sample_lowdim if k.startswith("action_")]
                for action_field in action_fields:
                    # Wrap the specific field window in a dict for the utility
                    temp_dict = {action_field: sample_lowdim[action_field]}
                    if is_still_sample(temp_dict, 0, window_size - 1, self.cfg.still_threshold):
                        if logger_actor is not None:
                            try:  # noqa: SIM105
                                logger_actor.increment_still_samples_filtered.remote()
                            except Exception:
                                pass
                        return (None,) * 6

        # Masking for padded steps at episode boundaries
        past_mask, future_mask = create_past_and_future_masks(
            idx=anchor_timestep,
            num_past=self.cfg.past_lowdim_steps,
            num_future=self.cfg.future_lowdim_steps,
            episode_length=episode_length,
        )

        sample_lowdim["past_mask"] = past_mask
        sample_lowdim["future_mask"] = future_mask

        # Image extraction: unique keys (e.g. head_t-1) fix the 'list has no attribute dtype' error
        sample_images = {}
        for img_offset in self.cfg.image_indices:
            img_t = np.clip(anchor_timestep + img_offset, 0, episode_length - 1)
            for camera_name, images in camera_data.items():
                key = f"{camera_name}_t{img_offset}"
                sample_images[key] = images[img_t]

        # Define language and metadata
        lang = metadata_data.get("language_instruction", metadata_data.get("task", "robot manipulation"))
        sample_metadata = {
            "episode_id": metadata_data.get("episode_id", "unknown"),
            "anchor_timestep": anchor_timestep,
            "timestamp": metadata_data["timestamps"][anchor_timestep],
            "camera_names": list(camera_data.keys()),
        }

        # Build stats_sample for batched statistics update (don't send immediately)
        stats_sample = None
        if statistics_ray_actor is not None:
            stats_sample = {
                "lowdim": {k: v.copy() for k, v in sample_lowdim.items()},
                "past_mask": past_mask,
                "future_mask": future_mask,
            }

        return (
            sample_images,
            sample_lowdim,
            sample_metadata,
            {"original": lang},
            anchor_timestep,
            stats_sample,
        )

    def _extract_window(self, data: np.ndarray, start_idx: int, end_idx: int, window_size: int) -> np.ndarray:
        """
        Extract temporal window with edge-padding if needed.

        Args:
            data: Full episode data [T, dim]
            start_idx: Window start index (can be negative)
            end_idx: Window end index (can exceed data length)
            window_size: Expected window size

        Returns:
            Window data [window_size, dim] with edge-padding if needed
        """
        if len(data.shape) == 1:
            data = data[:, np.newaxis]

        # Extract available window
        window_data = data[max(0, start_idx) : min(len(data), end_idx)]

        # Pad if needed
        if len(window_data) < window_size:
            pad_before = max(0, -start_idx)
            pad_after = max(0, end_idx - len(data))
            window_data = np.pad(window_data, ((pad_before, pad_after), (0, 0)), mode="edge")

        return window_data
