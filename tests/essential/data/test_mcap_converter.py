"""
Unit tests for MCAP converter and temporal alignment pipeline.

This test suite validates the MCAP-to-WebDataset preprocessing pipeline, covering:
- Message extraction from ROS 2 CompressedImage and raw Image topics
- Nearest-neighbor time-alignment across data channels
- Low-dimensional data extraction and normalization
- Sample generation with temporal padding and stillness filtering
- Integration with VLA Foundry's training data format

The tests ensure correctness of frequency-dependent temporal parameters, particularly
for 30Hz control policies where temporal window sizes must scale appropriately from
standard 10Hz baselines.
"""

import builtins
import io
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest
import yaml

from vla_foundry.data.preprocessing.robotics.converters.mcap import (
    MCAPConverter,
    extract_array_from_msg,
    extract_field_path,
    extract_image_from_msg,
    extract_structured_msg,
    parse_episode_path,
)
from vla_foundry.data.preprocessing.utils import nearest_indices

# Default action field config returned by load_action_field_config
DEFAULT_ACTION_FIELD_CONFIG = {
    "action_key_fields": ["action_joints"],
    "action_index_fields": [1],
    "pose_groups": [],
}

# Default language annotations file content
DEFAULT_LANGUAGE_ANNOTATIONS = {
    "language_dict": {
        "stack_cubes_ordered": {
            "original": ["Stack the cubes in order"],
            "randomized": ["Put the cubes on top of each other"],
        }
    }
}


def _make_base_config(**overrides):
    """Create a minimal mock config for MCAPConverter."""
    cfg = MagicMock()
    defaults = dict(
        language_annotations_path="/tmp/lang.yaml",
        topics_to_fields_path="/tmp/topics.yaml",
        action_fields_config_path="/tmp/action_fields.yaml",
        past_lowdim_steps=1,
        future_lowdim_steps=1,
        filter_still_samples=False,
        still_threshold=0.001,
        padding_strategy="zero",
        max_padding_left=5,
        max_padding_right=10,
        image_indices=[0],
        resize_images_size=[384, 384],
        jpeg_quality=90,
        camera_names=["camera1"],
        stride=1,
        num_workers=1,
        # Filters default to None/empty
        source_filter=None,
        task_filter=None,
        domain_filter=None,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_topics_config(**overrides):
    """Create a minimal topics config dict."""
    base = {
        "output_mode": "separate",
        "pivot_source_field": "camera1",
        "gap_threshold_dt": 0.1,
        "snap_threshold_dt": 0.1,
        "action_topics": ["/action/joints"],
        "state_topics": ["/state/joints"],
        "camera_topics": {"camera1": "/cam1/compressed"},
        "camera_topics_field_map": {"/cam1/compressed": "camera1"},
        "action_field_map": {"/action/joints": "action_joints"},
        "state_field_map": {"/state/joints": "obs_joints"},
    }
    base.update(overrides)
    return base


def _create_converter(cfg=None, topics_cfg=None, action_field_config=None, language_annotations=None):
    """Create an MCAPConverter with all external dependencies mocked."""
    cfg = cfg or _make_base_config()
    topics_cfg = topics_cfg or _make_topics_config()
    action_field_config = action_field_config or DEFAULT_ACTION_FIELD_CONFIG
    language_annotations = language_annotations or DEFAULT_LANGUAGE_ANNOTATIONS

    yaml_content = yaml.dump(language_annotations)

    # Patch to no-op and run without touching the file system
    with (
        patch("vla_foundry.data.preprocessing.robotics.converters.mcap.file_exists", return_value=True),
        patch("vla_foundry.data.preprocessing.robotics.converters.mcap.yaml_load", return_value=topics_cfg),
        patch(
            "vla_foundry.data.preprocessing.robotics.converters.mcap.load_action_field_config",
            return_value=action_field_config,
        ),
        patch(
            "vla_foundry.data.preprocessing.robotics.converters.mcap.validate_pose_groups",
        ),
        patch(builtins.__name__ + ".open", return_value=io.StringIO(yaml_content)),
    ):
        return MCAPConverter(cfg)


def _make_episode_data(episode_len=20, action_dim=1, state_dim=1, n_cameras=1):
    """Create episode_data in the format returned by load_episode_data."""
    observations = {"obs_joints": np.random.randn(episode_len, state_dim).astype(np.float32)}
    for i in range(n_cameras):
        observations[f"camera{i + 1}"] = np.random.randint(0, 255, (episode_len, 64, 64, 3), dtype=np.uint8)

    actions = {"actions": np.random.randn(episode_len, action_dim).astype(np.float32)}
    timestamps = np.arange(episode_len, dtype=np.float32) / 30.0

    return {"observations": observations, "actions": actions, "timestamps": timestamps}


def _make_logger():
    """Create a mock logger actor."""
    logger = MagicMock()
    logger.increment_total_potential_samples = MagicMock(return_value=MagicMock())
    logger.increment_padding_samples_filtered = MagicMock(return_value=MagicMock())
    logger.increment_still_samples_filtered = MagicMock(return_value=MagicMock())
    return logger


class TestExtractImageFromMsg:
    """Tests for image extraction from ROS2 Image and CompressedImage messages."""

    def test_compressed_jpeg(self):
        """Verify JPEG decoding from CompressedImage messages."""
        mock_msg = Mock(spec=["format", "data"])
        mock_msg.format = "jpeg"

        # Generate synthetic 2x2 RGB image and encode to JPEG
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        img[:, :] = [0, 0, 255]  # BGR red for OpenCV
        _, encoded = cv2.imencode(".jpg", img)
        mock_msg.data = encoded

        # Test JPEG decoding - should return numpy array in RGB format
        result = extract_image_from_msg(mock_msg)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2, 3)
        assert result.dtype == np.uint8

    def test_raw_rgb8(self):
        """Verify conversion from raw Image messages with RGB8 encoding."""
        mock_msg = Mock(spec=["height", "width", "encoding", "data"])
        mock_msg.height = 2
        mock_msg.width = 2
        mock_msg.encoding = "rgb8"

        # Create 2x2 RGB image with single red pixel
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        img[0, 0] = [255, 0, 0]
        mock_msg.data = img

        # Test raw image decoding to numpy array
        result = extract_image_from_msg(mock_msg)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2, 3)
        assert result.dtype == np.uint8
        # Verify red pixel is preserved
        assert np.array_equal(result[0, 0], [255, 0, 0])

    def test_depth_image_rejected(self):
        """
        Verify that depth images (16UC1 encoding) are rejected (raise ValueError).

        Depth images require a separate processing pipeline and cannot yet be
        mixed with RGB camera streams in the standard image extraction path.
        """
        mock_msg = Mock(spec=["height", "width", "encoding", "data"])
        mock_msg.height = 10
        mock_msg.width = 10
        mock_msg.encoding = "16UC1"
        mock_msg.data = np.zeros((10, 10), dtype=np.uint16)

        with pytest.raises(ValueError, match="Depth images"):
            extract_image_from_msg(mock_msg)


class TestExtractFieldPath:
    """Tests for nested field extraction from ROS2 messages."""

    def test_scalar_field(self):
        """Verify extraction of top-level scalar fields."""
        mock_msg = Mock(spec=["temperature"])
        mock_msg.temperature = 25.5

        result = extract_field_path(mock_msg, "temperature")
        assert result is not None
        np.testing.assert_array_equal(result, np.array([25.5], dtype=np.float32))

    def test_nested_field(self):
        """Verify extraction from nested message structures using dot notation."""
        mock_imu = Mock(spec=["x"])
        mock_imu.x = 1.0
        mock_msg = Mock(spec=["imu"])
        mock_msg.imu = mock_imu

        result = extract_field_path(mock_msg, "imu.x")
        assert result is not None
        np.testing.assert_array_equal(result, np.array([1.0], dtype=np.float32))

    def test_invalid_path_returns_none(self):
        """Non-existent paths return None."""
        mock_msg = Mock(spec=["valid_field"])
        mock_msg.valid_field = 42

        assert extract_field_path(mock_msg, "nonexistent") is None
        assert extract_field_path(mock_msg, "nonexistent.field") is None

    def test_array_iteration(self):
        """motor_state[*].q extracts q from each element."""
        item0 = Mock(spec=["q"])
        item0.q = 1.0
        item1 = Mock(spec=["q"])
        item1.q = 2.0
        mock_msg = Mock(spec=["motor_state"])
        mock_msg.motor_state = [item0, item1]

        result = extract_field_path(mock_msg, "motor_state[*].q")
        np.testing.assert_array_equal(result, np.array([1.0, 2.0], dtype=np.float32))


class TestExtractArrayFromMsg:
    """Tests for flattening arbitrary ROS 2 message structures to numpy arrays."""

    def test_wrench_message(self):
        """
        Verify recursive flattening of nested message structures.

        Used for extracting force-torque sensor data, IMU readings, and other
        multi-field messages into flat arrays suitable for observation vectors.
        Example: Wrench message with 3D force and 3D torque becomes 6D float32 array.
        """
        mock_force = Mock(spec=["x", "y", "z"])
        mock_force.x, mock_force.y, mock_force.z = 1.0, 2.0, 3.0
        mock_torque = Mock(spec=["x", "y", "z"])
        mock_torque.x, mock_torque.y, mock_torque.z = 0.1, 0.2, 0.3
        mock_msg = Mock(spec=["force", "torque"])
        mock_msg.force = mock_force
        mock_msg.torque = mock_torque

        result = extract_array_from_msg(mock_msg)
        assert result is not None
        assert result.shape == (6,)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32))


class TestExtractStructuredMsg:
    """Tests for extract_structured_msg covering JointState, Pose, Skeleton, and ReferenceMotion shapes."""

    def test_joint_state(self):
        """JointState messages emit one __<joint_name> entry per joint position."""
        msg = Mock(spec=["name", "position", "velocity", "effort"])
        msg.name = ["shoulder", "elbow"]
        msg.position = [0.1, 0.2]
        msg.velocity = [0.0, 0.0]
        msg.effort = [0.0, 0.0]

        result = extract_structured_msg(msg)
        assert result is not None
        assert set(result.keys()) == {"__shoulder", "__elbow"}
        np.testing.assert_array_equal(result["__shoulder"], np.array([0.1], dtype=np.float32))
        np.testing.assert_array_equal(result["__elbow"], np.array([0.2], dtype=np.float32))
        assert all(v.dtype == np.float32 for v in result.values())

    def test_skeleton_style(self):
        """Skeleton messages (parallel name[] + points[Point]) emit __<joint>_{x,y,z}."""
        p_pelvis = Mock(spec=["x", "y", "z"])
        p_pelvis.x, p_pelvis.y, p_pelvis.z = 0.0, 0.0, 0.9
        p_hip = Mock(spec=["x", "y", "z"])
        p_hip.x, p_hip.y, p_hip.z = 0.1, -0.2, 0.8
        msg = Mock(spec=["name", "points"])
        # Names carry a "namespace/" prefix that the extractor must strip.
        msg.name = ["smpl/pelvis", "smpl/left_hip"]
        msg.points = [p_pelvis, p_hip]

        result = extract_structured_msg(msg)
        assert result is not None
        assert set(result.keys()) == {
            "__pelvis_x",
            "__pelvis_y",
            "__pelvis_z",
            "__left_hip_x",
            "__left_hip_y",
            "__left_hip_z",
        }
        np.testing.assert_array_equal(result["__pelvis_z"], np.array([0.9], dtype=np.float32))
        np.testing.assert_array_equal(result["__left_hip_x"], np.array([0.1], dtype=np.float32))
        assert all(v.shape == (1,) and v.dtype == np.float32 for v in result.values())

    def test_skeleton_style_no_namespace_prefix(self):
        """Skeleton names without a "namespace/" prefix pass through unchanged."""
        p = Mock(spec=["x", "y", "z"])
        p.x, p.y, p.z = 1.0, 2.0, 3.0
        msg = Mock(spec=["name", "points"])
        msg.name = ["head"]
        msg.points = [p]

        result = extract_structured_msg(msg)
        assert result is not None
        assert set(result.keys()) == {"__head_x", "__head_y", "__head_z"}

    def test_reference_motion_composite(self):
        """ReferenceMotion (root_pose + joint_positions) emits __root_rot_6d + per-joint scalars."""
        # Identity quaternion -> identity rotation matrix -> rot_6d = [1,0,0, 0,1,0]
        orientation = Mock(spec=["x", "y", "z", "w"])
        orientation.x, orientation.y, orientation.z, orientation.w = 0.0, 0.0, 0.0, 1.0
        root_pose = Mock(spec=["orientation"])
        root_pose.orientation = orientation
        joint_positions = Mock(spec=["name", "position"])
        joint_positions.name = ["left_wrist_roll_joint", "right_wrist_roll_joint"]
        joint_positions.position = [0.5, -0.5]
        msg = Mock(spec=["root_pose", "joint_positions"])
        msg.root_pose = root_pose
        msg.joint_positions = joint_positions

        result = extract_structured_msg(msg)
        assert result is not None
        assert set(result.keys()) == {
            "__root_rot_6d",
            "__left_wrist_roll_joint",
            "__right_wrist_roll_joint",
        }
        assert result["__root_rot_6d"].shape == (6,)
        assert result["__root_rot_6d"].dtype == np.float32
        np.testing.assert_allclose(
            result["__root_rot_6d"], np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6
        )
        np.testing.assert_array_equal(result["__left_wrist_roll_joint"], np.array([0.5], dtype=np.float32))

    def test_unstructured_message_returns_none(self):
        """Messages that don't match any structured shape return None."""
        msg = Mock(spec=["unrelated_field"])
        msg.unrelated_field = 42
        assert extract_structured_msg(msg) is None


class TestNearestIndices:
    """Tests for nearest_indices snap-to-source lookup."""

    def test_exact_matches(self):
        source = np.array([0.0, 1.0, 2.0, 3.0])
        target = np.array([0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(nearest_indices(source, target), [0, 1, 2, 3])

    def test_snaps_to_closer_neighbor(self):
        source = np.array([0.0, 1.0, 2.0])
        target = np.array([0.3, 0.7, 1.6])
        np.testing.assert_array_equal(nearest_indices(source, target), [0, 1, 2])

    def test_single_source(self):
        source = np.array([5.0])
        target = np.array([1.0, 5.0, 9.0])
        # Everything must map to index 0 regardless of distance
        np.testing.assert_array_equal(nearest_indices(source, target), [0, 0, 0])

    def test_target_outside_source_range(self):
        source = np.array([1.0, 2.0, 3.0])
        # Target times before the first and after the last source time
        target = np.array([0.0, 4.0])
        # Verifies clip logic clamps correctly instead of going out of bounds
        np.testing.assert_array_equal(nearest_indices(source, target), [0, 2])

    def test_empty_source_raises(self):
        """Empty source_times is not allowed."""
        with pytest.raises(ValueError, match="source_times must not be empty"):
            nearest_indices(np.array([]), np.array([1.0]))

    def test_empty_target_returns_empty(self):
        """Empty target_times returns a zero-length index array."""
        result = nearest_indices(np.array([1.0, 2.0]), np.array([]))
        assert len(result) == 0

    def test_max_snap_distance_raises_on_violation(self):
        """Exceeding max_snap_distance raises ValueError."""
        source = np.array([0.0, 10.0])
        target = np.array([5.0])  # 5s from nearest source
        with pytest.raises(ValueError, match="snapped"):
            nearest_indices(source, target, max_snap_distance=1.0)

    def test_max_snap_distance_passes_within_tolerance(self):
        """Snaps within tolerance do not raise."""
        source = np.array([0.0, 1.0, 2.0])
        target = np.array([0.1, 0.9, 2.0])
        result = nearest_indices(source, target, max_snap_distance=0.5)
        np.testing.assert_array_equal(result, [0, 1, 2])


class TestParseEpisodePath:
    """Tests for parse_episode_path utility."""

    def test_valid_s3_path(self):
        path = "s3://bucket/platform/robot/mcap/stack_cubes/sim/teleop/0026_20260129_161531"
        result = parse_episode_path(path)
        assert result is not None
        assert result["task"] == "stack_cubes"
        assert result["domain"] == "sim"
        assert result["source"] == "teleop"
        assert result["episode_dir"] == "0026_20260129_161531"
        assert result["episode_index"] == 26

    def test_valid_local_path(self):
        path = "/data/pick_and_place/real/filtered/0001_20260201_100000"
        result = parse_episode_path(path)
        assert result is not None
        assert result["task"] == "pick_and_place"
        assert result["domain"] == "real"
        assert result["source"] == "filtered"
        assert result["episode_index"] == 1

    def test_trailing_slash_stripped(self):
        path = "s3://bucket/task/sim/teleop/0005_ts/"
        result = parse_episode_path(path)
        assert result is not None
        assert result["episode_index"] == 5

    def test_too_short_path_returns_none(self):
        assert parse_episode_path("a/b/c") is None
        assert parse_episode_path("single") is None

    def test_non_numeric_prefix_returns_none(self):
        path = "/data/task/sim/teleop/bad_episode_name"
        assert parse_episode_path(path) is None

    def test_zero_padded_index(self):
        """Leading zeros are stripped: '0000' to 0."""
        path = "/data/task/sim/teleop/0000_20260101_000000"
        result = parse_episode_path(path)
        assert result is not None
        assert result["episode_index"] == 0


class TestValidateFilters:
    """Tests for _validate_filters at init time."""

    def test_valid_domain_filter(self):
        cfg = _make_base_config(domain_filter=["sim"])
        _ = _create_converter(cfg=cfg)
        # No exception raised

    def test_valid_source_filter(self):
        cfg = _make_base_config(source_filter=["teleop", "filtered"])
        _ = _create_converter(cfg=cfg)
        # No exception raised

    def test_invalid_domain_filter_raises(self):
        cfg = _make_base_config(domain_filter=["sim", "virtual"])
        with pytest.raises(ValueError, match="Invalid domain_filter"):
            _create_converter(cfg=cfg)

    def test_invalid_source_filter_raises(self):
        cfg = _make_base_config(source_filter=["teleop", "manual"])
        with pytest.raises(ValueError, match="Invalid source_filter"):
            _create_converter(cfg=cfg)

    def test_none_filters_pass(self):
        cfg = _make_base_config(domain_filter=None, source_filter=None)
        _ = _create_converter(cfg=cfg)
        # No exception raised


class TestFilterEpisodes:
    """Tests for _filter_episodes method."""

    SAMPLE_EPISODES = [
        "s3://bucket/platform/robot/mcap/stack_cubes/sim/teleop/0000_20260101_000000",
        "s3://bucket/platform/robot/mcap/stack_cubes/sim/teleop/0001_20260101_000100",
        "s3://bucket/platform/robot/mcap/stack_cubes/sim/filtered/0002_20260101_000200",
        "s3://bucket/platform/robot/mcap/stack_cubes/real/teleop/0003_20260101_000300",
        "s3://bucket/platform/robot/mcap/pick_and_place/sim/teleop/0004_20260101_000400",
        "s3://bucket/platform/robot/mcap/pick_and_place/real/teleop/0005_20260101_000500",
    ]

    def test_no_filters_returns_all(self):
        converter = _create_converter()
        result = converter._filter_episodes(self.SAMPLE_EPISODES)
        assert len(result) == len(self.SAMPLE_EPISODES)

    def test_source_filter(self):
        cfg = _make_base_config(source_filter=["teleop"])
        converter = _create_converter(cfg=cfg)
        result = converter._filter_episodes(self.SAMPLE_EPISODES)
        assert len(result) == 5  # all except the 'filtered' one
        assert all("filtered" not in ep.split("/")[-2] for ep in result)

    def test_domain_filter(self):
        cfg = _make_base_config(domain_filter=["sim"])
        converter = _create_converter(cfg=cfg)
        result = converter._filter_episodes(self.SAMPLE_EPISODES)
        assert len(result) == 4  # sim only
        assert all("real" not in ep.split("/")[-3] for ep in result)

    def test_task_filter(self):
        cfg = _make_base_config(task_filter=["pick_and_place"])
        converter = _create_converter(cfg=cfg)
        result = converter._filter_episodes(self.SAMPLE_EPISODES)
        assert len(result) == 2
        assert all("pick_and_place" in ep for ep in result)

    def test_combined_filters(self):
        """source=teleop + domain=sim + task=stack_cubes, thus episodes 0 and 1."""
        cfg = _make_base_config(
            source_filter=["teleop"],
            domain_filter=["sim"],
            task_filter=["stack_cubes"],
        )
        converter = _create_converter(cfg=cfg)
        result = converter._filter_episodes(self.SAMPLE_EPISODES)
        assert len(result) == 2
        indices = [parse_episode_path(ep)["episode_index"] for ep in result]
        assert set(indices) == {0, 1}

    def test_no_matches_returns_empty(self):
        cfg = _make_base_config(task_filter=["nonexistent_task"])
        converter = _create_converter(cfg=cfg)
        result = converter._filter_episodes(self.SAMPLE_EPISODES)
        assert result == []

    def test_unparseable_paths_skipped(self):
        """Short paths that can't be parsed are skipped with a warning."""
        cfg = _make_base_config(source_filter=["teleop"])
        converter = _create_converter(cfg=cfg)
        episodes = ["too/short"] + self.SAMPLE_EPISODES[:2]
        result = converter._filter_episodes(episodes)
        # Only the two valid teleop episodes pass
        assert len(result) == 2


class TestMCAPConverterConfiguration:
    """Tests for configuration loading and validation."""

    def test_topics_config_loaded(self):
        topics_cfg = _make_topics_config(target_hz=15.0, action_topics=["/custom/action"])
        converter = _create_converter(topics_cfg=topics_cfg)
        assert converter.action_topics == ["/custom/action"]

    def test_invalid_output_mode_raises(self):
        topics_cfg = _make_topics_config(output_mode="invalid")
        with pytest.raises(ValueError, match="output_mode"):
            _create_converter(topics_cfg=topics_cfg)

    def test_concatenated_mode_requires_state_key_fields(self):
        topics_cfg = _make_topics_config(output_mode="concatenated", state_key_fields=[])
        with pytest.raises(ValueError, match="state_key_fields"):
            _create_converter(topics_cfg=topics_cfg)

    def test_action_field_sizes_computed(self):
        action_cfg = {
            "action_key_fields": ["ee_xyz", "ee_rot", "gripper"],
            "action_index_fields": [3, 9, 10],
            "pose_groups": [],
        }
        converter = _create_converter(action_field_config=action_cfg)
        assert converter.action_field_sizes == [3, 6, 1]

    def test_missing_pivot_source_raises(self):
        topics_cfg = _make_topics_config(pivot_source_field=None)
        with pytest.raises(ValueError, match="pivot_source_field"):
            _create_converter(topics_cfg=topics_cfg)


class TestMCAPConverterLowdimExtraction:
    """Tests for extract_lowdim_data in both output modes."""

    def test_separate_mode(self):
        """Separate mode preserves individual field keys from observations + slices actions."""
        action_cfg = {
            "action_key_fields": ["action_joints"],
            "action_index_fields": [2],
            "pose_groups": [],
        }
        converter = _create_converter(action_field_config=action_cfg)

        episode_data = {
            "observations": {
                "obs_joints": np.ones((10, 3), dtype=np.float32),
            },
            "actions": {
                "actions": np.ones((10, 2), dtype=np.float32),
            },
            "timestamps": np.arange(10, dtype=np.float32) / 30.0,
        }

        result = converter.extract_lowdim_data(episode_data)
        assert "obs_joints" in result
        assert result["obs_joints"].shape == (10, 3)
        # Action sliced into 'action_joints' (dim=2)
        assert "action_joints" in result
        assert result["action_joints"].shape == (10, 2)

    def test_concatenated_mode(self):
        """Concatenated mode merges state fields and passes actions through."""
        topics_cfg = _make_topics_config(
            output_mode="concatenated",
            state_key_fields=["obs_joints", "obs_force"],
        )
        converter = _create_converter(topics_cfg=topics_cfg)

        episode_data = {
            "observations": {
                "obs_joints": np.ones((10, 3), dtype=np.float32),
                "obs_force": np.ones((10, 6), dtype=np.float32),
            },
            "actions": {
                "actions": np.ones((10, 1), dtype=np.float32),
            },
            "timestamps": np.arange(10, dtype=np.float32) / 30.0,
        }

        result = converter.extract_lowdim_data(episode_data)
        assert "state" in result
        assert result["state"].shape == (10, 9)  # 3 + 6
        assert result["state"].dtype == np.float32
        assert "actions" in result
        assert result["actions"].shape == (10, 1)

    def test_action_dim_mismatch_raises(self):
        """Action tensor smaller than expected raises ValueError."""
        action_cfg = {
            "action_key_fields": ["big_action"],
            "action_index_fields": [100],
            "pose_groups": [],
        }
        converter = _create_converter(action_field_config=action_cfg)

        episode_data = {
            "observations": {},
            "actions": {"actions": np.ones((10, 5), dtype=np.float32)},
            "timestamps": np.arange(10, dtype=np.float32) / 30.0,
        }

        with pytest.raises(ValueError, match="insufficient dimension"):
            converter.extract_lowdim_data(episode_data)


class TestMCAPConverterSampleExtraction:
    """Tests for extract_sample_data with temporal windowing."""

    def _extract_sample(self, converter, episode_data, anchor=5, episode_len=20, **kwargs):
        """Helper to call extract_sample_data with sensible defaults."""
        camera_data = converter.extract_camera_data(episode_data)
        lowdim_data = converter.extract_lowdim_data(episode_data)
        metadata_data = converter.extract_metadata_data(episode_data)

        return converter.extract_sample_data(
            anchor_timestep=anchor,
            episode_path="s3://bucket/mcap/stack_cubes/sim/teleop/0001_20260101",
            episode_length=episode_len,
            camera_data=camera_data,
            lowdim_data=lowdim_data,
            intrinsics_data=None,
            extrinsics_data=None,
            metadata_data=metadata_data,
            statistics_ray_actor=kwargs.get("statistics_ray_actor"),
            logger_actor=kwargs.get("logger_actor", _make_logger()),
        )

    def test_basic_extraction(self):
        """Middle-of-episode sample returns valid data."""
        converter = _create_converter()
        episode_data = _make_episode_data(episode_len=20)

        result = self._extract_sample(converter, episode_data, anchor=10, episode_len=20)
        assert isinstance(result, tuple)
        # Verify extract_sample_data always returns a 7-element tuple.
        assert len(result) == 7

        sample_images, sample_lowdim, sample_metadata, lang, _, _, stats = result
        assert sample_images is not None
        assert sample_lowdim is not None
        assert sample_metadata is not None

    def test_still_sample_filtered(self):
        """Stationary robot returns all Nones."""
        cfg = _make_base_config(
            filter_still_samples=True, still_threshold=0.001, past_lowdim_steps=2, future_lowdim_steps=2
        )
        converter = _create_converter(cfg=cfg)

        # All-zeros actions, i.e., still
        episode_data = {
            "observations": {
                "obs_joints": np.zeros((20, 1), dtype=np.float32),
                "camera1": np.zeros((20, 64, 64, 3), dtype=np.uint8),
            },
            "actions": {"actions": np.zeros((20, 1), dtype=np.float32)},
            "timestamps": np.arange(20, dtype=np.float32) / 30.0,
        }

        result = self._extract_sample(converter, episode_data, anchor=10, episode_len=20)
        # All 6 return values should be None
        assert all(v is None for v in result)

    def test_padding_at_episode_start(self):
        """Sample near start gets left-padded, not rejected."""
        cfg = _make_base_config(past_lowdim_steps=2, future_lowdim_steps=2, max_padding_left=5)
        converter = _create_converter(cfg=cfg)
        episode_data = _make_episode_data(episode_len=20)

        result = self._extract_sample(converter, episode_data, anchor=0, episode_len=20)
        sample_images = result[0]
        sample_metadata = result[2]
        assert sample_images is not None
        assert sample_metadata.past_padding == 2

    def test_excessive_padding_rejected(self):
        """Too much padding required means filtered out."""
        cfg = _make_base_config(past_lowdim_steps=10, max_padding_left=2)
        converter = _create_converter(cfg=cfg)
        episode_data = _make_episode_data(episode_len=20)

        result = self._extract_sample(converter, episode_data, anchor=0, episode_len=20)
        # Needs 10 left padding, but max is 2 so rejected sample
        sample_images = result[0]
        assert sample_images is None

    def test_deployment_window_49_steps(self):
        """30 Hz config: 1 past + 1 current + 47 future = 49 total steps."""
        cfg = _make_base_config(past_lowdim_steps=1, future_lowdim_steps=47, filter_still_samples=False)
        converter = _create_converter(cfg=cfg)
        episode_data = _make_episode_data(episode_len=100)

        result = self._extract_sample(converter, episode_data, anchor=50, episode_len=100)
        sample_lowdim = result[1]

        for key, val in sample_lowdim.items():
            if key in ("past_mask", "future_mask"):
                continue
            assert val.shape[0] == 49, f"{key} has {val.shape[0]} steps, expected 49"


class TestMCAPConverterRelativeCoordinates:
    """Tests for relative coordinate computation using pose groups."""

    POSE_ACTION_CFG = {
        "action_key_fields": ["action_ee_pose_left__xyz", "action_ee_pose_left__rot_6d"],
        "action_index_fields": [3, 9],
        "pose_groups": [
            {
                "name": "left_ee_action",
                "position_key": "action_ee_pose_left__xyz",
                "rotation_key": "action_ee_pose_left__rot_6d",
            }
        ],
    }

    def test_relative_keys_added_in_separate_mode(self):
        topics_cfg = _make_topics_config(
            output_mode="separate",
            action_field_map={"/ee_target_left": "action_ee_pose_left"},
            state_field_map={"/current_left_ee": "obs_ee_pose_left"},
            reference_field_prefixes={"action_ee_pose_left": "obs_ee_pose_left"},
        )
        converter = _create_converter(topics_cfg=topics_cfg, action_field_config=self.POSE_ACTION_CFG)

        episode_len = 10
        episode_data = {
            "observations": {
                "obs_ee_pose_left__xyz": np.zeros((episode_len, 3), dtype=np.float32),
                "obs_ee_pose_left__rot_6d": np.tile(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), (episode_len, 1)),
            },
            "actions": {
                "actions": np.ones((episode_len, 9), dtype=np.float32),
            },
            "timestamps": np.arange(episode_len, dtype=np.float32) / 30.0,
        }

        camera_data = {"camera1": np.zeros((episode_len, 64, 64, 3), dtype=np.uint8)}
        lowdim_data = converter.extract_lowdim_data(episode_data)
        metadata_data = converter.extract_metadata_data(episode_data)

        result = converter.extract_sample_data(
            anchor_timestep=5,
            episode_path="s3://bucket/mcap/stack_cubes/sim/teleop/0001_ts",
            episode_length=episode_len,
            camera_data=camera_data,
            lowdim_data=lowdim_data,
            intrinsics_data=None,
            extrinsics_data=None,
            metadata_data=metadata_data,
            statistics_ray_actor=None,
            logger_actor=_make_logger(),
        )

        sample_lowdim = result[1]
        assert "action_ee_pose_left__xyz_relative" in sample_lowdim
        assert "action_ee_pose_left__rot_6d_relative" in sample_lowdim

    def test_no_relative_in_concatenated_mode(self):
        topics_cfg = _make_topics_config(
            output_mode="concatenated",
            state_key_fields=["obs_ee_pose_left__xyz", "obs_ee_pose_left__rot_6d"],
        )
        converter = _create_converter(topics_cfg=topics_cfg, action_field_config=self.POSE_ACTION_CFG)

        episode_len = 10
        episode_data = {
            "observations": {
                "obs_ee_pose_left__xyz": np.zeros((episode_len, 3), dtype=np.float32),
                "obs_ee_pose_left__rot_6d": np.zeros((episode_len, 6), dtype=np.float32),
            },
            "actions": {"actions": np.ones((episode_len, 9), dtype=np.float32)},
            "timestamps": np.arange(episode_len, dtype=np.float32) / 30.0,
        }

        camera_data = {"camera1": np.zeros((episode_len, 64, 64, 3), dtype=np.uint8)}
        lowdim_data = converter.extract_lowdim_data(episode_data)
        metadata_data = converter.extract_metadata_data(episode_data)

        result = converter.extract_sample_data(
            anchor_timestep=5,
            episode_path="s3://bucket/mcap/stack_cubes/sim/teleop/0001_ts",
            episode_length=episode_len,
            camera_data=camera_data,
            lowdim_data=lowdim_data,
            intrinsics_data=None,
            extrinsics_data=None,
            metadata_data=metadata_data,
            statistics_ray_actor=None,
            logger_actor=_make_logger(),
        )

        sample_lowdim = result[1]
        assert set(sample_lowdim.keys()) == {"state", "actions", "past_mask", "future_mask"}


class TestLanguageInstructions:
    """Tests for language instruction lookup from episode paths."""

    def test_known_task_returns_instructions(self):
        converter = _create_converter()
        result = converter.get_language_instructions("s3://bucket/mcap/stack_cubes_ordered/sim/teleop/0001_ts")
        assert "original" in result
        assert len(result["original"]) > 0

    def test_unknown_task_returns_empty(self):
        converter = _create_converter()
        result = converter.get_language_instructions("s3://bucket/mcap/unknown_task/sim/teleop/0001_ts")
        assert result == {}

    def test_filtered_instruction_types(self):
        converter = _create_converter()
        result = converter.get_language_instructions(
            "s3://bucket/mcap/stack_cubes_ordered/sim/teleop/0001_ts",
            instruction_types=["original"],
        )
        assert "original" in result
        assert "randomized" not in result
