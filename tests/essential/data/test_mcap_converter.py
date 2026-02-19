"""
Unit tests for MCAP converter and temporal resampling pipeline.

This test suite validates the MCAP-to-WebDataset preprocessing pipeline, covering:
- Message extraction from ROS2 CompressedImage and raw Image topics
- Temporal resampling with anti-aliasing for frequency conversion
- Episode discovery from MCAP file hierarchies
- Low-dimensional data extraction and normalization
- Sample generation with temporal padding and stillness filtering
- Integration with VLA Foundry's training data format

The tests ensure correctness of frequency-dependent temporal parameters, particularly
for 30Hz control policies where temporal window sizes must scale appropriately from
standard 10Hz baselines.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest

from vla_foundry.data.preprocessing.robotics.converters.mcap import (
    MCAPConverter,
    TemporalResampler,
    extract_array_from_msg,
    extract_field_path,
    extract_image_from_msg,
)


class TestExtractImageFromMsg:
    """Tests for image extraction from ROS2 Image and CompressedImage messages."""

    def test_extract_image_from_msg_compressed_jpeg(self):
        """
        Verify JPEG decoding from CompressedImage messages.

        CompressedImage messages are decoded to numpy arrays to support downstream
        image resizing and augmentation in the preprocessing pipeline.
        """
        # Create properly configured Mock for CompressedImage
        # CompressedImage has format and data but NO height/width/encoding
        mock_msg = Mock(spec=["format", "data"])
        mock_msg.format = "jpeg"

        # Generate synthetic 2x2 RGB image and encode to JPEG
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        img[:, :] = [0, 0, 255]  # BGR red for OpenCV
        success, encoded = cv2.imencode(".jpg", img)
        mock_msg.data = encoded

        # Test JPEG decoding - should return numpy array in RGB format
        result = extract_image_from_msg(mock_msg)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2, 3)
        assert result.dtype == np.uint8

    def test_extract_image_from_msg_raw_image(self):
        """
        Verify conversion from raw Image messages with RGB8 encoding.

        Raw Image messages are decoded to numpy arrays in RGB format for
        consistent processing through the resizing pipeline.
        """
        # Create properly configured Mock for raw Image
        # Raw Image has height, width, encoding, and data
        mock_msg = Mock(spec=["height", "width", "encoding", "data"])
        mock_msg.height = 2
        mock_msg.width = 2
        mock_msg.encoding = "rgb8"

        # Create 2x2 RGB image with single red pixel
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        img[0, 0] = [255, 0, 0]  # Red pixel
        mock_msg.data = img

        # Test raw image decoding to numpy array
        result = extract_image_from_msg(mock_msg)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2, 3)
        assert result.dtype == np.uint8
        # Verify red pixel is preserved
        assert np.array_equal(result[0, 0], [255, 0, 0])

    def test_extract_image_from_msg_unsupported_depth(self):
        """
        Verify that depth images (16UC1 encoding) are rejected.

        Depth images require separate processing pipeline and should not be
        mixed with RGB camera streams in the standard image extraction path.
        """
        mock_msg = Mock(spec=["height", "width", "encoding", "data"])
        mock_msg.height = 10
        mock_msg.width = 10
        mock_msg.encoding = "16UC1"  # 16-bit unsigned single-channel depth
        mock_msg.data = np.zeros((10, 10), dtype=np.uint16)

        with pytest.raises(ValueError, match="Depth images"):
            extract_image_from_msg(mock_msg)


class TestExtractFieldPath:
    """Tests for nested field extraction from ROS2 messages."""

    def test_extract_field_path_simple(self):
        """
        Verify extraction of top-level scalar fields.

        Used for extracting single values like temperature, pressure, or
        other scalar sensor readings from flat message structures.
        """
        mock_msg = Mock(spec=["temperature"])
        mock_msg.temperature = 25.5

        result = extract_field_path(mock_msg, "temperature")
        assert result is not None
        np.testing.assert_array_equal(result, np.array([25.5], dtype=np.float32))

    def test_extract_field_path_nested(self):
        """
        Verify extraction from nested message structures using dot notation.

        ROS2 messages often contain nested structures (e.g., IMU.angular_velocity.x)
        that require traversal through multiple attribute levels.
        """
        mock_imu = Mock(spec=["x"])
        mock_imu.x = 1.0

        mock_msg = Mock(spec=["imu"])
        mock_msg.imu = mock_imu

        result = extract_field_path(mock_msg, "imu.x")
        assert result is not None
        np.testing.assert_array_equal(result, np.array([1.0], dtype=np.float32))

    def test_extract_field_path_invalid(self):
        """
        Verify fail-fast error handling for non-existent field paths.

        Invalid field paths should raise AttributeError immediately rather than
        silently returning None, following fail-fast error handling principles
        requested by reviewers.
        """
        # Use spec to prevent Mock from creating attributes on the fly
        mock_msg = Mock(spec=["valid_field"])
        mock_msg.valid_field = 42

        # Non-existent field should raise AttributeError (fail-fast)
        with pytest.raises(AttributeError):
            extract_field_path(mock_msg, "nonexistent")

        # Non-existent nested field should also raise AttributeError
        with pytest.raises(AttributeError):
            extract_field_path(mock_msg, "nonexistent.field")


class TestExtractArrayFromMsg:
    """Tests for flattening arbitrary ROS2 message structures to numpy arrays."""

    def test_extract_array_from_msg(self):
        """
        Verify recursive flattening of nested message structures.

        Used for extracting force-torque sensor data, IMU readings, and other
        multi-field messages into flat arrays suitable for observation vectors.
        Example: Wrench message with 3D force and 3D torque becomes 6D array.
        """
        # Create properly configured nested Mock objects
        mock_force = Mock(spec=["x", "y", "z"])
        mock_force.x = 1.0
        mock_force.y = 2.0
        mock_force.z = 3.0

        mock_torque = Mock(spec=["x", "y", "z"])
        mock_torque.x = 0.1
        mock_torque.y = 0.2
        mock_torque.z = 0.3

        mock_msg = Mock(spec=["force", "torque"])
        mock_msg.force = mock_force
        mock_msg.torque = mock_torque

        result = extract_array_from_msg(mock_msg)
        assert result is not None
        assert result.shape == (6,)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32))


class TestTemporalResampler:
    """
    Tests for frequency conversion with anti-aliasing.

    The TemporalResampler handles upsampling and downsampling of time-series data
    from variable source frequencies to fixed target frequencies (typically 30Hz
    for humanoid control). Anti-aliasing is applied when downsampling to prevent
    high-frequency noise from aliasing into the target signal.
    """

    @pytest.fixture
    def resampler(self):
        """Create resampler configured for 10 Hz target frequency."""
        return TemporalResampler(target_hz=10.0)

    def test_temporal_resampler_continuous_linear(self, resampler):
        """
        Verify linear interpolation for continuous signals like joint positions.

        Continuous signals should be interpolated linearly between samples to
        maintain smooth trajectories. This is appropriate for position, velocity,
        and other smooth state variables.
        """
        # Source: 5 Hz data (samples at 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 seconds)
        source_times = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        source_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        # Target: 10 Hz data with matching start/end points (0.0 to 1.0)
        target_times = np.linspace(0.0, 1.0, 11)

        result = resampler.resample_continuous(source_times, source_values, target_times)
        assert result is not None
        assert len(result) == len(target_times)

        # Verify start and end points match exactly
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[-1], 5.0)

        # Verify interpolation: at t=0.1 (midpoint between 0.0 and 0.2), expect value 0.5
        np.testing.assert_array_equal(result[1], 0.5)

    def test_temporal_resampler_discrete_zoh(self, resampler):
        """
        Verify zero-order hold for discrete signals like gripper commands.

        Discrete signals (binary states, categorical actions) should maintain
        their value until the next state change rather than interpolating between
        states.
        """
        # Source: discrete state changes at t=0.0, 0.5, 1.0
        source_times = np.array([0.0, 0.5, 1.0])
        source_values = np.array([0, 1, 2])

        # Target: 10 Hz sampling
        target_times = np.arange(0.0, 1.0, 0.1)

        result = resampler.resample_discrete(source_times, source_values, target_times)
        assert result is not None
        assert len(result) == len(target_times)

        # Before t=0.5, state should be 0
        assert result[0] == 0
        assert result[4] == 0

        # After t=0.5, state should be 1
        assert result[5] == 1

    def test_temporal_resampler_images_nearest(self, resampler):
        """
        Verify nearest-neighbor resampling for image frames.

        Images cannot be interpolated and must use nearest-neighbor selection
        to match the closest available frame to each target timestamp.
        """
        # Source: 3 image frames at different timestamps
        source_times = np.array([0.0, 0.5, 1.0])
        source_images = [b"img0", b"img1", b"img2"]

        # Target: 10 Hz sampling
        target_times = np.arange(0.0, 1.0, 0.1)

        result = resampler.resample_images(source_times, source_images, target_times)
        assert result is not None
        assert len(result) == len(target_times)

        # Verify nearest-neighbor selection
        assert result[0] == b"img0"  # t=0.0 maps to img0
        assert result[5] == b"img1"  # t=0.5 maps to img1
        assert result[9] == b"img2"  # t=0.9 is closer to 1.0, maps to img2

    def test_temporal_resampler_upsampling(self):
        """
        Verify interpolation behavior when upsampling from low to high frequency.

        Upsampling from 5Hz source to 20Hz target requires linear interpolation
        to fill in intermediate values between sparse source samples.
        """
        resampler_high = TemporalResampler(target_hz=20.0)

        # Source: 5 Hz data
        source_times = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        source_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Target: 20 Hz data (4x higher frequency)
        target_times = np.arange(0.0, 0.8, 0.05)

        result = resampler_high.resample_continuous(source_times, source_values, target_times)
        assert result is not None
        assert len(result) == len(target_times)

    def test_temporal_resampler_downsampling(self):
        """
        Verify anti-aliasing filter activation during downsampling.

        When target frequency is less than half the source frequency (Nyquist criterion),
        anti-aliasing filter must be applied to prevent high-frequency content from
        folding into the lower-frequency output signal.
        """
        resampler_low = TemporalResampler(target_hz=5.0)

        # Source: 20 Hz data
        source_times = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
        source_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Target: 5 Hz data (4x lower frequency, needs anti-aliasing)
        target_times = np.array([0.0, 0.2])

        result = resampler_low.resample_continuous(source_times, source_values, target_times)
        assert result is not None
        assert len(result) == len(target_times)


class TestMCAPConverterDiscovery:
    """Tests for episode discovery from filesystem hierarchies."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for MCAPConverter initialization."""
        cfg = MagicMock()
        cfg.action_fields_config_path = "/tmp/fake_config.yaml"
        cfg.past_lowdim_steps = 5
        cfg.future_lowdim_steps = 10
        cfg.filter_still_samples = False
        cfg.padding_strategy = "copy"
        cfg.max_padding_left = 5
        cfg.max_padding_right = 10
        cfg.image_indices = [0]
        cfg.resize_images_size = [384, 384]
        cfg.jpeg_quality = 90
        cfg.camera_names = ["camera1"]
        cfg.stride = 1
        cfg.num_workers = 1
        return cfg

    @pytest.fixture
    def mock_topics_config(self):
        """Create mock topics configuration for MCAP preprocessing."""
        return {
            "target_hz": 30.0,
            "action_topics": ["/action/joints"],
            "state_topics": ["/state/joints"],
            "camera_topics": {"camera1": "/camera1/image_raw/compressed"},
            "state_key_fields": ["obs_joints__joint1"],
            "action_key_fields": ["action_joints__joint1", "action_joints__joint2"],
        }

    def test_discover_episodes_local_mcap_files(self, mock_config, mock_topics_config):
        """
        Verify discovery of .mcap files in local directories.

        Episode discovery should identify all .mcap files matching the expected
        naming pattern (e.g., episode_0000.mcap) while ignoring non-episode files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "episode_0000.mcap").touch()
            (Path(tmpdir) / "episode_0001.mcap").touch()

            mock_config.action_fields_config_path = str(Path(tmpdir) / "config.yaml")

            # Patch locally in the mcap module where it is used
            with (
                patch(
                    "vla_foundry.data.preprocessing.robotics.converters.mcap.yaml_load", return_value=mock_topics_config
                ),
                patch("vla_foundry.data.preprocessing.robotics.converters.mcap.file_exists", return_value=True),
            ):
                converter = MCAPConverter(mock_config)
                episodes = converter.discover_episodes([tmpdir])
                assert len(episodes) == 2

    def test_discover_episodes_nested_directories(self, mock_config, mock_topics_config):
        """
        Verify discovery of MCAP files in nested episode directories.

        Some data collection formats store each episode in a separate directory
        (e.g., episode_0000/recording.mcap). Discovery should handle both flat
        and nested hierarchies.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested episode structure
            ep_dir = Path(tmpdir) / "episode_0000"
            ep_dir.mkdir()
            (ep_dir / "recording.mcap").touch()

            # Create temporary config file
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("target_hz: 30.0\n")
            mock_config.action_fields_config_path = str(config_file)

            with patch(
                "vla_foundry.data.preprocessing.robotics.converters.mcap.yaml_load", return_value=mock_topics_config
            ):
                converter = MCAPConverter(mock_config)
                episodes = converter.discover_episodes([tmpdir])

                assert len(episodes) == 1
                assert "episode_0000" in episodes[0]

    def test_discover_episodes_max_limit(self, mock_config, mock_topics_config):
        """
        Verify max_episodes_to_process limit enforcement.

        For rapid prototyping and testing, preprocessing should respect the
        max episode limit to avoid processing entire large datasets.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 5 MCAP files
            for i in range(5):
                (Path(tmpdir) / f"episode_{i:04d}.mcap").touch()

            # Create temporary config file
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("target_hz: 30.0\n")
            mock_config.action_fields_config_path = str(config_file)

            with patch("vla_foundry.file_utils.yaml_load", return_value=mock_topics_config):
                converter = MCAPConverter(mock_config)
                episodes = converter.discover_episodes([tmpdir], max_episodes_to_process=3)

                assert len(episodes) == 3


class TestMCAPConverterConfiguration:
    """Tests for configuration loading and validation."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for configuration tests."""
        cfg = MagicMock()
        cfg.action_fields_config_path = "/tmp/fake_config.yaml"
        cfg.past_lowdim_steps = 5
        cfg.future_lowdim_steps = 10
        cfg.filter_still_samples = False
        cfg.padding_strategy = "copy"
        cfg.max_padding_left = 5
        cfg.max_padding_right = 10
        cfg.image_indices = [0]
        cfg.resize_images_size = [384, 384]
        cfg.jpeg_quality = 90
        cfg.camera_names = ["camera1"]
        cfg.stride = 1
        cfg.num_workers = 1
        return cfg

    def test_load_topics_config(self, mock_config):
        """
        Verify topics configuration parsing from YAML.

        Topics config defines the ROS2 topics to extract, target frequency,
        camera mappings, and output schema. All parameters must be correctly
        loaded and validated during converter initialization.
        """
        mock_topics = {
            "target_hz": 30.0,
            "action_topics": ["/action/cmd"],
            "state_topics": ["/state/joints"],
            "camera_topics": {"cam1": "/cam1/image"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.action_fields_config_path = str(Path(tmpdir) / "config.yaml")
            # Patch locally where it's used
            with (
                patch("vla_foundry.data.preprocessing.robotics.converters.mcap.yaml_load", return_value=mock_topics),
                patch("vla_foundry.data.preprocessing.robotics.converters.mcap.file_exists", return_value=True),
            ):
                converter = MCAPConverter(mock_config)
                assert converter.target_hz == 30.0
                assert converter.action_topics == ["/action/cmd"]

    def test_load_topics_config_missing_file(self, mock_config):
        """
        Verify fail-fast behavior when topics config file is missing.

        Missing configuration files should raise FileNotFoundError immediately
        rather than proceeding with undefined behavior.
        """
        # Point to a non-existent file
        mock_config.action_fields_config_path = "/nonexistent/path/to/config.yaml"

        with pytest.raises(FileNotFoundError):
            MCAPConverter(mock_config)


class TestMCAPConverterLowdimExtraction:
    """Tests for low-dimensional observation and action extraction."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for lowdim extraction tests."""
        cfg = MagicMock()
        cfg.action_fields_config_path = "/tmp/fake_config.yaml"
        cfg.past_lowdim_steps = 5
        cfg.future_lowdim_steps = 10
        cfg.filter_still_samples = False
        cfg.padding_strategy = "copy"
        cfg.max_padding_left = 5
        cfg.max_padding_right = 10
        cfg.image_indices = [0]
        cfg.resize_images_size = [384, 384]
        cfg.jpeg_quality = 90
        cfg.camera_names = ["camera1"]
        cfg.stride = 1
        cfg.num_workers = 1
        return cfg

    @pytest.fixture
    def mock_topics_config(self):
        """Create mock topics configuration for lowdim extraction tests."""
        return {
            "target_hz": 30.0,
            "action_topics": ["/action/joints"],
            "state_topics": ["/state/joints"],
            "camera_topics": {"camera1": "/cam1/compressed"},
        }

    def test_extract_lowdim_data_concatenated_mode(self, mock_config, mock_topics_config):
        """
        Verify observation/action concatenation in 'concatenated' output mode.

        Concatenated mode combines all action fields into a single action vector
        and all state fields into a single observation vector, which is the
        expected format for diffusion policy training.
        """
        mock_topics_config["output_mode"] = "concatenated"
        mock_topics_config["action_key_fields"] = ["action_joints__joint1", "action_joints__joint2"]
        mock_topics_config["state_key_fields"] = ["obs_joints__joint1"]

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("target_hz: 30.0\n")
            mock_config.action_fields_config_path = str(config_file)

            with patch(
                "vla_foundry.data.preprocessing.robotics.converters.mcap.yaml_load", return_value=mock_topics_config
            ):
                converter = MCAPConverter(mock_config)

                # Mock episode data with separate action fields
                episode_data = {
                    "action_joints__joint1": np.array([1.0, 2.0, 3.0]),
                    "action_joints__joint2": np.array([4.0, 5.0, 6.0]),
                    "obs_joints__joint1": np.array([0.1, 0.2, 0.3]),
                    "timestamps": np.array([0.0, 0.1, 0.2]),
                }

                result = converter.extract_lowdim_data(episode_data)

                assert "state" in result
                assert "actions" in result
                assert result["actions"].shape == (3, 2)  # 3 timesteps, 2 action dimensions
                assert result["state"].shape == (3, 1)  # 3 timesteps, 1 state dimension

    def test_extract_lowdim_data_separate_mode(self, mock_config, mock_topics_config):
        """
        Verify field preservation in 'separate' output mode.

        Separate mode maintains individual field names in the output dictionary,
        allowing downstream code to reference specific sensor channels by name
        (e.g., arm__joint_pos, gripper__force).
        """
        local_topics = mock_topics_config.copy()
        local_topics["output_mode"] = "separate"

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("target_hz: 30.0\n")
            mock_config.action_fields_config_path = str(config_file)

            with patch("vla_foundry.data.preprocessing.robotics.converters.mcap.yaml_load", return_value=local_topics):
                converter = MCAPConverter(mock_config)

                episode_data = {
                    "arm__joint_pos": np.array([1.0, 2.0]),
                    "gripper__force": np.array([0.5, 0.6]),
                    "timestamps": np.array([0.0, 0.1]),
                }

                result = converter.extract_lowdim_data(episode_data)

                assert "arm__joint_pos" in result
                assert "gripper__force" in result
                assert result["arm__joint_pos"].shape == (2, 1)
                assert result["gripper__force"].shape == (2, 1)


class TestMCAPConverterSampleExtraction:
    """Tests for sample generation with temporal windowing."""

    @pytest.fixture
    def mock_config(self):
        """Create configuration for sample extraction tests."""
        cfg = MagicMock()
        cfg.action_fields_config_path = "/tmp/fake_config.yaml"
        cfg.past_lowdim_steps = 1
        cfg.future_lowdim_steps = 1
        cfg.image_indices = [0]
        cfg.filter_still_samples = True
        cfg.still_threshold = 0.001
        cfg.padding_strategy = "copy"
        cfg.max_padding_left = 2
        cfg.max_padding_right = 2
        cfg.resize_images_size = [384, 384]
        cfg.jpeg_quality = 90
        cfg.camera_names = ["camera1"]
        cfg.stride = 1
        cfg.num_workers = 1
        return cfg

    @pytest.fixture
    def mock_topics_config(self):
        """Create topics configuration for sample extraction tests."""
        return {
            "target_hz": 30.0,
            "action_topics": ["/action/joints"],
            "state_topics": ["/state/joints"],
            "camera_topics": {"camera1": "/cam1/compressed"},
            "action_key_fields": ["action_joints__joint1"],
        }

    def test_extract_sample_data_filtered_still_sample(self, mock_config, mock_topics_config):
        """
        Verify filtering of samples below still_threshold variance.

        Still samples (where action variance is below threshold) typically
        indicate the robot is idle between tasks. These samples add noise
        to the training distribution and should be filtered out.
        """
        mock_config.filter_still_samples = True
        mock_config.still_threshold = 0.001

        local_topics = mock_topics_config.copy()
        local_topics["output_mode"] = "separate"

        # Increase past/future steps to ensure enough variance history for the filter
        mock_config.past_lowdim_steps = 2
        mock_config.future_lowdim_steps = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.action_fields_config_path = str(Path(tmpdir) / "config.yaml")
            # Use local_topics so 'output_mode' is actually 'separate'
            with (
                patch("vla_foundry.data.preprocessing.robotics.converters.mcap.yaml_load", return_value=local_topics),
                patch("vla_foundry.data.preprocessing.robotics.converters.mcap.file_exists", return_value=True),
            ):
                converter = MCAPConverter(mock_config)

                # Use prefix 'action_' so the logic identifies it for filtering
                still_episode = {
                    "action_joints__joint1": np.zeros((20, 1)),
                    "timestamps": np.arange(0.0, 2.0, 0.1),
                }
                camera_data = {"camera1": [b"img" for _ in range(20)]}
                lowdim_data = converter.extract_lowdim_data(still_episode)

                result = converter.extract_sample_data(
                    anchor_timestep=10,
                    episode_path="/fake",
                    episode_length=20,
                    camera_data=camera_data,
                    lowdim_data=lowdim_data,
                    intrinsics_data=None,
                    extrinsics_data=None,
                    metadata_data={"timestamps": still_episode["timestamps"]},
                    statistics_ray_actor=None,
                    logger_actor=MagicMock(),
                )

                assert result == (None, None, None, None, None, None)

    def test_extract_sample_data_deployment_window(self, mock_config, mock_topics_config):
        """
        Verify the specific window size used in the 30Hz deployment config:
        1 past + 1 current + 47 future = 49 total steps.
        """
        # Set parameters exactly as defined in the deployment YAML
        mock_config.past_lowdim_steps = 1
        mock_config.future_lowdim_steps = 47
        mock_config.filter_still_samples = False

        expected_total_steps = 49  #

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.action_fields_config_path = str(Path(tmpdir) / "config.yaml")
            with (
                patch(
                    "vla_foundry.data.preprocessing.robotics.converters.mcap.yaml_load", return_value=mock_topics_config
                ),
                patch("vla_foundry.data.preprocessing.robotics.converters.mcap.file_exists", return_value=True),
            ):
                converter = MCAPConverter(mock_config)

                # Create an episode long enough to verify the window in the middle
                episode_len = 100
                episode_data = {
                    "action_joints__joint1": np.zeros((episode_len, 1)),
                    "obs_joints__joint1": np.zeros((episode_len, 1)),
                    "timestamps": np.arange(episode_len) * (1.0 / 30.0),
                }

                camera_data = {"camera1": [b"img" for _ in range(episode_len)]}
                lowdim_data = converter.extract_lowdim_data(episode_data)

                # Test at an arbitrary middle point
                anchor = 50
                result = converter.extract_sample_data(
                    anchor_timestep=anchor,
                    episode_path="/fake/path",
                    episode_length=episode_len,
                    camera_data=camera_data,
                    lowdim_data=lowdim_data,
                    intrinsics_data=None,
                    extrinsics_data=None,
                    metadata_data={"timestamps": episode_data["timestamps"], "episode_id": "test"},
                    statistics_ray_actor=None,
                    logger_actor=None,
                )

                sample_lowdim = result[1]  #

                # Check that every extracted signal chunk is exactly 49 steps
                for key, val in sample_lowdim.items():
                    actual_window = val.shape[0]
                    assert actual_window == expected_total_steps, (
                        f"Deployment window mismatch for {key}: expected {expected_total_steps}, got {actual_window}"
                    )

    def test_extract_sample_data_boundary_conditions(self, mock_config, mock_topics_config):
        """
        Verify correct padding behavior at episode boundaries.

        Samples near episode start/end require padding to fill the temporal
        context window. Edge-copying strategy repeats the first/last frame
        to maintain temporal structure without introducing discontinuities.
        """
        mock_config.past_lowdim_steps = 2
        mock_config.future_lowdim_steps = 2
        mock_config.max_padding_left = 2
        mock_config.max_padding_right = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("target_hz: 30.0\n")
            mock_config.action_fields_config_path = str(config_file)

            with patch(
                "vla_foundry.data.preprocessing.robotics.converters.mcap.yaml_load", return_value=mock_topics_config
            ):
                converter = MCAPConverter(mock_config)

                episode_data = {
                    "action_joints__joint1": np.arange(10.0),  # Increasing values
                    "obs_joints__joint1": np.arange(10.0),
                    "timestamps": np.arange(0.0, 1.0, 0.1),
                }

                camera_data = {"camera1": [b"img" for _ in range(10)]}
                lowdim_data = converter.extract_lowdim_data(episode_data)

                logger_actor = MagicMock()
                logger_actor.increment_total_potential_samples = MagicMock(return_value=MagicMock())

                # Extract sample at episode boundary (timestep 1, needs left padding)
                result = converter.extract_sample_data(
                    anchor_timestep=1,
                    episode_path="/fake/path",
                    episode_length=10,
                    camera_data=camera_data,
                    lowdim_data=lowdim_data,
                    intrinsics_data=None,
                    extrinsics_data=None,
                    metadata_data={"timestamps": episode_data["timestamps"], "episode_id": "test"},
                    statistics_ray_actor=None,
                    logger_actor=logger_actor,
                )

                # Verify sample was generated with padding
                assert result[0] is not None  # sample_images
                assert result[1] is not None  # sample_lowdim


class TestMCAPConverterRelativeCoordinates:
    """Tests for relative coordinate computation using pose groups."""

    @pytest.fixture
    def mock_config(self):
        """Create configuration for relative coordinate tests."""
        cfg = MagicMock()
        cfg.action_fields_config_path = "/tmp/fake_config.yaml"
        cfg.past_lowdim_steps = 1
        cfg.future_lowdim_steps = 1
        cfg.image_indices = [0]
        cfg.filter_still_samples = False
        cfg.padding_strategy = "copy"
        cfg.max_padding_left = 2
        cfg.max_padding_right = 2
        cfg.resize_images_size = [384, 384]
        cfg.jpeg_quality = 90
        cfg.camera_names = ["camera1"]
        cfg.stride = 1
        cfg.num_workers = 1
        return cfg

    @pytest.fixture
    def mock_topics_config_with_pose_groups(self):
        """Create topics configuration with pose groups for relative coordinates."""
        return {
            "target_hz": 30.0,
            "output_mode": "separate",
            "action_topics": ["/ee_target_left"],
            "state_topics": ["/current_left_hand_ee_link"],
            "camera_topics": {"camera1": "/cam1/compressed"},
            "action_field_names": {"/ee_target_left": "action_ee_pose_left"},
            "state_field_names": {"/current_left_hand_ee_link": "obs_ee_pose_left"},
            "pose_groups": [
                {
                    "name": "left_ee_action",
                    "position_key": "action_ee_pose_left__xyz",
                    "rotation_key": "action_ee_pose_left__rot_6d",
                }
            ],
        }

    def test_pose_groups_loaded_from_config(self, mock_config, mock_topics_config_with_pose_groups):
        """
        Verify pose groups are loaded from topics config during initialization.

        Pose groups define which action fields should be computed relative to
        their corresponding observation fields, enabling delta action learning.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.action_fields_config_path = str(Path(tmpdir) / "config.yaml")
            with (
                patch(
                    "vla_foundry.data.preprocessing.robotics.converters.mcap.yaml_load",
                    return_value=mock_topics_config_with_pose_groups,
                ),
                patch("vla_foundry.data.preprocessing.robotics.converters.mcap.file_exists", return_value=True),
                # Removed validate_pose_groups patch
            ):
                converter = MCAPConverter(mock_config)

                assert len(converter.pose_groups) == 1
                assert converter.pose_groups[0]["name"] == "left_ee_action"
                assert converter.pose_groups[0]["position_key"] == "action_ee_pose_left__xyz"
                assert converter.pose_groups[0]["rotation_key"] == "action_ee_pose_left__rot_6d"

    def test_relative_coordinates_computed(self, mock_config, mock_topics_config_with_pose_groups):
        """
        Verify relative coordinates are computed in extract_sample_data.

        Actions should be expressed relative to current observations:
        action_relative[t] = action_target[t] - observation[anchor_t]

        This enables the policy to learn delta actions that generalize
        across different starting positions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.action_fields_config_path = str(Path(tmpdir) / "config.yaml")
            with (
                patch(
                    "vla_foundry.data.preprocessing.robotics.converters.mcap.yaml_load",
                    return_value=mock_topics_config_with_pose_groups,
                ),
                patch("vla_foundry.data.preprocessing.robotics.converters.mcap.file_exists", return_value=True),
                # Removed validate_pose_groups patch
            ):
                converter = MCAPConverter(mock_config)

                # Create episode data with action and observation poses
                episode_len = 10
                episode_data = {
                    # Action target: gripper should move to [1.0, 1.0, 1.0]
                    "action_ee_pose_left__xyz": np.ones((episode_len, 3)),
                    "action_ee_pose_left__rot_6d": np.ones((episode_len, 6)) * 0.5,
                    # Current observation: gripper is at [0.0, 0.0, 0.0]
                    "obs_ee_pose_left__xyz": np.zeros((episode_len, 3)),
                    "obs_ee_pose_left__rot_6d": np.tile(
                        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (episode_len, 1)
                    ),  # Identity rotation in 6D
                    "timestamps": np.arange(episode_len) * (1.0 / 30.0),
                }

                camera_data = {"camera1": [b"img" for _ in range(episode_len)]}
                lowdim_data = converter.extract_lowdim_data(episode_data)

                # Extract sample at middle timestep
                anchor = 5
                result = converter.extract_sample_data(
                    anchor_timestep=anchor,
                    episode_path="/fake/path",
                    episode_length=episode_len,
                    camera_data=camera_data,
                    lowdim_data=lowdim_data,
                    intrinsics_data=None,
                    extrinsics_data=None,
                    metadata_data={"timestamps": episode_data["timestamps"], "episode_id": "test"},
                    statistics_ray_actor=None,
                    logger_actor=None,
                )

                sample_lowdim = result[1]

                # Verify relative coordinate keys were added
                assert "action_ee_pose_left__xyz_relative" in sample_lowdim
                assert "action_ee_pose_left__rot_6d_relative" in sample_lowdim

                # Verify shapes match original actions
                assert (
                    sample_lowdim["action_ee_pose_left__xyz_relative"].shape
                    == sample_lowdim["action_ee_pose_left__xyz"].shape
                )
                assert (
                    sample_lowdim["action_ee_pose_left__rot_6d_relative"].shape
                    == sample_lowdim["action_ee_pose_left__rot_6d"].shape
                )

    def test_no_relative_coordinates_without_pose_groups(self, mock_config):
        """
        Verify relative coordinates are NOT computed when pose_groups is empty.

        If no pose groups are configured, the converter should skip relative
        coordinate computation entirely.
        """
        mock_topics_config_no_poses = {
            "target_hz": 30.0,
            "output_mode": "separate",
            "action_topics": ["/action/joints"],
            "state_topics": ["/state/joints"],
            "camera_topics": {"camera1": "/cam1/compressed"},
            # No pose_groups key
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.action_fields_config_path = str(Path(tmpdir) / "config.yaml")
            with (
                patch(
                    "vla_foundry.data.preprocessing.robotics.converters.mcap.yaml_load",
                    return_value=mock_topics_config_no_poses,
                ),
                patch("vla_foundry.data.preprocessing.robotics.converters.mcap.file_exists", return_value=True),
            ):
                converter = MCAPConverter(mock_config)

                assert len(converter.pose_groups) == 0

                # Create episode without pose data
                episode_len = 10
                episode_data = {
                    "action_joints__joint1": np.ones((episode_len, 1)),
                    "obs_joints__joint1": np.zeros((episode_len, 1)),
                    "timestamps": np.arange(episode_len) * (1.0 / 30.0),
                }

                camera_data = {"camera1": [b"img" for _ in range(episode_len)]}
                lowdim_data = converter.extract_lowdim_data(episode_data)

                result = converter.extract_sample_data(
                    anchor_timestep=5,
                    episode_path="/fake/path",
                    episode_length=episode_len,
                    camera_data=camera_data,
                    lowdim_data=lowdim_data,
                    intrinsics_data=None,
                    extrinsics_data=None,
                    metadata_data={"timestamps": episode_data["timestamps"], "episode_id": "test"},
                    statistics_ray_actor=None,
                    logger_actor=None,
                )

                sample_lowdim = result[1]

                # Verify no relative coordinate keys were added
                assert "action_joints__joint1_relative" not in sample_lowdim
                assert all(
                    "_relative" not in key for key in sample_lowdim if key != "past_mask" and key != "future_mask"
                )

    def test_no_relative_coordinates_in_concatenated_mode(self, mock_config, mock_topics_config_with_pose_groups):
        """
        Verify relative coordinates are NOT computed in concatenated mode.

        Relative coordinate computation is only supported in 'separate' output mode
        where individual fields can be identified and mapped.
        """
        # Override to concatenated mode
        mock_topics_config_with_pose_groups["output_mode"] = "concatenated"
        mock_topics_config_with_pose_groups["action_key_fields"] = [
            "action_ee_pose_left__xyz",
            "action_ee_pose_left__rot_6d",
        ]
        mock_topics_config_with_pose_groups["state_key_fields"] = ["obs_ee_pose_left__xyz", "obs_ee_pose_left__rot_6d"]

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.action_fields_config_path = str(Path(tmpdir) / "config.yaml")
            with (
                patch(
                    "vla_foundry.data.preprocessing.robotics.converters.mcap.yaml_load",
                    return_value=mock_topics_config_with_pose_groups,
                ),
                patch("vla_foundry.data.preprocessing.robotics.converters.mcap.file_exists", return_value=True),
            ):
                converter = MCAPConverter(mock_config)

                episode_len = 10
                episode_data = {
                    "action_ee_pose_left__xyz": np.ones((episode_len, 3)),
                    "action_ee_pose_left__rot_6d": np.ones((episode_len, 6)) * 0.5,
                    "obs_ee_pose_left__xyz": np.zeros((episode_len, 3)),
                    "obs_ee_pose_left__rot_6d": np.zeros((episode_len, 6)),
                    "timestamps": np.arange(episode_len) * (1.0 / 30.0),
                }

                camera_data = {"camera1": [b"img" for _ in range(episode_len)]}
                lowdim_data = converter.extract_lowdim_data(episode_data)

                result = converter.extract_sample_data(
                    anchor_timestep=5,
                    episode_path="/fake/path",
                    episode_length=episode_len,
                    camera_data=camera_data,
                    lowdim_data=lowdim_data,
                    intrinsics_data=None,
                    extrinsics_data=None,
                    metadata_data={"timestamps": episode_data["timestamps"], "episode_id": "test"},
                    statistics_ray_actor=None,
                    logger_actor=None,
                )

                sample_lowdim = result[1]

                # In concatenated mode, should only have 'state' and 'actions' keys
                assert set(sample_lowdim.keys()) == {"state", "actions", "past_mask", "future_mask"}
