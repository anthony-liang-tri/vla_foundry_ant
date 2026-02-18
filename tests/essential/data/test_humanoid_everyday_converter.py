"""Tests for HumanoidEverydayConverter using truncated real data."""

import io
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from vla_foundry.data.preprocessing.robotics.converters.humanoid_everyday import HumanoidEverydayConverter
from vla_foundry.data.preprocessing.robotics.preprocess_params import HumanoidEverydayPreprocessParams

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_DATA_DIR = str(
    Path(__file__).resolve().parent.parent / "test_assets" / "small_truncated_humanoid_everyday_dataset"
)

NUM_FRAMES = 3  # frames per truncated episode in above test data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config(output_dir: str, **overrides: Any) -> HumanoidEverydayPreprocessParams:
    """Create a HumanoidEverydayPreprocessParams with test defaults.
    Args:
        output_dir: Path to the output directory.
        overrides: Additional overrides for the config.

    Returns:
        HumanoidEverydayPreprocessParams with test defaults.
    """
    kwargs = dict(
        source_episodes=[TEST_DATA_DIR],
        output_dir=output_dir,
        use_depth_data=True,
        type="humanoid_everyday",
        # Match production YAML values (dataclass defaults are stricter and
        # would filter out every sample from a 3-frame episode).
        max_padding_left=3,
        max_padding_right=15,
        resize_images_size=[384, 384],
        depth_resolution=[480, 640],
    )
    kwargs.update(overrides)
    return HumanoidEverydayPreprocessParams(**kwargs)


def _mock_logger_actor() -> MagicMock:
    """Create a mock logger actor that satisfies .remote() calls.
    Returns:
        MagicMock of the logger actor.
    """
    actor = MagicMock()
    actor.increment_total_potential_samples.remote = MagicMock()
    actor.increment_padding_samples_filtered.remote = MagicMock()
    actor.increment_still_samples_filtered.remote = MagicMock()
    return actor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiscoverEpisodes:
    """Tests for episode discovery with various filters."""

    def test_task_filter(self) -> None:
        """Test task filter restricts discovery to matching tasks."""
        with tempfile.TemporaryDirectory() as tmp_out:
            # Temp dir satisfies the required output_dir param; not written to by discover_episodes.
            cfg = _mock_config(tmp_out)
            converter = HumanoidEverydayConverter(cfg)

            # Without filter: both episodes found
            all_episodes = converter.discover_episodes([TEST_DATA_DIR])
            assert len(all_episodes) == 2
            task_names = {os.path.basename(os.path.dirname(ep)) for ep in all_episodes}
            assert "drag_a_white_board" in task_names
            assert "pick_a_bag_of_fork_and_place_it_in_a_container" in task_names

            # With task filter: only the matching task
            cfg_filtered = _mock_config(tmp_out, task_filter=["drag_a_white_board"])
            converter_filtered = HumanoidEverydayConverter(cfg_filtered)
            episodes_filtered = converter_filtered.discover_episodes([TEST_DATA_DIR])
            assert len(episodes_filtered) == 1
            task_names = {os.path.basename(os.path.dirname(ep)) for ep in episodes_filtered}
            assert "drag_a_white_board" in task_names

    def test_episode_filter(self) -> None:
        """Test episode filter restricts discovery to matching episodes."""
        with tempfile.TemporaryDirectory() as tmp_out:
            # Filter with a non-existent episode returns an empty list
            cfg_miss = _mock_config(tmp_out, episode_filter=["episode_99"])
            converter_miss = HumanoidEverydayConverter(cfg_miss)
            assert len(converter_miss.discover_episodes([TEST_DATA_DIR])) == 0

            # Filter with episode_0 finds both tasks (each has an episode_0)
            cfg_hit = _mock_config(tmp_out, episode_filter=["episode_0"])
            converter_hit = HumanoidEverydayConverter(cfg_hit)
            episodes_hit = converter_hit.discover_episodes([TEST_DATA_DIR])
            assert len(episodes_hit) == 2
            assert all("episode_0" in ep for ep in episodes_hit)

    def test_embodiment_filter(self) -> None:
        """Test embodiment filter restricts discovery by robot_type."""
        with tempfile.TemporaryDirectory() as tmp_out:
            # Without filter: both episodes found
            cfg = _mock_config(tmp_out)
            converter = HumanoidEverydayConverter(cfg)
            all_episodes = converter.discover_episodes([TEST_DATA_DIR])
            assert len(all_episodes) == 2

            # Filter for h1 only — picks up the episode missing robot_type (defaults to h1)
            cfg_h1 = _mock_config(tmp_out, embodiment_filter=["h1"])
            converter_h1 = HumanoidEverydayConverter(cfg_h1)
            episodes_h1 = converter_h1.discover_episodes([TEST_DATA_DIR])
            assert len(episodes_h1) == 1
            assert "pick_a_bag_of_fork_and_place_it_in_a_container" in episodes_h1[0]

            # Filter for g1 only — picks up the episode with explicit robot_type="g1"
            cfg_g1 = _mock_config(tmp_out, embodiment_filter=["g1"])
            converter_g1 = HumanoidEverydayConverter(cfg_g1)
            episodes_g1 = converter_g1.discover_episodes([TEST_DATA_DIR])
            assert len(episodes_g1) == 1
            assert "drag_a_white_board" in episodes_g1[0]

            # Filter for both: all episodes returned
            cfg_both = _mock_config(tmp_out, embodiment_filter=["h1", "g1"])
            converter_both = HumanoidEverydayConverter(cfg_both)
            episodes_both = converter_both.discover_episodes([TEST_DATA_DIR])
            assert len(episodes_both) == 2


class TestConvertSmoke:
    """End-to-end smoke test: load, convert, and verify output tars."""

    # Common lowdim keys → expected second dimension for both embodiments.
    # When the dimension differs by embodiment, a dict {robot_type: dim} is used.
    COMMON_LOWDIM_KEYS = {
        "humanoid__state__arm": 14,
        "humanoid__state__leg": {"g1": 15, "h1": 13},
        "humanoid__state__hand": {"g1": 14, "h1": 12},
        "humanoid__state__imu_quaternion": 4,
        "humanoid__state__imu_accelerometer": 3,
        "humanoid__state__imu_gyroscope": 3,
        "humanoid__state__imu_rpy": 3,
        "humanoid__action__left_angles": {"g1": 7, "h1": 12},
        "humanoid__action__right_angles": {"g1": 7, "h1": 12},
        "humanoid__action__sol_q": 14,
        "humanoid__action__tau_ff": 14,
        "humanoid__action__head_rmat": 9,
        "humanoid__action__left_pose": 16,
        "humanoid__action__right_pose": 16,
    }

    # G1-only lowdim keys → expected second dimension for G1 episodes
    G1_ONLY_LOWDIM_KEYS = {
        "humanoid__state__hand_pressure_a": 48,
        "humanoid__state__hand_pressure_b": 18,
        "humanoid__state__odometry_position": 3,
        "humanoid__state__odometry_velocity": 3,
        "humanoid__state__odometry_rpy": 3,
        "humanoid__state__odometry_quat": 4,
    }

    def test_process_episode_produces_valid_tars(self) -> None:
        """Run process_episode on both embodiments and verify output tars."""
        with tempfile.TemporaryDirectory() as tmp_out:
            cfg = _mock_config(tmp_out)

            converter = HumanoidEverydayConverter(cfg)

            episodes = converter.discover_episodes([TEST_DATA_DIR])
            assert len(episodes) == 2, f"Expected 2 episodes from test data, got {len(episodes)}"

            logger_actor = _mock_logger_actor()
            expected_lowdim_len = cfg.past_lowdim_steps + cfg.future_lowdim_steps + 1
            frames_dir = Path(tmp_out) / "frames"

            for episode_path in episodes:
                # Determine embodiment before processing
                robot_type = converter._get_episode_robot_type(episode_path)

                existing_tars = set(frames_dir.glob("*.tar")) if frames_dir.exists() else set()

                # Run the full pipeline for the given episode
                _ = converter.process_episode(
                    episode_path,
                    statistics_ray_actor=None,
                    logger_actor=logger_actor,
                )

                # Find the newly created tars (one per anchor timestep)
                new_tars = set(frames_dir.glob("*.tar")) - existing_tars
                assert len(new_tars) >= 1, f"Expected at least 1 new tar for {episode_path}, got {len(new_tars)}"

                for tar_path in sorted(new_tars):
                    with tarfile.open(tar_path, "r") as tar:
                        members = tar.getnames()

                        # Check for expected file types
                        jpg_files = [m for m in members if m.endswith(".jpg")]
                        png_files = [m for m in members if m.endswith(".png")]
                        npz_files = [m for m in members if m.endswith(".npz")]
                        json_files = [m for m in members if m.endswith(".json")]

                        # RGB images: head_rgb_t-1 and head_rgb_t0
                        assert len(jpg_files) == 2, f"Expected 2 .jpg files, got {jpg_files}"
                        assert any("head_rgb_t-1" in f for f in jpg_files)
                        assert any("head_rgb_t0" in f for f in jpg_files)

                        # Depth images: head_rgb_depth_t-1 and head_rgb_depth_t0
                        assert len(png_files) == 2, f"Expected 2 .png files, got {png_files}"
                        assert any("head_rgb_depth_t-1" in f for f in png_files)
                        assert any("head_rgb_depth_t0" in f for f in png_files)

                        # Lowdim NPZ
                        lowdim_npz = [m for m in npz_files if "lowdim" in m]
                        assert len(lowdim_npz) == 1, f"Expected 1 lowdim .npz, got {lowdim_npz}"

                        # Metadata and language instructions JSON
                        metadata_json = [m for m in json_files if "metadata" in m]
                        lang_json = [m for m in json_files if "language_instructions" in m]
                        assert len(metadata_json) == 1, f"Expected 1 metadata .json, got {metadata_json}"
                        assert len(lang_json) == 1, f"Expected 1 language .json, got {lang_json}"

                        # ------ Verify lowdim shape ------
                        # past_lowdim_steps=1, future_lowdim_steps=14 => window length = 1 + 14 + 1 = 16
                        lowdim_member = tar.extractfile(lowdim_npz[0])
                        lowdim_data = np.load(io.BytesIO(lowdim_member.read()))
                        lowdim_keys = set(lowdim_data.files)

                        # 1. Common keys: present for both embodiments with correct shape
                        for key, expected_dim in self.COMMON_LOWDIM_KEYS.items():
                            dim = expected_dim[robot_type] if isinstance(expected_dim, dict) else expected_dim
                            assert key in lowdim_keys, f"Missing common key {key} in lowdim, got keys: {lowdim_keys}"
                            arr = lowdim_data[key]
                            assert arr.shape == (expected_lowdim_len, dim), (
                                f"{key}: expected shape ({expected_lowdim_len}, {dim}), got {arr.shape}"
                            )

                        # 2. Embodiment-specific keys (G1-only): check shape for G1, absent for others
                        for key, expected_dim in self.G1_ONLY_LOWDIM_KEYS.items():
                            if robot_type == "g1":
                                assert key in lowdim_keys, f"G1 episode missing {key}, got keys: {lowdim_keys}"
                                arr = lowdim_data[key]
                                assert arr.shape == (expected_lowdim_len, expected_dim), (
                                    f"{key}: expected shape ({expected_lowdim_len}, {expected_dim}), got {arr.shape}"
                                )
                            else:
                                assert key not in lowdim_keys, (
                                    f"Non-G1 episode should not have {key}, got keys: {lowdim_keys}"
                                )

                        # 3. Verify masks are present
                        assert "past_mask" in lowdim_keys
                        assert "future_mask" in lowdim_keys
