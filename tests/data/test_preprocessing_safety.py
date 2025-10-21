"""Tests for preprocessing safety checks."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


def _check_pyarrow_available():
    """Check if pyarrow is available."""
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


class TestPreprocessLbmToTarSafety:
    """Tests for preprocessing safety checks in preprocess_lbm_to_tar.py."""

    @patch("vla_foundry.data.scripts.preprocessing.preprocess_lbm_to_tar.check_directory_has_files_with_prefix")
    @patch("vla_foundry.data.scripts.preprocessing.preprocess_lbm_to_tar.draccus.parse")
    def test_main_fails_with_existing_episode_files(self, mock_parse, mock_check_dir):
        """Test that main() fails when output directory has existing episode files."""
        # Mock the config
        mock_cfg = MagicMock()
        mock_cfg.source_episodes = "/some/source"
        mock_cfg.output_dir = "s3://bucket/output"
        mock_parse.return_value = mock_cfg

        # Mock existing files
        mock_check_dir.return_value = ["episode_001.tar", "episode_002.tar"]

        from vla_foundry.data.scripts.preprocessing.preprocess_lbm_to_tar import main

        # Should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            main()

        assert "Output directory is not empty" in str(exc_info.value)
        assert "episode_001.tar" in str(exc_info.value)
        mock_check_dir.assert_called_once_with("s3://bucket/output", "episode_")

    @patch("vla_foundry.data.scripts.preprocessing.preprocess_lbm_to_tar.check_directory_has_files_with_prefix")
    @patch("vla_foundry.data.scripts.preprocessing.preprocess_lbm_to_tar.draccus.parse")
    @patch("vla_foundry.data.scripts.preprocessing.preprocess_lbm_to_tar.ray.init")
    def test_main_continues_with_empty_directory(self, mock_ray_init, mock_parse, mock_check_dir):
        """Test that main() continues when output directory is empty."""
        # Mock the config
        mock_cfg = MagicMock()
        mock_cfg.source_episodes = "/some/source"
        mock_cfg.output_dir = "s3://bucket/output"
        mock_cfg.ray_address = None
        mock_cfg.ray_num_cpus = 4
        mock_cfg.camera_names = ["camera1"]
        mock_cfg.language_annotations_path = "/tmp/test_annotations.yaml"
        mock_parse.return_value = mock_cfg

        # Mock empty directory
        mock_check_dir.return_value = []

        # Create a dummy language annotations file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("language_dict: {}\n")
            temp_file = f.name

        try:
            mock_cfg.language_annotations_path = temp_file

            from vla_foundry.data.scripts.preprocessing.preprocess_lbm_to_tar import main

            # Should not raise, but will fail later due to missing other dependencies
            # We're just testing that it passes the safety check
            try:
                main()
            except Exception as e:
                # It's ok if it fails later, we just want to ensure it didn't fail on our safety check
                assert "Output directory is not empty" not in str(e)

            mock_check_dir.assert_called_once_with("s3://bucket/output", "episode_")
        finally:
            os.unlink(temp_file)


@pytest.mark.skipif(
    not _check_pyarrow_available(),
    reason="pyarrow not available - required for preprocess_lerobot_to_tar",
)
class TestPreprocessLerobotToTarSafety:
    """Tests for preprocessing safety checks in preprocess_lerobot_to_tar.py."""

    @patch("vla_foundry.data.scripts.preprocessing.preprocess_lerobot_to_tar.check_directory_has_files_with_prefix")
    @patch("vla_foundry.data.scripts.preprocessing.preprocess_lerobot_to_tar.parse_args")
    def test_main_fails_with_existing_shard_files(self, mock_parse_args, mock_check_dir):
        """Test that main() fails when output directory has existing shard files."""
        # Mock the args
        mock_args = MagicMock()
        mock_args.s3_output_path = "s3://bucket/output"
        mock_parse_args.return_value = mock_args

        # Mock existing files
        mock_check_dir.return_value = ["shard_00001.tar", "shard_00002.tar"]

        from vla_foundry.data.scripts.preprocessing.preprocess_lerobot_to_tar import main

        # Should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            main()

        assert "Output directory is not empty" in str(exc_info.value)
        assert "shard_00001.tar" in str(exc_info.value)
        mock_check_dir.assert_called_once_with("s3://bucket/output", "shard_")

    @patch("vla_foundry.data.scripts.preprocessing.preprocess_lerobot_to_tar.check_directory_has_files_with_prefix")
    @patch("vla_foundry.data.scripts.preprocessing.preprocess_lerobot_to_tar.parse_args")
    @patch("vla_foundry.data.scripts.preprocessing.preprocess_lerobot_to_tar.ray.init")
    def test_main_continues_with_empty_directory(self, mock_ray_init, mock_parse_args, mock_check_dir):
        """Test that main() continues when output directory is empty."""
        # Mock the args
        mock_args = MagicMock()
        mock_args.s3_output_path = "s3://bucket/output"
        mock_args.ray_address = None
        mock_args.ray_num_cpus = 4
        mock_parse_args.return_value = mock_args

        # Mock empty directory
        mock_check_dir.return_value = []

        from vla_foundry.data.scripts.preprocessing.preprocess_lerobot_to_tar import main

        # Should not raise, but will fail later due to missing other dependencies
        # We're just testing that it passes the safety check
        try:
            main()
        except Exception as e:
            # It's ok if it fails later, we just want to ensure it didn't fail on our safety check
            assert "Output directory is not empty" not in str(e)

        mock_check_dir.assert_called_once_with("s3://bucket/output", "shard_")
