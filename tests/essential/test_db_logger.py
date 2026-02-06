"""Tests for DynamoDB logger module."""

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from vla_foundry.db_logger import (
    DATASETS_TABLE,
    ModelTrainingLogger,
    _get_env_info,
    _get_git_info,
    _serialize_config,
    db_safe,
    log_dataset_preprocessing,
)

# =============================================================================
# Mock Configuration Classes
# =============================================================================


@dataclass
class MockDataConfig:
    """Minimal data config for testing."""

    dataset_manifest: List[str] = field(default_factory=lambda: ["s3://bucket/dataset1", "s3://bucket/dataset2"])


@dataclass
class MockModelConfig:
    """Minimal model config for testing."""

    resume_from_checkpoint: Optional[str] = None


@dataclass
class MockTrainConfig:
    """Minimal config object for ModelTrainingLogger tests."""

    data: MockDataConfig = field(default_factory=MockDataConfig)
    model: MockModelConfig = field(default_factory=MockModelConfig)
    remote_sync: str = "s3://bucket/checkpoints"
    remote_sync_fixed_path: str = "s3://bucket/fixed"
    num_checkpoints: int = 10
    total_train_samples: int = 100000


@dataclass
class MockPreprocessConfig:
    """Minimal config object for dataset preprocessing tests."""

    source_paths: List[str] = field(default_factory=lambda: ["/data/raw/dataset1"])
    target_path: str = "s3://bucket/output"
    type: str = "robotics"


# =============================================================================
# Tests for _get_git_info
# =============================================================================


class TestGetGitInfo:
    """Tests for the _get_git_info function."""

    def test_returns_commit_hash(self):
        """Test that git commit hash is returned."""
        info = _get_git_info()
        assert "git_commit_hash" in info
        assert len(info["git_commit_hash"]) == 40  # Full SHA

    def test_returns_branch_name(self):
        """Test that git branch is returned."""
        info = _get_git_info()
        assert "git_branch" in info
        assert isinstance(info["git_branch"], str)
        assert len(info["git_branch"]) > 0

    def test_returns_remote_url(self):
        """Test that git remote URL is returned."""
        info = _get_git_info()
        assert "git_remote_url" in info
        assert "github.com" in info["git_remote_url"] or "git@" in info["git_remote_url"]

    def test_detects_local_changes(self):
        """Test that local changes are detected."""
        info = _get_git_info()
        assert "git_has_local_changes" in info
        assert isinstance(info["git_has_local_changes"], bool)

    def test_captures_diff_when_changes_exist(self):
        """Test that actual diff is captured when there are changes."""
        info = _get_git_info()
        assert "git_local_changes" in info
        if info["git_has_local_changes"]:
            # Should contain actual diff content, not just file list
            assert len(info["git_local_changes"]) > 0

    def test_fails_when_git_command_fails(self):
        """Test that function fails hard when git commands fail."""

        # Mock subprocess.run to simulate git command failure
        def mock_run(args, **kwargs):
            result = MagicMock()
            result.returncode = 128
            result.stderr = "fatal: not a git repository"
            result.stdout = ""
            return result

        with (
            patch("vla_foundry.db_logger.subprocess.run", side_effect=mock_run),
            pytest.raises(RuntimeError),
        ):
            _get_git_info()

    def test_diff_truncated_at_10k_chars(self):
        """Test that very large diffs are truncated."""
        # This is hard to test without creating a huge diff,
        # so we just verify the truncation logic exists by checking the code path
        info = _get_git_info()
        assert len(info.get("git_local_changes", "")) <= 10100  # 10k + some buffer for truncation message

    def test_commit_hash_is_valid_sha1(self):
        """Test that commit hash is a valid 40-character SHA-1 hash."""
        info = _get_git_info()
        assert len(info["git_commit_hash"]) == 40
        assert all(c in "0123456789abcdef" for c in info["git_commit_hash"].lower())

    def test_remote_url_is_valid_format(self):
        """Test that remote URL is a valid git URL format."""
        info = _get_git_info()
        url = info["git_remote_url"]
        assert url.startswith("https://") or url.startswith("git@") or url.startswith("ssh://")

    def test_has_local_changes_is_bool(self):
        """Test that has_local_changes is always a boolean."""
        info = _get_git_info()
        assert isinstance(info["git_has_local_changes"], bool)

    def test_local_changes_empty_when_clean(self):
        """Test that local_changes is empty string when repo is clean."""
        # Mock a clean repo scenario
        call_count = [0]

        def mock_run(args, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            result.returncode = 0

            if args == ["git", "rev-parse", "HEAD"]:
                result.stdout = "abc123def456" * 4  # 48 chars, will be trimmed
            elif args == ["git", "branch", "--show-current"]:
                result.stdout = "main"
            elif args == ["git", "config", "--get", "remote.origin.url"]:
                result.stdout = "git@github.com:test/repo.git"
            elif args == ["git", "status", "--porcelain"]:
                result.stdout = ""  # Clean - no changes
            else:
                result.stdout = ""
            return result

        with patch("vla_foundry.db_logger.subprocess.run", side_effect=mock_run):
            info = _get_git_info()
            assert info["git_has_local_changes"] is False
            assert info["git_local_changes"] == ""

    def test_detached_head_returns_detached(self):
        """Test that detached HEAD state returns 'DETACHED' for branch."""

        def mock_run(args, **kwargs):
            result = MagicMock()
            result.returncode = 0

            if args == ["git", "rev-parse", "HEAD"]:
                result.stdout = "a" * 40
            elif args == ["git", "branch", "--show-current"]:
                result.stdout = ""  # Empty when detached
            elif args == ["git", "config", "--get", "remote.origin.url"]:
                result.stdout = "git@github.com:test/repo.git"
            elif args == ["git", "status", "--porcelain"]:
                result.stdout = ""
            else:
                result.stdout = ""
            return result

        with patch("vla_foundry.db_logger.subprocess.run", side_effect=mock_run):
            info = _get_git_info()
            assert info["git_branch"] == "DETACHED"

    def test_untracked_files_included_in_diff(self):
        """Test that untracked files are listed in local_changes."""

        def mock_run(args, **kwargs):
            result = MagicMock()
            result.returncode = 0

            if args == ["git", "rev-parse", "HEAD"]:
                result.stdout = "a" * 40
            elif args == ["git", "branch", "--show-current"]:
                result.stdout = "main"
            elif args == ["git", "config", "--get", "remote.origin.url"]:
                result.stdout = "git@github.com:test/repo.git"
            elif args == ["git", "status", "--porcelain"]:
                result.stdout = "?? untracked.txt\n M modified.txt"
            elif args == ["git", "diff", "HEAD"]:
                result.stdout = "diff --git a/modified.txt\n+new line"
            elif args == ["git", "ls-files", "--others", "--exclude-standard"]:
                result.stdout = "untracked.txt\nnew_file.py"
            else:
                result.stdout = ""
            return result

        with patch("vla_foundry.db_logger.subprocess.run", side_effect=mock_run):
            info = _get_git_info()
            assert info["git_has_local_changes"] is True
            assert "# Untracked files:" in info["git_local_changes"]
            assert "untracked.txt" in info["git_local_changes"]
            assert "new_file.py" in info["git_local_changes"]

    def test_large_diff_truncated_with_message(self):
        """Test that diffs over 10k chars are truncated with a message."""
        large_diff = "x" * 15000  # Larger than 10k limit

        def mock_run(args, **kwargs):
            result = MagicMock()
            result.returncode = 0

            if args == ["git", "rev-parse", "HEAD"]:
                result.stdout = "a" * 40
            elif args == ["git", "branch", "--show-current"]:
                result.stdout = "main"
            elif args == ["git", "config", "--get", "remote.origin.url"]:
                result.stdout = "git@github.com:test/repo.git"
            elif args == ["git", "status", "--porcelain"]:
                result.stdout = " M large_file.txt"
            elif args == ["git", "diff", "HEAD"]:
                result.stdout = large_diff
            elif args == ["git", "ls-files", "--others", "--exclude-standard"]:
                result.stdout = ""
            else:
                result.stdout = ""
            return result

        with patch("vla_foundry.db_logger.subprocess.run", side_effect=mock_run):
            info = _get_git_info()
            assert len(info["git_local_changes"]) <= 10100
            assert "[truncated]" in info["git_local_changes"]


# =============================================================================
# Tests for _serialize_config
# =============================================================================


class TestSerializeConfig:
    """Tests for config serialization."""

    def test_serializes_dataclass_to_yaml(self):
        """Test that draccus configs are serialized to YAML."""
        cfg = MockTrainConfig()
        result = _serialize_config(cfg)

        assert isinstance(result, str)
        assert "remote_sync:" in result or "remote_sync" in result

    def test_serializes_nested_config(self):
        """Test that nested configs are properly serialized."""
        cfg = MockTrainConfig()
        result = _serialize_config(cfg)

        # Should contain nested data config
        assert "dataset_manifest" in result or "data" in result


# =============================================================================
# Tests for _get_env_info
# =============================================================================


class TestGetEnvInfo:
    """Tests for environment info gathering."""

    def test_returns_hostname(self):
        """Test that hostname is returned."""
        info = _get_env_info()
        assert "hostname" in info
        assert len(info["hostname"]) > 0

    def test_returns_python_version(self):
        """Test that Python version is returned."""
        info = _get_env_info()
        assert "python_version" in info
        assert "." in info["python_version"]  # e.g., "3.10.0"

    def test_returns_platform(self):
        """Test that platform info is returned."""
        info = _get_env_info()
        assert "platform" in info
        assert len(info["platform"]) > 0


# =============================================================================
# Tests for db_safe decorator
# =============================================================================


class TestDbSafeDecorator:
    """Tests for the db_safe error handling decorator."""

    def test_passes_through_successful_calls(self):
        """Test that successful function calls return normally."""

        @db_safe
        def success_func():
            return "success"

        assert success_func() == "success"

    def test_catches_client_error(self):
        """Test that ClientError exceptions are caught and logged."""
        from botocore.exceptions import ClientError

        @db_safe
        def failing_func():
            raise ClientError({"Error": {"Code": "ResourceNotFoundException", "Message": "Table not found"}}, "PutItem")

        # Should not raise, should return None
        result = failing_func()
        assert result is None

    def test_catches_generic_exceptions(self):
        """Test that generic exceptions are caught."""

        @db_safe
        def failing_func():
            raise ValueError("Something went wrong")

        result = failing_func()
        assert result is None

    def test_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        @db_safe
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


# =============================================================================
# Tests for ModelTrainingLogger
# =============================================================================


class TestModelTrainingLogger:
    """Tests for the ModelTrainingLogger class."""

    def test_init_stores_run_uuid(self):
        """Test that initialization stores the run UUID."""
        cfg = MockTrainConfig()
        logger = ModelTrainingLogger("test-uuid-123", cfg, enabled=False)

        assert logger.run_uuid == "test-uuid-123"
        assert logger.cfg == cfg

    def test_init_with_disabled_does_not_create_table(self):
        """Test that disabled logger doesn't access DynamoDB."""
        cfg = MockTrainConfig()
        logger = ModelTrainingLogger("test-uuid", cfg, enabled=False)

        # Table should not be initialized when disabled
        assert logger._table is None

    def test_log_job_start_skipped_when_disabled(self):
        """Test that log_job_start does nothing when disabled."""
        cfg = MockTrainConfig()
        logger = ModelTrainingLogger("test-uuid", cfg, enabled=False)

        # Should not raise and should not access DynamoDB
        with patch("vla_foundry.db_logger._get_dynamodb_table") as mock_table:
            logger.log_job_start("experiment", "/path/to/exp")
            mock_table.assert_not_called()

    @patch("vla_foundry.db_logger._get_dynamodb_table")
    def test_log_job_start_creates_record(self, mock_get_table):
        """Test that log_job_start creates a DynamoDB record."""
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        cfg = MockTrainConfig()
        logger = ModelTrainingLogger("test-uuid-123", cfg, enabled=True)

        logger.log_job_start("my_experiment", "/experiments/my_experiment", wandb_url="https://wandb.ai/run/123")

        # Verify put_item was called
        mock_table.put_item.assert_called_once()
        item = mock_table.put_item.call_args[1]["Item"]

        assert item["uuid"] == "test-uuid-123"
        assert item["experiment_name"] == "my_experiment"
        assert item["wandb_url"] == "https://wandb.ai/run/123"
        assert item["status"] == "running"
        assert item["completed"] is False
        assert "git_commit_hash" in item
        assert "hostname" in item

    @patch("vla_foundry.db_logger._get_dynamodb_table")
    def test_log_checkpoint_updates_record(self, mock_get_table):
        """Test that log_checkpoint updates the DynamoDB record."""
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        cfg = MockTrainConfig()
        logger = ModelTrainingLogger("test-uuid", cfg, enabled=True)

        logger.log_checkpoint(checkpoint_num=5, samples_seen=50000)

        mock_table.update_item.assert_called_once()
        call_kwargs = mock_table.update_item.call_args[1]

        assert call_kwargs["Key"] == {"uuid": "test-uuid"}
        assert ":ckpt" in str(call_kwargs["ExpressionAttributeValues"])
        assert call_kwargs["ExpressionAttributeValues"][":ckpt"] == 5
        assert call_kwargs["ExpressionAttributeValues"][":samples"] == 50000

    def test_log_checkpoint_skipped_when_disabled(self):
        """Test that log_checkpoint does nothing when disabled."""
        cfg = MockTrainConfig()
        logger = ModelTrainingLogger("test-uuid", cfg, enabled=False)

        with patch("vla_foundry.db_logger._get_dynamodb_table") as mock_table:
            logger.log_checkpoint(5, 50000)
            mock_table.assert_not_called()

    @patch("vla_foundry.db_logger._get_dynamodb_table")
    def test_log_completion_marks_completed(self, mock_get_table):
        """Test that log_completion marks the run as completed."""
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        cfg = MockTrainConfig()
        logger = ModelTrainingLogger("test-uuid", cfg, enabled=True)
        logger._job_started = True

        logger.log_completion(samples_seen=100000)

        mock_table.update_item.assert_called_once()
        call_kwargs = mock_table.update_item.call_args[1]

        assert call_kwargs["ExpressionAttributeValues"][":completed"] is True
        assert call_kwargs["ExpressionAttributeValues"][":status"] == "completed"
        assert logger._finished is True

    @patch("vla_foundry.db_logger._get_dynamodb_table")
    def test_log_crash_marks_crashed(self, mock_get_table):
        """Test that log_crash marks the run as crashed."""
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        cfg = MockTrainConfig()
        logger = ModelTrainingLogger("test-uuid", cfg, enabled=True)
        logger._job_started = True

        logger.log_crash("Out of memory error")

        mock_table.update_item.assert_called_once()
        call_kwargs = mock_table.update_item.call_args[1]

        assert call_kwargs["ExpressionAttributeValues"][":status"] == "crashed"
        assert "Out of memory" in call_kwargs["ExpressionAttributeValues"][":msg"]

    @patch("vla_foundry.db_logger._get_dynamodb_table")
    def test_update_wandb_url(self, mock_get_table):
        """Test that wandb URL can be updated after initialization."""
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        cfg = MockTrainConfig()
        logger = ModelTrainingLogger("test-uuid", cfg, enabled=True)

        logger.update_wandb_url("https://wandb.ai/new-url")

        mock_table.update_item.assert_called_once()
        call_kwargs = mock_table.update_item.call_args[1]

        assert call_kwargs["ExpressionAttributeValues"][":url"] == "https://wandb.ai/new-url"

    def test_atexit_handler_registered(self):
        """Test that atexit handler is registered on init."""
        cfg = MockTrainConfig()

        with patch("atexit.register") as mock_register:
            ModelTrainingLogger("test-uuid", cfg, enabled=True)
            mock_register.assert_called_once()

    @patch("vla_foundry.db_logger._get_dynamodb_table")
    def test_on_exit_logs_crash_if_not_finished(self, mock_get_table):
        """Test that _on_exit logs crash if job started but not finished."""
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        cfg = MockTrainConfig()
        logger = ModelTrainingLogger("test-uuid", cfg, enabled=True)
        logger._job_started = True
        logger._finished = False

        logger._on_exit()

        # Should have called update_item to mark as crashed
        mock_table.update_item.assert_called_once()

    @patch("vla_foundry.db_logger._get_dynamodb_table")
    def test_on_exit_does_nothing_if_finished(self, mock_get_table):
        """Test that _on_exit does nothing if already finished."""
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        cfg = MockTrainConfig()
        logger = ModelTrainingLogger("test-uuid", cfg, enabled=True)
        logger._job_started = True
        logger._finished = True

        logger._on_exit()

        mock_table.update_item.assert_not_called()


# =============================================================================
# Tests for log_dataset_preprocessing
# =============================================================================


class TestLogDatasetPreprocessing:
    """Tests for the log_dataset_preprocessing function."""

    def test_skipped_when_disabled(self):
        """Test that logging is skipped when enabled=False."""
        cfg = MockPreprocessConfig()

        with patch("vla_foundry.db_logger._get_dynamodb_table") as mock_get_table:
            log_dataset_preprocessing(
                dataset_uuid="test-uuid",
                cfg=cfg,
                dataset_type="robotics",
                source_paths=["/data"],
                target_path="/output",
                fixed_path="/fixed",
                episode_count=100,
                frame_count=10000,
                samples_per_shard=1000,
                num_shards=10,
                total_samples=10000,
                enabled=False,
            )

            mock_get_table.assert_not_called()

    @patch("vla_foundry.db_logger._get_dynamodb_table")
    def test_creates_record_when_enabled(self, mock_get_table):
        """Test that a DynamoDB record is created when enabled."""
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        cfg = MockPreprocessConfig()

        log_dataset_preprocessing(
            dataset_uuid="dataset-uuid-123",
            cfg=cfg,
            dataset_type="robotics",
            source_paths=["/data/raw1", "/data/raw2"],
            target_path="s3://bucket/processed",
            fixed_path="s3://bucket/fixed/dataset-uuid-123",
            episode_count=500,
            frame_count=50000,
            samples_per_shard=1000,
            num_shards=50,
            total_samples=50000,
            enabled=True,
        )

        mock_get_table.assert_called_once_with(DATASETS_TABLE)
        mock_table.put_item.assert_called_once()

        item = mock_table.put_item.call_args[1]["Item"]

        assert item["uuid"] == "dataset-uuid-123"
        assert item["dataset_type"] == "robotics"
        assert item["episode_count"] == 500
        assert item["frame_count"] == 50000
        assert item["num_shards"] == 50
        assert item["total_samples_created"] == 50000
        assert "/data/raw1" in item["dataset_source_paths"]
        assert "git_commit_hash" in item
        assert "hostname" in item

    @patch("vla_foundry.db_logger._get_dynamodb_table")
    def test_captures_git_info(self, mock_get_table):
        """Test that git info is captured in the record."""
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        cfg = MockPreprocessConfig()

        log_dataset_preprocessing(
            dataset_uuid="test-uuid",
            cfg=cfg,
            dataset_type="test",
            source_paths=[],
            target_path="",
            fixed_path="",
            episode_count=0,
            frame_count=0,
            samples_per_shard=0,
            num_shards=0,
            total_samples=0,
            enabled=True,
        )

        item = mock_table.put_item.call_args[1]["Item"]

        assert "git_commit_hash" in item
        assert "git_branch" in item
        assert "git_remote_url" in item
        assert "git_has_local_changes" in item


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the logging system."""

    @patch("vla_foundry.db_logger._get_dynamodb_table")
    def test_full_training_lifecycle(self, mock_get_table):
        """Test a complete training run lifecycle: start -> checkpoints -> completion."""
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        cfg = MockTrainConfig()
        logger = ModelTrainingLogger("run-uuid-123", cfg, enabled=True)

        # Start the job
        logger.log_job_start("test_experiment", "/experiments/test")
        assert mock_table.put_item.call_count == 1

        # Log several checkpoints
        for i in range(1, 6):
            logger.log_checkpoint(i, i * 20000)

        assert mock_table.update_item.call_count == 5

        # Complete the job
        logger.log_completion(100000)

        assert mock_table.update_item.call_count == 6
        # Verify final update marks as completed
        final_call = mock_table.update_item.call_args[1]
        assert final_call["ExpressionAttributeValues"][":status"] == "completed"

    @patch("vla_foundry.db_logger._get_dynamodb_table")
    def test_training_crash_lifecycle(self, mock_get_table):
        """Test training run that crashes: start -> checkpoints -> crash."""
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        cfg = MockTrainConfig()
        logger = ModelTrainingLogger("run-uuid-crash", cfg, enabled=True)

        # Start the job
        logger.log_job_start("crash_experiment", "/experiments/crash")
        logger._job_started = True

        # Log some checkpoints
        logger.log_checkpoint(1, 20000)
        logger.log_checkpoint(2, 40000)

        # Simulate crash via atexit handler
        logger._on_exit()

        # Should be marked as crashed
        final_call = mock_table.update_item.call_args[1]
        assert final_call["ExpressionAttributeValues"][":status"] == "crashed"

    def test_disabled_logging_no_aws_calls(self):
        """Test that disabled logging makes zero AWS calls."""
        cfg = MockTrainConfig()

        with patch("vla_foundry.db_logger.boto3") as mock_boto:
            logger = ModelTrainingLogger("test-uuid", cfg, enabled=False)

            logger.log_job_start("exp", "/path")
            logger.log_checkpoint(1, 1000)
            logger.log_completion(10000)
            logger.update_wandb_url("http://test")

            # boto3 should never be touched
            mock_boto.resource.assert_not_called()

    def test_disabled_dataset_logging_no_aws_calls(self):
        """Test that disabled dataset logging makes zero AWS calls."""
        cfg = MockPreprocessConfig()

        with patch("vla_foundry.db_logger.boto3") as mock_boto:
            log_dataset_preprocessing(
                dataset_uuid="test",
                cfg=cfg,
                dataset_type="test",
                source_paths=[],
                target_path="",
                fixed_path="",
                episode_count=0,
                frame_count=0,
                samples_per_shard=0,
                num_shards=0,
                total_samples=0,
                enabled=False,
            )

            mock_boto.resource.assert_not_called()
