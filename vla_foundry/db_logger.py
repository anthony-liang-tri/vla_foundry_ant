"""
DynamoDB Logger for VLA Foundry

Logs training runs and dataset preprocessing jobs to DynamoDB tables
for tracking and monitoring via the dashboard.
"""

import atexit
import datetime
import functools
import logging
import os
import platform
import signal
import socket
import subprocess
import sys
from collections.abc import Callable
from io import StringIO
from typing import Any

import boto3
import draccus
from botocore.exceptions import ClientError

# DynamoDB configuration
REGION = "us-west-2"
MODELS_TABLE = "vla_foundry_models"
DATASETS_TABLE = "vla_foundry_datasets"


def db_safe(func: Callable) -> Callable:
    """Decorator that catches and logs DynamoDB errors without raising."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ClientError, Exception) as e:
            logging.warning(f"DynamoDB operation failed in {func.__name__}: {e}")
            return None

    return wrapper


def _get_dynamodb_table(table_name: str):
    """Get a DynamoDB table resource."""
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    return dynamodb.Table(table_name)


def get_git_env_vars() -> dict[str, str]:
    """Capture git info as environment variables for passing to remote workers.

    Use this to capture git info on a head node and pass to workers that don't
    have access to .git (e.g., SageMaker, Ray workers).

    Returns:
        Dictionary of VLA_GIT_* env vars, or empty dict if not in a git repo.
    """
    cwd = os.path.dirname(__file__)

    def run_git(args: list[str]) -> str | None:
        result = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run_git(["git", "rev-parse", "HEAD"])
    if not commit:
        return {}

    changes = run_git(["git", "status", "--porcelain"])
    return {
        "VLA_GIT_COMMIT_HASH": commit,
        "VLA_GIT_BRANCH": run_git(["git", "branch", "--show-current"]) or "DETACHED",
        "VLA_GIT_REMOTE_URL": run_git(["git", "config", "--get", "remote.origin.url"]) or "unknown",
        "VLA_GIT_HAS_LOCAL_CHANGES": "true" if changes else "false",
    }


def _get_git_info() -> dict[str, Any]:
    """Get git repository information for code version tracking.

    Tries git commands first (for local runs), falls back to VLA_GIT_*
    environment variables (for SageMaker where no .git directory exists).

    Returns:
        Dictionary with git commit hash, branch, remote URL, and local changes.

    Raises:
        RuntimeError: If git info cannot be obtained from either source.
    """
    cwd = os.path.dirname(__file__)

    def run_git(args: list[str], fail_on_error: bool = True) -> str | None:
        result = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0:
            if not fail_on_error:
                return None
            raise RuntimeError(f"Git command failed: {' '.join(args)}")
        return result.stdout.strip()

    # Try git commands first (works for local training)
    commit_hash = run_git(["git", "rev-parse", "HEAD"], fail_on_error=False)
    if commit_hash:
        git_info = {
            "git_commit_hash": commit_hash,
            "git_branch": run_git(["git", "branch", "--show-current"]) or "DETACHED",
            "git_remote_url": run_git(["git", "config", "--get", "remote.origin.url"]) or "unknown",
            "git_has_local_changes": False,
            "git_local_changes": "",
        }
        changes = run_git(["git", "status", "--porcelain"])
        git_info["git_has_local_changes"] = bool(changes)
        if changes:
            diff = run_git(["git", "diff", "HEAD"]) or ""
            untracked = run_git(["git", "ls-files", "--others", "--exclude-standard"])
            if untracked:
                diff += "\n\n# Untracked files:\n" + untracked
            if len(diff) > 10000:
                diff = diff[:10000] + "\n... [truncated]"
            git_info["git_local_changes"] = diff
        return git_info

    # Fall back to environment variables (SageMaker)
    if os.environ.get("VLA_GIT_COMMIT_HASH"):
        git_info = {
            "git_commit_hash": os.environ["VLA_GIT_COMMIT_HASH"],
            "git_branch": os.environ.get("VLA_GIT_BRANCH", "unknown"),
            "git_remote_url": os.environ.get("VLA_GIT_REMOTE_URL", "unknown"),
            "git_has_local_changes": os.environ.get("VLA_GIT_HAS_LOCAL_CHANGES", "false") == "true",
            "git_local_changes": "",
        }
        git_diff_file = os.environ.get("VLA_GIT_DIFF_FILE")
        if git_diff_file and os.path.exists(git_diff_file):
            with open(git_diff_file) as f:
                diff = f.read()
                if len(diff) > 10000:
                    diff = diff[:10000] + "\n... [truncated]"
                git_info["git_local_changes"] = diff
        return git_info

    raise RuntimeError("Git command failed: git rev-parse HEAD; not in a git repo and VLA_GIT_* env vars not set")


def _serialize_config(cfg: Any) -> str:
    """Serialize a draccus config to YAML string."""
    buffer = StringIO()
    draccus.dump(cfg, buffer)
    return buffer.getvalue()


def _get_env_info() -> dict[str, str]:
    """Get environment information for reproducibility."""
    return {
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


class ModelTrainingLogger:
    """Logger for tracking model training runs in DynamoDB."""

    def __init__(self, run_uuid: str, cfg: Any, enabled: bool = True):
        """
        Initialize the model training logger.

        Args:
            run_uuid: Unique identifier for this training run
            cfg: TrainExperimentParams configuration object
            enabled: Whether to actually log to DynamoDB (can disable for testing)
        """
        self.run_uuid = run_uuid
        self.cfg = cfg
        self.enabled = enabled
        self._table = None
        self._job_started = False
        self._finished = False  # True if log_completion() was called

        # Register atexit handler to mark as crashed if not completed normally
        atexit.register(self._on_exit)

        # Register signal handlers for SIGINT (Ctrl+C) and SIGTERM
        # atexit handlers don't run on signal termination, so we need these
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM - log crash and re-raise for normal termination."""
        sig_name = signal.Signals(signum).name
        if self._job_started and not self._finished:
            self.log_crash(f"Process terminated by {sig_name}")
        # Re-raise the signal with default handler so process exits normally
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def _on_exit(self):
        """Called on process exit - marks run as crashed if not completed normally."""
        if self._job_started and not self._finished:
            self.log_crash("Process exited without completing")

    @property
    def table(self):
        """Lazy load the DynamoDB table."""
        if self._table is None:
            self._table = _get_dynamodb_table(MODELS_TABLE)
        return self._table

    @db_safe
    def log_job_start(self, experiment_name: str, experiment_path: str, wandb_url: str | None = None) -> None:
        """Log the start of a training job."""
        if not self.enabled:
            return

        git_info = _get_git_info()
        env_info = _get_env_info()
        cfg = self.cfg

        item = {
            "uuid": self.run_uuid,
            "datetime_job_started": datetime.datetime.now().isoformat(),
            "datetime": datetime.datetime.now().isoformat(),
            "created_by": os.environ.get("VLA_LAUNCHED_BY") or os.environ.get("USER", "unknown"),
            "cfg": _serialize_config(cfg),
            "experiment_name": experiment_name,
            **git_info,
            **env_info,
            "dataset_uuid": "",
            "dataset_source_paths": getattr(cfg.data, "dataset_manifest", []),
            "model_checkpoints_path": os.path.join(experiment_path, "checkpoints"),
            "fixed_model_path": os.path.join(cfg.remote_sync_fixed_path, self.run_uuid)
            if getattr(cfg, "remote_sync_fixed_path", None)
            else "",
            "resume_from_checkpoint_path": getattr(cfg.model, "resume_from_checkpoint", "") or "",
            "wandb_url": wandb_url or "",
            "checkpoint_number_current": 0,
            "checkpoint_number_total": getattr(cfg, "num_checkpoints", 0),
            "sample_number_current": 0,
            "sample_number_total": getattr(cfg, "total_train_samples", 0),
            "completed": False,
            "status": "running",  # running, completed, or crashed
        }

        self.table.put_item(Item=item)
        self._job_started = True
        logging.info(f"Logged training job start to DynamoDB: {self.run_uuid}")

    @db_safe
    def log_checkpoint(self, checkpoint_num: int, samples_seen: int) -> None:
        """Log progress after each checkpoint."""
        if not self.enabled:
            return

        self.table.update_item(
            Key={"uuid": self.run_uuid},
            UpdateExpression="SET checkpoint_number_current = :ckpt, sample_number_current = :samples, #dt = :dt",
            ExpressionAttributeNames={"#dt": "datetime"},
            ExpressionAttributeValues={
                ":ckpt": checkpoint_num,
                ":samples": samples_seen,
                ":dt": datetime.datetime.now().isoformat(),
            },
        )
        logging.debug(f"Logged checkpoint {checkpoint_num} to DynamoDB")

    @db_safe
    def log_completion(self, samples_seen: int) -> None:
        """Log training completion."""
        if not self.enabled:
            return

        self._finished = True  # Prevent atexit from marking as crashed
        self.table.update_item(
            Key={"uuid": self.run_uuid},
            UpdateExpression=(
                "SET completed = :completed, #status = :status, sample_number_current = :samples, #dt = :dt"
            ),
            ExpressionAttributeNames={"#dt": "datetime", "#status": "status"},
            ExpressionAttributeValues={
                ":completed": True,
                ":status": "completed",
                ":samples": samples_seen,
                ":dt": datetime.datetime.now().isoformat(),
            },
        )
        logging.info(f"Logged training completion to DynamoDB: {self.run_uuid}")

    @db_safe
    def log_crash(self, error_message: str = "") -> None:
        """Log that training crashed (called automatically on abnormal exit)."""
        if not self.enabled:
            return

        self._finished = True  # Prevent duplicate crash logs
        self.table.update_item(
            Key={"uuid": self.run_uuid},
            UpdateExpression="SET #status = :status, crash_message = :msg, #dt = :dt",
            ExpressionAttributeNames={"#dt": "datetime", "#status": "status"},
            ExpressionAttributeValues={
                ":status": "crashed",
                ":msg": error_message[:500] if error_message else "Unknown error",
                ":dt": datetime.datetime.now().isoformat(),
            },
        )
        logging.warning(f"Logged training crash to DynamoDB: {self.run_uuid} - {error_message}")

    @db_safe
    def update_wandb_url(self, wandb_url: str) -> None:
        """Update the W&B URL after wandb.init()."""
        if not self.enabled:
            return

        self.table.update_item(
            Key={"uuid": self.run_uuid},
            UpdateExpression="SET wandb_url = :url",
            ExpressionAttributeValues={":url": wandb_url},
        )
        logging.debug(f"Updated W&B URL in DynamoDB: {wandb_url}")


@db_safe
def log_dataset_preprocessing(
    dataset_uuid: str,
    cfg: Any,
    dataset_type: str,
    source_paths: list[str],
    target_path: str,
    fixed_path: str,
    episode_count: int,
    frame_count: int,
    samples_per_shard: int,
    num_shards: int,
    total_samples: int,
    enabled: bool = True,
) -> None:
    """
    Log a completed dataset preprocessing job to DynamoDB.

    Args:
        dataset_uuid: Unique identifier for this dataset
        cfg: Preprocessing configuration object
        dataset_type: Type of dataset (e.g., "robotics", "captions")
        source_paths: List of source data paths
        target_path: Output path for the processed dataset
        fixed_path: Fixed/permanent path for the dataset
        episode_count: Number of episodes processed
        frame_count: Total number of frames
        samples_per_shard: Samples per tar shard
        num_shards: Number of shards created
        total_samples: Total samples created
        enabled: Whether to actually log to DynamoDB
    """
    if not enabled:
        return

    table = _get_dynamodb_table(DATASETS_TABLE)
    git_info = _get_git_info()
    env_info = _get_env_info()

    item = {
        "uuid": dataset_uuid,
        "datetime": datetime.datetime.now().isoformat(),
        "created_by": os.environ.get("USER", "unknown"),
        "cfg": _serialize_config(cfg),
        **git_info,
        **env_info,
        "dataset_type": dataset_type,
        "dataset_source_paths": source_paths,
        "dataset_target_path": target_path,
        "dataset_fixed_path": fixed_path,
        "episode_count": episode_count,
        "frame_count": frame_count,
        "samples_per_shard": samples_per_shard,
        "num_shards": num_shards,
        "total_samples_created": total_samples,
    }

    table.put_item(Item=item)
    logging.info(f"Logged dataset preprocessing to DynamoDB: {dataset_uuid}")
