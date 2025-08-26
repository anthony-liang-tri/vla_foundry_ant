#!/usr/bin/env python3
"""
Usage:
    python preprocess_lbm_data_optimized.py \
        --source_episodes \
          "[s3://robotics-manip-lbm/efs/data/tasks/PickAndPlaceBox/cabot/sim/bc/teleop/2025-02-11T17-04-00-05-00/,]" \
        --output_dir s3://my-bucket/processed-dataset/ \
        --past_lowdim_steps 4 \
        --future_lowdim_steps 16 \
        --image_indices -1,0 \
        --max_padding_per_side 5 \
        --padding_strategy copy \
        --filter_still_samples True \
        --still_threshold 0.01 \
        --samples_per_shard 1000 \
        --stride 1 \
        --jpeg_quality 95 \
        --num_workers 16 \
        --no_statistics True \
        --batch_episodes 8 \
        --resize_images_size 224 \
        --use_gpu_resize True \
        --resume False
"""

import ast
import datetime
import hashlib
import io
import json
import os
import platform
import random
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from queue import Queue
from typing import Any, Dict, Iterator, List, Optional, Tuple

import draccus
import fsspec
import numpy as np

# Optional torch for GPU-accelerated resize
import torch
import torch.nn.functional as F
import yaml
from draccus.parsers import encoding as _draccus_encoding
from PIL import Image
from tqdm import tqdm
from turbojpeg import TurboJPEG

from lbm2.data.preprocessing.preprocess_statistics import StreamingDatasetStatistics

# Add import for relative coordinate utilities
from lbm2.data.robotics.utils import rot_6d_to_relative, xyz_to_relative

# Params base class
from lbm2.params.base_params import BaseParams

# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    print(f"\n⚠️  Received signal {signum}, initiating graceful shutdown...")
    print("📋 Saving current progress and cleaning up...")
    _shutdown_requested = True


@dataclass
class SampleMetadata:
    """Metadata for each preprocessed sample."""

    episode_id: str
    sample_id: str
    anchor_timestep: int
    anchor_relative_idx: int
    image_timesteps: List[int]
    lowdim_start_timestep: int
    lowdim_end_timestep: int
    past_padding: int
    future_padding: int
    camera_names: List[str]
    original_episode_length: int
    original_image_sizes: Dict[str, Tuple[int, int]]
    is_padded: bool


@dataclass(frozen=True)
class PreprocessParams(BaseParams):
    """Dataclass for preprocessing configuration parsed by draccus."""

    # Core I/O
    source_episodes: Optional[List[str]] = field(default=None)
    output_dir: Optional[str] = field(default=None)

    # Sampling/windowing
    past_lowdim_steps: int = field(default=5)
    future_lowdim_steps: int = field(default=20)
    image_indices: List[int] = field(default_factory=lambda: [-5, 0])
    stride: int = field(default=1)
    max_padding_left: int = field(default=5)
    max_padding_right: int = field(default=5)
    padding_strategy: str = field(default="copy")  # one of: copy, zero, reflect

    # Filtering
    filter_still_samples: bool = field(default=True)
    still_threshold: float = field(default=0.01)

    # Cameras
    camera_names: Optional[List[str]] = field(default=None)
    discard_keys: Optional[List[str]] = field(default=None)

    # Sharding / compression
    samples_per_shard: int = field(default=100)
    jpeg_quality: int = field(default=95)

    # Runtime
    num_workers: int = field(default=40)
    max_episodes: int = field(default=-1)
    fail_on_nan: bool = field(default=False)
    shuffle_buffer_size: int = field(default=1000)
    shuffle_input_files: bool = field(default=True)

    # Statistics and reproducibility
    no_statistics: bool = field(default=False)
    no_auto_tag: bool = field(default=False)

    # Incremental updates and resume capability
    enable_incremental_updates: bool = field(default=True)
    update_frequency: int = field(default=5)  # Update metadata every N shards
    resume: bool = field(default=False)  # Whether to resume from existing progress

    # Testing flags
    skip_git_tagging: bool = field(default=False)  # Skip git operations for testing

    # Image preprocessing
    resize_images_size: int = field(default=0)
    use_gpu_resize: bool = field(default=True)


# Global JPEG encoder pool to avoid repeated PIL overhead
_jpeg_encoder = None
_jpeg_quality = 95


def init_jpeg_encoder(quality: int = 95):
    """Initialize global JPEG encoder settings."""
    global _jpeg_quality, _jpeg_encoder
    _jpeg_quality = quality
    if _jpeg_encoder is None:
        _jpeg_encoder = TurboJPEG()


@lru_cache(maxsize=32)
def get_pil_image_cached(shape: Tuple[int, int], mode: str = "RGB"):
    """Cache PIL Image objects to reduce allocation overhead."""
    return Image.new(mode, shape)


def image_to_bytes(
    image: np.ndarray, quality: int = None, target_size: tuple = (224, 224)
) -> Tuple[bytes, Tuple[int, int]]:
    """Optimized image to JPEG conversion with resize and minimal allocations."""
    if quality is None:
        quality = _jpeg_quality

    # Ensure uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    h, w = image.shape[:2]
    original_image_size = (w, h)
    # turbojpeg expects BGR
    if image.shape[-1] == 3:
        bgr = image[:, :, ::-1]
    else:
        # Convert non-RGB by going through PIL (rare)
        pil_image = Image.fromarray(image)
        bgr = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    out = _jpeg_encoder.encode(bgr, quality=quality)
    return out, original_image_size


def find_repo_root(start_path):
    current_path = os.path.abspath(start_path)
    while current_path != "/":
        if (
            os.path.isdir(os.path.join(current_path, ".git"))
            or os.path.isfile(os.path.join(current_path, "pyproject.toml"))
            or os.path.isfile(os.path.join(current_path, "setup.py"))
        ):
            return current_path
        current_path = os.path.dirname(current_path)
    return None


def get_python_dependencies(file_path: str, repo_root: Optional[str] = None, visited: Optional[set] = None) -> set:
    """Dynamically extract Python file dependencies by parsing imports."""
    if visited is None:
        visited = set()

    if file_path in visited:
        return set()

    visited.add(file_path)
    dependencies = {file_path}

    # Try to determine repo root if not provided
    if repo_root is None:
        repo_root = find_repo_root(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom):
                    module_name = node.module
                    if module_name and module_name.startswith("lbm2"):
                        # Convert module path to file path relative to repo root
                        module_parts = module_name.split(".")
                        potential_paths = [
                            os.path.join(repo_root, "/".join(module_parts) + ".py"),
                            os.path.join(repo_root, "/".join(module_parts), "__init__.py"),
                        ]

                        for potential_path in potential_paths:
                            if os.path.exists(potential_path):
                                # Recursively get dependencies
                                sub_deps = get_python_dependencies(potential_path, repo_root, visited)
                                dependencies.update(sub_deps)
                                break

                    # Handle relative imports from current directory
                    if not module_name:  # relative import like "from . import"
                        current_dir = os.path.dirname(file_path)
                        for alias in node.names:
                            if alias.name != "*":
                                rel_path = os.path.join(current_dir, alias.name + ".py")
                                if os.path.exists(rel_path):
                                    sub_deps = get_python_dependencies(rel_path, repo_root, visited)
                                    dependencies.update(sub_deps)

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        if module_name.startswith("lbm2"):
                            # Convert module path to file path relative to repo root
                            module_parts = module_name.split(".")
                            potential_paths = [
                                os.path.join(repo_root, "/".join(module_parts) + ".py"),
                                os.path.join(repo_root, "/".join(module_parts), "__init__.py"),
                            ]

                            for potential_path in potential_paths:
                                if os.path.exists(potential_path):
                                    sub_deps = get_python_dependencies(potential_path, repo_root, visited)
                                    dependencies.update(sub_deps)
                                    break

        # Also check for direct file imports (like spartan_data_explorer)
        current_dir = os.path.dirname(file_path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and not node.module.startswith(".")
                and "." not in node.module
            ):
                # Handle direct imports from files in same directory
                # Check if it's a module file in the same directory
                potential_file = os.path.join(current_dir, node.module + ".py")
                if os.path.exists(potential_file):
                    sub_deps = get_python_dependencies(potential_file, repo_root, visited)
                    dependencies.update(sub_deps)

    except Exception as e:
        print(f"Warning: Could not analyze dependencies for {file_path}: {e}")

    return dependencies


def check_preprocessing_related_changes() -> Tuple[bool, List[str]]:
    """Check if uncommitted changes affect preprocessing code or dependencies."""

    # Get the main preprocessing script path
    script_path = os.path.abspath(__file__)

    # Dynamically discover Python dependencies
    print("🔍 Analyzing Python dependencies...")
    repo_root = find_repo_root(script_path)
    python_dependencies = get_python_dependencies(script_path, repo_root)

    # Also include package management files
    additional_critical_files = ["requirements.txt", "pyproject.toml", "setup.py", "environment.yml"]

    # Convert to relative paths and add additional files
    critical_files = set()

    for dep_path in python_dependencies:
        if os.path.exists(dep_path):
            try:
                rel_path = os.path.relpath(dep_path, repo_root)
                critical_files.add(rel_path)
            except ValueError:
                # If relative path calculation fails, use absolute path
                critical_files.add(dep_path)

    # Add additional critical files
    for additional_file in additional_critical_files:
        additional_file_path = os.path.join(repo_root, additional_file)
        if os.path.exists(additional_file_path):
            critical_files.add(additional_file)

    print(f"📋 Found {len(critical_files)} critical dependency files")

    # Files that are relevant but less critical
    relevant_files = [
        "lbm2/data/preprocessing/create_data.sh",  # Processing configuration
    ]

    # Files to explicitly exclude (documentation, tests, etc.)
    exclude_patterns = [
        "README",
        ".md",
        "_test.py",
        "test_",
        "/tests/",
        ".txt",  # Like episode_description.txt
        "validate_",
        "example_",
        "_old.py",
        "pickle_to_raw.py",  # Utility script, not used by preprocessing
        "create_test_data.py",  # Test utility
    ]

    try:
        # Get list of changed files
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=repo_root)
        if result.returncode != 0:
            return False, []

        changed_files = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                # Extract filename (skip the status prefix)
                filename = line[3:].strip()
                # Make filename relative to repo_root (in case it's not already)
                abs_path = os.path.abspath(os.path.join(repo_root, filename))
                rel_path = os.path.relpath(abs_path, repo_root)
                changed_files.append(rel_path)

        # Check for critical changes (require tagging)
        critical_changes = []
        relevant_changes = []

        for file in changed_files:
            # First check if file should be excluded
            should_exclude = False
            for exclude_pattern in exclude_patterns:
                if exclude_pattern in file:
                    should_exclude = True
                    break

            if should_exclude:
                continue

            # Check if file is in critical dependencies (exact match or path contains)
            file_matched = False

            # Exact match first
            if file in critical_files:
                critical_changes.append(file)
                file_matched = True
            else:
                # Check if any critical file path contains this changed file
                for critical_path in critical_files:
                    if critical_path in file or file in critical_path:
                        critical_changes.append(file)
                        file_matched = True
                        break

            if not file_matched:
                # Check relevant but non-critical files
                for relevant_path in relevant_files:
                    if relevant_path in file:
                        relevant_changes.append(file)
                        break

        # Return True if there are critical changes, and all relevant files
        all_relevant = critical_changes + relevant_changes
        return len(critical_changes) > 0, all_relevant, critical_changes, relevant_changes

    except Exception:
        return False, [], [], []


def create_preprocessing_tag(dataset_name: str = None, preprocessing_type: str = None) -> str:
    """Create a git tag for the current preprocessing code state with more explicit naming."""
    try:
        # Generate tag name with timestamp and context
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        tag_parts = ["preprocessing"]

        # Add preprocessing type for clarity
        if preprocessing_type:
            clean_type = preprocessing_type.replace("_", "-")
            tag_parts.append(clean_type)

        # Add dataset name
        if dataset_name:
            # Clean dataset name for tag
            clean_name = dataset_name.replace("/", "_").replace(" ", "_").replace(":", "_")
            tag_parts.append(clean_name[:20])  # Limit length

        # Add timestamp
        tag_parts.append(f"v{timestamp}")

        tag_name = "_".join(tag_parts)

        # Create descriptive commit message
        commit_msg_parts = [f"Preprocessing code snapshot {timestamp}"]
        if preprocessing_type:
            commit_msg_parts.append(f"for {preprocessing_type}")
        if dataset_name:
            commit_msg_parts.append(f"dataset: {dataset_name}")

        commit_message = " ".join(commit_msg_parts)

        # Create the tag
        result = subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", commit_message],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )
        if result.returncode != 0:
            print(f"Warning: Failed to create git tag: {result.stderr}")
            return ""

        # Try to push the tag (don't fail if this doesn't work)
        result = subprocess.run(
            ["git", "push", "origin", tag_name], capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        if result.returncode != 0:
            print(f"Warning: Failed to push git tag (continuing anyway): {result.stderr}")
        else:
            print(f"✅ Created and pushed git tag: {tag_name}")

        return tag_name

    except Exception as e:
        print(f"Warning: Failed to create git tag: {e}")
        return ""


def get_git_info(auto_tag: bool = True) -> Dict[str, str]:
    """Get git repository information for code version tracking."""
    git_info = {}

    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        if result.returncode == 0:
            git_info["commit_hash"] = result.stdout.strip()
    except Exception:
        git_info["commit_hash"] = "unknown"

    try:
        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()
    except Exception:
        git_info["branch"] = "unknown"

    try:
        # Check if there are uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        if result.returncode == 0:
            has_changes = bool(result.stdout.strip())
            git_info["has_uncommitted_changes"] = has_changes

            # If there are changes, check if they're preprocessing-related
            if has_changes:
                requires_tag, all_files, critical_files, relevant_files = check_preprocessing_related_changes()
                git_info["has_preprocessing_related_changes"] = len(all_files) > 0
                git_info["preprocessing_related_files"] = all_files
                git_info["critical_preprocessing_files"] = critical_files
                git_info["relevant_preprocessing_files"] = relevant_files

                # Show all files but distinguish between critical and relevant
                if all_files:
                    if critical_files:
                        print("⚠️  Found uncommitted changes to CRITICAL preprocessing code:")
                        for file in critical_files:
                            print(f"    🔴 {file}")

                    if relevant_files:
                        print("📝 Found uncommitted changes to relevant preprocessing files:")
                        for file in relevant_files:
                            print(f"    🟡 {file}")

                # Create a tag only if there are critical changes and auto_tag is enabled
                if requires_tag:
                    if auto_tag:
                        print("🏷️  Creating git tag for reproducibility (critical changes detected)...")
                        # Extract dataset info for more descriptive tag
                        dataset_name = None
                        preprocessing_type = "robotics_data"  # Default type
                        tag_name = create_preprocessing_tag(dataset_name, preprocessing_type)
                        if tag_name:
                            git_info["preprocessing_tag"] = tag_name
                            git_info["commit_hash_for_reproduction"] = git_info["commit_hash"]
                            print(f"📌 Use git tag '{tag_name}' to reproduce this exact code state")
                        else:
                            git_info["preprocessing_tag"] = "failed_to_create"
                    else:
                        print(
                            "🏷️  Auto-tagging disabled. "
                            "Consider manually creating a git tag for reproducibility of critical changes."
                        )
                        git_info["auto_tag_disabled"] = True
                elif all_files:
                    print("ℹ️  Only non-critical files changed. No git tag needed for reproducibility.")

            else:
                git_info["has_preprocessing_related_changes"] = False
                git_info["preprocessing_related_files"] = []
                git_info["critical_preprocessing_files"] = []
                git_info["relevant_preprocessing_files"] = []

    except Exception:
        git_info["has_uncommitted_changes"] = "unknown"
        git_info["has_preprocessing_related_changes"] = "unknown"

    try:
        # Get latest commit message
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%s"], capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        if result.returncode == 0:
            git_info["latest_commit_message"] = result.stdout.strip()
    except Exception:
        git_info["latest_commit_message"] = "unknown"

    # Add remote URL for full reproducibility info
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )
        if result.returncode == 0:
            git_info["remote_url"] = result.stdout.strip()
    except Exception:
        git_info["remote_url"] = "unknown"

    return git_info


def get_source_data_info(source_path: str, episodes: List[str]) -> Dict[str, Any]:
    """Get information about the source data."""
    source_info = {
        "source_path": source_path,
        "num_episodes": len(episodes),
        "episode_paths": episodes[:10],  # Store first 10 for reference
        "total_episodes_available": len(episodes),
    }

    # Try to get modification times of some episodes for data versioning
    try:
        fs, _ = fsspec.core.url_to_fs(source_path)
        sample_episodes = episodes[:3]  # Check first 3 episodes
        mod_times = []

        for episode in sample_episodes:
            try:
                if source_path.startswith("s3://"):
                    # For S3, try to get object info
                    fs_path = episode[5:] if episode.startswith("s3://") else episode
                    info = fs.info(fs_path)
                    if "LastModified" in info:
                        mod_times.append(info["LastModified"].isoformat())
                else:
                    # For local files
                    stat = fs.stat(episode)
                    mod_times.append(datetime.datetime.fromtimestamp(stat["mtime"]).isoformat())
            except Exception:
                continue

        source_info["sample_episode_modification_times"] = mod_times

        # Create a simple hash of episode paths for data version tracking
        episode_hash = hashlib.md5("\n".join(sorted(episodes)).encode()).hexdigest()
        source_info["episode_list_hash"] = episode_hash

    except Exception as e:
        source_info["source_data_info_error"] = str(e)

    return source_info


def create_processing_metadata(
    args: PreprocessParams, episodes: List[str], total_samples: int, processing_stats: Dict[str, int]
) -> Dict[str, Any]:
    """Create comprehensive metadata about the processing run."""

    # Get command line information
    command_line = {
        "script_name": sys.argv[0],
        "full_command": " ".join(sys.argv),
        "arguments": _draccus_encoding.encode(args),
    }

    # Get environment information
    environment = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "processor": platform.processor(),
        "python_executable": sys.executable,
        "working_directory": os.getcwd(),
        "user": os.environ.get("USER", "unknown"),
        "timestamp_captured": datetime.datetime.now().isoformat(),
    }

    # Get git information (skip if testing flag is set)
    if args.skip_git_tagging:
        git_info = {"skip_git_tagging": True, "commit_hash": "test", "branch": "test"}
    else:
        git_info = get_git_info(auto_tag=not args.no_auto_tag)

    # Get source data information
    source_data_info = get_source_data_info(args.source_episodes, episodes)

    # Processing statistics
    processing_info = {
        "total_samples_created": total_samples,
        "filtering_statistics": processing_stats,
        "estimated_dataset_size_gb": 0,  # Will be updated later
    }

    # Package versions (try to get key dependencies)
    try:
        # Use modern importlib.metadata instead of deprecated pkg_resources
        try:
            from importlib.metadata import PackageNotFoundError, version
        except ImportError:
            # Fallback for Python < 3.8
            from importlib_metadata import PackageNotFoundError, version

        key_packages = ["numpy", "fsspec", "PIL", "tqdm", "boto3", "webdataset"]
        package_versions = {}
        for pkg in key_packages:
            try:
                # Handle special case for PIL package name
                pkg_name = "Pillow" if pkg == "PIL" else pkg
                package_versions[pkg] = version(pkg_name)
            except PackageNotFoundError:
                package_versions[pkg] = "not_found"
            except Exception:
                package_versions[pkg] = "unknown"
        environment["package_versions"] = package_versions
    except ImportError:
        # If importlib.metadata is not available, fall back gracefully
        environment["package_versions"] = "unavailable_importlib_metadata_missing"
    except Exception:
        environment["package_versions"] = "unavailable"

    # Create reproducibility instructions based on git state
    reproducibility_notes = []

    if git_info.get("preprocessing_tag"):
        # If we created a tag, use that for reproduction
        reproducibility_notes.extend(
            [
                f"EXACT REPRODUCTION: Use git tag '{git_info['preprocessing_tag']}'",
                "Commands to reproduce:",
                f"  git clone {git_info.get('remote_url', 'REPO_URL')}",
                f"  git checkout {git_info['preprocessing_tag']}",
                f"  {command_line['full_command']}",
                "",
                "This tag captures the exact code state including uncommitted changes used for this dataset.",
            ]
        )
    elif git_info.get("has_uncommitted_changes"):
        # If there are uncommitted changes but no tag was created
        reproducibility_notes.extend(
            [
                f"WARNING: Dataset created with uncommitted changes to commit {git_info.get('commit_hash', 'unknown')}",
                "For exact reproduction, the following files had uncommitted changes:",
            ]
        )
        for file in git_info.get("preprocessing_related_files", []):
            reproducibility_notes.append(f"  - {file}")
        reproducibility_notes.extend(
            [
                "",
                "Basic reproduction (may differ due to uncommitted changes):",
                f"  git clone {git_info.get('remote_url', 'REPO_URL')}",
                f"  git checkout {git_info.get('commit_hash', 'COMMIT_HASH')}",
                f"  {command_line['full_command']}",
            ]
        )
    else:
        # Clean state - straightforward reproduction
        reproducibility_notes.extend(
            [
                "CLEAN REPRODUCTION: No uncommitted changes",
                "Commands to reproduce:",
                f"  git clone {git_info.get('remote_url', 'REPO_URL')}",
                f"  git checkout {git_info.get('commit_hash', 'COMMIT_HASH')}",
                f"  {command_line['full_command']}",
            ]
        )

    reproducibility_notes.extend(
        [
            "",
            "Additional requirements:",
            "- Ensure the source data at the specified paths is unchanged (check episode_list_hash)",
            "- Use the same package versions if possible for identical results",
            "- Use the same hardware/OS for identical performance characteristics",
        ]
    )

    # Combine all metadata
    metadata = {
        "metadata_version": "1.0",
        "created_at": datetime.datetime.now().isoformat(),
        "command_line": command_line,
        "environment": environment,
        "git_info": git_info,
        "source_data": source_data_info,
        "processing": processing_info,
        "reproducibility_notes": reproducibility_notes,
    }

    return metadata


class PaddingStrategy:
    """Optimized padding strategies with vectorized operations."""

    @staticmethod
    def copy_edge(data: np.ndarray, pad_before: int, pad_after: int) -> np.ndarray:
        """Vectorized edge padding."""
        if pad_before == 0 and pad_after == 0:
            return data

        pad_width = [(pad_before, pad_after)] + [(0, 0)] * (data.ndim - 1)
        return np.pad(data, pad_width, mode="edge")

    @staticmethod
    def zero_pad(data: np.ndarray, pad_before: int, pad_after: int) -> np.ndarray:
        """Vectorized zero padding."""
        if pad_before == 0 and pad_after == 0:
            return data

        pad_width = [(pad_before, pad_after)] + [(0, 0)] * (data.ndim - 1)
        return np.pad(data, pad_width, mode="constant", constant_values=0)

    @staticmethod
    def reflect_pad(data: np.ndarray, pad_before: int, pad_after: int) -> np.ndarray:
        """Vectorized reflect padding."""
        if pad_before == 0 and pad_after == 0:
            return data

        pad_width = [(pad_before, pad_after)] + [(0, 0)] * (data.ndim - 1)
        return np.pad(data, pad_width, mode="reflect")


class EpisodeProcessor:
    """Episode processor with streaming."""

    def __init__(
        self,
        past_lowdim_steps: int = 4,
        future_lowdim_steps: int = 16,
        image_indices: List[int] = None,
        max_padding_left: int = 5,
        max_padding_right: int = 5,
        padding_strategy: str = "copy",
        filter_still_samples: bool = False,
        still_threshold: float = 0.01,
        stride: int = 1,
        jpeg_quality: int = 95,
        fail_on_nan: bool = True,
        camera_names: Optional[List[str]] = None,
        discard_keys: Optional[List[str]] = None,
        compute_statistics: bool = True,
        resize_images_size: int = 0,
    ):
        if image_indices is None:
            image_indices = [-1, 0]
        self.past_lowdim_steps = past_lowdim_steps
        self.future_lowdim_steps = future_lowdim_steps
        self.image_indices = sorted(image_indices)
        self.max_padding_left = max_padding_left
        self.max_padding_right = max_padding_right
        self.stride = stride
        self.jpeg_quality = jpeg_quality
        self.fail_on_nan = fail_on_nan
        self.camera_names = camera_names
        self.filter_still_samples = filter_still_samples
        self.still_threshold = still_threshold
        self.discard_keys = discard_keys or []
        self.compute_statistics = compute_statistics
        self.resize_images_size = resize_images_size

        # Statistics tracking
        self.total_potential_samples = 0
        self.still_samples_filtered = 0
        self.padding_samples_filtered = 0

        # Initialize JPEG encoder
        init_jpeg_encoder(jpeg_quality)

        # Set padding function
        self.pad_functions = {
            "copy": PaddingStrategy.copy_edge,
            "zero": PaddingStrategy.zero_pad,
            "reflect": PaddingStrategy.reflect_pad,
        }
        self.pad_fn = self.pad_functions[padding_strategy]

    def load_episode_data(self, episode_path: str) -> Dict[str, Any]:
        """Load episode data with optimization and retry logic."""
        processed_path = os.path.join(episode_path, "processed")

        # Load metadata with retry
        metadata_path = os.path.join(processed_path, "metadata.yaml")
        for attempt in range(3):
            try:
                with fsspec.open(metadata_path, "r") as f:
                    metadata = yaml.safe_load(f)
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                print(f"Retry {attempt + 1} loading metadata for {episode_path}")
                import time

                time.sleep(1)

        # Load observations with retry
        obs_path = os.path.join(processed_path, "observations.npz")
        for attempt in range(3):
            try:
                with fsspec.open(obs_path, "rb") as f:
                    observations = np.load(f, allow_pickle=True)
                    observations = {k: v for k, v in observations.items() if k not in self.discard_keys}
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                print(f"Retry {attempt + 1} loading observations for {episode_path}")
                import time

                time.sleep(1)

        # Skip actions (as in original)
        actions = {}

        # Load camera params (optional)
        intrinsics, extrinsics = {}, {}
        try:
            intrinsics_path = os.path.join(processed_path, "intrinsics.npz")
            extrinsics_path = os.path.join(processed_path, "extrinsics.npz")
            with fsspec.open(intrinsics_path, "rb") as f:
                intrinsics_archive = np.load(f)
                # Convert to regular dict to avoid lazy loading issues
                intrinsics = {key: intrinsics_archive[key] for key in intrinsics_archive.files}
            with fsspec.open(extrinsics_path, "rb") as f:
                extrinsics_archive = np.load(f)
                # Convert to regular dict to avoid lazy loading issues
                extrinsics = {key: extrinsics_archive[key] for key in extrinsics_archive.files}
        except Exception:
            pass

        # Transform camera calibration keys from camera IDs to semantic names
        intrinsics, extrinsics = self.transform_camera_calibration_keys(intrinsics, extrinsics, metadata)

        return {
            "metadata": metadata,
            "observations": observations,
            "actions": actions,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }

    def extract_camera_data(
        self, observations: Dict[str, np.ndarray], metadata: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Extract camera data with filtering."""
        camera_mapping = metadata.get("camera_id_to_semantic_name", {})

        if self.camera_names:
            filtered_mapping = {
                cid: sname
                for cid, sname in camera_mapping.items()
                if sname in self.camera_names and cid in observations
            }
        else:
            filtered_mapping = {cid: sname for cid, sname in camera_mapping.items() if cid in observations}

        return {sname: observations[cid] for cid, sname in filtered_mapping.items()}

    def extract_lowdim_data(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract low-dimensional data."""
        return {
            key: value
            for key, value in observations.items()
            if len(value.shape) <= 2 or key.startswith(("robot__", "language_"))
        }

    def is_still_sample(self, lowdim_data: Dict[str, np.ndarray], start_idx: int, end_idx: int) -> bool:
        """Check if sample is still."""
        if not self.filter_still_samples:
            return False

        movement_keys = [k for k in lowdim_data if any(x in k.lower() for x in ["joint_position", "poses", "xyz"])]

        # If no movement keys found, don't filter the sample
        if not movement_keys:
            # Debug: Print available keys once to help troubleshoot
            if not hasattr(self, "_debug_keys_printed"):
                print(f"🔍 Debug: No movement keys found. Available keys: {list(lowdim_data.keys())[:10]}")
                self._debug_keys_printed = True
            return False

        for key in movement_keys:
            data = lowdim_data[key][start_idx : end_idx + 1]
            if len(data) > 1:
                movement = np.std(data, axis=0).max()
                if movement > self.still_threshold:
                    return False

        return True

    def create_relative_lowdim_data(
        self, lowdim_data: Dict[str, np.ndarray], anchor_relative_idx: int
    ) -> Dict[str, np.ndarray]:
        """Create relative coordinate data."""
        relative_data = {}

        for key, data in lowdim_data.items():
            if not np.issubdtype(data.dtype, np.number):
                continue

            try:
                if "xyz" in key.lower() and data.shape[-1] == 3:
                    relative_data[f"{key}_relative"] = xyz_to_relative(data, anchor_relative_idx)
                elif "rot_6d" in key.lower() and data.shape[-1] == 6:
                    relative_data[f"{key}_relative"] = rot_6d_to_relative(data, anchor_relative_idx)
                elif any(pos_word in key.lower() for pos_word in ["position", "pose", "pos"]) and data.shape[-1] == 3:
                    relative_data[f"{key}_relative"] = xyz_to_relative(data, anchor_relative_idx)
            except Exception:
                continue

        return relative_data

    def transform_camera_calibration_keys(
        self, intrinsics: Dict[str, Any], extrinsics: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Transform camera calibration keys from camera IDs to semantic names."""
        # Get camera mapping from metadata
        camera_mapping = metadata.get("camera_id_to_semantic_name", {})
        if not camera_mapping:
            # If no mapping available, return original data
            return intrinsics, extrinsics

        # Transform intrinsics keys
        transformed_intrinsics = {}
        for camera_id, semantic_name in camera_mapping.items():
            if camera_id in intrinsics:
                transformed_intrinsics[semantic_name] = intrinsics[camera_id]

        # Transform extrinsics keys
        transformed_extrinsics = {}
        for camera_id, semantic_name in camera_mapping.items():
            if camera_id in extrinsics:
                transformed_extrinsics[semantic_name] = extrinsics[camera_id]

        return transformed_intrinsics, transformed_extrinsics

    def extract_sample_camera_calibration(
        self,
        episode_intrinsics: Dict[str, Any],
        episode_extrinsics: Dict[str, Any],
        valid_start: int,
        valid_end: int,
        past_padding: int,
        future_padding: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract camera calibration data for the lowdim sequence timespan."""
        sample_intrinsics = {}
        sample_extrinsics = {}

        # Extract intrinsics for the sequence timespan
        for camera_name, intrinsics_data in episode_intrinsics.items():
            if isinstance(intrinsics_data, np.ndarray):
                if intrinsics_data.ndim == 3:  # Time-varying intrinsics (timesteps, 3, 3)
                    # Extract the sequence from valid_start to valid_end (inclusive)
                    valid_intrinsics = intrinsics_data[valid_start : valid_end + 1]

                    # Apply padding to match lowdim data
                    if past_padding > 0 or future_padding > 0:
                        # Use edge padding for camera calibration
                        pad_width = [(past_padding, future_padding)] + [(0, 0)] * (valid_intrinsics.ndim - 1)
                        valid_intrinsics = np.pad(valid_intrinsics, pad_width, mode="edge")

                    sample_intrinsics[camera_name] = valid_intrinsics
                elif intrinsics_data.ndim == 2:  # Static intrinsics (3, 3)
                    # For static calibration, repeat for the sequence length
                    sequence_length = (valid_end - valid_start + 1) + past_padding + future_padding
                    sample_intrinsics[camera_name] = np.tile(intrinsics_data[np.newaxis, :, :], (sequence_length, 1, 1))

        # Extract extrinsics for the sequence timespan
        for camera_name, extrinsics_data in episode_extrinsics.items():
            if isinstance(extrinsics_data, np.ndarray):
                if extrinsics_data.ndim == 3:  # Time-varying extrinsics (timesteps, 4, 4)
                    # Extract the sequence from valid_start to valid_end (inclusive)
                    valid_extrinsics = extrinsics_data[valid_start : valid_end + 1]

                    # Apply padding to match lowdim data
                    if past_padding > 0 or future_padding > 0:
                        # Use edge padding for camera calibration
                        pad_width = [(past_padding, future_padding)] + [(0, 0)] * (valid_extrinsics.ndim - 1)
                        valid_extrinsics = np.pad(valid_extrinsics, pad_width, mode="edge")

                    sample_extrinsics[camera_name] = valid_extrinsics
                elif extrinsics_data.ndim == 2:  # Static extrinsics (4, 4)
                    # For static calibration, repeat for the sequence length
                    sequence_length = (valid_end - valid_start + 1) + past_padding + future_padding
                    sample_extrinsics[camera_name] = np.tile(extrinsics_data[np.newaxis, :, :], (sequence_length, 1, 1))

        return sample_intrinsics, sample_extrinsics

    def process_episode_streaming(self, episode_path: str) -> Iterator[Dict[str, Any]]:
        """Process episode with streaming output."""
        try:
            episode_data = self.load_episode_data(episode_path)
            episode_id = os.path.basename(episode_path.rstrip("/"))

            observations = episode_data["observations"]
            first_obs_key = next(iter(observations.keys()))
            episode_length = observations[first_obs_key].shape[0]

            # Pre-extract data
            camera_data = self.extract_camera_data(observations, episode_data["metadata"])
            lowdim_data = self.extract_lowdim_data(observations)

            # Generate samples
            for anchor_timestep in range(0, episode_length, self.stride):
                self.total_potential_samples += 1

                # Calculate windows
                lowdim_start = anchor_timestep - self.past_lowdim_steps
                lowdim_end = anchor_timestep + self.future_lowdim_steps

                # Check padding
                past_padding = max(0, -lowdim_start)
                future_padding = max(0, lowdim_end - episode_length + 1)

                if past_padding > self.max_padding_left or future_padding > self.max_padding_right:
                    self.padding_samples_filtered += 1
                    continue

                valid_start = max(0, lowdim_start)
                valid_end = min(episode_length - 1, lowdim_end)

                # Check stillness
                if self.is_still_sample(lowdim_data, valid_start, valid_end):
                    self.still_samples_filtered += 1
                    continue

                # Extract images
                sample_images = {}
                actual_image_timesteps = []

                for img_offset in self.image_indices:
                    img_timestep = np.clip(anchor_timestep + img_offset, 0, episode_length - 1)
                    actual_image_timesteps.append(int(img_timestep))

                    for camera_name, camera_images in camera_data.items():
                        key = f"{camera_name}_t{img_offset}"
                        sample_images[key] = camera_images[img_timestep]

                # Process lowdim data
                sample_lowdim = {}
                for key, data in lowdim_data.items():
                    valid_data = data[valid_start : valid_end + 1]
                    if past_padding > 0 or future_padding > 0:
                        valid_data = self.pad_fn(valid_data, past_padding, future_padding)
                    sample_lowdim[key] = valid_data

                # Add relative coordinates
                anchor_relative_idx = self.past_lowdim_steps
                relative_data = self.create_relative_lowdim_data(sample_lowdim, anchor_relative_idx)
                sample_lowdim.update(relative_data)

                # Create masks
                total_length = self.past_lowdim_steps + self.future_lowdim_steps + 1
                past_mask = np.ones(total_length, dtype=bool)
                future_mask = np.ones(total_length, dtype=bool)
                # Current time step is part of the future for low dim data
                past_mask[anchor_relative_idx:] = False
                future_mask[:anchor_relative_idx] = False

                if past_padding > 0:
                    past_mask[:past_padding] = False
                    future_mask[:past_padding] = False
                if future_padding > 0:
                    future_mask[-future_padding:] = False
                    past_mask[-future_padding:] = False

                # Extract sequence-specific camera calibration
                sample_intrinsics, sample_extrinsics = self.extract_sample_camera_calibration(
                    episode_data.get("intrinsics", {}),
                    episode_data.get("extrinsics", {}),
                    valid_start,
                    valid_end,
                    past_padding,
                    future_padding,
                )

                # Create metadata
                sample_metadata = SampleMetadata(
                    episode_id=episode_id,
                    sample_id=f"{uuid.uuid4()}_{episode_id}_t{anchor_timestep:04d}",
                    anchor_timestep=int(anchor_timestep),
                    anchor_relative_idx=int(anchor_relative_idx),
                    image_timesteps=actual_image_timesteps,
                    lowdim_start_timestep=int(lowdim_start),
                    lowdim_end_timestep=int(lowdim_end),
                    past_padding=int(past_padding),
                    future_padding=int(future_padding),
                    camera_names=list(camera_data.keys()),
                    original_episode_length=int(episode_length),
                    original_image_sizes={},
                    is_padded=bool(past_padding > 0 or future_padding > 0),
                )

                yield {
                    "images": sample_images,
                    "lowdim": sample_lowdim,
                    "actions": {},
                    "past_mask": past_mask,
                    "future_mask": future_mask,
                    "metadata": sample_metadata,
                    "intrinsics": sample_intrinsics,
                    "extrinsics": sample_extrinsics,
                }

        except Exception as e:
            if self.fail_on_nan:
                raise e
            print(f"Warning: Failed to process episode {episode_path}: {e}")


class StreamingShardWriter:
    """Memory-efficient streaming shard writer with optional background S3 uploads and incremental updates."""

    def __init__(
        self,
        output_dir: str,
        samples_per_shard: int,
        jpeg_quality: int = 95,
        gpu_resize: bool = False,
        upload_workers: int = 16,
        enable_incremental_updates: bool = True,
        update_frequency: int = 5,  # Update metadata every N shards
        resume: bool = False,  # Whether to resume from existing progress
    ):
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.jpeg_quality = jpeg_quality
        self.gpu_resize = gpu_resize and (torch is not None) and torch.cuda.is_available()
        self.current_shard_idx = 0
        self.current_shard_samples = 0
        self.current_shard_file = None
        self.current_shard_tar = None
        self.manifest_data = []
        self.enable_incremental_updates = enable_incremental_updates
        self.update_frequency = update_frequency
        self.resume = resume
        self.total_samples_written = 0
        self.processed_episodes = set()  # Track processed episodes for resume capability

        # Filtering statistics for recovery
        self.total_potential_samples = 0
        self.total_still_filtered = 0
        self.total_padding_filtered = 0

        # Setup output directory
        self.is_s3_output = output_dir.startswith("s3://")
        if self.is_s3_output:
            self.temp_dir = tempfile.mkdtemp()
            self.shard_dir = self.temp_dir
        else:
            self.shard_dir = output_dir
            os.makedirs(self.shard_dir, exist_ok=True)

        # Initialize manifest file for incremental updates
        self.manifest_path = os.path.join(self.shard_dir, "manifest.jsonl")
        self.progress_path = os.path.join(self.shard_dir, "processing_progress.json")

        # Try to resume from existing progress
        self._try_resume_from_existing()

        # Initialize async S3 upload machinery if needed
        if self.is_s3_output:
            self._init_s3_transfer(upload_workers)

    def _try_resume_from_existing(self):
        """Try to resume from existing processing progress."""
        if not self.enable_incremental_updates or not self.resume:
            return

        try:
            # Check if progress file exists
            if os.path.exists(self.progress_path):
                with open(self.progress_path, "r") as f:
                    progress = json.load(f)

                # Resume from where we left off
                self.current_shard_idx = progress.get("next_shard_idx", 0)
                self.total_samples_written = progress.get("total_samples_written", 0)
                self.processed_episodes = set(progress.get("processed_episodes", []))

                # Restore filtering statistics if available
                filtering_stats = progress.get("filtering_statistics", {})
                self.total_potential_samples = filtering_stats.get("total_potential", 0)
                self.total_still_filtered = filtering_stats.get("total_still_filtered", 0)
                self.total_padding_filtered = filtering_stats.get("total_padding_filtered", 0)

                # Load existing manifest data
                if os.path.exists(self.manifest_path):
                    self.manifest_data = []
                    with open(self.manifest_path, "r") as f:
                        for line in f:
                            if line.strip():
                                self.manifest_data.append(json.loads(line.strip()))

                print(
                    f"📂 Resuming from shard {self.current_shard_idx}, {self.total_samples_written} samples processed"
                )
                print(f"📂 {len(self.processed_episodes)} episodes already processed")
                if self.total_potential_samples > 0:
                    print(
                        f"📊 Filtering stats recovered: {self.total_potential_samples} potential, "
                        f"{self.total_still_filtered} still filtered, {self.total_padding_filtered} padding filtered"
                    )

        except Exception as e:
            print(f"Warning: Could not resume from existing progress: {e}")
            # Reset to start from beginning
            self.current_shard_idx = 0
            self.total_samples_written = 0
            self.manifest_data = []

    def _update_progress(self):
        """Update processing progress file."""
        if not self.enable_incremental_updates:
            return

        progress = {
            "next_shard_idx": self.current_shard_idx,
            "total_samples_written": self.total_samples_written,
            "processed_episodes": list(self.processed_episodes),
            "filtering_statistics": {
                "total_potential": self.total_potential_samples,
                "total_still_filtered": self.total_still_filtered,
                "total_padding_filtered": self.total_padding_filtered,
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }

        with open(self.progress_path, "w") as f:
            json.dump(progress, f, indent=2)

        # Upload progress file if using S3
        if self.is_s3_output:
            self._schedule_upload(self.progress_path, os.path.basename(self.progress_path))

    def _append_to_manifest(self, shard_entry: Dict[str, Any]):
        """Append a shard entry to the manifest file."""
        if not self.enable_incremental_updates:
            return

        # Append to manifest file
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(shard_entry, separators=(",", ":")) + "\n")

        # Upload updated manifest if using S3
        if self.is_s3_output:
            self._schedule_upload(self.manifest_path, os.path.basename(self.manifest_path))

    def update_metadata_files(
        self, metadata: Dict[str, Any], statistics: Optional[Dict[str, Any]] = None, statistics_state=None
    ):
        """Update metadata and statistics files incrementally."""
        if not self.enable_incremental_updates:
            return

        # Update processing metadata
        metadata_path = os.path.join(self.shard_dir, "processing_metadata.json")

        # Update current progress in metadata
        if "processing" in metadata:
            metadata["processing"]["total_samples_created"] = self.total_samples_written
            metadata["processing"]["shards_completed"] = len(self.manifest_data)
            metadata["processing"]["last_updated"] = datetime.datetime.now().isoformat()

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if self.is_s3_output:
            self._schedule_upload(metadata_path, os.path.basename(metadata_path))

        # Update statistics if provided
        if statistics:
            stats_path = os.path.join(self.shard_dir, "dataset_statistics.json")
            with open(stats_path, "w") as f:
                json.dump(statistics, f, indent=2)

            if self.is_s3_output:
                self._schedule_upload(stats_path, os.path.basename(stats_path))

        # Save statistics state for recovery if provided
        if statistics_state:
            self.save_statistics_state(statistics_state)

    def save_statistics_state(self, statistics_state):
        """Save the current statistics computation state for recovery."""
        if not self.enable_incremental_updates:
            return

        stats_state_path = os.path.join(self.shard_dir, "processing_statistics.json")
        statistics_state.save_state(stats_state_path)

        if self.is_s3_output:
            self._schedule_upload(stats_state_path, os.path.basename(stats_state_path))

    def update_filtering_statistics(self, potential_samples: int, still_filtered: int, padding_filtered: int):
        """Update filtering statistics for recovery."""
        self.total_potential_samples += potential_samples
        self.total_still_filtered += still_filtered
        self.total_padding_filtered += padding_filtered

    def mark_episode_completed(self, episode_path: str):
        """Mark an episode as completed for resume capability."""
        if self.enable_incremental_updates:
            episode_id = os.path.basename(episode_path.rstrip("/"))
            self.processed_episodes.add(episode_id)

    def is_episode_processed(self, episode_path: str) -> bool:
        """Check if an episode has already been processed."""
        if not self.enable_incremental_updates:
            return False
        episode_id = os.path.basename(episode_path.rstrip("/"))
        return episode_id in self.processed_episodes

    def get_unprocessed_episodes(self, all_episodes: List[str]) -> List[str]:
        """Filter out already processed episodes."""
        if not self.enable_incremental_updates:
            return all_episodes

        unprocessed = []
        skipped_count = 0

        for episode in all_episodes:
            if not self.is_episode_processed(episode):
                unprocessed.append(episode)
            else:
                skipped_count += 1

        if skipped_count > 0:
            print(f"📂 Skipping {skipped_count} already processed episodes")

        return unprocessed

    def _init_s3_transfer(self, upload_workers: int):
        s3_path_clean = self.output_dir[5:]
        self.s3_bucket = s3_path_clean.split("/")[0]
        self.s3_prefix = "/".join(s3_path_clean.split("/")[1:])
        if self.s3_prefix and not self.s3_prefix.endswith("/"):
            self.s3_prefix += "/"

        import boto3
        from boto3.s3.transfer import S3Transfer, TransferConfig

        self._s3_client = boto3.client("s3")
        self._s3_transfer = S3Transfer(
            self._s3_client,
            config=TransferConfig(
                use_threads=True,
                max_concurrency=min(64, max(8, upload_workers)),
                multipart_threshold=8 * 1024 * 1024,
            ),
        )
        self._upload_executor = ThreadPoolExecutor(max_workers=min(64, max(8, upload_workers)))
        self._upload_futures = []

    def _schedule_upload(self, local_file: str, relative_path: Optional[str] = None):
        if not self.is_s3_output:
            return
        if relative_path is None:
            relative_path = os.path.relpath(local_file, self.shard_dir)
        s3_key = f"{self.s3_prefix}{relative_path}"
        future = self._upload_executor.submit(self._s3_transfer.upload_file, local_file, self.s3_bucket, s3_key)
        self._upload_futures.append(future)

    def upload_path(self, local_file: str, relative_path: Optional[str] = None):
        """Public API to queue an upload of any file under `shard_dir`."""
        self._schedule_upload(local_file, relative_path)

    def wait_for_uploads(self):
        if not self.is_s3_output:
            return
        for f in self._upload_futures:
            f.result()
        self._upload_futures = []

    def _start_new_shard(self):
        """Start a new shard file."""
        if self.current_shard_tar:
            self.current_shard_tar.close()

        shard_name = f"shard_{self.current_shard_idx:08d}"
        self.current_shard_file = os.path.join(self.shard_dir, f"{shard_name}.tar")
        self.current_shard_tar = tarfile.open(self.current_shard_file, "w")  # noqa SIM115
        self.current_shard_samples = 0

    def add_sample(self, sample: Dict[str, Any], target_image_size: Optional[Tuple[int, int]] = None):
        """Add a sample to current shard."""
        # Start new shard if needed
        if self.current_shard_tar is None or (
            self.samples_per_shard > 0 and self.current_shard_samples >= self.samples_per_shard
        ):
            if self.current_shard_tar:
                # Close previous shard and update manifest
                closed_shard_path = self.current_shard_file
                self.current_shard_tar.close()

                # Create shard entry
                shard_entry = {
                    "shard": f"shard_{self.current_shard_idx:08d}",
                    "num_sequences": self.current_shard_samples,
                }
                self.manifest_data.append(shard_entry)

                # Incrementally update manifest file
                self._append_to_manifest(shard_entry)

                # Queue upload of the closed shard if writing to S3
                if self.is_s3_output and closed_shard_path is not None:
                    self._schedule_upload(closed_shard_path, os.path.basename(closed_shard_path))

                self.current_shard_idx += 1

                # Update progress after completing a shard
                self._update_progress()

            self._start_new_shard()

        sample_id = sample["metadata"].sample_id

        # Write images as JPEG (optimized)
        original_image_sizes = sample["metadata"].original_image_sizes

        # Optional: batch GPU resize for speed
        resized_images: Dict[str, np.ndarray]
        if self.gpu_resize and target_image_size is not None:
            try:
                keys = list(sample["images"].keys())
                imgs = sample["images"]
                # Build batch tensor [N, C, H, W] on GPU
                tensors = []
                for k in keys:
                    arr = imgs[k]
                    if arr.dtype != np.uint8:
                        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
                    t = torch.from_numpy(arr).to(device="cuda", non_blocking=True)
                    if t.ndim == 2:
                        t = t.unsqueeze(-1).expand(-1, -1, 3)
                    t = t.permute(2, 0, 1).float() / 255.0
                    tensors.append(t)
                batch = torch.stack(tensors, dim=0)
                th, tw = target_image_size[1], target_image_size[0]
                batch = F.interpolate(batch, size=(th, tw), mode="bilinear", align_corners=False)
                batch = (batch.clamp(0, 1) * 255.0).to(torch.uint8).permute(0, 2, 3, 1).cpu()
                resized_images = {k: batch[i].numpy() for i, k in enumerate(keys)}
                # After GPU resize, skip per-image PIL resize by setting target_size=None
                per_image_target = None
            except Exception:
                # Fallback to CPU path if anything fails
                resized_images = sample["images"]
                per_image_target = target_image_size
        else:
            resized_images = sample["images"]
            per_image_target = target_image_size

        for img_key, img_data in resized_images.items():
            img_bytes, original_image_size = image_to_bytes(img_data, self.jpeg_quality, per_image_target)
            info = tarfile.TarInfo(name=f"{sample_id}.{img_key}.jpg")
            info.size = len(img_bytes)
            self.current_shard_tar.addfile(tarinfo=info, fileobj=io.BytesIO(img_bytes))
            img_key_no_t = img_key.split("_t")[0]
            original_image_sizes[img_key_no_t] = original_image_size

        # Write low-dim data as NPZ
        lowdim_buf = io.BytesIO()
        np.savez_compressed(lowdim_buf, **sample["lowdim"])  # Use compression
        lowdim_buf.seek(0)
        info = tarfile.TarInfo(name=f"{sample_id}.lowdim.npz")
        info.size = len(lowdim_buf.getbuffer())
        self.current_shard_tar.addfile(tarinfo=info, fileobj=lowdim_buf)

        # Write masks as NPZ
        masks_buf = io.BytesIO()
        np.savez_compressed(masks_buf, past_mask=sample["past_mask"], future_mask=sample["future_mask"])
        masks_buf.seek(0)
        info = tarfile.TarInfo(name=f"{sample_id}.masks.npz")
        info.size = len(masks_buf.getbuffer())
        self.current_shard_tar.addfile(tarinfo=info, fileobj=masks_buf)

        # Write intrinsics as NPZ (if available)
        if "intrinsics" in sample and sample["intrinsics"]:
            intrinsics_buf = io.BytesIO()
            np.savez_compressed(intrinsics_buf, **sample["intrinsics"])
            intrinsics_buf.seek(0)
            info = tarfile.TarInfo(name=f"{sample_id}.intrinsics.npz")
            info.size = len(intrinsics_buf.getbuffer())
            self.current_shard_tar.addfile(tarinfo=info, fileobj=intrinsics_buf)

        # Write extrinsics as NPZ (if available)
        if "extrinsics" in sample and sample["extrinsics"]:
            extrinsics_buf = io.BytesIO()
            np.savez_compressed(extrinsics_buf, **sample["extrinsics"])
            extrinsics_buf.seek(0)
            info = tarfile.TarInfo(name=f"{sample_id}.extrinsics.npz")
            info.size = len(extrinsics_buf.getbuffer())
            self.current_shard_tar.addfile(tarinfo=info, fileobj=extrinsics_buf)

        metadata_dict = sample["metadata"].__dict__
        metadata_dict["original_image_sizes"] = original_image_sizes

        metadata_json = json.dumps(metadata_dict, separators=(",", ":")).encode("utf-8")  # Compact JSON
        metadata_buf = io.BytesIO(metadata_json)
        info = tarfile.TarInfo(name=f"{sample_id}.metadata.json")
        info.size = len(metadata_buf.getbuffer())
        self.current_shard_tar.addfile(tarinfo=info, fileobj=metadata_buf)

        self.current_shard_samples += 1
        self.total_samples_written += 1

    def finalize(self) -> Tuple[List[Dict[str, Any]], str]:
        """Finalize writing and return manifest."""
        if self.current_shard_tar:
            closed_shard_path = self.current_shard_file
            self.current_shard_tar.close()

            # Create final shard entry
            shard_entry = {"shard": f"shard_{self.current_shard_idx:08d}", "num_sequences": self.current_shard_samples}
            self.manifest_data.append(shard_entry)

            # Update manifest incrementally if enabled
            if self.enable_incremental_updates:
                self._append_to_manifest(shard_entry)

            if self.is_s3_output and closed_shard_path is not None:
                self._schedule_upload(closed_shard_path, os.path.basename(closed_shard_path))

        # Write complete manifest (for non-incremental mode or as backup)
        if not self.enable_incremental_updates:
            with open(self.manifest_path, "w") as f:
                for entry in self.manifest_data:
                    f.write(json.dumps(entry, separators=(",", ":")) + "\n")

            # Queue manifest upload as well
            if self.is_s3_output:
                self._schedule_upload(self.manifest_path, os.path.basename(self.manifest_path))

        # Final progress update
        self._update_progress()

        return self.manifest_data, self.shard_dir

    def cleanup(self):
        """Cleanup temporary files. Ensures all uploads are complete first."""
        if self.is_s3_output and hasattr(self, "temp_dir"):
            self.wait_for_uploads()
            import shutil

            shutil.rmtree(self.temp_dir)


def make_full_path(relative_path: str, is_s3: bool) -> str:
    return f"s3://{relative_path}" if is_s3 else relative_path


def make_fs_path(full_path: str, is_s3: bool) -> str:
    return full_path[5:] if is_s3 and full_path.startswith("s3://") else full_path


def discover_episodes_targeted(source_paths: List[str], max_episodes: int = -1) -> List[str]:
    """Discover episodes efficiently."""
    if isinstance(source_paths, str):
        source_paths = [source_paths]
    episodes = []
    for source_path in source_paths:
        fs, fsspec_path = fsspec.core.url_to_fs(source_path)

        is_s3 = source_path.startswith("s3://")

        try:
            # Handle the case where source_path is already an episode directory
            source_base = os.path.basename(source_path.rstrip("/"))
            if source_base.startswith("episode_"):
                processed_path = os.path.join(source_path, "processed")
                if fs.exists(make_fs_path(processed_path, is_s3)):
                    episodes.append(source_path)
                    if max_episodes > 0 and len(episodes) >= max_episodes:
                        return sorted(episodes)
                continue

            # Handle the case where source_path is exactly a diffusion_spartan directory
            if source_base == "diffusion_spartan":
                try:
                    episode_items = fs.listdir(fsspec_path)
                    for episode_item in episode_items:
                        episode_relative = episode_item["name"] if isinstance(episode_item, dict) else episode_item
                        episode_path = make_full_path(episode_relative, is_s3)
                        episode_name = os.path.basename(episode_path.rstrip("/"))
                        if episode_name.startswith("episode_"):
                            processed_path = os.path.join(episode_path, "processed")
                            if fs.exists(make_fs_path(processed_path, is_s3)):
                                episodes.append(episode_path)
                                if max_episodes > 0 and len(episodes) >= max_episodes:
                                    return sorted(episodes)
                except Exception:
                    pass
                continue

            # General case: list items under source_path
            date_dirs = fs.listdir(fsspec_path)

            for date_item in date_dirs:
                date_relative = date_item["name"] if isinstance(date_item, dict) else date_item
                date_path = make_full_path(date_relative, is_s3)

                base_name = os.path.basename(date_path.rstrip("/"))

                # If the item itself is an episode directory, add directly
                if base_name.startswith("episode_"):
                    processed_path = os.path.join(date_path, "processed")
                    try:
                        if fs.exists(make_fs_path(processed_path, is_s3)):
                            episodes.append(date_path)
                            if max_episodes > 0 and len(episodes) >= max_episodes:
                                return sorted(episodes)
                    except Exception:
                        continue
                    continue

                # Determine diffusion_spartan path for this item
                if base_name == "diffusion_spartan":
                    diffusion_path = date_path
                else:
                    diffusion_path = os.path.join(date_path, "diffusion_spartan")
                fs_diffusion_path = make_fs_path(diffusion_path, is_s3)

                try:
                    if fs.exists(fs_diffusion_path):
                        episode_items = fs.listdir(fs_diffusion_path)

                        for episode_item in episode_items:
                            episode_relative = episode_item["name"] if isinstance(episode_item, dict) else episode_item
                            episode_path = make_full_path(episode_relative, is_s3)
                            episode_name = os.path.basename(episode_path.rstrip("/"))

                            if episode_name.startswith("episode_"):
                                processed_path = os.path.join(episode_path, "processed")
                                fs_processed_path = make_fs_path(processed_path, is_s3)

                                if fs.exists(fs_processed_path):
                                    episodes.append(episode_path)
                                    if max_episodes > 0 and len(episodes) >= max_episodes:
                                        return sorted(episodes)

                except Exception:
                    continue

        except Exception as e:
            print(f"Error scanning source path {source_path}: {e}")

    return sorted(episodes)


def process_episode_worker(args):
    """Worker function for processing episodes."""
    episode_path, processor_config = args

    try:
        # Initialize processor in worker thread to avoid S3 connection sharing issues
        processor = EpisodeProcessor(**processor_config)
        samples = []

        # print(f"Processing episode: {os.path.basename(episode_path)}")
        for sample in processor.process_episode_streaming(episode_path):
            samples.append(sample)

        # print(f"Completed episode: {os.path.basename(episode_path)} - {len(samples)} samples")
        return (
            samples,
            processor.total_potential_samples,
            processor.still_samples_filtered,
            processor.padding_samples_filtered,
        )
    except Exception as e:
        print(f"Error processing {episode_path}: {e}")
        import traceback

        traceback.print_exc()
        return [], 0, 0, 0


def streaming_episode_worker(
    episode_path: str, processor_config: Dict[str, Any], out_queue: Queue
) -> Tuple[str, int, int, int]:
    """Stream samples from one episode into a queue and return counters."""
    processor = EpisodeProcessor(**processor_config)
    for sample in processor.process_episode_streaming(episode_path):
        out_queue.put(sample)
    return (
        episode_path,
        processor.total_potential_samples,
        processor.still_samples_filtered,
        processor.padding_samples_filtered,
    )


def main():
    """Optimized robotics preprocessing using draccus-configured params."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse CLI into dataclass
    cfg = draccus.parse(config_class=PreprocessParams)

    # Validate required paths
    assert cfg.source_episodes is not None, "--source_episodes is required (or set in config_path)"
    assert cfg.output_dir is not None, "--output_dir is required (or set in config_path)"

    camera_names = cfg.camera_names

    processor_config = {
        "past_lowdim_steps": cfg.past_lowdim_steps,
        "future_lowdim_steps": cfg.future_lowdim_steps,
        "image_indices": cfg.image_indices,
        "max_padding_left": cfg.max_padding_left,
        "max_padding_right": cfg.max_padding_right,
        "padding_strategy": cfg.padding_strategy,
        "filter_still_samples": cfg.filter_still_samples,
        "still_threshold": cfg.still_threshold,
        "stride": cfg.stride,
        "jpeg_quality": cfg.jpeg_quality,
        "fail_on_nan": cfg.fail_on_nan,
        "camera_names": camera_names,
        "discard_keys": cfg.discard_keys,
        "compute_statistics": not cfg.no_statistics,
        "resize_images_size": cfg.resize_images_size,
    }

    print("🚀 Starting optimized preprocessing")
    print(f"Workers: {cfg.num_workers}")
    print(f"Statistics: {'disabled' if cfg.no_statistics else 'enabled'}")
    print(f"Incremental updates: {'enabled' if cfg.enable_incremental_updates else 'disabled'}")
    print(f"Resume: {'enabled' if cfg.resume else 'disabled'}")
    if cfg.enable_incremental_updates:
        print(f"Metadata update frequency: every {cfg.update_frequency} shards")
    if cfg.resize_images_size > 0:
        print(f"Image resize to {cfg.resize_images_size}x{cfg.resize_images_size}")
    else:
        print("Image resize disabled")

    # Create initial metadata with git info (before any code changes during processing)
    print("📋 Capturing initial metadata...")
    start_time = datetime.datetime.now()

    # Discover episodes
    print("🔍 Discovering episodes...")
    episodes = discover_episodes_targeted(cfg.source_episodes, cfg.max_episodes)
    print(f"Found {len(episodes)} episodes")

    # If running on SageMaker Processing with multiple instances, split work across hosts
    try:
        sm_hosts = os.environ.get("SM_HOSTS")
        sm_current_host = os.environ.get("SM_CURRENT_HOST")
        if sm_hosts and sm_current_host:
            hosts = json.loads(sm_hosts)
            if isinstance(hosts, list) and len(hosts) > 1:
                host_index = hosts.index(sm_current_host)
                total_hosts = len(hosts)
                # Chunk episodes evenly across hosts
                episodes = episodes[host_index::total_hosts]
                print(
                    f"Distributed across {total_hosts} hosts. Current host {host_index}"
                    f" processing {len(episodes)} episodes"
                )
    except Exception as e:
        print(f"Warning: failed to split episodes across hosts: {e}")

    # Optionally shuffle input episode processing order
    if cfg.shuffle_input_files and len(episodes) > 1:
        random.shuffle(episodes)

    # Create initial processing metadata with placeholder stats
    initial_processing_stats = {
        "total_potential_samples": 0,
        "samples_created": 0,
        "still_samples_filtered": 0,
        "padding_samples_filtered": 0,
        "episodes_processed": len(episodes),
    }

    # Create metadata with git info captured at start
    metadata = create_processing_metadata(cfg, episodes, 0, initial_processing_stats)
    metadata["processing"]["timestamp_start"] = start_time.isoformat()

    if len(episodes) == 0:
        print("❌ No episodes found!")
        return

    # Initialize shard writer with incremental updates
    shard_writer = StreamingShardWriter(
        cfg.output_dir,
        cfg.samples_per_shard,
        cfg.jpeg_quality,
        gpu_resize=(cfg.resize_images_size > 0 and cfg.use_gpu_resize),
        enable_incremental_updates=cfg.enable_incremental_updates,
        update_frequency=cfg.update_frequency,
        resume=cfg.resume,
    )

    # Filter out already processed episodes for resume capability
    original_episode_count = len(episodes)
    if cfg.resume:
        episodes = shard_writer.get_unprocessed_episodes(episodes)
        if len(episodes) == 0:
            print("✅ All episodes already processed! Processing complete.")
            if original_episode_count > 0:
                print(f"📊 Skipped {original_episode_count} already processed episodes")
            return
        print(f"📂 Resume mode: {len(episodes)} episodes remaining to process")
    else:
        print("🔄 Starting fresh processing (resume disabled)")

    # Process episodes in parallel with streaming to a bounded queue
    print("⚙️  Processing episodes...")

    # Try to load existing statistics state for resume capability
    stats_state_path = os.path.join(shard_writer.shard_dir, "processing_statistics.json")
    if cfg.resume and cfg.enable_incremental_updates and os.path.exists(stats_state_path):
        print(f"📊 Loading existing statistics state from {stats_state_path}")
        global_stats = StreamingDatasetStatistics.from_saved_state(
            stats_state_path, compute_stats=not cfg.no_statistics
        )
        print("✅ Statistics state loaded successfully")
    else:
        global_stats = StreamingDatasetStatistics(not cfg.no_statistics)

    # Initialize filtering counters from recovered state or start fresh
    total_samples = 0
    total_potential = shard_writer.total_potential_samples
    total_still_filtered = shard_writer.total_still_filtered
    total_padding_filtered = shard_writer.total_padding_filtered

    sample_queue: Queue = Queue(maxsize=max(1, cfg.num_workers * 4))

    counters: Dict[str, int] = {"total_samples": 0}

    def consumer():
        target = (cfg.resize_images_size, cfg.resize_images_size) if cfg.resize_images_size > 0 else None
        buffer_size = max(0, int(cfg.shuffle_buffer_size))

        # Reservoir sampling buffer for true random shuffle with bounded memory
        reservoir: List[Dict[str, Any]] = []
        samples_seen = 0

        def write_one(out_sample: Dict[str, Any]):
            shard_writer.add_sample(out_sample, target_image_size=target)
            counters["total_samples"] += 1
            global_stats.merge_from_samples([out_sample])

            # Periodic metadata and statistics updates
            if (
                shard_writer.enable_incremental_updates
                and len(shard_writer.manifest_data) > 0
                and len(shard_writer.manifest_data) % shard_writer.update_frequency == 0
            ):
                # Update metadata with current progress
                current_metadata = dict(metadata)  # Copy the original metadata
                current_stats = global_stats.get_statistics() if not cfg.no_statistics else None
                shard_writer.update_metadata_files(current_metadata, current_stats, global_stats)

        def maybe_write_from_reservoir():
            """Write a random sample from reservoir if it's full"""
            if len(reservoir) > buffer_size:
                # Randomly select and remove one item to write
                idx = random.randrange(len(reservoir))
                sample_to_write = reservoir.pop(idx)
                write_one(sample_to_write)

        while True:
            # Check for shutdown signal
            if _shutdown_requested:
                print("⚠️  Consumer shutting down due to signal...")
                # Flush remaining reservoir samples before shutdown
                while reservoir:
                    idx = random.randrange(len(reservoir)) if len(reservoir) > 1 else 0
                    write_one(reservoir.pop(idx))
                sample_queue.task_done()
                break

            sample = sample_queue.get()
            if sample is None:
                # Flush all remaining reservoir samples in random order
                while reservoir:
                    idx = random.randrange(len(reservoir)) if len(reservoir) > 1 else 0
                    write_one(reservoir.pop(idx))
                sample_queue.task_done()
                break

            samples_seen += 1

            if buffer_size <= 0:
                # No shuffling - write immediately
                write_one(sample)
            else:
                if len(reservoir) < buffer_size:
                    # Reservoir not full yet - just add
                    reservoir.append(sample)
                else:
                    # Reservoir full - use reservoir sampling algorithm
                    # Probability of replacing an existing item: buffer_size / samples_seen
                    if random.random() < buffer_size / samples_seen:
                        # Replace a random item in reservoir with new sample
                        replace_idx = random.randrange(buffer_size)
                        write_one(reservoir[replace_idx])  # Write the displaced sample
                        reservoir[replace_idx] = sample  # Replace with new sample
                    else:
                        # Don't add to reservoir - write immediately
                        write_one(sample)

                # Always try to write from reservoir to maintain flow
                maybe_write_from_reservoir()

            sample_queue.task_done()

    consumer_thread = threading.Thread(target=consumer, daemon=True)
    consumer_thread.start()

    with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
        futures = [
            executor.submit(streaming_episode_worker, episode, processor_config, sample_queue) for episode in episodes
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Episodes"):
            # Check for shutdown signal
            if _shutdown_requested:
                print("⚠️  Shutdown requested, canceling remaining episodes...")
                for remaining_fut in futures:
                    if not remaining_fut.done():
                        remaining_fut.cancel()
                break

            episode_path, potential, still_filtered, padding_filtered = fut.result()
            total_potential += potential
            total_still_filtered += still_filtered
            total_padding_filtered += padding_filtered

            # Update shard writer's filtering statistics for recovery
            shard_writer.update_filtering_statistics(potential, still_filtered, padding_filtered)

            # Mark episode as completed for resume capability
            shard_writer.mark_episode_completed(episode_path)

    # Stop consumer and wait for processing to drain
    sample_queue.put(None)
    sample_queue.join()
    consumer_thread.join()

    total_samples = counters["total_samples"]

    print(f"✅ Processed {total_samples} samples")

    # Print stats
    if total_potential > 0:
        print("📊 Filtering stats:")
        print(f"  Potential: {total_potential}")
        print(f"  Created: {total_samples}")
        print(f"  Still filtered: {total_still_filtered}")
        print(f"  Padding filtered: {total_padding_filtered}")

    # Finalize
    manifest_data, shard_dir = shard_writer.finalize()

    # Save statistics
    if not cfg.no_statistics:
        stats = global_stats.get_statistics()
        stats_path = os.path.join(shard_dir, "dataset_statistics.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        # Queue upload ASAP
        if cfg.output_dir.startswith("s3://"):
            shard_writer.upload_path(stats_path, os.path.basename(stats_path))

        # Save final statistics state for potential future resume
        if cfg.enable_incremental_updates:
            stats_state_path = os.path.join(shard_dir, "processing_statistics.json")
            global_stats.save_state(stats_state_path)
            if cfg.output_dir.startswith("s3://"):
                shard_writer.upload_path(stats_state_path, os.path.basename(stats_state_path))

    # Update metadata with final processing statistics
    print("📋 Updating processing metadata with final statistics...")
    metadata["processing"]["total_samples_created"] = total_samples
    metadata["processing"]["filtering_statistics"] = {
        "total_potential_samples": total_potential,
        "samples_created": total_samples,
        "still_samples_filtered": total_still_filtered,
        "padding_samples_filtered": total_padding_filtered,
        "episodes_processed": len(episodes),
    }

    # Estimate dataset size
    try:
        dataset_size_bytes = 0
        for root, _dirs, files in os.walk(shard_dir):
            for file in files:
                if file.endswith(".tar"):
                    dataset_size_bytes += os.path.getsize(os.path.join(root, file))
        metadata["processing"]["estimated_dataset_size_gb"] = dataset_size_bytes / (1024**3)
    except Exception:
        metadata["processing"]["estimated_dataset_size_gb"] = "unknown"

    # Add completion timestamp
    metadata["processing"]["timestamp_completed"] = datetime.datetime.now().isoformat()

    # Save metadata
    metadata_path = os.path.join(shard_dir, "processing_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    if cfg.output_dir.startswith("s3://"):
        shard_writer.upload_path(metadata_path, os.path.basename(metadata_path))

    print("✅ Saved processing metadata to processing_metadata.json")

    # Handle shutdown state
    if _shutdown_requested:
        print("⚠️  Processing interrupted by signal - saving current state...")
        print(f"📊 Processed {total_samples} samples in {len(manifest_data)} shards before interruption")

    # Ensure any background uploads complete and cleanup temp space
    if cfg.output_dir.startswith("s3://"):
        print("☁️  Waiting for background uploads to complete...")
        shard_writer.wait_for_uploads()
        shard_writer.cleanup()

    if _shutdown_requested:
        print("✅ Graceful shutdown completed - processing can be resumed later")
        sys.exit(1)  # Exit with error code to indicate interruption
    else:
        print(f"🎉 Complete! {total_samples} samples in {len(manifest_data)} shards")
    print("📋 Metadata files:")
    print("  - manifest.jsonl: Shard manifest")
    if not cfg.no_statistics:
        print("  - dataset_statistics.json: Dataset statistics")
    if cfg.enable_incremental_updates and not cfg.no_statistics:
        print("  - processing_statistics.json: Statistics state for resume capability")
    print("  - processing_metadata.json: Full processing metadata for reproducibility")
    print("📊 Dataset info:")
    print(f"  - Git commit: {metadata['git_info'].get('commit_hash', 'unknown')[:8]}...")
    if metadata["git_info"].get("preprocessing_tag"):
        print(f"  - Git tag: {metadata['git_info']['preprocessing_tag']}")
    print(f"  - Command: {' '.join(sys.argv)}")
    dataset_size = metadata.get("processing", {}).get("estimated_dataset_size_gb", "unknown")
    if isinstance(dataset_size, (int, float)):
        print(f"  - Size: {dataset_size:.2f} GB")
    else:
        print(f"  - Size: {dataset_size}")


if __name__ == "__main__":
    main()
