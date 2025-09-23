# This file is a dump of some of the functions we use in functions such as preprocess_lbm_to_tar.py
# We keep these one-off functions here to keep the main files clean and digestible.

import datetime
import hashlib
import os
import platform
import sys
from typing import Any, Dict, List

import fsspec
from draccus.parsers import encoding as _draccus_encoding

from lbm2.data.scripts.preprocessing.git_utils import get_git_info
from lbm2.data.scripts.preprocessing.params import PreprocessParams


def make_fs_path(full_path: str, is_s3: bool) -> str:
    return full_path[5:] if is_s3 and full_path.startswith("s3://") else full_path


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
        git_info = get_git_info(auto_tag=args.auto_tag)

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


def discover_episodes_targeted(source_paths: List[str], max_episodes_to_process: int = -1) -> List[str]:
    """Discover episodes efficiently, with different behavior based on whether 'diffusion_spartan' is in the path."""
    if isinstance(source_paths, str):
        source_paths = [source_paths]
    episodes = []

    def check_episode_validity(fs, episode_path: str, is_s3: bool) -> bool:
        """Check if an episode directory has valid processed data."""
        processed_path = os.path.join(episode_path, "processed")
        fs_processed_path = make_fs_path(processed_path, is_s3)

        try:
            if not fs.exists(fs_processed_path):
                return False

            # Check for required files
            required_files = ["metadata.yaml", "observations.npz"]
            for required_file in required_files:
                file_path = os.path.join(processed_path, required_file)
                fs_file_path = make_fs_path(file_path, is_s3)
                if not fs.exists(fs_file_path):
                    return False
            return True
        except Exception:
            return False

    def search_diffusion_spartan_directory(fs, diffusion_spartan_path: str, is_s3: bool) -> None:
        """Search within a diffusion_spartan directory for episode_* folders."""
        fs_path = make_fs_path(diffusion_spartan_path, is_s3)

        try:
            items = fs.listdir(fs_path)
            print(f"Found {len(items)} items in diffusion_spartan directory")
        except Exception as e:
            print(f"Warning: Cannot list directory {diffusion_spartan_path}: {e}")
            return

        episode_dirs = []
        # First pass: identify episode directories only
        for item in items:
            item_name = item["name"] if isinstance(item, dict) else item
            item_basename = os.path.basename(item_name.rstrip("/"))

            # Only process directories that start with "episode_" - skip all files
            if item_basename.startswith("episode_") and not any(
                item_basename.endswith(ext) for ext in [".pkl", ".npz", ".txt", ".json", ".yaml", ".tar", ".gz"]
            ):
                episode_dirs.append(item_basename)

        print(f"Found {len(episode_dirs)} potential episode directories")

        # Second pass: validate episode directories
        for episode_basename in episode_dirs:
            if max_episodes_to_process > 0 and len(episodes) >= max_episodes_to_process:
                break

            # Construct full episode path
            episode_path = os.path.join(diffusion_spartan_path, episode_basename)

            if check_episode_validity(fs, episode_path, is_s3):
                episodes.append(episode_path)
                print(f"Added valid episode: {episode_basename}")
            else:
                print(f"Skipped invalid episode: {episode_basename}")

        print(f"Total valid episodes found: {len(episodes)}")

    def crawl_directory_for_diffusion_spartan(
        fs, current_path: str, is_s3: bool, depth: int = 0, max_depth: int = 5
    ) -> None:
        """Recursively search for diffusion_spartan directories, but don't recurse into files."""
        if depth > max_depth:
            return

        if max_episodes_to_process > 0 and len(episodes) >= max_episodes_to_process:
            return

        fs_current_path = make_fs_path(current_path, is_s3)

        try:
            items = fs.listdir(fs_current_path)
        except Exception as e:
            print(f"Warning: Cannot list directory {current_path}: {e}")
            return

        # Check if current directory is diffusion_spartan
        current_basename = os.path.basename(current_path.rstrip("/"))
        if current_basename == "diffusion_spartan":
            search_diffusion_spartan_directory(fs, current_path, is_s3)
            return

        # Only recurse into directories, skip all files
        for item in items:
            if max_episodes_to_process > 0 and len(episodes) >= max_episodes_to_process:
                break

            item_name = item["name"] if isinstance(item, dict) else item
            item_basename = os.path.basename(item_name.rstrip("/"))

            # Skip all files by extension
            if any(
                item_basename.endswith(ext) for ext in [".pkl", ".npz", ".txt", ".json", ".yaml", ".tar", ".gz", ".log"]
            ):
                continue

            # Skip hidden directories and obvious non-directories
            if item_basename.startswith("."):
                continue

            item_path = os.path.join(current_path, item_basename)

            try:
                # Check if it's actually a directory before recursing
                fs_item_path = make_fs_path(item_path, is_s3)
                if fs.isdir(fs_item_path):
                    crawl_directory_for_diffusion_spartan(fs, item_path, is_s3, depth + 1, max_depth)
            except Exception:
                # If we can't check if it's a directory, skip it
                continue

    for source_path in source_paths:
        print(f"Scanning source path: {source_path}")
        fs, fsspec_path = fsspec.core.url_to_fs(source_path)
        is_s3 = source_path.startswith("s3://")

        # Check if 'diffusion_spartan' is in the source path
        if "diffusion_spartan" in source_path:
            print("Found 'diffusion_spartan' in source path - searching only this directory")
            search_diffusion_spartan_directory(fs, source_path, is_s3)
        else:
            print("No 'diffusion_spartan' in source path - performing recursive search")
            crawl_directory_for_diffusion_spartan(fs, source_path, is_s3)

    print(f"Total episodes discovered: {len(episodes)}")
    return sorted(episodes)
