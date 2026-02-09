"""
Utility functions for the evaluation campaign system.

These replace various Anzu-specific utilities with standalone implementations.
"""

import functools
import glob as glob_module
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Union


def wrap_called_process_error(func):
    """Simple wrapper for subprocess error handling."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Command failed with exit code {e.returncode}: {' '.join(e.cmd)}\n"
                f"stdout: {e.stdout}\nstderr: {e.stderr}"
            ) from e

    return wrapper


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3 path."""
    return path.startswith("s3://")


def resolve_glob_to_list(
    pattern: Union[str, List[str]],
    allow_empty: bool = False,
) -> List[str]:
    """
    Resolve glob pattern(s) to a list of file paths.

    Replaces anzu.intuitive.path_utils.resolve_glob_type_to_list.

    Args:
        pattern: A glob pattern string or list of patterns
        allow_empty: If False, raises ValueError when no matches found

    Returns:
        Sorted list of matching file paths
    """
    if isinstance(pattern, str):
        results = sorted(glob_module.glob(os.path.expanduser(pattern), recursive=True))
    elif isinstance(pattern, list):
        results = []
        for p in pattern:
            results.extend(glob_module.glob(os.path.expanduser(p), recursive=True))
        results = sorted(set(results))
    else:
        results = []

    if not allow_empty and not results:
        raise ValueError(f"No files found matching pattern: {pattern}")

    return results


def exec_s3_ls(
    path: str,
    aws_profile: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """
    List S3 objects at the given path.

    Args:
        path: S3 path to list
        aws_profile: Optional AWS profile to use

    Returns:
        CompletedProcess with stdout containing the listing
    """
    cmd = ["aws", "s3", "ls", path]
    env = os.environ.copy()
    if aws_profile:
        env["AWS_PROFILE"] = aws_profile

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )


def exec_s3_sync(
    src: str,
    dest: str,
    extras: Optional[List[str]] = None,
    aws_profile: Optional[str] = None,
    print_to_console: bool = True,
) -> subprocess.CompletedProcess:
    """
    Sync files between local filesystem and S3.

    Args:
        src: Source path (local or S3)
        dest: Destination path (local or S3)
        extras: Additional arguments to pass to aws s3 sync
        aws_profile: Optional AWS profile to use
        print_to_console: Whether to print output to console

    Returns:
        CompletedProcess result
    """
    cmd = ["aws", "s3", "sync", src, dest]
    if extras:
        cmd.extend(extras)

    env = os.environ.copy()
    if aws_profile:
        env["AWS_PROFILE"] = aws_profile

    result = subprocess.run(
        cmd,
        capture_output=not print_to_console,
        text=True,
        env=env,
    )
    result.check_returncode()
    return result


def exec_s3_cp(
    src: str,
    dest: str,
    aws_profile: Optional[str] = None,
    print_to_console: bool = True,
) -> subprocess.CompletedProcess:
    """
    Copy a file to/from S3.

    Args:
        src: Source path (local or S3)
        dest: Destination path (local or S3)
        aws_profile: Optional AWS profile to use
        print_to_console: Whether to print output to console

    Returns:
        CompletedProcess result
    """
    cmd = ["aws", "s3", "cp", src, dest]

    env = os.environ.copy()
    if aws_profile:
        env["AWS_PROFILE"] = aws_profile

    result = subprocess.run(
        cmd,
        capture_output=not print_to_console,
        text=True,
        env=env,
    )
    result.check_returncode()
    return result


def get_package_root() -> Path:
    """Get the root directory of this package."""
    return Path(__file__).parent.resolve()


def get_repo_root() -> Path:
    """
    Get the repository root directory.

    Falls back to package root if not in a git repository.
    """
    # Check for BUILD_WORKSPACE_DIRECTORY (Bazel environment)
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_dir:
        return Path(workspace_dir).resolve()

    # Try to find git root
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent

    # Fall back to package root
    return get_package_root()
