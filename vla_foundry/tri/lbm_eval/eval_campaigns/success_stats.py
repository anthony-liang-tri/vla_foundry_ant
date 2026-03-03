"""
Success statistics collection and confidence interval calculation.

Extracted from intuitive/visuomotor/collect_episode_success_stats.py
with dependencies on Anzu removed.
"""

import os
from fnmatch import fnmatch
from textwrap import indent

import numpy as np
import yaml
from scipy.stats import binomtest

from .utils import (
    exec_s3_ls,
    exec_s3_sync,
    is_s3_path,
    resolve_glob_to_list,
    wrap_called_process_error,
)

# Supported confidence interval types
INTERVAL_CHOICES = ["two-sided", "lower", "upper"]


def _common_subpath(paths: list[str]) -> str:
    """Find the common prefix subpath among a list of paths."""
    if not paths:
        return ""

    split_paths = [path.split("/") for path in paths]

    split_level = 0
    common_prefix = True
    while common_prefix:
        prefix = split_paths[0][split_level]
        for path in split_paths:
            # Ensure we keep the last 2 parts of the path even if there is only
            # one file. I.e. "demonstration_0/summary.yaml".
            if split_level + 2 >= len(path) or path[split_level] != prefix:
                common_prefix = False
        if common_prefix:
            split_level += 1

    subpath = "/".join(split_paths[0][:split_level]) + "/"

    # Sanity check
    for path in paths:
        assert path[: len(subpath)] == subpath

    return subpath


def _aggregate_success_stats(
    summary_files: list[str],
    verbose: bool = False,
) -> tuple[int, int]:
    """
    Aggregate success statistics from summary YAML files.

    Args:
        summary_files: List of paths to summary.yaml files
        verbose: Whether to print detailed success/failure lists

    Returns:
        Tuple of (successes, total_rollouts)
    """
    if not summary_files:
        if verbose:
            print("No episodes found.")
        return 0, 0

    successes = 0
    success_list = []
    failure_list = []
    common_subpath_len = len(_common_subpath(summary_files))

    # Add handler for unknown YAML tags (e.g., !EnvMetadata)
    def ignore_unknown_tags(loader, tag_suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        elif isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        else:
            return loader.construct_scalar(node)

    yaml.add_multi_constructor("!", ignore_unknown_tags, Loader=yaml.SafeLoader)

    for filename in summary_files:
        with open(filename) as f:
            summary_data = yaml.safe_load(f)

        if summary_data.get("success", False):
            successes += 1
            success_list.append(filename[common_subpath_len:])
        else:
            failure_list.append(filename[common_subpath_len:])

    if verbose:
        print("Successes:")
        print("  " + "\n  ".join(success_list) if success_list else "  (none)")
        print("Failures:")
        print("  " + "\n  ".join(failure_list) if failure_list else "  (none)")

    return successes, len(summary_files)


@wrap_called_process_error
def download_summary_yamls_from_s3(
    s3_path: str,
    local_path: str,
    print_to_console: bool = True,
) -> list[str]:
    """
    Download summary.yaml files from S3 from a single experiment folder.

    Args:
        s3_path: S3 path containing demonstration_*/summary.yaml files
        local_path: Local directory to download files to
        print_to_console: Whether to print progress

    Returns:
        List of paths to downloaded summary.yaml files
    """
    assert is_s3_path(s3_path)

    # Ensure trailing slash
    if not s3_path.endswith("/"):
        s3_path += "/"

    # Check for demonstration folders
    demonstration_proc = exec_s3_ls(path=s3_path, aws_profile=None)
    demonstration_proc.check_returncode()

    expected_demonstration_folder = "* demonstration_*/"
    found_demonstration = False
    for line in demonstration_proc.stdout.splitlines():
        if fnmatch(line, expected_demonstration_folder):
            found_demonstration = True
            break

    if not found_demonstration:
        raise ValueError(
            f"Cannot find `demonstration_*` folder at {s3_path}:\n" + indent(demonstration_proc.stdout, "  ")
        )

    # Check for summary.yaml files
    found_summary = False
    for line in demonstration_proc.stdout.splitlines():
        if "demonstration_" in line:
            subkey = line.split()[-1]
            assert "demonstration_" in subkey
            summary_path = os.path.join(s3_path, subkey)
            assert summary_path.endswith("/")
            summary_proc = exec_s3_ls(path=summary_path, aws_profile=None)
            summary_proc.check_returncode()
            if "summary.yaml" in summary_proc.stdout:
                found_summary = True
                break

    if not found_summary:
        raise ValueError(f"Cannot find `summary.yaml` files at {s3_path}")

    # Download the files
    exec_s3_sync(
        src=s3_path,
        dest=local_path,
        extras=["--exclude", "*", "--include", "*summary.yaml"],
        aws_profile=None,
        print_to_console=print_to_console,
    )

    # Glob for downloaded summary files
    glob_path = os.path.join(local_path, "demonstration_*/summary.yaml")
    return resolve_glob_to_list(glob_path, allow_empty=True)


def collect_multi_rollout_success_stats(
    globs_list: str | list[str],
    verbose: bool = False,
) -> np.ndarray:
    """
    Collect success stats for multiple sets of globs.

    Args:
        globs_list: Glob pattern(s) for summary.yaml files
        verbose: Whether to print detailed information

    Returns:
        2 x N array where first row is successes, second row is total rollouts
        If a single pattern is provided, N=1.

    Example:
        >>> stats = collect_multi_rollout_success_stats([
        ...     "results/exp1/demonstration_*/summary.yaml",
        ...     "results/exp2/demonstration_*/summary.yaml",
        ... ])
        >>> print(stats)  # e.g., [[35, 37], [50, 50]]
    """
    # Handle single pattern
    if isinstance(globs_list, str):
        globs_list = [globs_list]

    success_rates = np.zeros((2, len(globs_list)), dtype=np.int32)

    for i, single_set in enumerate(globs_list):
        summary_files = resolve_glob_to_list(single_set, allow_empty=True)
        successes, rollouts = _aggregate_success_stats(
            summary_files=summary_files,
            verbose=verbose,
        )
        success_rates[0, i] = successes
        success_rates[1, i] = rollouts

    return success_rates


def calculate_confidence_interval(
    successes: int,
    rollouts: int,
    confidence_level: float,
    interval_type: str = "two-sided",
    scipy_interval: bool = True,
) -> tuple[float, float]:
    """
    Compute confidence interval for success rate.

    Args:
        successes: Number of successful rollouts
        rollouts: Total number of rollouts
        confidence_level: Confidence level (0 to 1), e.g., 0.95 for 95%
        interval_type: One of "two-sided", "lower", or "upper"
        scipy_interval: If True, use scipy's Clopper-Pearson method

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval

    Example:
        >>> lower, upper = calculate_confidence_interval(
        ...     successes=35, rollouts=50, confidence_level=0.95
        ... )
        >>> print(f"Success rate: {35/50:.2%}, 95% CI: [{lower:.2%}, {upper:.2%}]")
    """
    if confidence_level < 0 or confidence_level > 1:
        raise ValueError("confidence_level must be in the interval [0, 1]")

    if interval_type not in INTERVAL_CHOICES:
        raise ValueError(f"Unsupported interval type: {interval_type}. Must be one of {INTERVAL_CHOICES}")

    if not scipy_interval and interval_type != "two-sided":
        # binomial_cis supports other interval types but we simplify here
        raise ValueError("Non-scipy intervals only support 'two-sided' in this implementation")

    assert successes >= 0
    assert rollouts >= successes

    if rollouts == 0:
        return 0.0, 1.0

    if interval_type == "two-sided":
        # Use scipy's binomtest for Clopper-Pearson interval
        result = binomtest(successes, rollouts).proportion_ci(confidence_level)
        return result.low, result.high
    elif interval_type == "lower":
        # One-sided lower bound
        result = binomtest(successes, rollouts).proportion_ci(confidence_level, alternative="greater")
        return result.low, 1.0
    else:  # upper
        # One-sided upper bound
        result = binomtest(successes, rollouts).proportion_ci(confidence_level, alternative="less")
        return 0.0, result.high
