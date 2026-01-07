#!/usr/bin/env python3
"""
Download model checkpoint from S3 using a W&B run identifier.

This script looks up a W&B run to find the S3 path (from remote_sync config),
then downloads the same files as download_model.sh.

Usage:
    # By run name (searches in W&B project)
    python download_model_from_wandb.py --run-name "2024_01_15-my_experiment"

    # By run ID
    python download_model_from_wandb.py --run-id "abc123def"

    # By W&B run URL
    python download_model_from_wandb.py --run-url "https://wandb.ai/entity/project/runs/abc123def"

    # Specify checkpoint number (default: latest)
    python download_model_from_wandb.py --run-name "my_experiment" --checkpoint 100

    # Specify output directory
    python download_model_from_wandb.py --run-name "my_experiment" --output ./my_model

    # Specify W&B project/entity
    python download_model_from_wandb.py --run-name "my_experiment" --project "vla_foundry" --entity "my-team"

    # List matching runs
    python download_model_from_wandb.py --search "diffusion"

    # AWS profile (default: sagemaker)
    python download_model_from_wandb.py --run-name "my_experiment" --aws-profile default
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import tqdm
import wandb

# Files to download (same as download_model.sh)
FILES_TO_DOWNLOAD = [
    "config.yaml",
    "config_normalizer.yaml",
    "config_processor.yaml",
    "stats.json",
    "preprocessing_configs.yaml",
]

# Legacy file mappings for backward compatibility
LEGACY_FILE_MAPPINGS = [
    ("stats_normalizer.json", "stats.json"),
    ("preprocessing_configs.yaml", "preprocessing_config.yaml"),
]


def get_wandb_api():
    """Get wandb API instance."""

    return wandb.Api(timeout=30)


def parse_wandb_url(url: str) -> tuple[str, str, str]:
    """Parse a W&B URL to extract entity, project, and run ID."""
    pattern = r"https?://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?]+)"
    match = re.match(pattern, url)
    if not match:
        raise ValueError(f"Invalid W&B URL format: {url}")
    return match.group(1), match.group(2), match.group(3)


def find_run_by_name(api, project: str, run_name: str, entity: str | None = None):
    """Find a W&B run by its display name."""
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"display_name": run_name})
    run_list = list(runs)

    if not run_list:
        raise ValueError(f"No run found with name '{run_name}' in project '{path}'")

    if len(run_list) > 1:
        print(f"Warning: Found {len(run_list)} runs with name '{run_name}', using the most recent one")
        run_list.sort(key=lambda r: r.created_at, reverse=True)

    return run_list[0]


def find_run_by_id(api, project: str, run_id: str, entity: str | None = None):
    """Find a W&B run by its ID."""
    path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    return api.run(path)


def search_runs(api, project: str, search_pattern: str, entity: str | None = None, limit: int = 20) -> list:
    """Search for runs matching a pattern."""
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, order="-created_at")
    matches = []

    runs = tqdm.tqdm(runs, desc="Scanning W&B runs", unit="run")

    for run in runs:
        if search_pattern.lower() in run.name.lower():
            matches.append(run)
            if len(matches) >= limit:
                break

    return matches


def extract_s3_path(run) -> str | None:
    """Extract S3 path from run config."""

    config = getattr(run, "config", None)
    if not config:
        return None

    # Handle case where config is a JSON string
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError:
            return None

    # Try remote_sync first
    remote_sync = config.get("remote_sync")
    if remote_sync:
        # W&B often wraps values in {"value": ...} format
        if isinstance(remote_sync, dict) and "value" in remote_sync:
            return remote_sync["value"]
        if isinstance(remote_sync, str):
            return remote_sync

    return None


def construct_full_s3_path(base_s3_path: str, run_name: str) -> str:
    """Construct the full S3 path including the run name."""
    base_path = base_s3_path.rstrip("/")
    # Check if the run name is already in the path
    if run_name in base_path.split("/")[-1]:
        return base_path
    return f"{base_path}/{run_name}"


def run_aws_command(
    cmd: list[str],
    profile: str | None = None,
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run an AWS CLI command with optional profile."""
    if profile:
        cmd = cmd + ["--profile", profile]
    if capture_output:
        return subprocess.run(cmd, capture_output=True, text=True)
    return subprocess.run(cmd, text=True)


def find_highest_checkpoint(s3_path: str, profile: str | None = None) -> int | None:
    """Find the highest checkpoint number in the S3 checkpoints directory."""
    cmd = ["aws", "s3", "ls", f"{s3_path}/checkpoints/"]
    result = run_aws_command(cmd, profile, capture_output=True)

    if result.returncode != 0:
        return None

    # Parse checkpoint numbers from output
    checkpoint_numbers = []
    for line in result.stdout.strip().split("\n"):
        match = re.search(r"checkpoint_(\d+)\.pt", line)
        if match:
            checkpoint_numbers.append(int(match.group(1)))

    return max(checkpoint_numbers) if checkpoint_numbers else None


def download_files(s3_path: str, output_dir: str, profile: str | None = None) -> bool:
    """Download config files from S3."""
    success = True

    for file in tqdm.tqdm(FILES_TO_DOWNLOAD, desc="Downloading config files", unit="file"):
        cmd = ["aws", "s3", "cp", f"{s3_path}/{file}", f"{output_dir}/{file}"]
        result = run_aws_command(cmd, profile, capture_output=False)
        if result.returncode != 0:
            print(f"  Warning: Failed to download {file}")
            success = False
        else:
            print(f"  Downloaded {file}")

    # Try legacy file mappings for backward compatibility
    for src_file, dst_file in tqdm.tqdm(LEGACY_FILE_MAPPINGS, desc="Downloading legacy config files", unit="file"):
        cmd = ["aws", "s3", "cp", f"{s3_path}/{src_file}", f"{output_dir}/{dst_file}"]
        result = run_aws_command(cmd, profile, capture_output=False)
        if result.returncode == 0:
            print(f"  Downloaded {src_file} -> {dst_file} (legacy)")

    return success


def download_checkpoint(s3_path: str, output_dir: str, checkpoint_num: int, profile: str | None = None) -> bool:
    """Download checkpoint files from S3."""
    checkpoints_dir = f"{output_dir}/checkpoints"
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    success = True

    # Download main checkpoint
    src_checkpoint = f"{s3_path}/checkpoints/checkpoint_{checkpoint_num}.pt"
    dst_checkpoint = f"{checkpoints_dir}/checkpoint_{checkpoint_num}.pt"
    print(f"  Downloading: {src_checkpoint} -> {dst_checkpoint}")
    cmd = ["aws", "s3", "cp", src_checkpoint, dst_checkpoint]
    result = run_aws_command(cmd, profile, capture_output=False)
    if result.returncode == 0:
        print(f"  Downloaded checkpoint_{checkpoint_num}.pt")
    else:
        print(f"  Warning: Failed to download checkpoint_{checkpoint_num}.pt")
        success = False

    # Download EMA checkpoint
    src_ema = f"{s3_path}/checkpoints/ema_{checkpoint_num}.pt"
    dst_ema = f"{checkpoints_dir}/ema_{checkpoint_num}.pt"
    print(f"  Downloading: {src_ema} -> {dst_ema}")
    cmd = ["aws", "s3", "cp", src_ema, dst_ema]
    result = run_aws_command(cmd, profile, capture_output=False)
    if result.returncode == 0:
        print(f"  Downloaded ema_{checkpoint_num}.pt")
    else:
        print(f"  Note: ema_{checkpoint_num}.pt not found (may not exist)")

    return success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download model checkpoint from S3 using W&B run identifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Run identification (mutually exclusive)
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--run-name", type=str, help="W&B run display name")
    run_group.add_argument("--run-id", type=str, help="W&B run ID")
    run_group.add_argument("--run-url", type=str, help="Full W&B run URL")
    run_group.add_argument("--search", type=str, help="Search for runs matching this pattern")

    # Project/entity configuration
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "vla_foundry"),
        help="W&B project name (default: $WANDB_PROJECT or 'vla_foundry')",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity/team (default: $WANDB_ENTITY)",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: experiments/<run_name>)",
    )

    # Checkpoint selection
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=int,
        default=None,
        help="Checkpoint number to download (default: highest available)",
    )

    # AWS configuration
    parser.add_argument(
        "--aws-profile",
        type=str,
        default="sagemaker",
        help="AWS profile to use (default: sagemaker)",
    )

    args = parser.parse_args()

    # Validate input
    if not any([args.run_name, args.run_id, args.run_url, args.search]):
        parser.error("One of --run-name, --run-id, --run-url, or --search is required")

    try:
        api = get_wandb_api()
    except Exception as e:
        print(f"Error: Failed to initialize W&B API: {e}", file=sys.stderr)
        print("Make sure you have wandb installed and are logged in (wandb login)", file=sys.stderr)
        sys.exit(1)

    # Search mode
    if args.search:
        print(f"Searching for runs matching '{args.search}'...")
        runs = search_runs(api, args.project, args.search, args.entity)

        if not runs:
            print("No matching runs found.")
            sys.exit(0)

        print(f"\nFound {len(runs)} matching runs:\n")
        print(f"{'Name':<50} {'ID':<12} {'State':<10} {'Created'}")
        print("-" * 90)
        for run in runs:
            created = run.created_at[:19] if run.created_at else "N/A"
            print(f"{run.name:<50} {run.id:<12} {run.state:<10} {created}")
        sys.exit(0)

    # Find the run
    try:
        if args.run_url:
            entity, project, run_id = parse_wandb_url(args.run_url)
            print("Looking up run by URL...")
            run = find_run_by_id(api, project, run_id, entity)
        elif args.run_id:
            print(f"Looking up run by ID: {args.run_id}")
            run = find_run_by_id(api, args.project, args.run_id, args.entity)
        else:
            print(f"Looking up run by name: {args.run_name}")
            run = find_run_by_name(api, args.project, args.run_name, args.entity)

        print(f"Found run: {run.name} (ID: {run.id}, State: {run.state})")

    except Exception as e:
        print(f"Error: Failed to find run: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract S3 path
    s3_path = extract_s3_path(run)
    if not s3_path:
        print("Error: Could not find S3 path (remote_sync) in run config", file=sys.stderr)
        sys.exit(1)

    full_s3_path = construct_full_s3_path(s3_path, run.name)
    print(f"S3 path: {full_s3_path}")

    # Determine output directory
    output_dir = args.output
    if not output_dir:
        safe_name = re.sub(r"[^\w\-_.]", "_", run.name)
        output_dir = f"experiments/{safe_name}"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Find checkpoint number
    checkpoint_num = args.checkpoint
    if checkpoint_num is None:
        print("Finding highest checkpoint number...")
        checkpoint_num = find_highest_checkpoint(full_s3_path, args.aws_profile)
        if checkpoint_num is None:
            print("Error: No checkpoint files found in S3 directory", file=sys.stderr)
            sys.exit(1)
        print(f"Found highest checkpoint: {checkpoint_num}")
    else:
        print(f"Using provided checkpoint number: {checkpoint_num}")

    # Download files
    print("\nDownloading config files...")
    download_files(full_s3_path, output_dir, args.aws_profile)

    print(f"\nDownloading checkpoint {checkpoint_num}...")
    download_checkpoint(full_s3_path, output_dir, checkpoint_num, args.aws_profile)

    print(f"\nDownload complete! Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
