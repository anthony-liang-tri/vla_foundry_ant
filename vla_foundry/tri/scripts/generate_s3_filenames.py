#!/usr/bin/env python3
"""Generate a filenames.txt for use with data_generation_ray.py --tasks-file.

Usage:
    python generate_task_list.py \\
        --tasks_file tasks.txt \\
        --output_txt filenames.txt

The tasks file should contain one task name per line, exactly matching the S3 path
(including casing), e.g.:
    BimanualCleanUpBenchNeedleNosePliers
    BimanualCleanUpBenchPhillipsScrewdriver

For each task the script runs:
    aws s3 ls s3://<source_bucket_base>/<TaskName>/ --recursive
and collects every unique path that contains a leaf folder named "diffusion_spartan",
writing one S3 path per line to the output file.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_BUCKET_BASE = "s3://robotics-manip-lbm/efs/data/tasks"


# ---------------------------------------------------------------------------
# S3 discovery
# ---------------------------------------------------------------------------


def find_diffusion_spartan_paths(source_bucket_base: str, task_name: str) -> list[str]:
    """Return sorted list of all diffusion_spartan S3 prefixes for *task_name*.

    Runs `aws s3 ls --recursive` on
    `{source_bucket_base}/{task_name}/` and collects unique directory prefixes
    of the form `.../diffusion_spartan/`.

    Args:
        source_bucket_base: Base S3 URI up to (but not including) the task
            name, e.g. ``s3://robotics-manip-lbm/efs/data/tasks``.
        task_name: Exact task name matching the S3 key, e.g.
            ``BimanualCleanUpBenchNeedleNosePliers``.

    Returns:
        Sorted list of full S3 URIs ending with ``diffusion_spartan/``.
        Empty list if the listing fails or no matching paths are found.
    """
    s3_prefix = f"{source_bucket_base.rstrip('/')}/{task_name}/"

    # Bucket name is the second component of s3://bucket/...
    bucket = s3_prefix.removeprefix("s3://").split("/")[0]

    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_prefix, "--recursive"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(
            f"WARNING: aws s3 ls failed for task '{task_name}':\n  {e.stderr.strip()}",
            file=sys.stderr,
        )
        return []

    spartan_dirs: set[str] = set()
    for line in result.stdout.splitlines():
        # Output format: "DATE TIME SIZE key/relative/to/bucket/root"
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        key = parts[3]  # Full key from bucket root (no leading slash)
        marker = "/diffusion_spartan/"
        if marker in key:
            idx = key.index(marker) + len(marker)
            spartan_dir = f"s3://{bucket}/{key[:idx]}"
            if "rollout" in spartan_dir.lower():
                continue
            if "teleop" not in spartan_dir.lower():
                continue
            spartan_dirs.add(spartan_dir)

    if not spartan_dirs:
        print(
            f"WARNING: No diffusion_spartan paths found for task '{task_name}' under {s3_prefix}",
            file=sys.stderr,
        )

    return sorted(spartan_dirs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a filenames.txt for use with data_generation_ray.py --tasks-file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tasks_file",
        required=True,
        help="Path to a .txt file with one task name per non-empty line.",
    )
    p.add_argument(
        "--output_txt",
        default="filenames.txt",
        help="Path to write the output filenames file (one S3 episode path per line).",
    )
    p.add_argument(
        "--source_bucket_base",
        default=DEFAULT_SOURCE_BUCKET_BASE,
        help="Base S3 URI containing per-task subdirectories.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tasks_path = Path(args.tasks_file)
    if not tasks_path.exists():
        sys.exit(f"ERROR: tasks file not found: {tasks_path}")

    tasks = [line.strip() for line in tasks_path.read_text().splitlines() if line.strip()]
    if not tasks:
        sys.exit("ERROR: tasks file is empty or contains only blank lines.")

    print(f"Found {len(tasks)} task(s) in {tasks_path}")

    all_episodes: list[str] = []
    for task_name in tasks:
        print(f"  Discovering episodes for: {task_name} ...", end=" ", flush=True)
        episodes = find_diffusion_spartan_paths(args.source_bucket_base, task_name)
        if not episodes:
            print("SKIPPED (no episodes found)")
            continue
        print(f"{len(episodes)} folder(s) found")
        all_episodes.extend(episodes)

    if not all_episodes:
        sys.exit("ERROR: No episodes found for any task. The output file was not written.")

    output_txt = Path(args.output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(all_episodes) + "\n")
    print(f"\nWrote {len(all_episodes)} episode path(s) to: {output_txt}")


if __name__ == "__main__":
    main()
