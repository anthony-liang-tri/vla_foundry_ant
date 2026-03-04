#!/usr/bin/env python3
"""
Script to merge manifests from all task-specific manifest.jsonl files
in the stage3_singletask_sim dataset and upload the merged manifest to S3.

Usage:
    python vla_foundry/tri/stage3_singletask_sim/merge_manifests.py <base_s3_path>
    python vla_foundry/tri/stage3_singletask_sim/merge_manifests.py <base_s3_path> --ignore-file ignore_tasks.txt
"""

import argparse
import json
import os
import subprocess
import tempfile

from vla_foundry.tri.stage3_singletask_sim.utils import discover_tasks

DEFAULT_BASE_S3_PATH = "s3://tri-ml-datasets-uw2/vla_foundry_datasets_test/v0.4.2.1"


def load_ignore_list(ignore_file: str) -> set:
    """Load a set of task names to ignore from a text file (one task per line)."""
    ignore_tasks = set()
    with open(ignore_file) as f:
        for line in f:
            task = line.strip()
            if task and not task.startswith("#"):
                ignore_tasks.add(task)
    return ignore_tasks


def discover_tasks_with_fallback(base_s3_path):
    normalized = base_s3_path.rstrip("/")
    candidate_paths = [normalized]
    if not normalized.endswith("/real"):
        candidate_paths.append(f"{normalized}/real")

    for candidate_path in candidate_paths:
        print(f"Discovering tasks in {candidate_path}...")
        tasks = discover_tasks(candidate_path, "manifest.jsonl")
        if tasks:
            return candidate_path, tasks

    return candidate_paths[0], []


def main():
    """
    Download and merge manifest.jsonl files from all tasks.
    Adjusts shard paths to be relative from the merged manifest location.

    Each task manifest is at: {BASE_S3_PATH}/{task}/shards/manifest.jsonl
    Merged manifest will be at: {BASE_S3_PATH}/manifest.jsonl
    So shard paths need to be prefixed with: {task}/shards/
    """
    parser = argparse.ArgumentParser(description="Merge task manifests from S3.")
    parser.add_argument(
        "base_s3_path",
        nargs="?",
        default=DEFAULT_BASE_S3_PATH,
        help=(
            "Dataset S3 path. Can be either the task root (for example .../v0.4.3/) "
            "or the real split path (.../v0.4.3/real)."
        ),
    )
    parser.add_argument(
        "--ignore-file",
        type=str,
        default=None,
        help="Path to a text file with task names to ignore (one per line).",
    )
    args = parser.parse_args()

    ignore_tasks = set()
    if args.ignore_file:
        ignore_tasks = load_ignore_list(args.ignore_file)
        print(f"Ignoring {len(ignore_tasks)} tasks from {args.ignore_file}: {ignore_tasks}")

    # Discover tasks from S3
    base_s3_path, tasks = discover_tasks_with_fallback(args.base_s3_path)
    print(f"Found {len(tasks)} tasks: {tasks}")
    if not tasks:
        print("No tasks found. Please verify the S3 path.")
        return

    if ignore_tasks:
        tasks = [t for t in tasks if t not in ignore_tasks]
        print(f"After filtering: {len(tasks)} tasks")

    all_manifest_entries = []
    failed_tasks = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for task in tasks:
            base_path = base_s3_path.rstrip("/")
            local_path = os.path.join(tmpdir, f"{task}_manifest.jsonl")

            # Try manifest.jsonl directly under the task directory first,
            # then fall back to the shards/ subdirectory
            candidates = [
                (f"{base_path}/{task}/manifest.jsonl", f"{task}/"),
                (f"{base_path}/{task}/shards/manifest.jsonl", f"{task}/shards/"),
            ]

            downloaded = False
            shard_prefix = None
            for manifest_path, prefix in candidates:
                result = subprocess.run(
                    ["aws", "s3", "cp", manifest_path, local_path],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    downloaded = True
                    shard_prefix = prefix
                    break

            if not downloaded:
                print(f"Failed to download manifest for {task} (tried {[c[0] for c in candidates]})")
                failed_tasks.append(task)
                continue

            with open(local_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    # Adjust the shard path to be relative from the merged manifest location
                    if "shard" in entry:
                        entry["shard"] = f"{shard_prefix}{entry['shard']}"
                    all_manifest_entries.append(entry)

            print(f"Loaded manifest for {task} (from {shard_prefix}manifest.jsonl)")

        print(f"\nLoaded {len(all_manifest_entries)} manifest entries")
        if failed_tasks:
            print(f"Failed tasks: {failed_tasks}")

        # Save merged manifest
        local_manifest_output = os.path.join(tmpdir, "manifest.jsonl")
        with open(local_manifest_output, "w") as f:
            for entry in all_manifest_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved merged manifest to {local_manifest_output}")

        # Upload to S3
        s3_manifest_output = f"{base_s3_path}/manifest.jsonl"
        result = subprocess.run(
            ["aws", "s3", "cp", local_manifest_output, s3_manifest_output],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"Uploaded manifest to {s3_manifest_output}")
        else:
            print(f"Manifest upload failed: {result.stderr}")

    # Print summary
    print("\nMerged manifest summary:")
    print(f"  Total shards: {len(all_manifest_entries)}")
    total_sequences = sum(entry.get("num_sequences", 0) for entry in all_manifest_entries)
    print(f"  Total sequences: {total_sequences}")
    if all_manifest_entries:
        print("  Sample entries:")
        for entry in all_manifest_entries[:3]:
            print(f"    {entry}")
        if len(all_manifest_entries) > 3:
            print(f"    ... and {len(all_manifest_entries) - 3} more")


if __name__ == "__main__":
    main()
