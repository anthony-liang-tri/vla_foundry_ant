#!/usr/bin/env python3
"""
Extract all S3 paths matching the pattern:
s3://robotics-manip-lbm/efs/data/tasks/{TASK_NAME}/wollaston/real/bc/teleop/{TIMESTAMP}/diffusion_spartan/
"""

import os

import boto3

# Task names to process
TASKS = [
    "BimanualPlaceMalletOnPegboard",
    "BimanualPlaceGluegunOnPegboard",
    "BimanualPlaceTapeMeasureOnPegboard",
    "BimanualPlaceDustpanOnPegboard",
    "BimanualPlaceTtoolOnPegboard",
    "BimanualPutScrewdriverOnPegboard",
]

OUTPUT_FILE = "wollaston_real_filenames.txt"


def list_directories(s3_client, bucket, prefix):
    """List all directory prefixes under the given prefix."""
    directories = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        if "CommonPrefixes" in page:
            for prefix_info in page["CommonPrefixes"]:
                directories.append(prefix_info["Prefix"])
    return directories


def find_diffusion_spartan_paths(task_name):
    """Find all diffusion_spartan paths for a given task."""
    teleop_prefix = f"efs/data/tasks/{task_name}/wollaston/real/bc/teleop/"
    bucket = "robotics-manip-lbm"

    s3_client = boto3.client("s3")

    # List all timestamp directories under teleop/
    timestamp_dirs = list_directories(s3_client, bucket, teleop_prefix)

    diffusion_spartan_paths = []
    for timestamp_dir in timestamp_dirs:
        # List all subdirectories in the timestamp directory
        subdirs = list_directories(s3_client, bucket, timestamp_dir)

        # Check if diffusion_spartan/ is one of them
        diffusion_spartan_prefix = timestamp_dir + "diffusion_spartan/"
        # Check if any subdirectory matches the diffusion_spartan prefix
        if diffusion_spartan_prefix in subdirs:
            # Path exists, add it
            full_path = f"s3://{bucket}/{diffusion_spartan_prefix}"
            diffusion_spartan_paths.append(full_path)

    return diffusion_spartan_paths


def main():
    """Main function to extract all filenames."""
    all_paths = []

    for task_name in TASKS:
        print(f"Processing task: {task_name}")
        paths = find_diffusion_spartan_paths(task_name)
        print(f"  Found {len(paths)} paths")
        all_paths.extend(paths)

    # Write to output file
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)
    with open(output_path, "w") as f:
        for path in sorted(all_paths):
            f.write(path + "\n")

    print(f"\nTotal paths found: {len(all_paths)}")
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
