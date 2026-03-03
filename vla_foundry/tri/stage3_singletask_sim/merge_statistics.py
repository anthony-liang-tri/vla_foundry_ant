#!/usr/bin/env python3
"""
Script to merge statistics from all task-specific stats.json files
in the stage3_singletask_sim dataset and upload the merged stats to S3.

Usage:
    python vla_foundry/tri/stage3_singletask_sim/merge_statistics.py
"""

import json
import os
import subprocess
import tempfile

from vla_foundry.data.robotics.utils import merge_statistics

BASE_S3_PATH = "s3://tri-ml-datasets-uw2/vla_foundry_datasets/v0.3/stage3_singletask_sim"

TASKS = [
    "BimanualHangMugsOnMugHolderFromDryingRack",
    "BimanualHangMugsOnMugHolderFromTable",
    "BimanualLayCerealBoxOnCuttingBoardFromTopShelf",
    "BimanualLayCerealBoxOnCuttingBoardFromUnderShelf",
    "BimanualPlaceAppleFromBowlIntoBin",
    "BimanualPlaceAppleFromBowlOnCuttingBoard",
    "BimanualPlaceAvocadoFromBowlOnCuttingBoard",
    "BimanualPlaceFruitFromBowlIntoBin",
    "BimanualPlaceFruitFromBowlOnCuttingBoard",
    "BimanualPlacePearFromBowlIntoBin",
    "BimanualPlacePearFromBowlOnCuttingBoard",
    "BimanualPutMugsOnPlatesFromDryingRack",
    "BimanualPutMugsOnPlatesFromTable",
    "BimanualPutRedBellPepperInBin",
    "BimanualPutSpatulaOnPlateFromDryingRack",
    "BimanualPutSpatulaOnPlateFromTable",
    "BimanualPutSpatulaOnTableFromDryingRack",
    "BimanualPutSpatulaOnTableFromUtensilCrock",
    "BimanualStackPlatesOnTableFromDryingRack",
    "BimanualStackPlatesOnTableFromTable",
    "BimanualStoreCerealBoxUnderShelf",
    "PickAndPlaceBox",
    "PlaceCupByCoaster",
    "PlaceCupOnCoaster",
    "PushCoasterToCenterOfTable",
    "PushCoasterToMug",
    "PutBananaInCenterOfTable",
    "PutBananaOnSaucer",
    "PutCupInCenterOfTable",
    "PutCupOnSaucer",
    "PutGreenAppleInCenterOfTable",
    "PutGreenAppleOnSaucer",
    "PutKiwiInCenterOfTable",
    "PutKiwiOnSaucer",
    "PutMugOnSaucer",
    "PutOrangeInCenterOfTable",
    "PutOrangeOnSaucer",
    "PutSpatulaInUtensilCrock",
    "PutSpatulaInUtensilCrockFromDryingRack",
    "TurnCupUpsideDown",
    "TurnMugRightsideUp",
]


def main():
    # Download all stats.json files
    all_statistics = []
    failed_tasks = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for task in TASKS:
            stats_path = f"{BASE_S3_PATH}/{task}/shards/stats.json"
            local_path = os.path.join(tmpdir, f"{task}_stats.json")

            result = subprocess.run(
                ["aws", "s3", "cp", stats_path, local_path],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"Failed to download stats for {task}: {result.stderr}")
                failed_tasks.append(task)
                continue

            with open(local_path) as f:
                stats = json.load(f)
                all_statistics.append(stats)
                print(f"Loaded stats for {task}")

    print(f"\nLoaded {len(all_statistics)} stats files")
    if failed_tasks:
        print(f"Failed tasks: {failed_tasks}")

    # Merge all statistics
    print("\nMerging statistics...")
    merged_stats = merge_statistics(all_statistics)

    # Save locally
    local_output = "stage3_singletask_sim_stats.json"
    with open(local_output, "w") as f:
        json.dump(merged_stats, f, indent=2)
    print(f"Saved merged stats to {local_output}")

    # Upload to S3
    s3_output = f"{BASE_S3_PATH}/stats.json"
    result = subprocess.run(
        ["aws", "s3", "cp", local_output, s3_output],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"Uploaded to {s3_output}")
    else:
        print(f"Upload failed: {result.stderr}")

    # Print summary of merged stats
    print("\nMerged stats summary:")
    print(f"  Tensor names: {list(merged_stats.keys())}")
    for tensor_name in list(merged_stats.keys())[:3]:  # Show first 3 tensors
        print(f"  {tensor_name}:")
        for stat_name, value in merged_stats[tensor_name].items():
            if value is not None:
                if isinstance(value, list):
                    if isinstance(value[0], list):
                        print(f"    {stat_name}: shape {len(value)}x{len(value[0]) if value else 0}")
                    else:
                        print(f"    {stat_name}: shape {len(value)}")
                else:
                    print(f"    {stat_name}: {value}")
    print("  ...")


if __name__ == "__main__":
    main()
