#!/usr/bin/env python3
"""
Script to merge manifests from all task-specific manifest.jsonl files
in the stage3_singletask_sim dataset and upload the merged manifest to S3.

Usage:
    python vla_foundry/tri/stage3_singletask_sim/merge_manifests.py
"""

import json
import os
import subprocess
import tempfile

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
    """
    Download and merge manifest.jsonl files from all tasks.
    Adjusts shard paths to be relative from the merged manifest location.

    Each task manifest is at: {BASE_S3_PATH}/{task}/shards/manifest.jsonl
    Merged manifest will be at: {BASE_S3_PATH}/manifest.jsonl
    So shard paths need to be prefixed with: {task}/shards/
    """
    all_manifest_entries = []
    failed_tasks = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for task in TASKS:
            manifest_path = f"{BASE_S3_PATH}/{task}/shards/manifest.jsonl"
            local_path = os.path.join(tmpdir, f"{task}_manifest.jsonl")

            result = subprocess.run(
                ["aws", "s3", "cp", manifest_path, local_path],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"Failed to download manifest for {task}: {result.stderr}")
                failed_tasks.append(task)
                continue

            with open(local_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    # Adjust the shard path to be relative from the merged manifest location
                    # Original: "shard_000000" -> Adjusted: "{task}/shards/shard_000000"
                    if "shard" in entry:
                        entry["shard"] = f"{task}/shards/{entry['shard']}"
                    all_manifest_entries.append(entry)

            print(f"Loaded manifest for {task}")

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
        s3_manifest_output = f"{BASE_S3_PATH}/manifest.jsonl"
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
