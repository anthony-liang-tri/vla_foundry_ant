#!/usr/bin/env python3
"""Download a VLA Foundry dataset to local disk for offline training.

Supports two modes:
  1. Public HTTPS download from the VLA Foundry dataset registry (no credentials needed)
  2. Private S3 download via boto3 (requires AWS credentials)

The public registry hosts preprocessed datasets at:
  https://vla-foundry.s3.amazonaws.com/datasets/lbm_sim/preprocessed/<version>/<task>/shards/
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve

BASE_URL = "https://vla-foundry.s3.amazonaws.com/datasets/lbm_sim/preprocessed"
RAW_BASE_URL = "https://tri-ml-public.s3.amazonaws.com/datasets/lbm-eval-v1.1-sim-training-data"
DEFAULT_VERSION = "v0.1.0"
METADATA_FILES = ["manifest.jsonl", "stats.json", "preprocessing_config.yaml", "processing_metadata.json"]

# All available tasks in the public registry (v0.1.0).
# fmt: off
AVAILABLE_TASKS = [
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
    "PushBox",
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
# fmt: on


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_size(num_bytes: int | float) -> str:
    """Format a byte count as a human-readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def http_file_size(url: str) -> int | None:
    """Get the Content-Length of a URL via HEAD request, or None if unavailable."""
    from urllib.request import Request

    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=10) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download_http_file(url: str, local_path: str) -> int:
    """Download a single file via HTTP and return its size in bytes."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    urlretrieve(url, local_path)
    return os.path.getsize(local_path)


# ---------------------------------------------------------------------------
# Public HTTPS download
# ---------------------------------------------------------------------------


def download_folder(
    folder_name: str,
    folder_url: str,
    local_dir: Path,
    num_workers: int,
    dry_run: bool,
    download_metadata: bool = True,
) -> None:
    """Download a single folder (shards or episodes) with its manifest and tar files."""
    print(f"\n{'=' * 60}")
    print(f"Downloading {folder_name}/ folder")
    print(f"{'=' * 60}")

    # Step 1: Download metadata files (manifest is always downloaded, even in dry_run)
    manifest_path = None
    if download_metadata:
        print(f"Fetching metadata for {folder_name}/")
        for fname in METADATA_FILES:
            url = f"{folder_url}/{fname}"
            dest = local_dir / fname
            try:
                if dry_run and fname != "manifest.jsonl":
                    size = http_file_size(url)
                    size_str = format_size(size) if size else "unknown size"
                    print(f"  [metadata] {fname} ({size_str})")
                else:
                    os.makedirs(local_dir, exist_ok=True)
                    size = download_http_file(url, str(dest))
                    print(f"  Downloaded {fname} ({format_size(size)})")
                    if fname == "manifest.jsonl":
                        manifest_path = dest
            except Exception as e:
                if fname == "manifest.jsonl":
                    raise RuntimeError(f"Failed to download manifest.jsonl from {folder_name}/: {e}") from e
                print(f"  Skipped {fname} (not found)")
    else:
        # For episodes, just try to download manifest
        url = f"{folder_url}/manifest.jsonl"
        dest = local_dir / "manifest.jsonl"
        try:
            os.makedirs(local_dir, exist_ok=True)
            size = download_http_file(url, str(dest))
            print(f"  Downloaded manifest.jsonl ({format_size(size)})")
            manifest_path = dest
        except Exception as e:
            raise RuntimeError(f"Failed to download manifest.jsonl from {folder_name}/: {e}") from e

    # Step 2: Parse manifest to get tar file list
    if manifest_path is None:
        manifest_path = local_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise RuntimeError(f"manifest.jsonl not found at {manifest_path}")

    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                entries.append(entry)

    # Both shards and episodes manifests use "shard" as the key
    tar_key = "shard"
    item_name = "shard" if folder_name == "shards" else "episode"
    print(f"\nFound {len(entries)} {item_name}s in manifest")
    total_sequences = sum(e.get("num_sequences", 0) for e in entries)
    print(f"Total sequences: {total_sequences}")

    if dry_run:
        print(f"\n--- Dry run: {item_name}s that would be downloaded ---")
        for entry in entries:
            tar_name = entry[tar_key]
            url = f"{folder_url}/{tar_name}.tar"
            size = http_file_size(url)
            size_str = format_size(size) if size else "unknown size"
            print(f"  {tar_name}.tar ({size_str})")
        print(f"\nTotal: {len(entries)} {item_name}s")
        return

    # Step 3: Download tar files in parallel
    to_download = []
    skipped = 0
    for entry in entries:
        tar_name = entry[tar_key]
        tar_url = f"{folder_url}/{tar_name}.tar"
        tar_dest = local_dir / f"{tar_name}.tar"
        remote_size = http_file_size(tar_url)
        if tar_dest.exists() and remote_size and tar_dest.stat().st_size == remote_size:
            skipped += 1
        else:
            to_download.append({"name": tar_name, "url": tar_url, "dest": str(tar_dest), "size": remote_size})

    if skipped:
        print(f"Skipping {skipped} {item_name}s that already exist with correct size")

    if not to_download:
        print(f"All {item_name}s already downloaded.")
    else:
        known_sizes = [d["size"] for d in to_download if d["size"] is not None]
        total_download = sum(known_sizes) if known_sizes else None
        size_info = f" ({format_size(total_download)})" if total_download else ""
        print(f"Downloading {len(to_download)} {item_name}s{size_info} with {num_workers} workers ...")

        lock = threading.Lock()
        completed = [0, 0]  # [count, bytes]

        def download_one(entry: dict) -> str:
            download_http_file(entry["url"], entry["dest"])
            actual_size = os.path.getsize(entry["dest"])
            with lock:
                completed[0] += 1
                completed[1] += actual_size
                print(f"  [{completed[0]}/{len(to_download)}] {format_size(completed[1])} - {entry['name']}.tar")
            return entry["dest"]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(download_one, entry): entry for entry in to_download}
            for future in as_completed(futures):
                future.result()

    # Step 4: Rewrite manifest with just the tar filename (no directory path)
    # The dataloader will prepend the manifest directory automatically
    local_manifest = local_dir / "manifest.jsonl"
    with open(local_manifest, "w") as f:
        for entry in entries:
            # Keep just the filename, no path prefix
            entry[tar_key] = entry[tar_key]
            f.write(json.dumps(entry) + "\n")

    print(f"\n{folder_name}/ download complete!")


def list_available_tasks(version: str = DEFAULT_VERSION) -> None:
    """Print all available tasks in the public registry."""
    print(f"Available tasks in the VLA Foundry registry ({version}):\n")
    for task in AVAILABLE_TASKS:
        print(f"  {task}")
    print(f"\nTotal: {len(AVAILABLE_TASKS)} tasks")
    print("\nTo download:  python scripts/download_dataset.py --task <TaskName> --local_path <dir>")
    print("To download all: python scripts/download_dataset.py --all --local_path <dir>")


def download_public_dataset(task: str, local_path: str, version: str, num_workers: int, dry_run: bool) -> None:
    """Download a dataset from the public HTTPS registry (both shards and episodes folders)."""
    base_url = f"{BASE_URL}/{version}/{task}"
    local_dir = Path(local_path)

    print(f"Downloading task: {task}")
    print(f"Version: {version}")
    print(f"Local path: {local_dir.resolve()}")

    # Download shards/ folder (with metadata files)
    shards_url = f"{base_url}/shards"
    shards_local_dir = local_dir / "shards"
    download_folder(
        folder_name="shards",
        folder_url=shards_url,
        local_dir=shards_local_dir,
        num_workers=num_workers,
        dry_run=dry_run,
        download_metadata=True,
    )

    # Download episodes/ folder (just manifest and tar files)
    episodes_url = f"{base_url}/episodes"
    episodes_local_dir = local_dir / "episodes"
    try:
        download_folder(
            folder_name="episodes",
            folder_url=episodes_url,
            local_dir=episodes_local_dir,
            num_workers=num_workers,
            dry_run=dry_run,
            download_metadata=False,
        )
    except Exception as e:
        print(f"\nWarning: Failed to download episodes/ folder: {e}")
        print("Continuing with shards/ only...")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"Dataset saved to: {local_dir.resolve()}")
    print("\nFolder structure:")
    print(f"  {local_dir.resolve()}/shards/manifest.jsonl")
    print(f"  {local_dir.resolve()}/episodes/manifest.jsonl")
    print("\nTo use this dataset:")
    print(f"  - For training (shuffled): {shards_local_dir.resolve()}/manifest.jsonl")
    print(f"  - For visualization (ordered): {episodes_local_dir.resolve()}/manifest.jsonl")


# ---------------------------------------------------------------------------
# Raw data download
# ---------------------------------------------------------------------------


def download_raw_dataset(task: str, local_path: str, dry_run: bool) -> None:
    """Download raw Spartan-format training data for a task."""
    tar_url = f"{RAW_BASE_URL}/{task}.tar"
    local_dir = Path(local_path)
    tar_dest = local_dir / f"{task}.tar"

    size = http_file_size(tar_url)
    if size is None:
        raise RuntimeError(f"Could not find raw data at {tar_url}. Is the task name correct?")

    print(f"Raw data for: {task}")
    print(f"  URL:  {tar_url}")
    print(f"  Size: {format_size(size)}")

    if dry_run:
        print(f"\n--- Dry run: would download {task}.tar ({format_size(size)}) to {local_dir} ---")
        return

    # Check if already downloaded
    if tar_dest.exists() and tar_dest.stat().st_size == size:
        print(f"  Already downloaded: {tar_dest}")
    else:
        os.makedirs(local_dir, exist_ok=True)
        print(f"  Downloading to {tar_dest} ...")
        download_http_file(tar_url, str(tar_dest))
        print(f"  Downloaded {format_size(os.path.getsize(tar_dest))}")

    # Extract
    import tarfile

    print(f"  Extracting to {local_dir} ...")
    with tarfile.open(tar_dest) as tf:
        tf.extractall(path=str(local_dir))

    print(f"\nDone! Raw data extracted to: {local_dir.resolve()}")
    print("To preprocess, run:")
    print("  python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \\")
    print('    --type "spartan" \\')
    print(f"    --source_episodes \"['{local_dir.resolve()}/tasks/{task}/']\" \\")
    print(f"    --output_dir /data/preprocessed/{task}/ \\")
    print('    --config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml"')


# ---------------------------------------------------------------------------
# Private S3 download (existing functionality)
# ---------------------------------------------------------------------------


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Parse an S3 URI into (bucket, prefix)."""
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected an s3:// URI, got: {s3_uri}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


def download_s3_dataset(s3_path: str, local_path: str, num_workers: int, dry_run: bool) -> None:
    """Download a dataset from a private S3 bucket via boto3."""
    import boto3
    from botocore.config import Config

    bucket, prefix = parse_s3_uri(s3_path)

    s3_config = Config(
        max_pool_connections=max(num_workers + 10, 50),
        retries={"mode": "adaptive", "max_attempts": 10},
    )
    s3_client = boto3.client("s3", config=s3_config)

    if prefix and not prefix.endswith("/"):
        prefix += "/"
    paginator = s3_client.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append({"key": obj["Key"], "size": obj["Size"]})

    if not objects:
        print("No objects found under the given S3 path.")
        return

    prefix_stripped = prefix.rstrip("/") + "/"
    file_entries = []
    for obj in objects:
        rel_key = obj["key"][len(prefix_stripped) :] if obj["key"].startswith(prefix_stripped) else obj["key"]
        local_file = os.path.join(local_path, rel_key)
        file_entries.append({"key": obj["key"], "size": obj["size"], "local_file": local_file, "rel_key": rel_key})

    total_size = sum(e["size"] for e in file_entries)
    print(f"Found {len(file_entries)} files, total size: {format_size(total_size)}")

    if dry_run:
        print("\n--- Dry run: files that would be downloaded ---")
        for entry in file_entries:
            print(f"  {entry['rel_key']}  ({format_size(entry['size'])})")
        print(f"\nTotal: {len(file_entries)} files, {format_size(total_size)}")
        return

    to_download = []
    skipped = 0
    for entry in file_entries:
        if os.path.exists(entry["local_file"]) and os.path.getsize(entry["local_file"]) == entry["size"]:
            skipped += 1
        else:
            to_download.append(entry)

    if skipped:
        print(f"Skipping {skipped} files that already exist with correct size.")

    if not to_download:
        print("All files already downloaded.")
    else:
        download_size = sum(e["size"] for e in to_download)
        print(f"Downloading {len(to_download)} files ({format_size(download_size)}) with {num_workers} workers ...")

        lock = threading.Lock()
        completed_bytes = [0]
        completed_count = [0]

        def download_one(entry: dict) -> str:
            thread_client = boto3.client("s3", config=s3_config)
            os.makedirs(os.path.dirname(entry["local_file"]), exist_ok=True)
            thread_client.download_file(bucket, entry["key"], entry["local_file"])
            with lock:
                completed_bytes[0] += entry["size"]
                completed_count[0] += 1
                pct = completed_bytes[0] / download_size * 100
                print(
                    f"  [{completed_count[0]}/{len(to_download)}] "
                    f"{format_size(completed_bytes[0])}/{format_size(download_size)} "
                    f"({pct:.1f}%) - {entry['rel_key']}"
                )
            return entry["local_file"]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(download_one, entry): entry for entry in to_download}
            for future in as_completed(futures):
                future.result()

    manifest_path = None
    for entry in file_entries:
        if entry["rel_key"] == "manifest.jsonl" or entry["rel_key"].endswith("/manifest.jsonl"):
            manifest_path = entry["local_file"]
            break

    print("\nDownload complete!")
    print(f"Dataset saved to: {os.path.abspath(local_path)}")
    if manifest_path:
        print("\nTo use this dataset in training, set the manifest path to:")
        print(f"  {os.path.abspath(manifest_path)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download a VLA Foundry dataset to local disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # List available tasks
  python scripts/download_dataset.py --list

  # Download pre-processed data (no credentials needed)
  python scripts/download_dataset.py \\
      --task BimanualPutRedBellPepperInBin \\
      --local_path /data/datasets/BimanualPutRedBellPepperInBin

  # Download ALL pre-processed tasks
  python scripts/download_dataset.py \\
      --all --local_path /data/datasets

  # Download raw Spartan data (for custom preprocessing)
  python scripts/download_dataset.py \\
      --task BimanualPutRedBellPepperInBin \\
      --local_path /data/raw \\
      --raw

  # Preview what would be downloaded
  python scripts/download_dataset.py \\
      --task BimanualPutRedBellPepperInBin \\
      --local_path /data/datasets/BimanualPutRedBellPepperInBin \\
      --dry_run

  # Download from a private S3 bucket
  python scripts/download_dataset.py \\
      --s3_path s3://my-bucket/datasets/my_task/shards \\
      --local_path /data/my_task
""",
    )
    parser.add_argument("--task", help="Task name to download from the public registry")
    parser.add_argument("--all", action="store_true", help="Download all available tasks from the public registry")
    parser.add_argument("--raw", action="store_true", help="Download raw Spartan data instead of pre-processed shards")
    parser.add_argument("--list", action="store_true", help="List all available tasks and exit")
    parser.add_argument("--version", default=DEFAULT_VERSION, help=f"Dataset version (default: {DEFAULT_VERSION})")
    parser.add_argument("--s3_path", help="S3 URI for private bucket download (alternative to --task)")
    parser.add_argument("--local_path", help="Local directory to download files into")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel download threads (default: 8)")
    parser.add_argument("--dry_run", action="store_true", help="List files without downloading")
    args = parser.parse_args()

    if args.list:
        list_available_tasks(args.version)
        return

    if not args.local_path:
        parser.error("--local_path is required for downloading")

    if args.raw:
        # Raw data download mode
        if args.all:
            for i, task in enumerate(AVAILABLE_TASKS):
                print(f"\n{'=' * 60}")
                print(f"[{i + 1}/{len(AVAILABLE_TASKS)}] {task} (raw)")
                print(f"{'=' * 60}")
                download_raw_dataset(task=task, local_path=args.local_path, dry_run=args.dry_run)
            print(f"\n{'=' * 60}")
            print(f"All {len(AVAILABLE_TASKS)} raw datasets downloaded to: {os.path.abspath(args.local_path)}")
        elif args.task:
            download_raw_dataset(task=args.task, local_path=args.local_path, dry_run=args.dry_run)
        else:
            parser.error("--raw requires --task or --all")
    elif args.all:
        # Download all pre-processed tasks, each into its own subdirectory
        for i, task in enumerate(AVAILABLE_TASKS):
            print(f"\n{'=' * 60}")
            print(f"[{i + 1}/{len(AVAILABLE_TASKS)}] {task}")
            print(f"{'=' * 60}")
            task_path = os.path.join(args.local_path, task)
            download_public_dataset(
                task=task,
                local_path=task_path,
                version=args.version,
                num_workers=args.num_workers,
                dry_run=args.dry_run,
            )
        print(f"\n{'=' * 60}")
        print(f"All {len(AVAILABLE_TASKS)} tasks downloaded to: {os.path.abspath(args.local_path)}")
    elif args.task:
        download_public_dataset(
            task=args.task,
            local_path=args.local_path,
            version=args.version,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
        )
    elif args.s3_path:
        download_s3_dataset(
            s3_path=args.s3_path,
            local_path=args.local_path,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
        )
    else:
        parser.error("Provide --task, --all, or --s3_path")


if __name__ == "__main__":
    main()
