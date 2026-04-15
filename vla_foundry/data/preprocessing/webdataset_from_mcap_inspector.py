#!/usr/bin/env python3
"""
Dataset Inspector for VLA Foundry WebDataset

Inspect and validate preprocessed datasets against MCAP topics config.

Usage:
    uv run python webdataset_from_mcap_inspector.py <dataset_path> [OPTIONS]

Commands:
    # Inspect dataset with config validation
    uv run python webdataset_from_mcap_inspector.py s3://bucket/path/to/dataset/ --config g1_mcap_topics.yaml

    # Quick inspection without config
    uv run python webdataset_from_mcap_inspector.py s3://bucket/path/to/dataset/

    # Inspect single tar file
    uv run python webdataset_from_mcap_inspector.py s3://bucket/path/frames/sample.tar

Examples:
    uv run python webdataset_from_mcap_inspector.py s3://robotics-cam-data/.../tarfile/v3.2/task/real/teleop/ \
        --config vla_foundry/config_presets/data/unitree_g1/g1_mcap_topics.yaml
"""

import argparse
import io
import json
import tarfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from vla_foundry.file_utils import (
    copy_to_temp_file,
    file_exists,
    json_load,
    yaml_load,
)
from vla_foundry.file_utils import (
    list_directory as list_files,
)


def _print_section(title: str, level: int = 1) -> None:
    """Print formatted section header."""
    char = "=" if level == 1 else "-"
    print("\n" + char * 70)
    print(title)
    print(char * 70)


@contextmanager
def open_tar(tar_path: str):
    """
    Context manager to open tar files from local or S3 paths.

    Handles both local files and S3 downloads transparently,
    yielding an open tarfile object.
    """
    is_s3 = tar_path.startswith("s3://")

    if is_s3:
        with copy_to_temp_file(tar_path) as local_path, tarfile.open(local_path, mode="r") as tar:
            yield tar
    else:
        with tarfile.open(tar_path, mode="r") as tar:
            yield tar


def get_expected_keys_from_config(config: dict) -> dict[str, set[str]]:
    """
    Extract expected output keys from mcap topics config.

    Returns dict with 'actions', 'states', 'images' sets.
    """
    expected = {
        "actions": set(),
        "states": set(),
        "images": set(),
    }

    # Action field names
    action_field_names = config.get("action_field_names", {})
    for _topic, name in action_field_names.items():
        expected["actions"].add(name)

    # State field names (base topics without sub-field extraction)
    state_field_names = config.get("state_field_names", {})
    state_field_extraction = config.get("state_field_extraction", {})

    for topic, name in state_field_names.items():
        # Only add if topic doesn't have sub-field extraction
        if topic not in state_field_extraction:
            expected["states"].add(name)

    # State subfield names (from extraction)
    state_subfield_names = config.get("state_subfield_names", {})
    for _subtopic, name in state_subfield_names.items():
        expected["states"].add(name)

    # Camera topics -> image keys
    camera_topics = config.get("camera_topics", {})
    for camera_name in camera_topics:
        # With image_indices like [-3, 0], we get head_t-3, head_t0, etc.
        # We'll just note the base camera names
        expected["images"].add(camera_name)

    return expected


def inspect_tar(tar_path: str, verbose: bool = False) -> dict:
    """Inspect contents of a tar file."""

    with open_tar(tar_path) as tar:
        result = {
            "path": tar_path,
            "files": [],
            "keys": set(),
            "images": {},
            "arrays": {},
            "metadata": None,
            "language": None,
        }

        members = tar.getmembers()

        for m in members:
            name = m.name
            ext = Path(name).suffix
            base = Path(name).stem

            # Extract the key (remove sample_id prefix)
            parts = base.split(".", 1)
            key = parts[1] if len(parts) > 1 else base

            result["files"].append({"name": name, "size": m.size, "ext": ext})
            result["keys"].add(key)

            f = tar.extractfile(m)
            if f is None:
                continue
            data = f.read()

            if ext == ".json" and "metadata" in name:
                result["metadata"] = json.loads(data.decode("utf-8"))

            elif ext == ".txt":
                result["language"] = data.decode("utf-8").strip()

            elif ext == ".npy":
                arr = np.load(io.BytesIO(data))
                result["arrays"][key] = {
                    "shape": arr.shape,
                    "dtype": str(arr.dtype),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "mean": float(arr.mean()),
                }

            elif ext == ".npz":
                npz = np.load(io.BytesIO(data))
                for npz_key in npz.files:
                    arr = npz[npz_key]
                    # For lowdim.npz, use the inner key directly
                    full_key = npz_key if key == "lowdim" else f"{key}.{npz_key}"
                    is_numeric = arr.size > 0 and np.issubdtype(arr.dtype, np.number)
                    result["arrays"][full_key] = {
                        "shape": arr.shape,
                        "dtype": str(arr.dtype),
                        "min": float(arr.min()) if is_numeric else None,
                        "max": float(arr.max()) if is_numeric else None,
                        "mean": float(arr.mean()) if is_numeric else None,
                    }

            elif ext in [".jpg", ".jpeg", ".png"]:
                result["images"][key] = {"size_bytes": len(data)}
                try:
                    import cv2

                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        result["images"][key]["dimensions"] = f"{img.shape[1]}x{img.shape[0]}"
                except Exception:
                    pass

        return result


def print_tar_inspection(result: dict, verbose: bool = False):
    """Print tar inspection results."""
    print(f"\nPath: {result['path']}")
    print(f"Total files: {len(result['files'])}")

    if result["images"]:
        print(f"\nImages ({len(result['images'])}):")
        for key, info in sorted(result["images"].items()):
            dims = info.get("dimensions", "unknown")
            print(f"  {key}: {dims}, {info['size_bytes']:,} bytes")

    if result["arrays"]:
        print(f"\nArrays ({len(result['arrays'])}):")
        for key, info in sorted(result["arrays"].items()):
            print(f"  {key}: shape={info['shape']}, dtype={info['dtype']}")
            # Combined the nested if statements to satisfy SIM102
            if verbose and info["min"] is not None:
                print(f"    min={info['min']:.4f}, max={info['max']:.4f}, mean={info['mean']:.4f}")

    if result["metadata"]:
        print("\nMetadata:")
        for k, v in result["metadata"].items():
            print(f"  {k}: {v}")

    if result["language"]:
        print(f"\nLanguage instruction: {result['language']}")


def validate_against_config(tar_result: dict, stats: dict | None, config: dict, verbose: bool = False):
    """Validate tar contents and stats against mcap topics config."""

    expected = get_expected_keys_from_config(config)

    _print_section("CONFIG VALIDATION")

    # Get actual keys from tar
    actual_arrays = set(tar_result["arrays"].keys())
    actual_images = set(tar_result["images"].keys())

    # Separate actual arrays into actions/states based on prefix
    actual_actions = {k for k in actual_arrays if k.startswith("action_")}
    actual_states = {k for k in actual_arrays if k.startswith("obs_")}
    actual_masks = {k for k in actual_arrays if "mask" in k}
    actual_other = actual_arrays - actual_actions - actual_states - actual_masks

    # Extract base camera names from actual images (strip _t-3, _t0 suffixes)
    actual_cameras = set()
    for img_key in actual_images:
        # head_t-3 -> head, left_wrist_t0 -> left_wrist
        if "_t" in img_key:
            base = img_key.rsplit("_t", 1)[0]
            actual_cameras.add(base)
        else:
            actual_cameras.add(img_key)

    # Actions validation
    print("\nACTIONS:")
    print(f"  Expected from config: {sorted(expected['actions'])}")
    print(f"  Found in tar:         {sorted(actual_actions)}")

    missing_actions = expected["actions"] - actual_actions
    extra_actions = actual_actions - expected["actions"]

    if missing_actions:
        print(f"  [WARN] Missing: {sorted(missing_actions)}")
    if extra_actions:
        print(f"  [INFO] Extra:   {sorted(extra_actions)}")
    if not missing_actions and not extra_actions:
        print("  [OK] All expected actions present")

    # States validation
    print("\nSTATES:")
    print(f"  Expected from config: {sorted(expected['states'])}")
    print(f"  Found in tar:         {sorted(actual_states)}")

    missing_states = expected["states"] - actual_states
    extra_states = actual_states - expected["states"]

    if missing_states:
        print(f"  [WARN] Missing: {sorted(missing_states)}")
    if extra_states:
        print(f"  [INFO] Extra:   {sorted(extra_states)}")
    if not missing_states and not extra_states:
        print("  [OK] All expected states present")

    # Images validation
    print("\nIMAGES:")
    print(f"  Expected cameras: {sorted(expected['images'])}")
    print(f"  Found cameras:    {sorted(actual_cameras)}")
    print(f"  All image keys:   {sorted(actual_images)}")

    missing_cameras = expected["images"] - actual_cameras
    extra_cameras = actual_cameras - expected["images"]

    if missing_cameras:
        print(f"  [WARN] Missing cameras: {sorted(missing_cameras)}")
    if extra_cameras:
        print(f"  [INFO] Extra cameras:   {sorted(extra_cameras)}")
    if not missing_cameras:
        print("  [OK] All expected cameras present")

    # Masks validation
    print("\nMASKS:")
    print(f"  Found: {sorted(actual_masks)}")
    expected_masks = {"past_mask", "future_mask"}
    if expected_masks <= actual_masks:
        print("  [OK] past_mask and future_mask present")
    else:
        print(f"  [WARN] Missing masks: {expected_masks - actual_masks}")

    # Other arrays
    if actual_other:
        print("\nOTHER ARRAYS:")
        print(f"  {sorted(actual_other)}")

    # Stats validation
    if stats and "mean" in stats:
        _print_section("STATS.JSON VALIDATION", level=2)

        stats_keys = set(stats["mean"].keys())

        # Compare stats keys vs tar array keys
        in_tar_not_stats = actual_arrays - stats_keys - {"past_mask", "future_mask"}
        in_stats_not_tar = stats_keys - actual_arrays

        if in_tar_not_stats:
            print("\n  [WARN] In tar but NOT in stats.json:")
            for k in sorted(in_tar_not_stats):
                print(f"    - {k}")

        if in_stats_not_tar:
            print("\n  [WARN] In stats.json but NOT in tar:")
            for k in sorted(in_stats_not_tar):
                print(f"    - {k}")

        if not in_tar_not_stats and not in_stats_not_tar:
            print("\n  [OK] Stats keys match tar array keys")

        # Shape validation
        print("\n  Dimension validation:")
        dim_issues = []
        for key, info in tar_result["arrays"].items():
            if key in stats.get("mean", {}):
                mean_val = stats["mean"][key]
                stats_dim = len(mean_val) if isinstance(mean_val, list) else 1
                tar_dim = info["shape"][-1] if len(info["shape"]) > 0 else 1

                if stats_dim != tar_dim:
                    dim_issues.append(f"{key}: tar={tar_dim}, stats={stats_dim}")

        if dim_issues:
            for issue in dim_issues:
                print(f"    [MISMATCH] {issue}")
        else:
            print("    [OK] All dimensions match")


def inspect_dataset(dataset_path: str, config_path: str | None = None, verbose: bool = False):
    """Inspect entire dataset structure."""

    dataset_path = dataset_path.rstrip("/")

    _print_section("DATASET INSPECTION")

    # Load config if provided
    config = None
    if config_path is not None:
        print(f"Config: {config_path}")
        config = yaml_load(config_path)

    # Define expected files
    stats_path = f"{dataset_path}/shards/stats.json"
    preproc_config_path = f"{dataset_path}/shards/preprocessing_config.yaml"
    metadata_path = f"{dataset_path}/shards/processing_metadata.json"
    manifest_path = f"{dataset_path}/shards/manifest.jsonl"
    frames_path = f"{dataset_path}/frames/"
    episodes_path = f"{dataset_path}/episodes/"

    # Check structure
    print("\nDataset Structure:")
    structure = {
        "shards/stats.json": stats_path,
        "shards/preprocessing_config.yaml": preproc_config_path,
        "shards/processing_metadata.json": metadata_path,
        "shards/manifest.jsonl": manifest_path,
    }

    for name, path in structure.items():
        exists = file_exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name}")

    # Load stats.json
    stats = None
    try:
        stats = json_load(stats_path)
        _print_section("STATS.JSON", level=2)

        if "mean" in stats and "std" in stats:
            print(f"Keys: {len(stats['mean'])}")
            print("\nFields:")
            for key in sorted(stats["mean"].keys()):
                mean_val = stats["mean"][key]
                std_val = stats["std"].get(key, [])
                if isinstance(mean_val, list):
                    dim = len(mean_val)
                    mean_range = f"[{min(mean_val):.3f}, {max(mean_val):.3f}]"
                    std_range = f"[{min(std_val):.3f}, {max(std_val):.3f}]" if std_val else "N/A"
                else:
                    dim = 1
                    mean_range = f"{mean_val:.3f}"
                    std_range = f"{std_val:.3f}" if std_val else "N/A"
                print(f"  {key}: dim={dim}, mean={mean_range}, std={std_range}")

    except Exception as e:
        print(f"\nCould not load stats.json: {e}")

    # Load preprocessing config
    try:
        preproc_config = yaml_load(preproc_config_path)
        _print_section("PREPROCESSING CONFIG", level=2)

        important_keys = [
            "past_lowdim_steps",
            "future_lowdim_steps",
            "pivot_source_field",
            "image_indices",
            "filter_still_samples",
            "still_threshold",
        ]

        for key in important_keys:
            if key in preproc_config:
                print(f"  {key}: {preproc_config[key]}")

    except Exception as e:
        print(f"\nCould not load preprocessing_config.yaml: {e}")

    # Load processing metadata
    try:
        proc_meta = json_load(metadata_path)
        _print_section("PROCESSING METADATA", level=2)

        for key, val in proc_meta.items():
            print(f"  {key}: {val}")

    except Exception as e:
        print(f"\nCould not load processing_metadata.json: {e}")

    # Count frames and episodes
    _print_section("DATA COUNTS", level=2)

    try:
        frames = list_files(frames_path, max_files=1000)
        frame_count = len([f for f in frames if f.endswith(".tar")])
        print(f"  Frames: {frame_count}+ tar files")
    except Exception:
        print("  Frames: could not count")

    try:
        episodes = list_files(episodes_path, max_files=1000)
        ep_count = len([e for e in episodes if e.endswith(".tar")])
        print(f"  Episodes: {ep_count}+ tar files")
    except Exception:
        print("  Episodes: could not count")

    # Get a sample tar file for inspection
    sample_tar = None
    try:
        frames = list_files(frames_path, max_files=10)
        tar_files = [f for f in frames if f.endswith(".tar")]
        if tar_files:
            sample_tar = tar_files[0]
    except Exception:
        pass

    if sample_tar is not None:
        _print_section("SAMPLE TAR INSPECTION")
        tar_result = inspect_tar(sample_tar, verbose=verbose)
        print_tar_inspection(tar_result, verbose=verbose)

        # Validate against config if provided
        if config is not None:
            validate_against_config(tar_result, stats, config, verbose=verbose)
    else:
        print("\nCould not find sample tar file")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and validate VLA Foundry WebDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("path", help="Path to tar file or dataset directory (local or s3://)")
    parser.add_argument("--config", "-c", type=str, help="Path to mcap topics config (g1_mcap_topics.yaml)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    path = args.path.rstrip("/")

    if path.endswith(".tar"):
        result = inspect_tar(path, verbose=args.verbose)
        print("\n" + "=" * 70)
        print("TAR FILE CONTENTS")
        print("=" * 70)
        print_tar_inspection(result, verbose=args.verbose)

        if args.config is not None:
            config = yaml_load(args.config)
            validate_against_config(result, None, config, verbose=args.verbose)
    else:
        inspect_dataset(path, config_path=args.config, verbose=args.verbose)


if __name__ == "__main__":
    main()
