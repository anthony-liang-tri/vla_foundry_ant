#!/usr/bin/env python3
"""
Script to compare two stats.json files and generate a detailed report.

Differences are reported as absolute values and as a percentage of the
tensor's range (max - min from the reference) so that near-zero values
don't produce misleadingly large relative numbers.
"""

import contextlib
import json
import subprocess
import sys
import tempfile
from typing import Any, Dict

import numpy as np


def load_json(path: str) -> dict:
    """Load a JSON file from a local path or S3."""
    if path.startswith("s3://"):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        result = subprocess.run(
            ["aws", "s3", "cp", path, tmp_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise FileNotFoundError(f"Failed to download {path}: {result.stderr}")
        with open(tmp_path, "r") as f:
            return json.load(f)
    with open(path, "r") as f:
        return json.load(f)


def load_anchor(stats_path: str) -> int | None:
    """Load anchor_relative_idx from processing_metadata.json next to a stats file."""
    meta_path = stats_path.rsplit("/", 1)[0] + "/processing_metadata.json"
    try:
        meta = load_json(meta_path)
        if "anchor_relative_idx" in meta:
            return meta["anchor_relative_idx"]
        # Per-task metadata: stored under command_line.arguments
        return meta.get("command_line", {}).get("arguments", {}).get("past_lowdim_steps")
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def find_anchor_index(count_array: list) -> int:
    """Find the anchor (reference) index from a count array.

    The anchor is at `lowdim_past_timesteps`, which is the first index
    where the count reaches its maximum (start of the plateau).
    """
    arr = np.array(count_array, dtype=float)
    max_val = np.max(arr)
    return int(np.where(arr == max_val)[0][0])


def align_per_timestep(our_arr: np.ndarray, ref_arr: np.ndarray, our_anchor: int, ref_anchor: int):
    """Align two per-timestep arrays on their anchor indices.

    Returns the aligned slices (same length) covering only the overlapping window.
    """
    common_past = min(our_anchor, ref_anchor)
    our_future = our_arr.shape[0] - our_anchor - 1
    ref_future = ref_arr.shape[0] - ref_anchor - 1
    common_future = min(our_future, ref_future)

    our_start = our_anchor - common_past
    our_end = our_anchor + common_future + 1
    ref_start = ref_anchor - common_past
    ref_end = ref_anchor + common_future + 1

    return our_arr[our_start:our_end], ref_arr[ref_start:ref_end]


def get_tensor_range(ref_tensor_stats: dict) -> float:
    """Get the range (max - min) of a tensor from its reference stats.

    Used to normalize absolute differences into a scale-aware percentage.
    """
    ref_min = ref_tensor_stats.get("min")
    ref_max = ref_tensor_stats.get("max")
    if ref_min is None or ref_max is None:
        return 0.0
    try:
        min_arr = np.array(ref_min, dtype=float)
        max_arr = np.array(ref_max, dtype=float)
        return float(np.max(max_arr - min_arr))
    except (TypeError, ValueError):
        return 0.0


# Fields to skip entirely (internal state, not meaningful to compare)
SKIP_FIELDS = {
    "tdigest_state",
    "tdigest_state_per_timestep",
    "percentile_sample_count",
    "psquared_state",
    "psquared_state_per_timestep",
}

# Count fields: normalized by reference count value, not by tensor value range
COUNT_FIELDS = {"count", "count_per_timestep"}

# Tensors to skip (not meaningful statistics)
SKIP_TENSORS = {"robot__version"}


def compare_stats(
    our_stats: dict,
    ref_stats: dict,
    our_anchor: int | None = None,
    ref_anchor: int | None = None,
) -> Dict[str, Any]:
    """Compare two stats dictionaries and return detailed comparison."""
    our_tensors = set(our_stats.keys())
    ref_tensors = set(ref_stats.keys())

    common_tensors = our_tensors & ref_tensors
    only_in_ours = our_tensors - ref_tensors
    only_in_ref = ref_tensors - our_tensors

    comparison = {
        "common_tensors": common_tensors,
        "only_in_ours": only_in_ours,
        "only_in_ref": only_in_ref,
        "field_differences": {},
        "none_values": {"our": {}, "ref": {}},
        "timestep_info": {},
    }

    for tensor in common_tensors:
        if tensor in SKIP_TENSORS:
            continue
        our_tensor_stats = our_stats[tensor]
        ref_tensor_stats = ref_stats[tensor]

        our_fields = set(our_tensor_stats.keys())
        ref_fields = set(ref_tensor_stats.keys())
        common_fields = (our_fields & ref_fields) - SKIP_FIELDS

        # Determine anchor indices: use metadata if provided, fall back to count heuristic
        tensor_our_anchor, tensor_ref_anchor = our_anchor, ref_anchor
        our_count = our_tensor_stats.get("count")
        ref_count = ref_tensor_stats.get("count")
        if our_count is not None and ref_count is not None:
            if tensor_our_anchor is None:
                with contextlib.suppress(TypeError, ValueError):
                    tensor_our_anchor = find_anchor_index(our_count)
            if tensor_ref_anchor is None:
                with contextlib.suppress(TypeError, ValueError):
                    tensor_ref_anchor = find_anchor_index(ref_count)
            if tensor_our_anchor is not None and tensor_ref_anchor is not None:
                comparison["timestep_info"][tensor] = {
                    "our_len": len(our_count),
                    "ref_len": len(ref_count),
                    "our_anchor": tensor_our_anchor,
                    "ref_anchor": tensor_ref_anchor,
                }

        tensor_range = get_tensor_range(ref_tensor_stats)

        field_diffs = []

        for field in common_fields:
            our_val = our_tensor_stats[field]
            ref_val = ref_tensor_stats[field]

            # Track None values
            if our_val is None:
                comparison["none_values"]["our"].setdefault(tensor, []).append(field)
            if ref_val is None:
                comparison["none_values"]["ref"].setdefault(tensor, []).append(field)
            if our_val is None or ref_val is None:
                continue

            try:
                our_arr = np.array(our_val, dtype=float)
                ref_arr = np.array(ref_val, dtype=float)

                # If shapes differ, try to align on anchor
                if our_arr.shape != ref_arr.shape:
                    if (
                        tensor_our_anchor is not None
                        and tensor_ref_anchor is not None
                        and our_arr.ndim >= 1
                        and ref_arr.ndim >= 1
                    ):
                        our_arr, ref_arr = align_per_timestep(our_arr, ref_arr, tensor_our_anchor, tensor_ref_anchor)
                    else:
                        continue

                abs_diff = np.abs(our_arr - ref_arr)
                max_abs = float(np.max(abs_diff))
                mean_abs = float(np.mean(abs_diff))

                # Normalized diff: count fields use % of ref count, others use % of tensor range
                if field in COUNT_FIELDS:
                    ref_max_count = float(np.max(np.abs(ref_arr))) if np.any(ref_arr) else 0.0
                    norm_diff = max_abs / ref_max_count * 100 if ref_max_count > 0 else 0.0
                elif tensor_range > 0:
                    norm_diff = max_abs / tensor_range * 100
                else:
                    norm_diff = 0.0

                field_diffs.append(
                    {
                        "field": field,
                        "max_abs_diff": max_abs,
                        "mean_abs_diff": mean_abs,
                        "norm_diff": norm_diff,
                    }
                )
            except (TypeError, ValueError):
                continue

        if field_diffs:
            field_diffs.sort(key=lambda x: x["norm_diff"], reverse=True)
            comparison["field_differences"][tensor] = field_diffs

    return comparison


def print_report(our_path: str, ref_path: str, our_stats: dict, ref_stats: dict, comparison: dict):
    """Print detailed comparison report."""

    print("=" * 110)
    print("STATISTICS COMPARISON REPORT")
    print("=" * 110)

    print(f"\nOur stats file:   {our_path}")
    print(f"Reference file:   {ref_path}")

    print(f"\n{'SUMMARY':-^110}")
    print(f"Our tensors:              {len(our_stats)}")
    print(f"Reference tensors:        {len(ref_stats)}")
    print(f"Common tensors:           {len(comparison['common_tensors'])}")
    print(f"Only in our stats:        {len(comparison['only_in_ours'])}")
    print(f"Only in reference:        {len(comparison['only_in_ref'])}")

    # Timestep alignment info
    ts_info = comparison.get("timestep_info", {})
    mismatched = {t: v for t, v in ts_info.items() if v["our_len"] != v["ref_len"]}
    if mismatched:
        example = next(iter(mismatched.values()))
        common_past = min(example["our_anchor"], example["ref_anchor"])
        our_future = example["our_len"] - example["our_anchor"] - 1
        ref_future = example["ref_len"] - example["ref_anchor"] - 1
        common_future = min(our_future, ref_future)
        print("\nTimestep alignment (per-timestep arrays differ in length):")
        print(f"  Our window:  {example['our_len']} steps (anchor at {example['our_anchor']})")
        print(f"  Ref window:  {example['ref_len']} steps (anchor at {example['ref_anchor']})")
        print(f"  Overlap:     {common_past + common_future + 1} steps (past={common_past}, future={common_future})")

    if comparison["only_in_ours"]:
        print("\nTensors only in our stats:")
        for t in sorted(comparison["only_in_ours"])[:10]:
            print(f"  - {t}")
        if len(comparison["only_in_ours"]) > 10:
            print(f"  ... and {len(comparison['only_in_ours']) - 10} more")

    if comparison["only_in_ref"]:
        print("\nTensors only in reference:")
        for t in sorted(comparison["only_in_ref"])[:10]:
            print(f"  - {t}")
        if len(comparison["only_in_ref"]) > 10:
            print(f"  ... and {len(comparison['only_in_ref']) - 10} more")

    # Report None values
    if comparison["none_values"]["our"] or comparison["none_values"]["ref"]:
        print(f"\n{'NONE VALUES DETECTED':-^110}")
        if comparison["none_values"]["our"]:
            print(f"\nNone values in our stats ({len(comparison['none_values']['our'])} tensors):")
            for tensor, fields in sorted(comparison["none_values"]["our"].items())[:5]:
                print(f"  {tensor}: {len(fields)} fields are None")
                print(f"    Fields: {', '.join(sorted(fields)[:5])}{'...' if len(fields) > 5 else ''}")
            if len(comparison["none_values"]["our"]) > 5:
                print(f"  ... and {len(comparison['none_values']['our']) - 5} more tensors")

        if comparison["none_values"]["ref"]:
            print(f"\nNone values in reference ({len(comparison['none_values']['ref'])} tensors):")
            for tensor, fields in sorted(comparison["none_values"]["ref"].items())[:5]:
                print(f"  {tensor}: {len(fields)} fields are None")

    # Separate value diffs from count diffs
    value_diffs = []
    count_diffs = []
    for tensor, field_diffs in comparison["field_differences"].items():
        for diff_info in field_diffs:
            entry = {"tensor": tensor, **diff_info}
            if diff_info["field"] in COUNT_FIELDS:
                count_diffs.append(entry)
            else:
                value_diffs.append(entry)

    # Top value field differences
    print(f"\n{'TOP VALUE DIFFERENCES (by % of tensor range)':-^110}")

    if value_diffs:
        value_diffs.sort(key=lambda x: x["norm_diff"], reverse=True)

        print(f"\n{'Tensor':<50} {'Field':<25} {'Max Abs Diff':>14} {'% of Range':>12}")
        print(f"{'-' * 50} {'-' * 25} {'-' * 14} {'-' * 12}")

        for diff in value_diffs[:30]:
            tensor_short = diff["tensor"][-47:] if len(diff["tensor"]) > 47 else diff["tensor"]
            field_short = diff["field"][-22:] if len(diff["field"]) > 22 else diff["field"]
            print(f"{tensor_short:<50} {field_short:<25} {diff['max_abs_diff']:>14.6f} {diff['norm_diff']:>11.2f}%")

    # Count differences
    print(f"\n{'COUNT DIFFERENCES (by % of ref count)':-^110}")

    if count_diffs:
        count_diffs.sort(key=lambda x: x["norm_diff"], reverse=True)

        print(f"\n{'Tensor':<50} {'Field':<25} {'Max Abs Diff':>14} {'% of Ref':>12}")
        print(f"{'-' * 50} {'-' * 25} {'-' * 14} {'-' * 12}")

        for diff in count_diffs[:10]:
            tensor_short = diff["tensor"][-47:] if len(diff["tensor"]) > 47 else diff["tensor"]
            field_short = diff["field"][-22:] if len(diff["field"]) > 22 else diff["field"]
            print(f"{tensor_short:<50} {field_short:<25} {diff['max_abs_diff']:>14.0f} {diff['norm_diff']:>11.2f}%")

    # Tensor-wise summary (value fields only)
    print(f"\n{'TENSOR-WISE SUMMARY (value fields only)':-^110}")

    tensor_summary = []
    for tensor in sorted(comparison["common_tensors"]):
        if tensor not in comparison["field_differences"]:
            continue
        diffs = [d for d in comparison["field_differences"][tensor] if d["field"] not in COUNT_FIELDS]
        if not diffs:
            continue
        max_norm = max(d["norm_diff"] for d in diffs)
        avg_norm = np.mean([d["norm_diff"] for d in diffs])
        max_abs = max(d["max_abs_diff"] for d in diffs)
        tensor_summary.append(
            {
                "tensor": tensor,
                "num_fields": len(diffs),
                "avg_norm": avg_norm,
                "max_norm": max_norm,
                "max_abs": max_abs,
            }
        )

    if tensor_summary:
        tensor_summary.sort(key=lambda x: x["max_norm"], reverse=True)

        print(f"\n{'Tensor':<50} {'Fields':>7} {'Max Abs Diff':>14} {'Avg % Range':>13} {'Max % Range':>13}")
        print(f"{'-' * 50} {'-' * 7} {'-' * 14} {'-' * 13} {'-' * 13}")

        for item in tensor_summary[:20]:
            tensor_short = item["tensor"][-47:] if len(item["tensor"]) > 47 else item["tensor"]
            print(
                f"{tensor_short:<50} {item['num_fields']:>7} "
                f"{item['max_abs']:>14.6f} {item['avg_norm']:>12.2f}% {item['max_norm']:>12.2f}%"
            )


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_stats.py <our_stats.json> <reference_stats.json>")
        sys.exit(1)

    our_path = sys.argv[1]
    ref_path = sys.argv[2]

    print(f"Loading {our_path}...", end="", flush=True)
    our_stats = load_json(our_path)
    print(" done")

    print(f"Loading {ref_path}...", end="", flush=True)
    ref_stats = load_json(ref_path)
    print(" done")

    # Load anchor indices from processing_metadata.json
    our_anchor = load_anchor(our_path)
    ref_anchor = load_anchor(ref_path)
    if our_anchor is not None:
        print(f"Our anchor_relative_idx: {our_anchor} (from metadata)")
    if ref_anchor is not None:
        print(f"Ref anchor_relative_idx: {ref_anchor} (from metadata)")

    print("Comparing...", end="", flush=True)
    comparison = compare_stats(our_stats, ref_stats, our_anchor, ref_anchor)
    print(" done")

    print_report(our_path, ref_path, our_stats, ref_stats, comparison)


if __name__ == "__main__":
    main()
