#!/usr/bin/env python3
"""
Script to compare two stats.json files and generate a detailed report.

Differences are reported as absolute values and as a percentage of the
tensor's range (max - min from the reference) so that near-zero values
don't produce misleadingly large relative numbers.

Run with --no-interactive to skip the curses TUI.
"""

import argparse
import contextlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Tensor name parsing
# ---------------------------------------------------------------------------


@dataclass
class TensorInfo:
    """Parsed components of a tensor name."""

    name: str  # Full original name
    group: str  # actual, action, desired, or "" for special
    category: str  # poses, grippers, joint_torque, etc.
    limb: str  # left, right, or ""
    robot: str  # panda, panda_hand, or ""
    representation: str  # xyz, rot_6d, xyz_relative, rot_6d_relative, or ""


def parse_tensor_name(name: str) -> TensorInfo:
    """Parse a tensor name into its component axes.

    Handles:
      robot__<group>__<cat>__<limb>::<robot>__<repr>  (poses)
      robot__<group>__<cat>__<limb>::<robot>           (grippers, torques)
      robot__<group>__<cat>                            (timestamps)
      robot__version, timestamp_packaged               (special)
    """
    info = TensorInfo(name=name, group="", category="", limb="", robot="", representation="")

    if not name.startswith("robot__"):
        info.category = name
        return info

    remainder = name[len("robot__") :]

    if "::" in remainder:
        left_of_colons, right_of_colons = remainder.split("::", 1)
        left_parts = left_of_colons.split("__")
        info.group = left_parts[0] if len(left_parts) > 0 else ""
        info.category = left_parts[1] if len(left_parts) > 1 else ""
        info.limb = left_parts[2] if len(left_parts) > 2 else ""

        right_parts = right_of_colons.split("__", 1)
        info.robot = right_parts[0]
        info.representation = right_parts[1] if len(right_parts) > 1 else ""
    else:
        parts = remainder.split("__")
        info.group = parts[0] if len(parts) > 0 else ""
        info.category = parts[1] if len(parts) > 1 else ""

    return info


def extract_filter_axes(tensor_names: list) -> dict:
    """Extract all unique values for each filter axis from tensor names."""
    axes: dict[str, set] = {
        "group": set(),
        "category": set(),
        "representation": set(),
        "limb": set(),
        "robot": set(),
    }
    for name in tensor_names:
        info = parse_tensor_name(name)
        if info.group:
            axes["group"].add(info.group)
        if info.category:
            axes["category"].add(info.category)
        if info.representation:
            axes["representation"].add(info.representation)
        if info.limb:
            axes["limb"].add(info.limb)
        if info.robot:
            axes["robot"].add(info.robot)
    return axes


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
        with open(tmp_path) as f:
            return json.load(f)
    with open(path) as f:
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
    our_anchor_source: str = "unknown",
    ref_anchor_source: str = "unknown",
) -> dict[str, Any]:
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
        "anchor_info": {
            "our_anchor": our_anchor,
            "ref_anchor": ref_anchor,
            "our_source": our_anchor_source,
            "ref_source": ref_anchor_source,
        },
        "sample_counts": {},
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
                # Record sample count at anchor (max-count timestep)
                with contextlib.suppress(IndexError, TypeError):
                    our_count_at_anchor = int(our_count[tensor_our_anchor])
                    ref_count_at_anchor = int(ref_count[tensor_ref_anchor])
                    comparison["sample_counts"][tensor] = {
                        "our_count": our_count_at_anchor,
                        "ref_count": ref_count_at_anchor,
                        "our_count_array": [int(c) for c in our_count],
                        "ref_count_array": [int(c) for c in ref_count],
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


def _short_tensor(name: str, width: int = 40) -> str:
    """Shorten a tensor name to fit *width*, keeping the most informative suffix."""
    if len(name) <= width:
        return name
    return "..." + name[-(width - 3) :]


def print_report(our_path: str, ref_path: str, our_stats: dict, ref_stats: dict, comparison: dict):
    """Print compact comparison report."""

    W = 100  # total report width

    print(f"\n{'  STATS COMPARISON  ':=^{W}}")
    print(f"Ours: {our_path}")
    print(f"Ref:  {ref_path}")

    # ── Dataset composition (compact) ──
    anchor_info = comparison.get("anchor_info", {})
    our_src = anchor_info.get("our_source", "?")
    ref_src = anchor_info.get("ref_source", "?")

    n_common = len(comparison["common_tensors"])
    n_only_ours = len(comparison["only_in_ours"])
    n_only_ref = len(comparison["only_in_ref"])

    print(f"\n{'  COMPOSITION  ':-^{W}}")
    print(f"{'':25s} {'Ours':>15s}   {'Reference':>15s}")
    print(f"{'Tensors':<25s} {len(our_stats):>15d}   {len(ref_stats):>15d}")
    print(f"{'Common / only here':<25s} {n_common:>8d} / {n_only_ours:<5d}  {n_common:>8d} / {n_only_ref:<5d}")

    ts_info = comparison.get("timestep_info", {})
    if ts_info:
        ex = next(iter(ts_info.values()))
        our_past, our_future = ex["our_anchor"], ex["our_len"] - ex["our_anchor"] - 1
        ref_past, ref_future = ex["ref_anchor"], ex["ref_len"] - ex["ref_anchor"] - 1
        common_past = min(our_past, ref_past)
        common_future = min(our_future, ref_future)
        overlap = common_past + common_future + 1

        our_anc, ref_anc = ex["our_anchor"], ex["ref_anchor"]
        print(f"{'Anchor (source)':<25s} {our_anc:>10d} ({our_src:>8s})   {ref_anc:>10d} ({ref_src:>8s})")
        print(f"{'Window (past/future)':<25s} {our_past:>6d} / {our_future:<7d}   {ref_past:>6d} / {ref_future:<7d}")
        print(f"{'Compared overlap':<25s} {overlap:>15d} timesteps (past={common_past}, future={common_future})")

    sample_counts = comparison.get("sample_counts", {})
    if sample_counts:
        sc = next(iter(sample_counts.values()))
        our_c, ref_c = sc["our_count"], sc["ref_count"]
        diff = our_c - ref_c
        print(f"{'Samples (at anchor)':<25s} {our_c:>15,d}   {ref_c:>15,d}", end="")
        if diff != 0:
            pct = abs(diff) / ref_c * 100 if ref_c > 0 else float("inf")
            sign = "+" if diff > 0 else ""
            print(f"   ({sign}{diff:,d}, {sign}{pct:.1f}%)")
        else:
            print("   (identical)")

        our_cs = {v["our_count"] for v in sample_counts.values()}
        ref_cs = {v["ref_count"] for v in sample_counts.values()}
        if len(our_cs) > 1 or len(ref_cs) > 1:
            print(
                f"  !! Sample counts vary across tensors:"
                f" ours {min(our_cs):,d}-{max(our_cs):,d},"
                f" ref {min(ref_cs):,d}-{max(ref_cs):,d}"
            )

    # Tensor mismatches (compact)
    if n_only_ours or n_only_ref:
        if n_only_ours:
            tensors_str = ", ".join(sorted(comparison["only_in_ours"]))
            print(f"Only in ours ({n_only_ours}): {tensors_str}")
        if n_only_ref:
            tensors_str = ", ".join(sorted(comparison["only_in_ref"]))
            print(f"Only in ref ({n_only_ref}):  {tensors_str}")

    # None values (compact)
    n_none_ours = len(comparison["none_values"]["our"])
    n_none_ref = len(comparison["none_values"]["ref"])
    if n_none_ours or n_none_ref:
        parts = []
        if n_none_ours:
            parts.append(f"ours: {n_none_ours} tensors")
        if n_none_ref:
            parts.append(f"ref: {n_none_ref} tensors")
        print(f"None fields: {', '.join(parts)}")

    # ── Tensor-level overview (main table) ──
    print(f"\n{'  PER-TENSOR OVERVIEW (value fields, sorted by max % of range)  ':-^{W}}")

    tensor_summary = []
    for tensor in sorted(comparison["common_tensors"]):
        if tensor not in comparison["field_differences"]:
            continue
        diffs = [d for d in comparison["field_differences"][tensor] if d["field"] not in COUNT_FIELDS]
        if not diffs:
            continue
        worst = max(diffs, key=lambda d: d["norm_diff"])
        max_norm = worst["norm_diff"]
        avg_norm = float(np.mean([d["norm_diff"] for d in diffs]))
        tensor_summary.append(
            {
                "tensor": tensor,
                "n": len(diffs),
                "avg_pct": avg_norm,
                "max_pct": max_norm,
                "max_abs": worst["max_abs_diff"],
                "worst_field": worst["field"],
            }
        )

    if tensor_summary:
        tensor_summary.sort(key=lambda x: x["max_pct"], reverse=True)
        TW = 40
        print(f"{'Tensor':<{TW}s} {'#':>3s} {'Worst field':<22s} {'MaxAbs':>12s} {'Avg%':>7s} {'Max%':>7s}")
        print(f"{'-' * TW} {'-' * 3} {'-' * 22} {'-' * 12} {'-' * 7} {'-' * 7}")
        for item in tensor_summary:
            t = _short_tensor(item["tensor"], TW)
            f = item["worst_field"][-22:] if len(item["worst_field"]) > 22 else item["worst_field"]
            print(
                f"{t:<{TW}s} {item['n']:>3d} {f:<22s} "
                f"{item['max_abs']:>12.6f} {item['avg_pct']:>6.2f}% {item['max_pct']:>6.2f}%"
            )
    else:
        print("(no value differences)")

    # ── Count overview (compact) ──
    count_diffs = []
    for tensor, field_diffs in comparison["field_differences"].items():
        for d in field_diffs:
            if d["field"] in COUNT_FIELDS:
                count_diffs.append({"tensor": tensor, **d})

    if count_diffs:
        count_diffs.sort(key=lambda x: x["norm_diff"], reverse=True)
        print(f"\n{'  COUNT DIFFERENCES (% of ref count)  ':-^{W}}")
        TW = 40
        print(f"{'Tensor':<{TW}s} {'Field':<20s} {'MaxAbsDiff':>12s} {'%Ref':>7s}")
        print(f"{'-' * TW} {'-' * 20} {'-' * 12} {'-' * 7}")
        for d in count_diffs[:10]:
            t = _short_tensor(d["tensor"], TW)
            print(f"{t:<{TW}s} {d['field']:<20s} {d['max_abs_diff']:>12.0f} {d['norm_diff']:>6.2f}%")

    print(f"\n{'':=^{W}}")


def main():
    parser = argparse.ArgumentParser(description="Compare two stats.json files and generate a detailed report.")
    parser.add_argument("our_stats", help="Path to our stats.json (local or s3://)")
    parser.add_argument("ref_stats", help="Path to reference stats.json (local or s3://)")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip the interactive curses TUI (print report only)",
    )
    args = parser.parse_args()

    our_path = args.our_stats
    ref_path = args.ref_stats

    print(f"Loading {our_path}...", end="", flush=True)
    our_stats = load_json(our_path)
    print(" done")

    print(f"Loading {ref_path}...", end="", flush=True)
    ref_stats = load_json(ref_path)
    print(" done")

    # Load anchor indices from processing_metadata.json
    our_anchor = load_anchor(our_path)
    ref_anchor = load_anchor(ref_path)
    our_anchor_source = "metadata" if our_anchor is not None else "heuristic"
    ref_anchor_source = "metadata" if ref_anchor is not None else "heuristic"
    if our_anchor is not None:
        print(f"Our anchor_relative_idx: {our_anchor} (from metadata)")
    else:
        print("Our anchor: will use count-plateau heuristic (no metadata found)")
    if ref_anchor is not None:
        print(f"Ref anchor_relative_idx: {ref_anchor} (from metadata)")
    else:
        print("Ref anchor: will use count-plateau heuristic (no metadata found)")

    print("Comparing...", end="", flush=True)
    comparison = compare_stats(our_stats, ref_stats, our_anchor, ref_anchor, our_anchor_source, ref_anchor_source)
    print(" done")

    print_report(our_path, ref_path, our_stats, ref_stats, comparison)

    # Launch interactive TUI if conditions are met
    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if not args.no_interactive and is_tty:
        try:
            import curses

            # Try package import first, fall back to relative import for script execution
            try:
                from vla_foundry.tri.utils.compare_stats_tui import StatsComparisonViewer
            except ImportError:
                from compare_stats_tui import StatsComparisonViewer

            viewer = StatsComparisonViewer(
                our_path=our_path,
                ref_path=ref_path,
                our_stats=our_stats,
                ref_stats=ref_stats,
                comparison=comparison,
            )
            curses.wrapper(viewer.run)
        except ImportError as exc:
            print(f"\nCould not launch TUI (missing dependency): {exc}")
        except Exception as exc:
            print(f"\nTUI failed to launch: {exc}")


if __name__ == "__main__":
    main()
