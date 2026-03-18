#!/usr/bin/env python3
"""
Generate ablation experiment configuration YAML files for VLA Foundry training.

Reads ablation definitions from YAML files and generates resolved configs using main.py.
Supports sweep syntax for parameter sweeps:
  - List: [value1, value2, value3]
  - Linear range: linspace(start, end, n_steps)
  - Log range: logspace(start, end, n_steps)

Usage:
    uv run python vla_foundry/tri/ablations/generate_ablation_configs.py

    # With custom configs:
    uv run python vla_foundry/tri/ablations/generate_ablation_configs.py \
        --nominal-config path/to/nominal.yaml \
        --ablations-config path/to/ablations.yaml
"""

import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import yaml

from vla_foundry.file_utils import copy_to_temp_file
from vla_foundry.tri.ablations.ablation_utils import (
    expand_ablation_names,
    parse_sweep_value,
)

# Constants
MAX_WANDB_TAG_LENGTH = 64


def _is_numeric_or_version(s: str) -> bool:
    """Check if string is numeric, version-like (1p5), or size notation (30m, 10k)."""
    # Handle decimal notation with 'p' (e.g., '1p5' for 1.5)
    s_normalized = s.replace("p", ".")
    # Check pure numeric
    try:
        float(s_normalized)
        return True
    except ValueError:
        pass
    # Check size notation like '30m', '10k', '1b'
    return len(s) >= 2 and s[-1] in "kmbt" and s[:-1].replace("p", ".").replace(".", "").isdigit()


def _is_short_modifier(s: str) -> bool:
    """Check if string is a short modifier/suffix (2-4 chars, lowercase/numeric)."""
    return len(s) <= 4 and s.isalnum() and not s.isdigit()


def _starts_new_concept(parts: list[str], i: int) -> bool:
    """Check if position i starts a new semantic concept."""
    if i >= len(parts):
        return False
    part = parts[i]

    # Keywords that start new concepts
    if part in ("with", "no", "without", "per"):
        return True

    # Check if this part starts a 'X_range_Y' or 'X_<numeric>' pattern
    if i + 1 < len(parts):
        next_part = parts[i + 1]
        if next_part == "range" or _is_numeric_or_version(next_part):
            return True
        # If next part is a keyword, this part starts a new concept
        # (e.g., 'relative' before 'per_timestep')
        if next_part in ("with", "no", "without", "per"):
            return True

    return False


def parse_semantic_units(tag: str) -> list[str]:
    """
    Parse a tag string into semantic units using pattern-based heuristics.

    Recognizes patterns like:
    - 'with_X', 'no_X', 'without_X' → feature flags
    - 'X_range_Y_Z' → parameter with range values
    - 'X_<number>' → parameter with numeric value
    - 'per_X_Y' → compound concepts

    Args:
        tag: The tag string to parse

    Returns:
        List of semantic units
    """
    parts = tag.split("_")
    units = []
    i = 0

    while i < len(parts):
        part = parts[i]

        # Pattern: 'with_X', 'no_X', 'without_X' - feature flags
        if part in ("with", "no", "without") and i + 1 < len(parts):
            # Consume the feature name
            unit_parts = [part, parts[i + 1]]
            i += 2
            # Keep consuming only if next part doesn't start a new concept
            while i < len(parts) and not _starts_new_concept(parts, i):
                # Stop if this looks like a standalone short modifier at the end
                if _is_short_modifier(parts[i]) and (i == len(parts) - 1 or _starts_new_concept(parts, i + 1)):
                    break
                unit_parts.append(parts[i])
                i += 1
            units.append("_".join(unit_parts))
            continue

        # Pattern: 'per_X_Y' - compound concepts like 'per_timestep_norm'
        if part == "per" and i + 1 < len(parts):
            unit_parts = [part, parts[i + 1]]
            i += 2
            # May have additional qualifier
            if i < len(parts) and parts[i] in ("norm", "step", "sample", "batch"):
                unit_parts.append(parts[i])
                i += 1
            units.append("_".join(unit_parts))
            continue

        # Pattern: 'X_range_Y_Z' or 'X_Y_Z' where Y, Z are numeric
        if i + 1 < len(parts):
            next_part = parts[i + 1]

            # Handle 'X_range_Y_Z' pattern
            if next_part == "range" and i + 2 < len(parts):
                unit_parts = [part, "range", parts[i + 2]]
                i += 3
                # Consume additional numeric parts (e.g., range_1_5)
                while i < len(parts) and _is_numeric_or_version(parts[i]):
                    unit_parts.append(parts[i])
                    i += 1
                units.append("_".join(unit_parts))
                continue

            # Handle 'X_<numeric>' pattern (e.g., 'timesteps_1', 'nominal_30k')
            if _is_numeric_or_version(next_part):
                unit_parts = [part, next_part]
                i += 2
                # Consume additional numeric parts
                while i < len(parts) and _is_numeric_or_version(parts[i]):
                    unit_parts.append(parts[i])
                    i += 1
                units.append("_".join(unit_parts))
                continue

        # Default: single part as its own unit
        units.append(part)
        i += 1

    return units


def _join_units(units: list[str]) -> str:
    """Join semantic units with underscores."""
    return "_".join(units)


def _find_balanced_split(units: list[str], max_length: int) -> int:
    """
    Find the best split point that produces balanced parts while respecting max_length.

    Returns the index where the second part should start.
    """
    n = len(units)
    best_idx = 1
    best_score = float("inf")

    for i in range(1, n):
        left = _join_units(units[:i])
        right = _join_units(units[i:])

        # Both parts must fit within max_length
        if len(left) > max_length or len(right) > max_length:
            continue

        # Score: prefer balanced lengths, penalize very short parts
        length_diff = abs(len(left) - len(right))
        min_part_len = min(len(left), len(right))
        # Penalize if either part is very short (< 10 chars)
        short_penalty = max(0, 10 - min_part_len) * 5

        score = length_diff + short_penalty
        if score < best_score:
            best_score = score
            best_idx = i

    return best_idx


def split_long_tag(tag: str, max_length: int = MAX_WANDB_TAG_LENGTH) -> list[str]:
    """
    Split a long tag into multiple shorter tags, preserving semantic meaning.

    Wandb tags have a 64-character limit. This function splits long ablation
    names intelligently by:
    1. Parsing the tag into semantic units (known patterns like 'with_proprioception')
    2. Finding a balanced split point that keeps related concepts together

    Args:
        tag: The tag string to potentially split
        max_length: Maximum allowed length per tag (default 64)

    Returns:
        List of tags, each within the max_length limit

    Example:
        'relative_per_timestep_with_proprioception_clamp_range_1_5_30m_wcd_ema'
        -> ['relative_per_timestep_with_proprioception', 'clamp_range_1_5_30m_wcd_ema']
    """
    if len(tag) <= max_length:
        return [tag]

    # Parse into semantic units
    units = parse_semantic_units(tag)

    # If we only have one unit and it's too long, we can't split meaningfully
    if len(units) == 1:
        raise ValueError(
            f"Tag '{tag}' is {len(tag)} characters and cannot be split into semantic units. "
            f"Consider shorter ablation names."
        )

    # Try to find a balanced two-way split first
    split_idx = _find_balanced_split(units, max_length)
    left = _join_units(units[:split_idx])
    right = _join_units(units[split_idx:])

    # If the right part is still too long, recursively split it
    if len(right) > max_length:
        return [left] + split_long_tag(right, max_length)

    # Validate
    for t in [left, right]:
        if len(t) > max_length:
            raise ValueError(
                f"Tag component '{t}' is {len(t)} characters, exceeds {max_length} char limit. "
                f"Consider shorter ablation names or adding the pattern to SEMANTIC_UNITS."
            )

    return [left, right]


def validate_wandb_tags(tags: list[str]) -> None:
    """
    Validate that all wandb tags are within the 64-character limit.

    Raises ValueError early if any tag exceeds the limit.
    """
    for tag in tags:
        if len(tag) > MAX_WANDB_TAG_LENGTH:
            raise ValueError(
                f"Wandb tag '{tag}' is {len(tag)} characters. "
                f"Tags must be between 1 and {MAX_WANDB_TAG_LENGTH} characters."
            )


def _normalize_wandb_tags_value(value) -> list[str]:
    """
    Normalize a wandb_tags value from YAML/CLI into a list of strings.

    Intended inputs (from YAML) are usually:
      - list[str]
      - a single string tag

    We also support simple stringified lists like:
      - '["a", "b"]' or "['a', 'b']"
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]

    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if inner == "":
                return []
            parts = [p.strip() for p in inner.split(",")]
            # Strip surrounding quotes around each element if present
            return [p.strip().strip('"').strip("'") for p in parts if p.strip() != ""]
        return [s]

    return [str(value)]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate items while preserving first-seen order."""
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _format_value_for_cmdline(value) -> str:
    """
    Format a value for command line argument passing to draccus.

    - Lists: Convert to Python list syntax with trailing comma: ['item1', 'item2',]
    - Bools: Convert to "True" or "False" strings
    - Other types: Convert to string with str()

    Draccus expects list arguments in Python syntax when passed via subprocess.
    The trailing comma ensures single-element lists are parsed correctly.
    """
    if isinstance(value, list):
        # Format as Python list with single quotes and trailing comma
        # This matches the format used in launch_multiple_tasks.sh
        formatted_items = ", ".join(f"'{item}'" if isinstance(item, str) else str(item) for item in value)
        return f"[{formatted_items},]"
    elif isinstance(value, bool):
        # Draccus expects boolean strings
        return str(value)
    else:
        return str(value)


def expand_ablation(name: str, params: dict) -> list[tuple[str, dict]]:
    """
    Expand an ablation definition with potential sweeps into multiple ablations.

    Returns a list of (ablation_name, params_dict) tuples.
    For sweeps:
      - If name ends with '_sweep', use only param_value (e.g., 'scale_1p5')
      - Otherwise, prefix with base name (e.g., 'merged_stats_epsilon_1e-4')
    """
    # Find all sweep parameters
    sweep_params = {}
    fixed_params = {}

    for param, value in params.items():
        parsed = parse_sweep_value(value)
        if len(parsed) > 1:
            sweep_params[param] = parsed
        else:
            fixed_params[param] = parsed[0]

    # If no sweeps, return single ablation
    if not sweep_params:
        return [(name, fixed_params)]

    # Get the names using shared logic
    names = expand_ablation_names(name, params)

    # Generate all combinations with parameters
    ablations = []
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[p] for p in param_names]

    for ablation_name, combo in zip(names, product(*param_values), strict=True):
        combo_params = dict(fixed_params)
        for param, value in zip(param_names, combo, strict=True):
            combo_params[param] = value
        ablations.append((ablation_name, combo_params))

    return ablations


def generate_config(
    task: str | None,
    ablation_name: str,
    base_args: dict,
    ablation_args: dict,
    base_s3_path: str,
    checkpoint_base: str,
    output_base: str,
    runset_hash: str | None = None,
    multitask: bool = False,
) -> tuple[str, bool, str]:
    """
    Generate config for a specific task and ablation.

    When multitask is True, task must be None.

    Returns (config_label, success, message).
    """
    if multitask:
        assert task is None, "task must be None when multitask=True"
        output_dir = Path(output_base) / ablation_name / "mt"
        config_label = ablation_name
    else:
        output_dir = Path(output_base) / ablation_name / task
        config_label = f"{task}/{ablation_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge base_args and ablation_args (ablation overrides base)
    merged_args = dict(base_args)
    merged_args.update(ablation_args)

    # Handle S3 config_path: download locally since draccus doesn't support S3 paths
    if "config_path" in merged_args and merged_args["config_path"].startswith("s3://"):
        s3_path = merged_args["config_path"]
        local_config_path = output_dir / "checkpoint_base_config.yaml"
        with copy_to_temp_file(s3_path) as temp_path:
            import shutil

            shutil.copy(temp_path, local_config_path)
        # Update merged_args to point to local copy
        merged_args["config_path"] = str(local_config_path)

    if multitask:
        # Multitask mode: construct manifest/stats from base_s3_path root
        if "data.dataset_manifest" not in merged_args:
            merged_args["data.dataset_manifest"] = [f"{base_s3_path}/manifest.jsonl"]
        if "data.dataset_statistics" not in merged_args:
            merged_args["data.dataset_statistics"] = [f"{base_s3_path}/stats.json"]        
        merged_args["remote_sync"] = f"{checkpoint_base}/mt/{ablation_name}"
    else:
        # Single-task mode: construct per-task manifest/stats/remote_sync
        manifest_path = f"{base_s3_path}/{task}/shards/manifest.jsonl"
        merged_args["data.dataset_manifest"] = [manifest_path]
        merged_args["remote_sync"] = f"{checkpoint_base}/{task}/{ablation_name}"

        if "data.dataset_statistics" not in merged_args:
            merged_args["data.dataset_statistics"] = [f"{base_s3_path}/{task}/shards/stats.json"]

    # Build wandb tags
    yaml_tags = _normalize_wandb_tags_value(base_args.get("wandb_tags")) + _normalize_wandb_tags_value(
        ablation_args.get("wandb_tags")
    )
    if multitask:
        generated_tags = ["multitask", "ablation"] + split_long_tag(ablation_name)
    else:
        generated_tags = [task, "ablation"] + split_long_tag(ablation_name)
    if runset_hash:
        generated_tags.append(runset_hash)
    merged_args["wandb_tags"] = _dedupe_preserve_order([t for t in (yaml_tags + generated_tags) if str(t) != ""])
    validate_wandb_tags(merged_args["wandb_tags"])

    # Build command from merged args
    cmd = ["uv", "run", "python", "vla_foundry/main.py"]

    for key, value in merged_args.items():
        # Skip SageMaker-specific parameters (not recognized by main.py)
        if key.startswith("sagemaker."):
            continue
        formatted_value = _format_value_for_cmdline(value)
        # For shell execution, quote values that need protection:
        # - list syntax (to protect single quotes inside)
        # - values with spaces (e.g. "include path/to/config.yaml")
        if formatted_value.startswith("[") or " " in formatted_value:
            formatted_value = f'"{formatted_value}"'
        cmd.extend([f"--{key}", formatted_value])

    # Add resolve config args
    cmd.extend(["--resolve_configs", "True"])
    cmd.extend(["--resolve_configs_path", str(output_dir)])

    try:
        # Use shell=True and join command with proper escaping to match launch_multiple_tasks.sh behavior
        # Draccus expects the Python list syntax to be evaluated by the shell
        cmd_str = " ".join(cmd)
        _result = subprocess.run(cmd_str, capture_output=True, text=True, check=True, shell=True)

        # Post-process: fix resolve_configs in output
        config_file = output_dir / "resolved_config.yaml"
        if config_file.exists():
            content = config_file.read_text()
            content = re.sub(r"^resolve_configs: true$", "resolve_configs: false", content, flags=re.MULTILINE)
            content = re.sub(r"^resolve_configs_path:.*\n", "", content, flags=re.MULTILINE)
            config_file.write_text(content)

        return (output_dir,  True, "Generated successfully")

    except subprocess.CalledProcessError as e:
        return (output_dir, False, f"Failed: {e.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Generate ablation experiment configs")
    parser.add_argument(
        "--nominal-config",
        type=str,
        default="vla_foundry/tri/ablations/nominal_config.yaml",
        help="Path to nominal config YAML",
    )
    parser.add_argument(
        "--ablations-config",
        type=str,
        default="vla_foundry/tri/ablations/ablations.yaml",
        help="Path to ablations YAML",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=40,
        help="Maximum parallel config generation jobs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without running",
    )
    parser.add_argument(
        "--runset-hash",
        type=str,
        default=None,
        help="Optional runset hash to add to wandb tags for campaign tracking",
    )
    args = parser.parse_args()

    # Load configs
    with open(args.nominal_config) as f:
        nominal = yaml.safe_load(f)

    with open(args.ablations_config) as f:
        ablations_config = yaml.safe_load(f)

    multitask = nominal.get("multitask", False)
    
    base_args = nominal["base_args"]
    base_s3_path = nominal.get("base_s3_path", "")
    checkpoint_base = nominal["checkpoint_base"]
    output_base = nominal["output_base"]

    if multitask:
        # Multitask mode: no per-task iteration, manifest/stats come from base_args
        tasks = [None]
    else:
        # Tasks can be overridden in the ablations YAML for convenience.
        # If not provided there, fall back to the nominal config tasks.
        tasks = ablations_config.get("tasks", nominal["tasks"])

    # Expand all ablations (handle sweeps)
    expanded_ablations = []
    for name, params in ablations_config["ablations"].items():
        if params is None:
            params = {}
        expanded = expand_ablation(name, params)
        expanded_ablations.extend(expanded)

    # Report what will be generated
    total_configs = len(tasks) * len(expanded_ablations)
    if multitask:
        print("Mode: multitask")
    else:
        print(f"Tasks: {len(tasks)}")
    print(f"Ablations: {len(expanded_ablations)} (from {len(ablations_config['ablations'])} definitions)")
    print(f"Total configs to generate: {total_configs}")

    if args.dry_run:
        print("\n=== Dry Run - Would generate: ===")
        for ablation_name, _ in expanded_ablations:
            for task in tasks:
                if task is not None:
                    print(f"  {output_base}/{ablation_name}/{task}/resolved_config.yaml")
                else:
                    print(f"  {output_base}/{ablation_name}/mt/resolved_config.yaml")
        return

    print("\n=== Generating configs ===")

    # Generate configs in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {}
        for ablation_name, ablation_args in expanded_ablations:
            for task in tasks:
                future = executor.submit(
                    generate_config,
                    task,
                    ablation_name,
                    base_args,
                    ablation_args,
                    base_s3_path,
                    checkpoint_base,
                    output_base,
                    args.runset_hash,
                    multitask,
                )
                futures[future] = (task, ablation_name)

        for future in as_completed(futures):
            name, success, message = future.result()
            status = "✓" if success else "✗"
            print(f"  {status} {name}: {message}")
            results.append((name, success))

    # Summary
    succeeded = sum(1 for _, s in results if s)
    failed = len(results) - succeeded
    print("\n=== Summary ===")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed: {failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
