#!/usr/bin/env python3
"""
Download individual rollout summary.yaml files and compute success rates.

This script:
1. Downloads all summary.yaml files from S3 for each task
2. Parses them to extract success/failure information
3. Computes success rates per task
"""

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def parse_tasks_file(tasks_file: Path) -> List[Tuple[str, str, str]]:
    """Parse the tasks file and extract task information."""
    with open(tasks_file, "r") as f:
        content = f.read()

    # Remove f-string prefix
    content = content.replace('f"', '"')
    wrapped_content = f"[{content}]"

    try:
        tasks = ast.literal_eval(wrapped_content)
    except Exception as e:
        print(f"ERROR: Failed to parse tasks file: {e}", file=sys.stderr)
        sys.exit(1)

    return tasks


def download_summaries(
    checkpoint_s3_path: str,
    task_name: str,
    campaign_name: str,
    output_dir: Path,
) -> List[Path]:
    """Download all summary.yaml files for a task."""
    import shutil

    # Ensure checkpoint path ends with / and strip any trailing slashes first to normalize
    checkpoint_s3_path = checkpoint_s3_path.rstrip("/") + "/"

    # Create output directory (clear it first to avoid mixing data from different sources)
    task_dir = output_dir / task_name
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    # Try multiple paths to find the rollouts - stop after finding data
    # IMPORTANT: Check task-specific path FIRST for multi-task evaluation support
    # This is where run_inference_bundle.sh uploads results when LAUNCH_TASK_NAME is set.
    checkpoint_base = checkpoint_s3_path.rstrip("/")
    paths_to_check = [
        # Task-specific path (for multi-task evaluation on same checkpoint)
        f"{checkpoint_base}/evaluation/{task_name}/rollouts/",
        # Legacy path (for single-task evaluation)
        f"{checkpoint_base}/evaluation/rollouts/",
        f"{checkpoint_base}/evaluation/summary/",
        f"{checkpoint_base}/evaluation/{campaign_name}/artifacts/rollouts/",
    ]

    for s3_base in paths_to_check:
        print(f"  Checking: {s3_base}")

        # Clear task_dir before each attempt to avoid mixing data
        if task_dir.exists():
            shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)

        # Use aws s3 sync to download only summary.yaml files
        cmd = ["aws", "s3", "sync", s3_base, str(task_dir), "--exclude", "*", "--include", "*/summary.yaml"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                continue

            # Check if we got any files
            summary_files = list(task_dir.glob("demonstration_*/summary.yaml"))
            if summary_files:
                print(f"  ✓ Found {len(summary_files)} summaries from {s3_base}")
                return summary_files

        except Exception as e:
            print(f"  ERROR: {e}")

    # No files found from any path
    print("  ✗ No summaries found")
    return []


def parse_summary_file(summary_path: Path) -> Dict[str, Any]:
    """Parse a summary.yaml file and extract success information."""
    try:
        # Add a custom constructor to handle unknown tags
        def ignore_unknown_tags(loader, tag_suffix, node):
            if isinstance(node, yaml.MappingNode):
                return loader.construct_mapping(node)
            elif isinstance(node, yaml.SequenceNode):
                return loader.construct_sequence(node)
            else:
                return loader.construct_scalar(node)

        # Register multi constructor for all unknown tags
        yaml.add_multi_constructor("!", ignore_unknown_tags, Loader=yaml.SafeLoader)

        with open(summary_path, "r") as f:
            data = yaml.safe_load(f)

        # Extract success status
        # The success field should be a boolean
        success = data.get("success", False)
        demo_id = summary_path.parent.name

        return {"demonstration_id": demo_id, "success": success, "data": data}
    except Exception as e:
        print(f"    WARNING: Failed to parse {summary_path}: {e}")
        return None


def compute_success_rate(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute success rate from parsed summaries."""
    if not summaries:
        return {"total": 0, "successes": 0, "success_rate": 0.0}

    total = len(summaries)
    successes = sum(1 for s in summaries if s.get("success", False))
    success_rate = successes / total if total > 0 else 0.0

    return {"total": total, "successes": successes, "failures": total - successes, "success_rate": success_rate}


def main():
    parser = argparse.ArgumentParser(
        description="Download rollout summaries and compute success rates",
    )

    parser.add_argument(
        "tasks_file",
        type=Path,
        help="Path to the tasks configuration file",
    )

    parser.add_argument(
        "--campaign-name",
        type=str,
        required=True,
        help="Campaign name (e.g., stage3_singletask_diffusion_policy_nov2025)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./rollout_summaries"),
        help="Local directory to save results (default: ./rollout_summaries)",
    )

    args = parser.parse_args()

    # Parse tasks
    tasks = parse_tasks_file(args.tasks_file)
    print(f"Found {len(tasks)} tasks\n")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each task
    all_results = {}

    for i, (_scenario_name, task_name, checkpoint_s3_path) in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task_name}")

        # Download summaries
        summary_files = download_summaries(
            checkpoint_s3_path=checkpoint_s3_path,
            task_name=task_name,
            campaign_name=args.campaign_name,
            output_dir=args.output_dir,
        )

        if not summary_files:
            print("  ✗ No summaries found")
            print()
            continue

        # Parse summaries
        parsed_summaries = []
        for summary_file in summary_files:
            parsed = parse_summary_file(summary_file)
            if parsed:
                parsed_summaries.append(parsed)

        # Compute success rate
        stats = compute_success_rate(parsed_summaries)
        all_results[task_name] = stats

        print(f"  Success: {stats['successes']}/{stats['total']} = {stats['success_rate']:.1%}")
        print()

    # Print summary
    print("=" * 80)
    print(f"SUMMARY - {len(all_results)} tasks processed")
    print("=" * 80)

    for task_name in sorted(all_results.keys()):
        stats = all_results[task_name]
        print(f"  {task_name:50s}: {stats['successes']:3d}/{stats['total']:3d} = {stats['success_rate']:6.1%}")

    # Save results
    results_file = args.output_dir / "success_rates.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
