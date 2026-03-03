#!/usr/bin/env python3
"""
Verify campaign completion by checking which episode summaries exist on S3.

This script:
1. Reads the campaign configuration to understand expected episodes
2. Checks S3 for existing summary.yaml files
3. Reports missing episodes
4. Outputs a tasks file for re-running failed/missing jobs
"""

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskSpec:
    """Specification for a single task evaluation."""

    scenario: str
    task_name: str
    checkpoint: str
    branch: str = None

    @property
    def name(self) -> str:
        """Generate a unique name for this task."""
        # Extract meaningful checkpoint identifier
        ckpt_id = self.checkpoint.rstrip("/").split("/")[-1]
        return f"{self.task_name}-{ckpt_id}"


def parse_tasks_file(tasks_file: Path) -> list[TaskSpec]:
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

    print(tasks)
    return [
        TaskSpec(scenario=t[0], task_name=t[1], checkpoint=t[2], branch=t[3] if len(t) > 3 else None) for t in tasks
    ]


def list_s3_summaries(s3_path: str, aws_profile: str | None = None) -> set[int]:
    """List all demonstration indices that have summary.yaml on S3."""
    cmd = ["aws", "s3", "ls", s3_path, "--recursive"]
    if aws_profile:
        cmd.extend(["--profile", aws_profile])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return set()

        # Parse output to find demonstration_*/summary.yaml
        # Pattern: demonstration_123/summary.yaml
        pattern = re.compile(r"demonstration_(\d+)/summary\.yaml")
        indices = set()

        for line in result.stdout.splitlines():
            match = pattern.search(line)
            if match:
                indices.add(int(match.group(1)))

        return indices

    except subprocess.TimeoutExpired:
        print(f"  WARNING: Timeout listing {s3_path}")
        return set()
    except Exception as e:
        print(f"  WARNING: Error listing {s3_path}: {e}")
        return set()


def resolve_task_name_on_s3(checkpoint_base: str, task_name: str, aws_profile: str | None = None) -> str:
    """
    Resolve the task_name used for evaluation output paths on S3.

    Some launch flows accept a truncated/abbreviated task_name but write results under the
    full task name in S3 (e.g., "BimanualStoreCerealBoxUnderShel" -> "BimanualStoreCerealBoxUnderShelf").
    This function looks at the immediate prefixes under:
      {checkpoint_base}/evaluation/
    and returns the best match.
    """
    if not task_name:
        return task_name

    cmd = ["aws", "s3", "ls", f"{checkpoint_base}/evaluation/"]
    if aws_profile:
        cmd.extend(["--profile", aws_profile])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return task_name

    prefixes = []
    for line in result.stdout.splitlines():
        # Typical format: "                           PRE SomePrefix/"
        if " PRE " in line:
            token = line.split(" PRE ", 1)[1].strip()
            if token.endswith("/"):
                token = token[:-1]
            prefixes.append(token)

    if task_name in prefixes:
        return task_name

    # Common case: task_name is a truncated prefix of the true task name.
    startswith_matches = [p for p in prefixes if p.startswith(task_name)]
    if len(startswith_matches) == 1:
        return startswith_matches[0]

    return task_name


def find_existing_summaries(
    checkpoint: str,
    aws_profile: str | None = None,
    task_name: str | None = None,
) -> set[int]:
    """Find all existing summary indices for a checkpoint.

    Args:
        checkpoint: S3 path to the checkpoint
        aws_profile: AWS profile to use
        task_name: Task name for multi-task evaluation (results stored under task-specific path)
    """
    checkpoint_base = checkpoint.rstrip("/")

    # Check multiple possible paths - task-specific path first for multi-task support
    paths_to_check = []
    if task_name:
        # New task-specific path (for multi-task evaluation on same checkpoint)
        paths_to_check.append(f"{checkpoint_base}/evaluation/{task_name}/rollouts/")
        # Some launch scripts upload the per-episode summaries separately under a task-specific
        # summary prefix (mirrors the legacy evaluation/summary/ layout).
        paths_to_check.append(f"{checkpoint_base}/evaluation/{task_name}/summary/")
    # Legacy paths
    paths_to_check.extend(
        [
            f"{checkpoint_base}/evaluation/rollouts/",
            f"{checkpoint_base}/evaluation/summary/",
        ]
    )

    all_indices = set()
    for s3_path in paths_to_check:
        indices = list_s3_summaries(s3_path, aws_profile)
        if indices:
            all_indices.update(indices)

    return all_indices


def verify_task(
    task: TaskSpec,
    start_index: int,
    num_samples: int,
    aws_profile: str | None = None,
) -> tuple[set[int], set[int]]:
    """
    Verify a single task's completion.

    Returns:
        (existing_indices, missing_indices)
    """
    expected_indices = set(range(start_index, start_index + num_samples))
    existing_indices = find_existing_summaries(task.checkpoint, aws_profile, task.task_name)

    # Only consider indices in our expected range
    existing_in_range = existing_indices.intersection(expected_indices)
    missing_indices = expected_indices - existing_indices

    return existing_in_range, missing_indices


def format_index_ranges(indices: set[int]) -> str:
    """Format a set of indices as compact ranges (e.g., '0-5,10,15-20')."""
    if not indices:
        return ""

    sorted_indices = sorted(indices)
    ranges = []
    start = sorted_indices[0]
    end = start

    for idx in sorted_indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = idx

    # Don't forget the last range
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ",".join(ranges)


def indices_to_ranges(indices: set[int]) -> list[tuple[int, int]]:
    """Convert a set of indices to a list of (start, end) ranges."""
    if not indices:
        return []

    sorted_indices = sorted(indices)
    ranges = []
    start = sorted_indices[0]
    end = start

    for idx in sorted_indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append((start, end + 1))  # end is exclusive
            start = end = idx

    ranges.append((start, end + 1))
    return ranges


def generate_rerun_tasks_file(
    tasks: list[TaskSpec],
    missing_by_task: dict[str, set[int]],
    output_path: Path,
):
    """Generate a tasks file for re-running missing episodes."""
    with open(output_path, "w") as f:
        f.write("# Tasks file for re-running missing episodes\n")
        f.write("# Generated by verify_campaign_completion.py\n")
        f.write("#\n")
        f.write("# Each line: (scenario, task_name, checkpoint)\n")
        f.write("# Missing indices are listed in comments\n\n")

        for task in tasks:
            missing = missing_by_task.get(task.name, set())
            if missing:
                ranges = format_index_ranges(missing)
                f.write(f"# Missing indices: {ranges}\n")
                f.write(f'("{task.scenario}", "{task.task_name}", "{task.checkpoint}"),\n\n')


def generate_rerun_commands(
    tasks: list[TaskSpec],
    missing_by_task: dict[str, set[int]],
    campaign_config: Path | None = None,
) -> list[str]:
    """Generate shell commands to re-run missing episodes."""
    commands = []

    for task in tasks:
        missing = missing_by_task.get(task.name, set())
        if not missing:
            continue

        # Convert to ranges for efficient re-running
        ranges = indices_to_ranges(missing)

        for start, end in ranges:
            num_samples = end - start
            # Generate a command for each range
            cmd = (
                f"# Re-run {task.task_name}: indices {start}-{end - 1} ({num_samples} episodes)\n"
                f"python3 eval_campaigns/ray_policy_runner.py/ray_policy_runner.py \\\n"
                f"  --cluster-url $CLUSTER_URL \\\n"
                f"  --image $DOCKER_IMAGE \\\n"
                f"  --task {task.task_name} \\\n"
                f"  --checkpoints '{task.checkpoint}' \\\n"
                f"  --start-index {start} \\\n"
                f"  --num-samples {num_samples} \\\n"
                f"  --launch-scenario {task.scenario} \\\n"
                f"  --distribute"
            )
            commands.append(cmd)

    return commands


def main():
    parser = argparse.ArgumentParser(
        description="Verify campaign completion and identify missing episodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check completion for a campaign
  %(prog)s eval_campaigns/16_tasks_30ksteps_lr5e5.txt --start-index 100 --num-samples 100

  # Generate re-run file
  %(prog)s eval_campaigns/tasks.txt --start-index 100 --num-samples 100 --output-rerun rerun_tasks.txt
        """,
    )

    parser.add_argument(
        "tasks_file",
        type=Path,
        help="Path to the tasks configuration file",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting demonstration index (default: 0)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of samples per task (default: 200)",
    )

    parser.add_argument(
        "--aws-profile",
        type=str,
        default="sagemaker",
        help="AWS profile to use (default: sagemaker)",
    )

    parser.add_argument(
        "--output-rerun",
        type=Path,
        help="Output file for tasks that need re-running",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output JSON file with detailed results",
    )

    parser.add_argument(
        "--output-script",
        type=Path,
        help="Output shell script with commands to re-run missing episodes",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information for each task",
    )

    args = parser.parse_args()

    # Parse tasks
    tasks = parse_tasks_file(args.tasks_file)
    print(f"Found {len(tasks)} tasks")
    print(f"Expected indices: {args.start_index} to {args.start_index + args.num_samples - 1}")
    print(f"Expected episodes per task: {args.num_samples}")
    print()

    # Verify each task
    results = {}
    missing_by_task = {}
    total_expected = 0
    total_existing = 0
    total_missing = 0
    tasks_with_missing = []

    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task.task_name}", end="", flush=True)

        existing, missing = verify_task(
            task,
            args.start_index,
            args.num_samples,
            args.aws_profile,
        )

        total_expected += args.num_samples
        total_existing += len(existing)
        total_missing += len(missing)

        results[task.name] = {
            "task_name": task.task_name,
            "checkpoint": task.checkpoint,
            "expected": args.num_samples,
            "existing": len(existing),
            "missing": len(missing),
            "missing_indices": sorted(missing),
            "completion_rate": len(existing) / args.num_samples if args.num_samples > 0 else 0,
        }

        if missing:
            missing_by_task[task.name] = missing
            tasks_with_missing.append(task)

        # Print status
        if len(missing) == 0:
            print(f" ✓ {len(existing)}/{args.num_samples} (100%)")
        elif len(existing) == 0:
            print(f" ✗ {len(existing)}/{args.num_samples} (0%) - NO DATA")
        else:
            pct = len(existing) / args.num_samples * 100
            print(f" ! {len(existing)}/{args.num_samples} ({pct:.0f}%) - {len(missing)} missing")

        if args.verbose and missing:
            print(f"      Missing: {format_index_ranges(missing)}")

    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tasks:           {len(tasks)}")
    print(f"Tasks complete:        {len(tasks) - len(tasks_with_missing)}")
    print(f"Tasks with missing:    {len(tasks_with_missing)}")
    print()
    print(f"Total expected:        {total_expected}")
    print(f"Total existing:        {total_existing}")
    print(f"Total missing:         {total_missing}")
    print(f"Overall completion:    {total_existing / total_expected * 100:.1f}%")
    print()

    if tasks_with_missing:
        print("Tasks with missing episodes:")
        for task in tasks_with_missing:
            missing = missing_by_task[task.name]
            print(f"  - {task.task_name}: {len(missing)} missing ({format_index_ranges(missing)})")
        print()

    # Generate re-run file if requested
    if args.output_rerun and tasks_with_missing:
        generate_rerun_tasks_file(tasks, missing_by_task, args.output_rerun)
        print(f"Re-run tasks file written to: {args.output_rerun}")
        print()

    # Output JSON results if requested
    if args.output_json:
        import json

        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results written to: {args.output_json}")

    # Generate re-run script if requested
    if args.output_script and tasks_with_missing:
        commands = generate_rerun_commands(tasks, missing_by_task)
        with open(args.output_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Re-run script for missing episodes\n")
            f.write("# Generated by verify_campaign_completion.py\n")
            f.write("#\n")
            f.write("# Set these environment variables before running:\n")
            f.write("#   export CLUSTER_URL='http://<head-ip>:8265'\n")
            f.write("#   export DOCKER_IMAGE='682769330988.dkr.ecr.us-east-1.amazonaws.com/anzu-vla-foundry:latest'\n")
            f.write("#\n")
            f.write(f"# Total missing episodes: {total_missing}\n")
            f.write(f"# Tasks with missing data: {len(tasks_with_missing)}\n")
            f.write("\nset -e\n\n")

            for cmd in commands:
                f.write(cmd + "\n\n")

        # Make executable
        import os

        os.chmod(args.output_script, 0o755)
        print(f"Re-run script written to: {args.output_script}")

    # Return exit code based on completion
    if total_missing > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
