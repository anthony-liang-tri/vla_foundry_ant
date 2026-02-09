#!/usr/bin/env python3
"""
Manage ablation experiments by querying their status from wandb.

This script scans the ablation_configs directory to find all ablation/task
combinations and queries wandb to determine their status (running, finished,
failed, or not launched).

Usage:
    uv run python vla_foundry/tri/ablations/manage_ablations.py [--entity ENTITY] [--project PROJECT]

Requirements:
    pip install wandb pyyaml
"""

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

try:
    import wandb
except ImportError:
    print("Error: wandb is not installed. Install it with: pip install wandb")
    exit(1)


# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    GRAY = "\033[90m"


def get_status_color(status: str) -> str:
    """Return colored status string based on status type."""
    status_colors = {
        "running": Colors.BLUE,
        "finished": Colors.GREEN,
        "failed": Colors.RED,
        "crashed": Colors.RED,
        "killed": Colors.YELLOW,
        "not_launched": Colors.GRAY,
    }
    color = status_colors.get(status.lower(), Colors.ENDC)
    return f"{color}{status}{Colors.ENDC}"


def scan_ablation_configs(config_dir: str) -> dict[str, dict[str, list[str]]]:
    """
    Scan the ablation_configs directory to find all ablation/task combinations.

    Returns:
        dict mapping ablation_name -> {task_name: [wandb_tags]}
    """
    ablations = defaultdict(dict)
    config_path = Path(config_dir)

    if not config_path.exists():
        print(f"Error: Config directory not found: {config_dir}")
        return {}

    for ablation_dir in sorted(config_path.iterdir()):
        if not ablation_dir.is_dir():
            continue
        ablation_name = ablation_dir.name

        for task_dir in sorted(ablation_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            # Check if resolved_config.yaml exists
            config_file = task_dir / "resolved_config.yaml"
            if config_file.exists():
                # Read wandb_tags from config
                try:
                    with open(config_file) as f:
                        config = yaml.safe_load(f)
                    wandb_tags = config.get("wandb_tags", [])
                except Exception:
                    wandb_tags = []
                ablations[ablation_name][task_dir.name] = wandb_tags

    return dict(ablations)


def get_wandb_runs(entity: str | None, project: str) -> list[dict]:
    """
    Fetch all ablation runs from wandb.

    Returns:
        list of run info dicts, each containing tags as a frozenset for matching
    """
    api = wandb.Api()

    # Query all runs with "ablation" tag
    try:
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(path, filters={"tags": {"$in": ["ablation"]}})
    except Exception as e:
        print(f"Error fetching runs from wandb: {e}")
        print("Make sure you're logged in with: wandb login")
        return []

    run_list = []

    for run in runs:
        tags = run.tags
        if "ablation" not in tags:
            continue

        run_list.append(
            {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "url": run.url,
                "summary": run.summary._json_dict if hasattr(run.summary, "_json_dict") else {},
                "tags": frozenset(tags),
            }
        )

    return run_list


def format_run_info(run_info: dict) -> str:
    """Format run info for display."""
    state = run_info["state"]
    name = run_info["name"]

    # Get training progress if available
    summary = run_info.get("summary", {})
    step = summary.get("_step", summary.get("global_step"))
    progress = f" (step {step:,})" if step else ""

    return f"{get_status_color(state)}{progress} - {name}"


def main():
    parser = argparse.ArgumentParser(description="Manage ablation experiments status from wandb")
    parser.add_argument("--entity", type=str, default=None, help="Wandb entity (team/user)")
    parser.add_argument("--project", type=str, default="vla_foundry", help="Wandb project name")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="vla_foundry/tri/ablations/ablation_configs",
        help="Path to ablation configs directory",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show run details")
    args = parser.parse_args()

    print(f"{Colors.BOLD}Scanning ablation configs...{Colors.ENDC}")
    ablations = scan_ablation_configs(args.config_dir)

    if not ablations:
        print("No ablation configs found.")
        return

    total_configs = sum(len(tasks) for tasks in ablations.values())
    print(f"Found {len(ablations)} ablations with {total_configs} total configs")

    print(f"\n{Colors.BOLD}Fetching wandb runs...{Colors.ENDC}")
    wandb_runs = get_wandb_runs(args.entity, args.project)
    print(f"Found {len(wandb_runs)} ablation runs in wandb")

    # Status counters
    status_counts = defaultdict(int)

    # Display results grouped by ablation
    print(f"\n{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}ABLATION STATUS{Colors.ENDC}")
    print(f"{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")

    for ablation_name in sorted(ablations.keys()):
        tasks = ablations[ablation_name]
        print(f"{Colors.BOLD}{Colors.CYAN}[{ablation_name}]{Colors.ENDC}")

        for task_name in sorted(tasks):
            # Get the expected tags from the config
            expected_tags = frozenset(tasks[task_name])

            # Find runs that have all the expected tags
            matching_runs = [run for run in wandb_runs if expected_tags <= run["tags"]]

            if not matching_runs:
                status = "not_launched"
                status_counts["not_launched"] += 1
                print(f"  {task_name}: {get_status_color(status)}")
            else:
                # Get the most recent run
                latest_run = sorted(matching_runs, key=lambda x: x["created_at"], reverse=True)[0]
                status = latest_run["state"]
                status_counts[status] += 1

                if args.verbose:
                    print(f"  {task_name}: {format_run_info(latest_run)}")
                else:
                    print(f"  {task_name}: {get_status_color(status)}")

                # Show additional runs if multiple exist
                if len(matching_runs) > 1 and args.verbose:
                    print(f"    ({len(matching_runs) - 1} older runs)")

        print()

    # Summary
    print(f"{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}SUMMARY{Colors.ENDC}")
    print(f"{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"  Total configs:    {total_configs}")
    print(f"  {Colors.GREEN}Finished:{Colors.ENDC}        {status_counts.get('finished', 0)}")
    print(f"  {Colors.BLUE}Running:{Colors.ENDC}         {status_counts.get('running', 0)}")
    print(f"  {Colors.RED}Failed:{Colors.ENDC}          {status_counts.get('failed', 0)}")
    print(f"  {Colors.RED}Crashed:{Colors.ENDC}         {status_counts.get('crashed', 0)}")
    print(f"  {Colors.YELLOW}Killed:{Colors.ENDC}          {status_counts.get('killed', 0)}")
    print(f"  {Colors.GRAY}Not launched:{Colors.ENDC}    {status_counts.get('not_launched', 0)}")


if __name__ == "__main__":
    main()
