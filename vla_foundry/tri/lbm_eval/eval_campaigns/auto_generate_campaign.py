#!/usr/bin/env python3
"""
Generate ready-to-launch evaluation campaign from W&B training runs.

Usage:
  # Single-task runs: use runset tag
  python auto_generate_campaign.py --tags runset_20260212_143052_a3f9d8e1 --owner-email your.name@tri.global

  # Multitask runs: specify which task(s) to evaluate with --task
  python auto_generate_campaign.py --tags multitask_20260223_{1..15} --name multitask_eval \\
    --task BimanualPutSpatulaOnPlate BimanualStackPlates --owner-email your.name@tri.global

  # Multitask runs: use task preset to evaluate on all sim_16 tasks
  python auto_generate_campaign.py --tags multitask_20260223_{1..15} --name multitask_eval \\
    --task sim_16 --owner-email your.name@tri.global

  # Multiple explicit tags
  python auto_generate_campaign.py --tags tag1 tag2 tag3 --name my_campaign --owner-email your.name@tri.global

  # Specify checkpoint and launch immediately
  python auto_generate_campaign.py --tags runset_20260212_143052_a3f9d8e1 --checkpoint-num 3 --launch

  # Customize sample count
  python auto_generate_campaign.py --tags runset_20260212_143052_a3f9d8e1 --num-samples 200

Task Presets:
  - sim_16: All 16 sim tasks
  - realworld_toolhang: All toolhang tasks
  
  (Task sets defined in: vla_foundry/tri/ablations/task_sets.py)
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# Add ablations directory to import task_sets
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ablations"))
from task_sets import get_task_set, is_preset


def _wandb_api_path(entity: str | None, project: str) -> str:
    """Build W&B API path."""
    if entity:
        return f"{entity}/{project}"
    return project


def _extract_remote_sync(run) -> str | None:
    """Extract remote_sync from run config."""
    config = getattr(run, "config", None)
    if hasattr(config, "get"):
        remote_sync = config.get("remote_sync")
        if isinstance(remote_sync, dict) and "value" in remote_sync:
            return remote_sync["value"]
        if isinstance(remote_sync, str):
            return remote_sync
    return None


def _get_task_from_tags(tags: list[str]) -> str | None:
    """Extract task name from tags (assumes first tag is task name)."""
    if tags:
        return tags[0]
    return None


def _get_task_from_config(run) -> str | None:
    """Extract task name from W&B run config (data.manifest path)."""
    config = getattr(run, "config", None)
    if not hasattr(config, "get"):
        return None

    # Try to extract from data.manifest path (e.g., "s3://.../TaskName/manifest.jsonl")
    data_config = config.get("data")
    if isinstance(data_config, dict):
        manifest = data_config.get("manifest")
        if isinstance(manifest, str):
            # Extract task name from manifest path
            # Expected format: s3://bucket/path/TaskName/manifest.jsonl
            parts = manifest.rstrip("/").split("/")
            if len(parts) >= 2:
                # Get the second-to-last component (task directory)
                task_name = parts[-2]
                if task_name and not task_name.endswith(".jsonl"):
                    return task_name

    # Fallback: if data is a dict with a "value" wrapper
    if isinstance(data_config, dict) and "value" in data_config:
        manifest = data_config["value"].get("manifest")
        if isinstance(manifest, str):
            parts = manifest.rstrip("/").split("/")
            if len(parts) >= 2:
                task_name = parts[-2]
                if task_name and not task_name.endswith(".jsonl"):
                    return task_name

    return None


def _expand_task_presets(task_names: list[str] | None) -> list[str]:
    """Expand task preset names (like 'sim_16') to actual task lists.

    Uses presets defined in vla_foundry/tri/ablations/task_sets.py.

    Args:
        task_names: List of task names or preset names

    Returns:
        Expanded list of actual task names
    """
    if not task_names:
        return []

    expanded = []
    for name in task_names:
        if is_preset(name):
            tasks = get_task_set(name)
            if tasks:
                expanded.extend(tasks)
                print(f"Expanded preset '{name}' to {len(tasks)} tasks")
            else:
                print(f"Warning: Preset '{name}' has no tasks defined")
                expanded.append(name)
        else:
            expanded.append(name)

    return expanded


def query_wandb_runs(
    entity: str | None,
    project: str,
    tags: list[str] | None = None,
    hours: int | None = None,
    user: str | None = None,
    states: list[str] | None = None,
) -> list[dict]:
    """
    Query W&B API for runs matching criteria.

    Args:
        entity: W&B entity (team/org)
        project: W&B project name
        tags: Filter runs with any of these tags (OR logic)
        hours: Only include runs from last N hours
        user: Filter by username (e.g., 'firstname.lastname')
        states: Run states to include (e.g., ['finished', 'running'])

    Returns list of dicts with: {task, run_name, remote_sync, tags, created_at}
    """
    import wandb

    api = wandb.Api()
    path = _wandb_api_path(entity, project)

    # Build filter
    filters = {}
    if tags:
        filters["tags"] = {"$in": tags}
    if states:
        filters["state"] = {"$in": states}
    if hours:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        filters["created_at"] = {"$gte": cutoff.isoformat()}
    if user:
        filters["username"] = user

    print(f"Querying W&B: {path}")
    if filters:
        print(f"Filters: {filters}")

    runs = api.runs(path, filters=filters if filters else None)

    # Collect all matching runs
    result = []
    for run in runs:
        # Try to extract task name from config first (more reliable)
        task = _get_task_from_config(run)
        if not task:
            # Fallback to tags (legacy behavior)
            task = _get_task_from_tags(run.tags)

        if not task:
            print(f"Warning: Run {run.name} has no task name in config or tags, skipping")
            continue

        remote_sync = _extract_remote_sync(run)
        if not remote_sync:
            print(f"Warning: Run {run.name} has no remote_sync, skipping")
            continue

        result.append(
            {
                "task": task,
                "run_name": run.name,  # Use actual W&B run name
                "remote_sync": remote_sync,
                "tags": run.tags,
                "created_at": run.created_at,
            }
        )

    return result


def generate_task_tuples(
    runs: list[dict],
    checkpoint_num: int | None = None,
    eval_tasks: list[str] | None = None,
    job_name_prefix: str = "stage3_singletask_sim_",
) -> list[str]:
    """
    Generate task tuples from run metadata.

    Args:
        runs: List of run dicts with task, run_name, remote_sync, etc.
        checkpoint_num: Specific checkpoint number to use
        eval_tasks: Explicit task names to evaluate (overrides auto-detected task)
        job_name_prefix: Prefix for job names

    Returns list of tuple strings like:
    ("run_name", "TaskName", "s3://path/to/checkpoints/")
    """
    tuples = []

    for run_data in runs:
        # Use explicit eval_tasks if provided (for multitask runs)
        # Otherwise use auto-detected task from run
        tasks_to_eval = eval_tasks if eval_tasks else [run_data["task"]]

        remote_sync = run_data["remote_sync"]
        run_name = run_data["run_name"]

        for task in tasks_to_eval:
            # Build checkpoint path
            # remote_sync structure: s3://.../base_dir/run_name/ or s3://.../base_dir/TaskName/run_name/
            # Structure is: remote_sync/run_name/checkpoints/checkpoint_N.pt
            if checkpoint_num is not None:
                # For specific checkpoint
                checkpoint_path = f"{remote_sync.rstrip('/')}/{run_name}/checkpoints/checkpoint_{checkpoint_num}.pt"
            else:
                # For directory of checkpoints
                checkpoint_path = f"{remote_sync.rstrip('/')}/{run_name}/"

            # Build job name with prefix (matching csv_to_tuples.py behavior)
            job_name = f"{job_name_prefix}{task}"

            tuple_str = f'("{job_name}", "{task}", "{checkpoint_path}"),'
            tuples.append(tuple_str)

    return tuples


def create_campaign_yaml(
    campaign_name: str,
    task_tuples: list[str],
    output_path: Path,
    template_path: Path,
    num_samples: int = 100,
    num_workers: int = 128,
    owner_email: str | None = None,
    cluster_name: str | None = None,
) -> None:
    """Create campaign YAML from template, overriding specific fields."""

    # Load template YAML
    with open(template_path) as f:
        campaign = yaml.safe_load(f)

    # Create task_lists directory and write task tuples file
    task_lists_dir = output_path.parent.parent / "task_lists"
    task_lists_dir.mkdir(parents=True, exist_ok=True)

    task_file_name = f"{campaign_name}.txt"
    task_file_path = task_lists_dir / task_file_name

    # Write task tuples to separate .txt file
    with open(task_file_path, "w") as f:
        for tuple_str in task_tuples:
            f.write(f"    {tuple_str}\n")

    print(f"✓ Task tuples file: {task_file_path}")

    # Override template fields
    campaign["campaign_name"] = campaign_name
    campaign["description"] = f"Auto-generated campaign: {campaign_name}"

    # Override cluster settings
    if "cluster" not in campaign:
        campaign["cluster"] = {}
    campaign["cluster"]["num_workers"] = num_workers
    if owner_email:
        campaign["cluster"]["owner_email"] = owner_email
    if cluster_name:
        campaign["cluster"]["cluster_name"] = cluster_name

    # Override evaluation settings
    if "evaluation" not in campaign:
        campaign["evaluation"] = {}
    campaign["evaluation"]["tasks_file"] = f"vla_foundry/tri/lbm_eval/eval_campaigns/task_lists/{task_file_name}"
    campaign["evaluation"]["num_samples"] = num_samples

    # Override output settings
    if "output" not in campaign:
        campaign["output"] = {}
    campaign["output"]["results_dir"] = f"results/{campaign_name}"

    # Write modified YAML file
    with open(output_path, "w") as f:
        f.write(f"# Auto-generated evaluation campaign from template: {template_path.name}\n")
        if not owner_email and campaign.get("cluster", {}).get("owner_email") == "":
            f.write("# WARNING: Review and update owner_email before launching!\n")
        f.write("\n")
        yaml.dump(campaign, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Campaign YAML: {output_path}")

    if not owner_email and campaign.get("cluster", {}).get("owner_email") == "":
        print("\n⚠️  WARNING: owner_email is empty! Update it before launching:")
        print(f"   Edit {output_path} and set cluster.owner_email to your @tri.global email")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ready-to-launch evaluation campaign from W&B runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # W&B query options
    parser.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY"), help="W&B entity/team")
    parser.add_argument("--project", type=str, default="vla_foundry", help="W&B project name")
    parser.add_argument(
        "--tag",
        "--tags",
        nargs="+",
        dest="tags",
        help="Filter by tag(s) - accepts multiple tags or use bash expansion like multitask_{1..15}",
    )
    parser.add_argument("--hours", type=int, help="Filter runs from last N hours")
    parser.add_argument("--user", type=str, help="Filter by username")
    parser.add_argument("--states", nargs="+", default=["finished"], help="Run states (default: finished)")

    # Task filtering
    parser.add_argument(
        "--task",
        "--tasks",
        nargs="+",
        dest="tasks",
        help="Task name(s) for evaluation or preset (ex: 'sim_16' or 'BimanualPutSpatulaOnPlate')",
    )
    parser.add_argument("--checkpoint-num", type=int, help="Checkpoint number (e.g., 3)")
    parser.add_argument(
        "--job-name-prefix",
        type=str,
        default="stage3_singletask_sim_",
        help="Prefix for job names (default: stage3_singletask_sim_)",
    )

    # Campaign configuration
    parser.add_argument("--name", type=str, help="Campaign name (default: first tag from --tags)")
    parser.add_argument("--template", type=str, help="Template YAML path (default: campaigns/example_campaign.yaml)")
    parser.add_argument("--num-samples", type=int, default=200, help="Rollouts per checkpoint (default: 200)")
    parser.add_argument("--num-workers", type=int, default=128, help="Cluster workers (default: 128)")
    parser.add_argument("--owner-email", type=str, help="Owner email for AWS tagging (your @tri.global email)")
    parser.add_argument(
        "--cluster-name", type=str, help="Explicit cluster name (default: auto-derived from owner_email)"
    )

    # Output and action
    parser.add_argument("-o", "--output", type=str, help="Output path (default: campaigns/<name>.yaml)")
    parser.add_argument("--launch", action="store_true", help="Launch campaign immediately after generation")

    args = parser.parse_args()

    # Validate
    if not args.tags and not args.hours and not args.user:
        parser.error("Must specify at least one filter: --tags, --hours, or --user")

    # Default campaign name
    if args.name:
        campaign_name = args.name
    elif args.tags and len(args.tags) == 1:
        campaign_name = args.tags[0]
    elif args.tags:
        # Multiple tags: require explicit name
        parser.error(f"Multiple tags provided ({len(args.tags)} tags). Please specify --name for the campaign.")
    else:
        campaign_name = "campaign"

    # Sanitize campaign name for filesystem safety
    campaign_name = re.sub(r"[^\w\-.]", "_", campaign_name)  # Replace special chars with underscore

    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        campaigns_dir = Path("vla_foundry/tri/lbm_eval/eval_campaigns/campaigns")
        campaigns_dir.mkdir(parents=True, exist_ok=True)
        output_path = campaigns_dir / f"{campaign_name}.yaml"

    # Set template path
    if args.template:
        template_path = Path(args.template)
    else:
        template_path = Path("vla_foundry/tri/lbm_eval/eval_campaigns/campaigns/example_campaign.yaml")

    if not template_path.exists():
        print(f"Error: Template file not found: {template_path}")
        sys.exit(1)

    # Query W&B
    print(f"Querying W&B for {campaign_name}...")

    runs = query_wandb_runs(
        entity=args.entity,
        project=args.project,
        tags=args.tags,
        hours=args.hours,
        user=args.user,
        states=args.states,
    )

    if not runs:
        print("No runs found")
        sys.exit(1)

    print(f"Found {len(runs)} runs")

    # Expand task presets (e.g., 'sim_16' -> list of 16 tasks)
    eval_tasks = _expand_task_presets(args.tasks) if args.tasks else None

    # Generate task tuples
    task_tuples = generate_task_tuples(
        runs=runs,
        checkpoint_num=args.checkpoint_num,
        eval_tasks=eval_tasks,
        job_name_prefix=args.job_name_prefix,
    )

    if not task_tuples:
        print("No task tuples generated")
        sys.exit(1)

    print(f"Generated {len(task_tuples)} task tuples\n")

    # Create campaign YAML
    create_campaign_yaml(
        campaign_name=campaign_name,
        task_tuples=task_tuples,
        output_path=output_path,
        template_path=template_path,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        owner_email=args.owner_email,
        cluster_name=args.cluster_name,
    )

    # Launch if requested
    if args.launch:
        print("\nLaunching campaign...")
        launch_script = Path("vla_foundry/tri/lbm_eval/eval_campaigns/launch_campaign.sh")
        if not launch_script.exists():
            print(f"Error: {launch_script} not found")
            sys.exit(1)

        result = subprocess.run([str(launch_script), str(output_path)])
        sys.exit(result.returncode)
    else:
        print(f"\nTo launch: ./vla_foundry/tri/lbm_eval/eval_campaigns/launch_campaign.sh {output_path}")


if __name__ == "__main__":
    main()
