#!/usr/bin/env python3
"""
Export W&B runs for a given ablation-config directory and generate an eval task list.

Given an ablation config directory like:
  vla_foundry/tri/ablations/ablation_configs/fp32_30m/
    ├── TaskA/resolved_config.yaml
    ├── TaskB/resolved_config.yaml
    └── ...

This script:
1) Reads tasks + (project, remote_sync, tags) from each resolved_config.yaml
2) Queries W&B for runs matching tags: ["ablation", <ablation_name>, <task_name>]
3) Writes a CSV export to: vla_foundry/tri/lbm_eval/eval_campaigns/wandb_exports/
   with columns: Name, Tags, remote_sync
4) Calls csv_to_tuples.py to generate a task list in:
   vla_foundry/tri/lbm_eval/eval_campaigns/task_lists/

Notes:
- `csv_to_tuples.py` assumes the first tag in the CSV "Tags" column is the task name.
  This script enforces that ordering when writing the CSV.
"""

import argparse
import csv
import os
from pathlib import Path

import yaml
from csv_to_tuples import csv_to_tuples


def _load_resolved_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_task_dirs(ablation_config_dir: Path) -> list[Path]:
    task_dirs = []
    for child in sorted(ablation_config_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "resolved_config.yaml").exists():
            task_dirs.append(child)
    return task_dirs


def _wandb_api_path(entity: str | None, project: str) -> str:
    if entity:
        return f"{entity}/{project}"
    return project


def _extract_remote_sync(run) -> str | None:
    config = getattr(run, "config", None)
    if hasattr(config, "get"):
        remote_sync = config.get("remote_sync")
        if isinstance(remote_sync, dict) and "value" in remote_sync:
            return remote_sync["value"]
        if isinstance(remote_sync, str):
            return remote_sync
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export W&B runs for an ablation config dir and generate an eval task list."
    )
    parser.add_argument(
        "ablation_config_dir",
        type=str,
        help="Path to ablation config dir (e.g. vla_foundry/tri/ablations/ablation_configs/fp32_30m)",
    )
    parser.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY"), help="W&B entity/team")
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="W&B project (defaults to wandb_project_name from resolved_config.yaml)",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=["finished"],
        help="Only include runs with these states (default: finished). Use e.g. --states finished running",
    )
    parser.add_argument(
        "--max-runs-per-task",
        type=int,
        default=1,
        help="Max runs per task to export (latest by created_at). Use 0 to export all.",
    )
    parser.add_argument(
        "--export-name",
        type=str,
        default=None,
        help="Basename for outputs (default: ablation directory name).",
    )
    args = parser.parse_args()

    ablation_config_dir = Path(args.ablation_config_dir).expanduser()
    if not ablation_config_dir.exists() or not ablation_config_dir.is_dir():
        raise ValueError(f"ablation_config_dir does not exist or is not a directory: {ablation_config_dir}")

    ablation_name = ablation_config_dir.name
    export_name = args.export_name or ablation_name

    task_dirs = _find_task_dirs(ablation_config_dir)
    if not task_dirs:
        raise ValueError(f"No task subdirectories with resolved_config.yaml found under: {ablation_config_dir}")

    # Collect tasks + defaults from resolved configs.
    tasks: list[str] = []
    remote_sync_by_task: dict[str, str] = {}
    project_names: set[str] = set()

    for task_dir in task_dirs:
        task_name = task_dir.name
        cfg = _load_resolved_config(task_dir / "resolved_config.yaml")

        if "wandb_project_name" in cfg and cfg["wandb_project_name"]:
            project_names.add(cfg["wandb_project_name"])
        if "remote_sync" in cfg and cfg["remote_sync"]:
            remote_sync_by_task[task_name] = cfg["remote_sync"]

        tasks.append(task_name)

    if args.project is None:
        if len(project_names) != 1:
            raise ValueError(
                f"Could not infer a single wandb_project_name from configs. Found: {sorted(project_names)}. "
                "Pass --project explicitly."
            )
        project = next(iter(project_names))
    else:
        project = args.project

    # Lazy import so `--help` works even without wandb installed.
    import wandb  # noqa: PLC0415

    api = wandb.Api()
    path = _wandb_api_path(args.entity, project)

    # Broad query; we refine in Python to avoid relying on W&B filter dialect.
    runs = api.runs(path, filters={"tags": {"$in": ["ablation", ablation_name]}})

    tasks_set = set(tasks)
    allowed_states = set(args.states)

    runs_by_task: dict[str, list] = {t: [] for t in tasks}
    for run in runs:
        if run.state not in allowed_states:
            continue
        tags = list(run.tags)
        if "ablation" not in tags:
            continue
        if ablation_name not in tags:
            continue

        # Identify the task tag from the known set.
        task_tag = None
        for t in tags:
            if t in tasks_set:
                task_tag = t
                break
        if task_tag is None:
            continue

        runs_by_task[task_tag].append(run)

    # Select latest N runs per task.
    selected_runs: list = []
    for task_name in tasks:
        task_runs = runs_by_task[task_name]
        task_runs.sort(key=lambda r: r.created_at, reverse=True)
        if args.max_runs_per_task == 0:
            selected_runs.extend(task_runs)
        else:
            selected_runs.extend(task_runs[: args.max_runs_per_task])

    eval_campaigns_dir = Path(__file__).resolve().parent
    wandb_exports_dir = eval_campaigns_dir / "wandb_exports"
    task_lists_dir = eval_campaigns_dir / "task_lists"
    wandb_exports_dir.mkdir(parents=True, exist_ok=True)
    task_lists_dir.mkdir(parents=True, exist_ok=True)

    export_csv_path = wandb_exports_dir / f"{export_name}.csv"
    task_list_path = task_lists_dir / f"{export_name}.txt"

    # Write CSV with exact columns expected by csv_to_tuples.py
    with open(export_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Tags", "remote_sync"])
        writer.writeheader()

        for run in selected_runs:
            tags = list(run.tags)

            task_tag = None
            for t in tags:
                if t in tasks_set:
                    task_tag = t
                    break
            if task_tag is None:
                continue

            ordered_tags = [task_tag] + [t for t in tags if t != task_tag]
            remote_sync = _extract_remote_sync(run)
            if remote_sync is None:
                remote_sync = remote_sync_by_task.get(task_tag)

            writer.writerow(
                {
                    "Name": run.name,
                    "Tags": ",".join(ordered_tags),
                    "remote_sync": remote_sync,
                }
            )

    # Generate tasks file from the export.
    csv_to_tuples(export_csv_path, task_list_path)

    print(f"Wrote W&B export: {export_csv_path}")
    print(f"Wrote task list : {task_list_path}")
    print(f"Included tasks  : {len(tasks)}")
    print(f"Included runs   : {len(selected_runs)} (states={sorted(allowed_states)})")


if __name__ == "__main__":
    main()
