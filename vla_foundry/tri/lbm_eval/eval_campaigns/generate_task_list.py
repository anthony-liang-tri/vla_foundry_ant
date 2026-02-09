#!/usr/bin/env python3
"""
Generate a task list file for evaluation campaigns.

Takes a list of tasks and a checkpoint path, and creates a task list file
where each task uses the same checkpoint.

Usage:
    python generate_task_list.py --checkpoint s3://path/to/checkpoint --output tasks.txt
    python generate_task_list.py --checkpoint s3://path/to/checkpoint --output tasks.txt --tasks-file my_tasks.txt
    python generate_task_list.py --checkpoint s3://path/to/checkpoint --output tasks.txt --prefix "lbm_ablation"
"""

import argparse
from pathlib import Path

# Default full task list from stage3_singletask_sim
# DEFAULT_TASKS = [
#     "BimanualHangMugsOnMugHolderFromDryingRack",
#     "BimanualHangMugsOnMugHolderFromTable",
#     "BimanualLayCerealBoxOnCuttingBoardFromTopShelf",
#     "BimanualLayCerealBoxOnCuttingBoardFromUnderShelf",
#     "BimanualPlaceAppleFromBowlIntoBin",
#     "BimanualPlaceAppleFromBowlOnCuttingBoard",
#     "BimanualPlaceAvocadoFromBowlOnCuttingBoard",
#     "BimanualPlaceFruitFromBowlIntoBin",
#     "BimanualPlaceFruitFromBowlOnCuttingBoard",
#     "BimanualPlacePearFromBowlIntoBin",
#     "BimanualPlacePearFromBowlOnCuttingBoard",
#     "BimanualPutMugsOnPlatesFromDryingRack",
#     "BimanualPutMugsOnPlatesFromTable",
#     "BimanualPutRedBellPepperInBin",
#     "BimanualPutSpatulaOnPlateFromDryingRack",
#     "BimanualPutSpatulaOnPlateFromTable",
#     "BimanualPutSpatulaOnTableFromDryingRack",
#     "BimanualPutSpatulaOnTableFromUtensilCrock",
#     "BimanualStackPlatesOnTableFromDryingRack",
#     "BimanualStackPlatesOnTableFromTable",
#     "BimanualStoreCerealBoxUnderShelf",
#     "PickAndPlaceBox",
#     "PlaceCupByCoaster",
#     "PlaceCupOnCoaster",
#     "PushCoasterToCenterOfTable",
#     "PushCoasterToMug",
#     "PutBananaInCenterOfTable",
#     "PutBananaOnSaucer",
#     "PutCupInCenterOfTable",
#     "PutCupOnSaucer",
#     "PutGreenAppleInCenterOfTable",
#     "PutGreenAppleOnSaucer",
#     "PutKiwiInCenterOfTable",
#     "PutKiwiOnSaucer",
#     "PutMugOnSaucer",
#     "PutOrangeInCenterOfTable",
#     "PutOrangeOnSaucer",
#     "PutSpatulaInUtensilCrock",
#     "PutSpatulaInUtensilCrockFromDryingRack",
#     "TurnCupUpsideDown",
#     "TurnMugRightsideUp",
# ]

DEFAULT_TASKS = [
    "BimanualPlaceAppleFromBowlIntoBin",
    "BimanualPlaceFruitFromBowlIntoBin",
    "BimanualPutRedBellPepperInBin",
    "BimanualPutSpatulaOnPlateFromDryingRack",
    "BimanualPutSpatulaOnPlateFromTable",
    "BimanualStackPlatesOnTableFromDryingRack",
    "BimanualStoreCerealBoxUnderShelf",
    "PlaceCupByCoaster",
    "PushCoasterToCenterOfTable",
    "PushCoasterToMug",
    "PutBananaOnSaucer",
    "PutKiwiInCenterOfTable",
    "PutMugOnSaucer",
    "PutSpatulaInUtensilCrock",
    "TurnCupUpsideDown",
    "TurnMugRightsideUp",
]


def load_tasks_from_file(path: Path) -> list:
    """Load task names from a file (one per line, or comma-separated)."""
    content = path.read_text().strip()
    tasks = []
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Handle comma-separated tasks
        for task in line.split(","):
            task = task.strip().strip('"').strip("'")
            if task:
                tasks.append(task)
    return tasks


def generate_task_list(tasks: list, checkpoint: str, prefix: str = None) -> str:
    """Generate task list content in the tuple format."""
    lines = []
    lines.append(f"# Task list generated for checkpoint: {checkpoint}")
    lines.append(f"# Total tasks: {len(tasks)}")
    lines.append("")

    for task in tasks:
        # Create job name using prefix if provided, otherwise use task name
        job_name = f"{prefix}_{task}" if prefix else task
        line = f'("{job_name}", "{task}", "{checkpoint}"),'
        lines.append(line)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a task list file for evaluation campaigns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate task list for all default tasks
  python generate_task_list.py \\
    --checkpoint s3://robotics-manip-lbm-us-west-2/checkpoints/model.ckpt \\
    --output my_tasks.txt

  # Generate with a custom job name prefix
  python generate_task_list.py \\
    --checkpoint s3://path/to/checkpoint \\
    --output tasks.txt \\
    --prefix "lbm_step48k"

  # Use a subset of tasks from a file
  python generate_task_list.py \\
    --checkpoint s3://path/to/checkpoint \\
    --output tasks.txt \\
    --tasks-file subset_tasks.txt

  # Filter to specific tasks
  python generate_task_list.py \\
    --checkpoint s3://path/to/checkpoint \\
    --output tasks.txt \\
    --tasks BimanualPutRedBellPepperInBin PlaceCupByCoaster
""",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path (S3 or local) to use for all tasks",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path. If not specified, prints to stdout.",
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        help="File containing task names (one per line). If not specified, uses default task list.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific task names to include. Overrides --tasks-file.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefix for job names (e.g., 'lbm_ablation' -> 'lbm_ablation_TaskName')",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all default tasks and exit",
    )

    args = parser.parse_args()

    # List tasks mode
    if args.list_tasks:
        print("Default tasks:")
        for i, task in enumerate(DEFAULT_TASKS, 1):
            print(f"  {i:2d}. {task}")
        print(f"\nTotal: {len(DEFAULT_TASKS)} tasks")
        return

    # Determine task list
    if args.tasks:
        tasks = args.tasks
    elif args.tasks_file:
        tasks = load_tasks_from_file(args.tasks_file)
    else:
        tasks = DEFAULT_TASKS

    # Generate content
    content = generate_task_list(tasks, args.checkpoint, args.prefix)

    # Output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n")
        print(f"✓ Generated task list with {len(tasks)} tasks: {args.output}")
    else:
        print(content)


if __name__ == "__main__":
    main()
