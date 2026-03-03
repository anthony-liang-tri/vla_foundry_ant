#!/usr/bin/env python3
"""
Generate resolved ablation configs, then launch SageMaker jobs from those configs.

This is a thin wrapper around:
  - vla_foundry/tri/ablations/generate_ablation_configs.py
  - vla_foundry/tri/sagemaker/launch_from_configs.sh

Example:
  uv run python vla_foundry/tri/ablations/launch_ablation_sagemaker.py \
    vla_foundry/tri/ablations/ablations.yaml

Dry-run (shows what would be launched, without actually launching jobs):
  uv run python vla_foundry/tri/ablations/launch_ablation_sagemaker.py \
    vla_foundry/tri/ablations/ablations.yaml --dry-run
"""

import argparse
import secrets
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from vla_foundry.tri.ablations.ablation_utils import expand_ablation_names


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def resolve_path(repo_root: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ablation configs from an ablations YAML, then launch SageMaker jobs from the results."
    )
    parser.add_argument(
        "ablations_config",
        type=str,
        help="Path to ablations YAML (e.g. vla_foundry/tri/ablations/ablations.yaml)",
    )
    parser.add_argument(
        "--nominal-config",
        type=str,
        default="vla_foundry/tri/ablations/nominal_config.yaml",
        help="Path to nominal config YAML (default: vla_foundry/tri/ablations/nominal_config.yaml)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=40,
        help="Maximum parallel config generation jobs (forwarded to generate_ablation_configs.py)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not actually launch SageMaker jobs (still generates configs).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Filter launches by task name (substring match; forwarded to launch_from_configs.sh)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="",
        help="Filter launches by ablation name (substring match; forwarded to launch_from_configs.sh)",
    )
    parser.add_argument(
        "--generate-dry-run",
        action="store_true",
        help="Only print what would be generated (does not generate configs, and does not launch).",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())

    nominal_config_path = resolve_path(repo_root, args.nominal_config)
    ablations_config_path = resolve_path(repo_root, args.ablations_config)

    nominal = yaml.safe_load(nominal_config_path.read_text())
    output_base = nominal["output_base"]
    output_base_path = resolve_path(repo_root, str(output_base))

    # Generate runset hash for campaign tracking
    runset_hash = f"runset_{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(4)}"
    print(f"Runset identifier: {runset_hash}")
    print()

    generate_script = resolve_path(repo_root, "vla_foundry/tri/ablations/generate_ablation_configs.py")
    generate_cmd = [
        "uv",
        "run",
        "python",
        str(generate_script),
        "--nominal-config",
        str(nominal_config_path),
        "--ablations-config",
        str(ablations_config_path),
        "--max-parallel",
        str(args.max_parallel),
        "--runset-hash",
        runset_hash,
    ]
    if args.generate_dry_run:
        generate_cmd.append("--dry-run")

    subprocess.run(generate_cmd, cwd=repo_root, check=True)

    if args.generate_dry_run:
        return

    # Extract ablation names from the YAML to only launch what was just generated
    ablations_yaml = yaml.safe_load(ablations_config_path.read_text())
    all_ablation_names = []
    for name, params in ablations_yaml.get("ablations", {}).items():
        all_ablation_names.extend(expand_ablation_names(name, params))

    # Apply user's ablation filter if provided
    if args.ablation:
        all_ablation_names = [n for n in all_ablation_names if args.ablation in n]

    if not all_ablation_names:
        print("No ablations to launch (after filtering).")
        return

    # Pass ablation names as comma-separated list to the launch script
    ablation_filter = ",".join(all_ablation_names)

    launch_script = resolve_path(repo_root, "vla_foundry/tri/sagemaker/launch_from_configs.sh")
    launch_cmd = ["bash", str(launch_script), str(output_base_path)]
    if args.dry_run:
        launch_cmd.append("--dry-run")
    if args.task:
        launch_cmd.extend(["--task", args.task])
    launch_cmd.extend(["--ablation", ablation_filter])

    # Pass SageMaker args from nominal config
    sagemaker_args = nominal.get("sagemaker_args", {})
    for key in ["user", "profile", "queue_name", "instance_type", "instance_count", "priority", "max_run"]:
        value = sagemaker_args.get(key)
        if value is not None:
            launch_cmd.extend([f"--sagemaker.{key}", str(value)])

    subprocess.run(launch_cmd, cwd=repo_root, check=True)

    # Print runset hash info for evaluation
    if not args.dry_run:
        print("\n" + "=" * 80)
        print(f"Runset Hash: {runset_hash}")
        print("=" * 80)
        print("\nAll runs tagged with runset hash for evaluation tracking.")
        print(f"Use --tags {runset_hash} when running auto_generate_campaign.py")
        print()


if __name__ == "__main__":
    main()
