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

from __future__ import annotations

import argparse
import math
import re
import subprocess
from itertools import product
from pathlib import Path

import yaml


def parse_sweep_value(value: str | int | float | list) -> list:
    """Parse a value that might be a sweep specification."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        match = re.match(r"linspace\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*(\d+)\s*\)", value)
        if match:
            start, end, n = float(match.group(1)), float(match.group(2)), int(match.group(3))
            if n == 1:
                return [start]
            step = (end - start) / (n - 1)
            return [start + i * step for i in range(n)]
        match = re.match(r"logspace\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*(\d+)\s*\)", value)
        if match:
            start, end, n = float(match.group(1)), float(match.group(2)), int(match.group(3))
            if n == 1:
                return [start]
            log_start, log_end = math.log10(start), math.log10(end)
            log_step = (log_end - log_start) / (n - 1)
            return [10 ** (log_start + i * log_step) for i in range(n)]
    return [value]


def format_value_for_name(value) -> str:
    """Format a value for use in ablation name."""
    if isinstance(value, float):
        if abs(value) < 0.01 or abs(value) >= 1000:
            exp = int(math.floor(math.log10(abs(value)))) if value != 0 else 0
            mantissa = value / (10**exp)
            if abs(mantissa - round(mantissa)) < 0.01:
                return f"{int(round(mantissa))}e{exp}"
            return f"{mantissa:.1f}e{exp}".replace(".", "p")
        return f"{value:g}".replace(".", "p")
    return str(value).replace(".", "p")


def get_param_short_name(param: str) -> str:
    """Get a short name for a parameter."""
    for prefix in ["hparams.", "data.", "model.", "--"]:
        if param.startswith(prefix):
            param = param[len(prefix) :]
    if "." in param:
        param = param.split(".")[-1]
    return param


def expand_ablation_names(name: str, params: dict) -> list[str]:
    """Expand an ablation definition into the list of generated ablation names."""
    if params is None:
        params = {}

    sweep_params = {}
    for param, value in params.items():
        parsed = parse_sweep_value(value)
        if len(parsed) > 1:
            sweep_params[param] = parsed

    if not sweep_params:
        return [name]

    ablation_names = []
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[p] for p in param_names]
    include_base_name = not name.endswith("_sweep")

    for combo in product(*param_values):
        name_parts = []
        for param, value in zip(param_names, combo, strict=True):
            short_name = get_param_short_name(param)
            value_str = format_value_for_name(value)
            name_parts.append(f"{short_name}_{value_str}")
        ablation_name = f"{name}_" + "_".join(name_parts) if include_base_name else "_".join(name_parts)
        ablation_names.append(ablation_name)

    return ablation_names


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

    subprocess.run(launch_cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
