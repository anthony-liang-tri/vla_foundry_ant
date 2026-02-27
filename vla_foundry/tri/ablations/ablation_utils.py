"""
Shared utilities for ablation experiment generation and launching.

These functions are used by both generate_ablation_configs.py and launch_ablation_sagemaker.py
to ensure consistency in naming and parameter handling.
"""

import math
import re
from itertools import product


def parse_sweep_value(value: str | int | float | list) -> list:
    """
    Parse a value that might be a sweep specification.

    Returns a list of values. For non-sweep values, returns a single-element list.

    Supported sweep syntaxes:
        - List: [v1, v2, v3] -> [v1, v2, v3]
        - linspace(start, end, n) -> n linearly spaced values
        - logspace(start, end, n) -> n logarithmically spaced values
    """
    # Already a list - treat as explicit sweep values
    if isinstance(value, list):
        return value

    # Check for linspace/logspace syntax in strings
    if isinstance(value, str):
        # linspace(start, end, n)
        match = re.match(r"linspace\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*(\d+)\s*\)", value)
        if match:
            start, end, n = float(match.group(1)), float(match.group(2)), int(match.group(3))
            if n == 1:
                return [start]
            step = (end - start) / (n - 1)
            return [start + i * step for i in range(n)]

        # logspace(start, end, n) - logarithmic spacing
        match = re.match(r"logspace\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*(\d+)\s*\)", value)
        if match:
            start, end, n = float(match.group(1)), float(match.group(2)), int(match.group(3))
            if n == 1:
                return [start]
            log_start = math.log10(start)
            log_end = math.log10(end)
            log_step = (log_end - log_start) / (n - 1)
            return [10 ** (log_start + i * log_step) for i in range(n)]

    # Scalar value (int, float, or plain string)
    return [value]


def format_value_for_name(value) -> str:
    """
    Format a value for use in ablation name - safe for filesystem paths and command-line arguments.

    - Floats: Use scientific notation (1e-5) or 'p' for decimal point (1p5)
    - S3 paths: Extract timestamp and checkpoint number (2026_01_06-20_12_13_ckpt3)
    - Other strings: Replace special characters (: / \\ space) with underscores
    """
    if isinstance(value, float):
        # Use scientific notation for very small/large numbers
        if abs(value) < 0.01 or abs(value) >= 1000:
            exp = int(math.floor(math.log10(abs(value)))) if value != 0 else 0
            mantissa = value / (10**exp)
            if abs(mantissa - round(mantissa)) < 0.01:
                return f"{int(round(mantissa))}e{exp}"
            return f"{mantissa:.1f}e{exp}".replace(".", "p")
        # For regular decimals, replace dot with 'p'
        return f"{value:g}".replace(".", "p")

    # Handle strings - S3 paths and general sanitization
    value_str = str(value)

    # If it's an S3 path, extract meaningful identifiers
    if value_str.startswith("s3://"):
        # Extract timestamp (YYYY_MM_DD-HH_MM_SS) and checkpoint number
        timestamp_match = re.search(r"(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})", value_str)
        checkpoint_match = re.search(r"checkpoint[_-](\d+)", value_str)

        parts = []
        if timestamp_match:
            parts.append(timestamp_match.group(1))
        if checkpoint_match:
            parts.append(f"ckpt{checkpoint_match.group(1)}")

        if parts:
            return "_".join(parts)
        # Fallback if no patterns match
        return "unknown_ckpt"

    # Default: sanitize by replacing special characters
    return value_str.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


def get_param_short_name(param: str) -> str:
    """Get a short name for a parameter for use in ablation names."""
    # Remove common prefixes
    for prefix in ["hparams.", "data.", "model.", "--"]:
        if param.startswith(prefix):
            param = param[len(prefix) :]

    # Take last component if dotted
    if "." in param:
        param = param.split(".")[-1]

    return param


def expand_ablation_names(name: str, params: dict) -> list[str]:
    """
    Expand an ablation definition into the list of generated ablation names.

    For sweeps:
      - If name ends with '_sweep', use only param_value (e.g., 'scale_1p5')
      - Otherwise, prefix with base name (e.g., 'merged_stats_epsilon_1e-4')
    """
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
