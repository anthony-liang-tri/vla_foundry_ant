#!/usr/bin/env python3
"""Convert wandb CSV export to tuple format like gnagna file."""

import csv
import re
import sys
from pathlib import Path


def _looks_like_task_name(value):
    """Heuristic: stage-3 tasks are typically PascalCase identifiers."""
    if not value:
        return False
    token = value.strip()
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9]+", token))


def infer_task_name_from_s3_path(s3_path):
    """Infer task name from an S3 checkpoint path by scanning path segments."""
    if not s3_path:
        return None

    path = s3_path.strip().rstrip("/")
    parts = [p for p in path.split("/") if p]
    for part in parts:
        if _looks_like_task_name(part):
            return part
    return None


def infer_task_name_from_manifest_text(text):
    """Infer task name from a manifest-like text field."""
    if not text:
        return None
    # Prefer path segments under common dataset roots.
    for marker in ("/stage3_singletask_sim/", "/v0.4.1/", "/vla_foundry_datasets/"):
        if marker in text:
            tail = text.split(marker, 1)[1]
            segment = tail.split("/", 1)[0].strip().strip("\"'[]")
            if _looks_like_task_name(segment):
                return segment

    # Fallback: pick the first PascalCase token found in the text.
    match = re.search(r"\b([A-Z][A-Za-z0-9]+)\b", text)
    if match and _looks_like_task_name(match.group(1)):
        return match.group(1)

    return None


def extract_task_name(tags_str, s3_path=None, manifest_text=None):
    """Extract task name from tags or S3 path.

    Priority:
    1) First tag that looks like a concrete task name.
    2) Fallback to inferred task name from S3 path segments.
    3) Fallback to dataset-manifest text.
    4) Legacy fallback to first tag.
    """
    if not tags_str:
        inferred = infer_task_name_from_s3_path(s3_path)
        if inferred:
            return inferred
        return infer_task_name_from_manifest_text(manifest_text)
    tags = [tag.strip() for tag in tags_str.split(",")]
    for tag in tags:
        if _looks_like_task_name(tag):
            return tag

    inferred = infer_task_name_from_s3_path(s3_path)
    if inferred:
        return inferred

    inferred = infer_task_name_from_manifest_text(manifest_text)
    if inferred:
        return inferred

    return tags[0] if tags else None


def parse_tags(tags_str):
    """Parse tags string into a list of tags."""
    if not tags_str:
        return []
    return [tag.strip() for tag in tags_str.split(",")]


def find_differentiating_tag(runs_with_tags):
    """Find unique differentiating tag for each run.

    Args:
        runs_with_tags: List of (run_name, s3_path, tags_list) tuples

    Returns:
        Dict mapping run_name to its differentiating tag (or None if only one
        run, or numeric string "2", "3", etc. as fallback for duplicates)
    """
    if len(runs_with_tags) <= 1:
        return {run[0]: None for run in runs_with_tags}

    # Collect all tags for each run (excluding task name which is tags[0])
    run_tags = {}
    for run_name, _, tags in runs_with_tags:
        run_tags[run_name] = set(tags[1:]) if len(tags) > 1 else set()

    # Find tags common to ALL runs
    all_tag_sets = list(run_tags.values())
    common_tags = all_tag_sets[0].copy()
    for tag_set in all_tag_sets[1:]:
        common_tags &= tag_set

    # For each run, find a tag that's unique to it (not in common_tags)
    result = {}
    for run_name, tags in run_tags.items():
        unique_tags = tags - common_tags
        # Pick the first unique tag if any exist
        result[run_name] = next(iter(unique_tags), None) if unique_tags else None

    # Check if any runs lack a differentiating tag - if so, assign numeric fallbacks
    # to ALL runs without unique tags to maintain consistency
    runs_without_tag = [r for r, t in result.items() if t is None]
    if len(runs_without_tag) > 1:
        # Multiple runs have no unique tag - assign numeric suffixes
        # Preserve order from input list
        run_order = [r[0] for r in runs_with_tags]
        idx = 1
        for run_name in run_order:
            if result[run_name] is None:
                result[run_name] = str(idx)
                idx += 1

    return result


def build_s3_path(base_path, run_name):
    """Build full S3 path with run name directory."""
    if not base_path:
        return None
    if not base_path.endswith("/"):
        base_path += "/"
    return f"{base_path}{run_name}/"


def parse_timestamp(run_name):
    """Parse timestamp from run name for sorting (YYYY_MM_DD-HH_MM_SS)."""
    parts = run_name.split("-")
    if len(parts) >= 2:
        date_part = parts[0]  # YYYY_MM_DD
        time_part = parts[1]  # HH_MM_SS
        return (date_part, time_part)
    return ("", "")


def parse_task_filter(filter_input, line_range=None):
    """Parse task names from a file or string.

    Supports:
    - Bash array format: TASKS=("Task1" "Task2" ...)
    - Simple list: one task name per line
    - Comma-separated string
    - Line range: if line_range is (start, end), only parse those lines

    Args:
        filter_input: File path, string, or list of task names
        line_range: Optional tuple (start_line, end_line) for 1-based line numbers
    """
    task_set = set()

    if isinstance(filter_input, (str, Path)):
        path = Path(filter_input)
        if path.exists():
            # Read from file
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Apply line range if specified
                if line_range:
                    start, end = line_range
                    lines = lines[start - 1 : end]  # Convert to 0-based indexing
                content = "".join(lines)
        else:
            # Treat as comma-separated string
            content = filter_input
    else:
        # Assume it's already a list/set
        return set(filter_input)

    # First, try to parse bash array format: TASKS=("Task1" "Task2" ...)
    # This is the most specific pattern, so we prioritize it
    bash_array_match = re.search(r"TASKS=\((.*?)\)", content, re.DOTALL)
    if bash_array_match:
        array_content = bash_array_match.group(1)
        # Extract quoted strings only from within the TASKS array
        task_names = re.findall(r'"([^"]+)"', array_content)
        # Filter out strings that look like they're not task names
        # (e.g., variable names, paths, etc.)
        for name in task_names:
            name = name.strip()
            # Only add if it looks like a task name (starts with capital, no special chars like $, /, etc.)
            if name and not name.startswith(("$", "/", "s3://", "[")) and "::" not in name:
                task_set.add(name)
        # If we found valid task names, return early
        if task_set:
            return task_set

    # If no TASKS array found, try to parse as simple list
    # Look for lines that look like task names (quoted strings on their own line)
    lines = content.split("\n")
    in_tasks_array = False
    for line in lines:
        line = line.strip()
        # Detect if we're in a TASKS array context
        if re.match(r"^\s*TASKS\s*=\s*\(", line):
            in_tasks_array = True
            continue
        if in_tasks_array and line == ")":
            break
        if in_tasks_array:
            # Extract quoted string from line
            match = re.search(r'"([^"]+)"', line)
            if match:
                task_name = match.group(1).strip()
                if task_name and not task_name.startswith(("$", "/", "s3://", "[")) and "::" not in task_name:
                    task_set.add(task_name)

    # If still no tasks found, try comma-separated format
    if not task_set and "," in content:
        for task in content.split(","):
            task = task.strip().strip("\"'")
            if task and not task.startswith(("$", "/", "s3://", "[", "#")):
                task_set.add(task)

    return task_set


def load_run_names_from_csv(csv_path):
    """Collect run names present in a W&B CSV export."""
    run_names = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return run_names

        name_idx = header.index("Name")

        for row in reader:
            if len(row) <= name_idx:
                continue

            run_name = row[name_idx].strip('"')
            if not run_name:
                continue

            run_names.add(run_name)

    return run_names


def csv_to_tuples(csv_path, output_path=None, task_filter=None, exclude_runs=None):
    """Convert CSV to tuple format.

    Args:
        csv_path: Path to CSV file
        output_path: Optional output file path
        task_filter: Optional set of task names to filter by
        exclude_runs: Optional set of run names to exclude
    """
    entries_by_task = {}
    seen_runs = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Find column indices
        name_idx = header.index("Name")
        tags_idx = header.index("Tags")
        remote_sync_idx = header.index("remote_sync")

        manifest_col_names = [
            "data.dataset_manifest",
            "dataset_manifest",
            "data.val_dataset_manifest",
            "val_dataset_manifest",
        ]
        manifest_indices = [header.index(col) for col in manifest_col_names if col in header]

        for row in reader:
            if len(row) <= max(name_idx, tags_idx, remote_sync_idx):
                continue

            run_name = row[name_idx].strip('"')
            if not run_name or run_name in seen_runs:
                continue
            seen_runs.add(run_name)

            if exclude_runs is not None and run_name in exclude_runs:
                continue

            tags = row[tags_idx].strip('"')
            s3_base = row[remote_sync_idx].strip('"')

            manifest_text = ""
            for idx in manifest_indices:
                if idx < len(row) and row[idx]:
                    manifest_text = row[idx].strip('"')
                    break

            task_name = extract_task_name(tags, s3_base, manifest_text)
            if not task_name:
                continue

            tags_list = parse_tags(tags)

            # Build full S3 path
            s3_path = build_s3_path(s3_base, run_name)
            if not s3_path:
                continue

            if task_filter is not None and task_name not in task_filter:
                continue

            entries_by_task.setdefault(task_name, []).append((run_name, s3_path, tags_list))

    # Convert to tuples
    tuples = []
    for task_name in sorted(entries_by_task.keys()):
        runs = entries_by_task[task_name]
        runs.sort(key=lambda item: parse_timestamp(item[0]), reverse=True)

        # Find differentiating tags for runs with same task name
        diff_tags = find_differentiating_tag(runs)

        base_stage_id = f"stage3_singletask_sim_{task_name}"
        for run_name, s3_path, _ in runs:
            diff_tag = diff_tags.get(run_name)
            stage_id = f"{base_stage_id}_{diff_tag}" if diff_tag else base_stage_id
            tuples.append((stage_id, task_name, s3_path))

    # Sort by task name for consistency
    tuples.sort(key=lambda x: (x[1], x[0]))

    # Format output
    output_lines = []
    for stage_id, task_name, s3_path in tuples:
        output_lines.append("(")
        output_lines.append(f'    "{stage_id}",')
        output_lines.append(f'    "{task_name}",')
        output_lines.append(f'    f"{s3_path}",')
        output_lines.append("),")
        output_lines.append("")

    output_text = "\n".join(output_lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Wrote {len(tuples)} tuples to {output_path}")
    else:
        print(output_text)

    return tuples


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python csv_to_tuples.py <csv_file> [output_file] [--filter <task_filter_file> [--lines start-end]]"
        )
        print("  or:  python csv_to_tuples.py <csv_file> [output_file] [--filter-tasks Task1 Task2 ...]")
        print("  or:  python csv_to_tuples.py <csv_file> [output_file] [--exclude-csv other.csv]")
        print("\nExamples:")
        print("  python csv_to_tuples.py file.csv --filter tasks.sh --lines 48-65")
        print("  python csv_to_tuples.py file.csv output.txt --filter tasks.sh")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    output_path = None
    task_filter = None
    exclude_csv_paths = []

    # Parse arguments
    i = 2
    filter_input = None
    line_range = None
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--filter" and i + 1 < len(sys.argv):
            filter_input = sys.argv[i + 1]
            i += 2
        elif arg == "--lines" and i + 1 < len(sys.argv):
            # Line range for filter
            range_str = sys.argv[i + 1]
            if "-" in range_str:
                start, end = map(int, range_str.split("-"))
                line_range = (start, end)
            i += 2
        elif arg == "--filter-tasks":
            # Collect all remaining arguments as task names
            task_names = []
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith("--"):
                task_names.append(sys.argv[i])
                i += 1
            if task_names:
                task_filter = set(task_names)
        elif arg == "--exclude-csv" and i + 1 < len(sys.argv):
            exclude_csv_paths.append(Path(sys.argv[i + 1]))
            i += 2
        else:
            # Treat as output path if not a flag
            if not arg.startswith("--"):
                output_path = Path(arg)
            i += 1

    # Apply filter if specified
    if filter_input:
        task_filter = parse_task_filter(filter_input, line_range)

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    if task_filter:
        print(f"Filtering to {len(task_filter)} tasks: {sorted(task_filter)}")

    exclude_runs = set()
    for exclude_path in exclude_csv_paths:
        if not exclude_path.exists():
            print(f"Error: CSV file not found for exclusion: {exclude_path}")
            sys.exit(1)
        exclude_runs.update(load_run_names_from_csv(exclude_path))

    if exclude_runs:
        print(f"Excluding {len(exclude_runs)} runs: {sorted(exclude_runs)}")

    csv_to_tuples(
        csv_path,
        output_path,
        task_filter,
        exclude_runs if exclude_runs else None,
    )
