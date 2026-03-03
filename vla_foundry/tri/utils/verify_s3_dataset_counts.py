#!/usr/bin/env python3
"""Verify per-task S3 dataset integrity by comparing frames file counts to stats.json counts.

Example:
  python vla_foundry/data/scripts/scratch/verify_s3_dataset_counts.py \
    --base-path s3://tri-ml-datasets-uw2/vla_foundry_datasets/v0.4.2
"""

import argparse
import json
import os
import re
import signal
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError

TASK_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
TASK_LIKE_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")
STOP_EVENT = Event()
INTERRUPT_COUNT = 0


@dataclass
class TaskResult:
    task: str
    status: str  # ok | failed | missing
    shards: int = 0
    samples: int = 0
    episode_files: int = 0
    stats_max: int = 0
    missing_from_stats: int = 0
    reasons: list[str] | None = None
    image_resize_dim: tuple[int, int] | None = None
    settings_issues: list[str] | None = None
    past_steps: int | None = None
    future_steps: int | None = None
    camera_names: tuple[str, ...] | None = None
    image_indices: tuple[int, ...] | None = None
    image_resizing_method: str | None = None
    jpeg_quality: int | None = None


@dataclass
class RuntimeProgress:
    lock: Lock
    tasks_started: int = 0
    tasks_finished: int = 0
    s3_pages_scanned: int = 0
    s3_objects_scanned: int = 0

    def mark_task_started(self) -> None:
        with self.lock:
            self.tasks_started += 1

    def mark_task_finished(self) -> None:
        with self.lock:
            self.tasks_finished += 1

    def add_scan(self, pages: int, objects: int) -> None:
        with self.lock:
            self.s3_pages_scanned += pages
            self.s3_objects_scanned += objects

    def snapshot(self) -> tuple[int, int, int, int]:
        with self.lock:
            return (
                self.tasks_started,
                self.tasks_finished,
                self.s3_pages_scanned,
                self.s3_objects_scanned,
            )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    body = uri[len("s3://") :]
    parts = body.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix.rstrip("/")


def extract_task_name(line: str) -> str | None:
    token = line.strip().rstrip("/")
    if not token:
        return None
    if TASK_NAME_RE.match(token):
        return token
    if token.startswith("s3://"):
        # Example:
        # s3://.../tasks/<TaskName>/... or s3://.../<TaskName>/...
        parts = token.split("/")
        if "tasks" in parts:
            idx = parts.index("tasks")
            if idx + 1 < len(parts):
                candidate = parts[idx + 1]
                if TASK_LIKE_RE.match(candidate):
                    return candidate
        # Common dataset layout: .../<TaskName>/(frames|shards|episodes|...)
        leaf = parts[-1]
        if leaf in {"frames", "shards", "episodes", "manifest.jsonl", "stats.json", "processing_metadata.json"}:
            parent = parts[-2] if len(parts) >= 2 else ""
            if TASK_LIKE_RE.match(parent):
                return parent

        # Fallback: first task-like segment.
        for segment in parts:
            if TASK_LIKE_RE.match(segment):
                return segment
    return None


def list_common_prefixes(client: Any, bucket: str, prefix: str) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    out: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            full = p.get("Prefix", "")
            if full.endswith("/"):
                full = full[:-1]
            task = full.rsplit("/", 1)[-1]
            if TASK_NAME_RE.match(task):
                out.append(task)
    return sorted(set(out))


def has_objects_with_prefix(client: Any, bucket: str, prefix: str) -> bool:
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return bool(resp.get("KeyCount", 0))


def filter_dataset_like_tasks(client: Any, bucket: str, base_prefix: str, tasks: list[str]) -> list[str]:
    """Keep only prefixes that look like dataset task folders."""
    filtered: list[str] = []
    for task in tasks:
        task_prefix = f"{base_prefix}/{task}" if base_prefix else task
        if has_objects_with_prefix(client, bucket, f"{task_prefix}/frames/") or has_objects_with_prefix(
            client, bucket, f"{task_prefix}/shards/"
        ):
            filtered.append(task)
    return filtered


def iter_objects(client: Any, bucket: str, prefix: str):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key:
                yield key


def count_objects(
    client: Any,
    bucket: str,
    prefix: str,
    suffix: str | None = None,
    runtime_progress: RuntimeProgress | None = None,
) -> int:
    paginator = client.get_paginator("list_objects_v2")
    total = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        if suffix is None:
            total += len(contents)
        else:
            total += sum(1 for obj in contents if str(obj.get("Key", "")).endswith(suffix))
        if runtime_progress:
            runtime_progress.add_scan(pages=1, objects=len(contents))
    return total


def get_json_or_none(client: Any, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except ClientError as e:
        err = e.response.get("Error", {}).get("Code", "")
        if err in {"NoSuchKey", "404"}:
            return None
        raise


def get_manifest_entries_or_none(client: Any, bucket: str, key: str) -> list[dict[str, Any]] | None:
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        lines = resp["Body"].read().decode("utf-8").splitlines()
        entries: list[dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                entries.append(row)
        return entries
    except ClientError as e:
        err = e.response.get("Error", {}).get("Code", "")
        if err in {"NoSuchKey", "404"}:
            return None
        raise


def recursive_max_for_count_fields(value: Any) -> int:
    best = 0
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (int, float)) and "count" in k.lower():
                best = max(best, int(v))
            if isinstance(v, list) and "count" in k.lower():
                numeric = [int(x) for x in v if isinstance(x, (int, float))]
                if numeric:
                    best = max(best, max(numeric))
            best = max(best, recursive_max_for_count_fields(v))
    elif isinstance(value, list):
        for item in value:
            best = max(best, recursive_max_for_count_fields(item))
    return best


@dataclass
class ExtractedSettings:
    image_resize_dim: tuple[int, int] | None = None
    past_steps: int | None = None
    future_steps: int | None = None
    camera_names: tuple[str, ...] | None = None
    image_indices: tuple[int, ...] | None = None
    image_resizing_method: str | None = None
    jpeg_quality: int | None = None


def extract_settings(metadata: dict[str, Any] | None) -> ExtractedSettings:
    """Extract processing settings from processing_metadata.json."""
    s = ExtractedSettings()

    if metadata is None:
        return s

    # Try to find settings in command_line.arguments section (new format)
    command_line = metadata.get("command_line", {})
    if isinstance(command_line, dict):
        arguments = command_line.get("arguments", {})
        if isinstance(arguments, dict):
            # Image resize dimensions
            for key in ["resize_images_size", "image_resize_dim", "image_resize_size", "resize_size", "image_size"]:
                if key in arguments:
                    val = arguments[key]
                    if isinstance(val, (list, tuple)) and len(val) >= 2:
                        with suppress(ValueError, TypeError):
                            s.image_resize_dim = (int(val[0]), int(val[1]))
                            break

            # Past/future steps
            if "past_lowdim_steps" in arguments:
                val = arguments["past_lowdim_steps"]
                if isinstance(val, (int, float)):
                    with suppress(ValueError, TypeError):
                        s.past_steps = int(val)

            if "future_lowdim_steps" in arguments:
                val = arguments["future_lowdim_steps"]
                if isinstance(val, (int, float)):
                    with suppress(ValueError, TypeError):
                        s.future_steps = int(val)

            # Camera names
            if "camera_names" in arguments:
                val = arguments["camera_names"]
                if isinstance(val, (list, tuple)) and all(isinstance(v, str) for v in val):
                    s.camera_names = tuple(sorted(val))

            # Image indices
            if "image_indices" in arguments:
                val = arguments["image_indices"]
                if isinstance(val, (list, tuple)) and all(isinstance(v, (int, float)) for v in val):
                    s.image_indices = tuple(int(v) for v in val)

            # Image resizing method
            for key in ["image_resizing_method", "resize_method"]:
                if key in arguments:
                    val = arguments[key]
                    if isinstance(val, str):
                        s.image_resizing_method = val
                        break

            # JPEG quality
            if "jpeg_quality" in arguments:
                val = arguments["jpeg_quality"]
                if isinstance(val, (int, float)):
                    with suppress(ValueError, TypeError):
                        s.jpeg_quality = int(val)

    # Fallback: try processing section (old format)
    if s.image_resize_dim is None:
        processing = metadata.get("processing", {})
        if isinstance(processing, dict):
            for key in ["image_resize_dim", "image_resize_size", "resize_size", "image_size"]:
                if key in processing:
                    val = processing[key]
                    if isinstance(val, (list, tuple)) and len(val) >= 2:
                        with suppress(ValueError, TypeError):
                            s.image_resize_dim = (int(val[0]), int(val[1]))
                            break

    return s


def verify_task(
    client: Any,
    bucket: str,
    base_prefix: str,
    task: str,
    runtime_progress: RuntimeProgress | None = None,
    count_source: str = "auto",
) -> TaskResult:
    if runtime_progress:
        runtime_progress.mark_task_started()
    task_prefix = f"{base_prefix}/{task}" if base_prefix else task
    frames_prefix = f"{task_prefix}/frames/"
    shards_prefix = f"{task_prefix}/shards/"

    reasons: list[str] = []
    settings_issues: list[str] = []

    episode_files = 0
    shards = 0
    used_manifest = False

    manifest_entries = None
    if count_source in {"auto", "manifest"}:
        manifest_entries = get_manifest_entries_or_none(client, bucket, f"{shards_prefix}manifest.jsonl")
        if manifest_entries is not None:
            used_manifest = True
            shards = len(manifest_entries)
            episode_files = sum(
                int(entry.get("num_sequences", 0))
                for entry in manifest_entries
                if isinstance(entry.get("num_sequences"), (int, float))
            )

    if not used_manifest:
        if count_source == "manifest":
            reasons.append("manifest.jsonl missing")
        # Slow fallback: scan all objects under frames/ and shards/.
        episode_files = count_objects(client, bucket, frames_prefix, runtime_progress=runtime_progress)
        shards = count_objects(client, bucket, shards_prefix, suffix=".tar", runtime_progress=runtime_progress)

    stats = get_json_or_none(client, bucket, f"{shards_prefix}stats.json")
    if stats is None:
        reasons.append("stats.json missing")
        stats_max = 0
    else:
        stats_max = recursive_max_for_count_fields(stats)

    metadata = get_json_or_none(client, bucket, f"{shards_prefix}processing_metadata.json")
    if metadata is None:
        reasons.append("processing_metadata.json missing")

    samples = episode_files
    if isinstance(metadata, dict):
        processing = metadata.get("processing")
        if isinstance(processing, dict):
            total = processing.get("total_samples_created")
            if isinstance(total, (int, float)):
                samples = int(total)

    # Extract settings for coherence checking
    settings = extract_settings(metadata)

    missing_from_stats = max(0, episode_files - stats_max)
    if missing_from_stats > 0:
        reasons.append(f"missing from stats: {missing_from_stats} (episodes={episode_files}, stats_max={stats_max})")

    status = "ok" if not reasons else "failed"
    result = TaskResult(
        task=task,
        status=status,
        shards=shards,
        samples=samples,
        episode_files=episode_files,
        stats_max=stats_max,
        missing_from_stats=missing_from_stats,
        reasons=reasons,
        image_resize_dim=settings.image_resize_dim,
        settings_issues=settings_issues,
        past_steps=settings.past_steps,
        future_steps=settings.future_steps,
        camera_names=settings.camera_names,
        image_indices=settings.image_indices,
        image_resizing_method=settings.image_resizing_method,
        jpeg_quality=settings.jpeg_quality,
    )
    if runtime_progress:
        runtime_progress.mark_task_finished()
    return result


def format_task_line(r: TaskResult) -> str:
    if r.status == "missing":
        return f"  {r.task}: NOT PROCESSED"

    settings_str = ""
    parts = []
    if r.image_resize_dim is not None:
        parts.append(f"img_resize={r.image_resize_dim}")
    if r.past_steps is not None or r.future_steps is not None:
        past = r.past_steps or "?"
        future = r.future_steps or "?"
        parts.append(f"seq=[{past}past,{future}future]")
    if r.camera_names is not None:
        parts.append(f"cams={len(r.camera_names)}")
    if parts:
        settings_str = " | " + ", ".join(parts)

    if r.status == "ok":
        return (
            f"  {r.task}: {r.shards} shards, {r.samples} samples, {r.episode_files} episode files, "
            f"stats_max={r.stats_max}, missing_from_stats={r.missing_from_stats}{settings_str}"
        )

    reason = "; ".join(r.reasons or ["unknown failure"])
    return (
        f"  {r.task}: {reason} | episode_files={r.episode_files}, "
        f"stats_max={r.stats_max}, missing_from_stats={r.missing_from_stats}{settings_str}"
    )


def main() -> None:
    def _handle_sigint(_signum, _frame):
        global INTERRUPT_COUNT
        INTERRUPT_COUNT += 1
        if INTERRUPT_COUNT == 1:
            STOP_EVENT.set()
            print("\nInterrupt received: stopping workers and cancelling pending tasks...")
        else:
            print("\nSecond interrupt received: forcing immediate exit.")
            os._exit(130)

    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", required=True, help="S3 path like s3://bucket/prefix")
    parser.add_argument(
        "--tasks-file",
        type=str,
        default=None,
        help="Optional newline-separated expected tasks. "
        "If provided, tasks missing in S3 are reported as NOT PROCESSED. "
        "Otherwise, tasks discovered in S3 are verified.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print verification progress every N completed tasks (default: 10).",
    )
    parser.add_argument(
        "--failed-tasks-file",
        type=str,
        default=None,
        help="Path to write failed/missing task names (one per line) for easy re-run.",
    )
    parser.add_argument(
        "--count-source",
        choices=["auto", "manifest", "frames"],
        default="auto",
        help=(
            "How to count generated samples/episode files: "
            "'auto' uses shards/manifest.jsonl when available (fast), "
            "'manifest' requires manifest.jsonl, "
            "'frames' scans frames/ objects directly (slow)."
        ),
    )
    args = parser.parse_args()

    bucket, base_prefix = parse_s3_uri(args.base_path.rstrip("/"))
    s3_config = Config(
        max_pool_connections=max(32, args.workers * 4),
        connect_timeout=5,
        read_timeout=30,
        retries={"mode": "adaptive", "max_attempts": 8},
    )
    client = boto3.client("s3", config=s3_config)

    try:
        existing_tasks = set(list_common_prefixes(client, bucket, f"{base_prefix}/" if base_prefix else ""))
    except EndpointConnectionError as e:
        raise SystemExit(
            "Failed to reach AWS endpoints while listing tasks. "
            "Check network connectivity/AWS auth setup and retry.\n"
            f"Details: {e}"
        ) from e

    # Avoid checking non-dataset prefixes when auto-discovering tasks.
    existing_tasks = set(filter_dataset_like_tasks(client, bucket, base_prefix, sorted(existing_tasks)))

    expected_tasks: list[str]
    if args.tasks_file:
        with open(args.tasks_file, encoding="utf-8") as f:
            parsed = [extract_task_name(line) for line in f]
            expected_tasks = sorted({task for task in parsed if task})
    else:
        expected_tasks = sorted(existing_tasks)

    print(f"Base path: {args.base_path.rstrip('/')}")
    print(f"Tasks to verify: {len(expected_tasks)}")
    print(f"Count source: {args.count_source}")
    print()

    missing_tasks = sorted(set(expected_tasks) - existing_tasks)
    to_check = sorted(set(expected_tasks) & existing_tasks)

    results_by_task: dict[str, TaskResult] = {}
    progress_every = max(1, args.progress_every)
    started = time.monotonic()
    last_progress = started
    total_to_check = len(to_check)
    runtime_progress = RuntimeProgress(lock=Lock())
    print(f"Starting verification of {total_to_check} discovered tasks with {max(1, args.workers)} workers...")
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    try:
        futs = {
            pool.submit(verify_task, client, bucket, base_prefix, task, runtime_progress, args.count_source): task
            for task in to_check
        }
        pending = set(futs.keys())
        completed = 0

        while pending:
            if STOP_EVENT.is_set():
                for fut in pending:
                    fut.cancel()
                break

            done, pending = wait(pending, timeout=5.0, return_when=FIRST_COMPLETED)

            for fut in done:
                task = futs[fut]
                try:
                    results_by_task[task] = fut.result()
                except (ClientError, BotoCoreError, ValueError) as e:
                    results_by_task[task] = TaskResult(task=task, status="failed", reasons=[f"error: {e}"])
                completed += 1

            now = time.monotonic()
            if completed % progress_every == 0 or completed == total_to_check or now - last_progress >= 10:
                elapsed = now - started
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (total_to_check - completed) / rate if rate > 0 else 0.0
                t_started, t_finished, pages_scanned, objects_scanned = runtime_progress.snapshot()
                print(
                    f"[progress] {completed}/{total_to_check} tasks done | "
                    f"in-flight={len(pending)} | started={t_started} finished={t_finished} | "
                    f"s3_pages={pages_scanned} objects={objects_scanned} | "
                    f"elapsed={elapsed:.1f}s | eta~{eta:.1f}s"
                )
                last_progress = now
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    if STOP_EVENT.is_set():
        print("Verification interrupted by user. Exiting.")
        raise SystemExit(130)

    for task in expected_tasks:
        if task in missing_tasks:
            result = TaskResult(task=task, status="missing", reasons=["task not found under base path"])
        else:
            result = results_by_task[task]
        print(format_task_line(result))

    successful = 0
    failed = 0
    missing = 0
    total_shards = 0
    total_samples = 0
    total_missing_from_stats = 0
    failed_tasks: list[TaskResult] = []

    # Collect settings for coherence checking
    image_resize_dims: dict[tuple[int, int], list[str]] = {}
    past_steps_map: dict[int, list[str]] = {}
    future_steps_map: dict[int, list[str]] = {}
    camera_names_map: dict[tuple[str, ...], list[str]] = {}
    image_indices_map: dict[tuple[int, ...], list[str]] = {}
    image_resizing_method_map: dict[str, list[str]] = {}
    jpeg_quality_map: dict[int, list[str]] = {}

    for task in expected_tasks:
        if task in missing_tasks:
            missing += 1
            continue
        r = results_by_task[task]
        total_shards += r.shards
        total_samples += r.samples
        total_missing_from_stats += r.missing_from_stats

        # Track settings
        if r.image_resize_dim is not None:
            image_resize_dims.setdefault(r.image_resize_dim, []).append(task)
        if r.past_steps is not None:
            past_steps_map.setdefault(r.past_steps, []).append(task)
        if r.future_steps is not None:
            future_steps_map.setdefault(r.future_steps, []).append(task)
        if r.camera_names is not None:
            camera_names_map.setdefault(r.camera_names, []).append(task)
        if r.image_indices is not None:
            image_indices_map.setdefault(r.image_indices, []).append(task)
        if r.image_resizing_method is not None:
            image_resizing_method_map.setdefault(r.image_resizing_method, []).append(task)
        if r.jpeg_quality is not None:
            jpeg_quality_map.setdefault(r.jpeg_quality, []).append(task)

        if r.status == "ok":
            successful += 1
        else:
            failed += 1
            failed_tasks.append(r)

    print()
    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"  ✅ Successful: {successful}/{len(expected_tasks)} tasks")
    print(f"  ❌ Failed:     {failed}/{len(expected_tasks)} tasks")
    print(f"  ⬜ Missing:    {missing}/{len(expected_tasks)} tasks")
    print()
    print(f"  Total shards:  {total_shards}")
    print(f"  Total samples: {total_samples}")
    print(f"  Missing from stats (episodes - stats_max_count): {total_missing_from_stats}")

    # Print settings coherence report
    print()
    print("=" * 60)
    print("⚙️  SETTINGS COHERENCE")
    print("=" * 60)

    def _print_coherence(label: str, value_map: dict, sort_key=None):
        if not value_map:
            print(f"  ℹ️  {label}: NOT FOUND in metadata")
            return
        if len(value_map) == 1:
            val = list(value_map.keys())[0]
            print(f"  ✅ {label}: COHERENT - all {len(value_map[val])} tasks use {val}")
        else:
            print(f"  ⚠️  {label}: INCOHERENT - {len(value_map)} different values found:")
            keys = sorted(value_map.keys(), key=sort_key) if sort_key else sorted(value_map.keys())
            for val in keys:
                tasks = value_map[val]
                print(
                    f"      {val}: {len(tasks)} task(s) - {', '.join(sorted(tasks)[:3])}"
                    + (f" ... ({len(tasks) - 3} more)" if len(tasks) > 3 else "")
                )

    _print_coherence("Image resize dimension", image_resize_dims)
    _print_coherence("Past steps", past_steps_map)
    _print_coherence("Future steps", future_steps_map)
    _print_coherence("Image indices", image_indices_map)
    _print_coherence("Image resizing method", image_resizing_method_map)
    _print_coherence("JPEG quality", jpeg_quality_map)

    # Camera names: show per-set counts and common subset
    if camera_names_map:
        if len(camera_names_map) == 1:
            cams = list(camera_names_map.keys())[0]
            print(f"  ✅ Camera names: COHERENT - all {len(camera_names_map[cams])} tasks use {list(cams)}")
        else:
            total_tasks_with_cams = sum(len(t) for t in camera_names_map.values())
            print(f"  ℹ️  Camera names: {len(camera_names_map)} different sets across {total_tasks_with_cams} tasks:")
            for cams in sorted(camera_names_map.keys(), key=lambda c: (-len(camera_names_map[c]), c)):
                tasks = camera_names_map[cams]
                print(
                    f"      {list(cams)}: {len(tasks)} task(s) - {', '.join(sorted(tasks)[:3])}"
                    + (f" ... ({len(tasks) - 3} more)" if len(tasks) > 3 else "")
                )
            # Common subset
            all_cam_sets = [set(cams) for cams in camera_names_map]
            common = sorted(set.intersection(*all_cam_sets)) if all_cam_sets else []
            if common:
                print(f"      Common subset ({len(common)} cameras): {common}")
            else:
                print("      Common subset: NONE")
    else:
        print("  ℹ️  Camera names: NOT FOUND in metadata")

    if failed_tasks:
        print()
        print("Failed tasks:")
        for r in sorted(failed_tasks, key=lambda x: x.task):
            print(f"  - {r.task}: {'; '.join(r.reasons or ['unknown failure'])}")

    if missing_tasks:
        print()
        print("Missing tasks:")
        for task in missing_tasks:
            print(f"  - {task}")

    if args.failed_tasks_file:
        all_failed_names = sorted([r.task for r in failed_tasks] + list(missing_tasks))
        os.makedirs(os.path.dirname(os.path.abspath(args.failed_tasks_file)), exist_ok=True)
        with open(args.failed_tasks_file, "w", encoding="utf-8") as f:
            for task_name in all_failed_names:
                f.write(task_name + "\n")
        print(f"\nWrote {len(all_failed_names)} failed/missing task names to {args.failed_tasks_file}")


if __name__ == "__main__":
    main()
