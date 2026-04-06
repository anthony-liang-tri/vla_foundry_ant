"""Filesystem scanning and aggregation for rollout results.

Handles both directory layouts:
- ``rollouts/{task}/{model}/results-*.json``
- ``rollouts/{task}/results-*.json``  (model defaults to ``"default"``)

No external dependencies beyond the standard library.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def find_recordings(task_dir: Path, skill_name: str, demo_id: int) -> dict[str, str | None]:
    """Return absolute paths to recording files (video and/or HTML), or *None* for each."""
    video = None
    html = None
    demo_dir = task_dir / skill_name / f"demonstration_{demo_id}"
    if demo_dir.is_dir():
        for v in demo_dir.glob("*.mp4"):
            video = str(v.resolve())
            break
        rec = demo_dir / "recording.html"
        if rec.exists():
            html = str(rec.resolve())
    return {"video": video, "html": html}


def detect_model(root: Path, results_path: Path) -> str:
    """Infer model name from directory structure.

    Handles both layouts:
    - ``Task/model/results-xxx.json`` → model is parts[-2]
    - ``Task/model/timestamp/results.json`` → model is parts[-3]
    - ``Task/results-xxx.json`` → "default"
    """
    rel = results_path.relative_to(root)
    # New layout: Task/model/timestamp/results.json (4 parts)
    if len(rel.parts) >= 4:
        return rel.parts[-3]
    # Old layout: Task/model/results-xxx.json (3 parts)
    if len(rel.parts) >= 3:
        return rel.parts[-2]
    return "default"


def _try_combine_files(
    directory: Path,
    files: list[Path],
    root: Path,
) -> tuple[str, bool]:
    """Check if multiple ``results-*.json`` files in a directory can be combined.

    Validates that all files share the same ``max_sample_size_per_model``
    (or all omit it) and that no ``(skill_type, scenario_index)`` pairs overlap.

    Returns ``(reason, can_combine)``; *reason* explains failures.
    """
    mss_values: set = set()
    seen_keys: set[tuple[str, int]] = set()

    for f in files:
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as e:
            return f"Cannot read {f.name}: {e}", False

        mss_values.add(data.get("max_sample_size_per_model"))

        for ep in data.get("evaluations", []):
            skill = ep.get("skill_type", "unknown")
            idx = ep.get("scenario_index", 0)
            key = (skill, idx)
            if key in seen_keys:
                return (
                    f"Overlapping episode ({skill}, index={idx}) across files in {directory.relative_to(root)}"
                ), False
            seen_keys.add(key)

    if len(mss_values) > 1:
        return (f"Mismatched max_sample_size_per_model values {mss_values} in {directory.relative_to(root)}"), False

    return "", True


def load_episodes(root: Path):
    """Parse all ``results-*.json`` under *root* into a flat episode list.

    When a directory contains multiple result files, they are combined if
    they share the same ``max_sample_size_per_model`` and have no overlapping
    ``(skill_type, scenario_index)`` pairs.  Otherwise only the newest file
    is kept (older ones are treated as stale).

    ``max_sample_size_per_model`` is read from inside each results JSON
    (injected by ``run_evaluation.py`` as soon as the file is created).

    Returns ``(episodes, pending_by, crashed_by, stale_info, max_sample_size_per_model)``.
    """
    # Support both naming conventions:
    #   - results-TIMESTAMP.json  (old Docker images)
    #   - TIMESTAMP/results.json  (new Docker images)
    all_result_files = sorted(set(root.rglob("results-*.json")) | set(root.rglob("results.json")))
    if not all_result_files:
        logger.warning("NO RESULTS FILES FOUND under %s — is the path correct?", root)
        return [], {}, {}, [], None

    files_by_dir: dict[Path, list[Path]] = {}
    for rj in all_result_files:
        files_by_dir.setdefault(rj.parent, []).append(rj)

    result_files: list[Path] = []
    stale_info: list[dict] = []
    max_sample_sizes_seen: set[int | None] = set()

    for d, files in files_by_dir.items():
        files_sorted = sorted(files, key=lambda p: p.name)

        if len(files_sorted) == 1:
            result_files.append(files_sorted[0])
            data = json.loads(files_sorted[0].read_text())
            max_sample_sizes_seen.add(data.get("max_sample_size_per_model"))
            continue

        # Multiple files — attempt to combine
        reason, can_combine = _try_combine_files(d, files_sorted, root)
        if can_combine:
            result_files.extend(files_sorted)
            for f in files_sorted:
                data = json.loads(f.read_text())
                max_sample_sizes_seen.add(data.get("max_sample_size_per_model"))
            logger.info(
                "Combined %d results files in %s",
                len(files_sorted),
                d.relative_to(root),
            )
        else:
            # Fallback: keep newest only
            result_files.append(files_sorted[-1])
            data = json.loads(files_sorted[-1].read_text())
            max_sample_sizes_seen.add(data.get("max_sample_size_per_model"))
            stale_info.append(
                {
                    "dir": str(d.relative_to(root)),
                    "kept": files_sorted[-1].name,
                    "skipped": [f.name for f in files_sorted[:-1]],
                    "reason": reason,
                }
            )

    if stale_info:
        details = "; ".join(f"{s['dir']}: using {s['kept']}, ignoring {', '.join(s['skipped'])}" for s in stale_info)
        logger.warning(
            "Found stale results file(s) from previous runs. "
            "Video recordings for older runs may have been overwritten. %s",
            details,
        )

    episodes: list[dict] = []
    pending_by: Counter[tuple[str, str]] = Counter()
    crashed_by: Counter[tuple[str, str]] = Counter()
    for rj in result_files:
        with open(rj) as f:
            data = json.load(f)
        task_dir = rj.parent
        model = detect_model(root, rj)
        # Each evaluation entry has an "is_success" bool set by the simulator
        # based on task-specific success criteria (e.g. object reached target
        # pose within tolerance). Episodes without is_success are skipped
        # (pending or crashed) with a warning.
        for ep in data.get("evaluations", []):
            skill = ep.get("skill_type", "unknown")
            if ep.get("is_success") is None:
                if ep.get("is_pending"):
                    pending_by[(skill, model)] += 1
                else:
                    crashed_by[(skill, model)] += 1
                continue
            demo_id = ep.get("scenario_index", 0)
            recs = find_recordings(task_dir, skill, demo_id)
            episodes.append(
                {
                    "task": skill,
                    "model": model,
                    "demo_id": demo_id,
                    "success": bool(ep.get("is_success", False)),
                    "duration": float(ep.get("total_time") or 0),
                    "recording": recs["video"] or recs["html"],
                    "recording_video": recs["video"],
                    "recording_html": recs["html"],
                    "failure": ep.get("failure_message"),
                }
            )

    if pending_by:
        total = sum(pending_by.values())
        breakdown = ", ".join(f"{task}/{model}: {n}" for (task, model), n in sorted(pending_by.items()))
        logger.info(
            "PENDING %d episode(s) still in progress. Breakdown: %s",
            total,
            breakdown,
        )
    if crashed_by:
        total = sum(crashed_by.values())
        breakdown = ", ".join(f"{task}/{model}: {n}" for (task, model), n in sorted(crashed_by.items()))
        logger.warning(
            "SKIPPED %d episode(s) that crashed (is_success=null, is_pending=false). Breakdown: %s",
            total,
            breakdown,
        )
    logger.info("Loaded %d episodes from %d results file(s)", len(episodes), len(result_files))

    # Derive a single max_sample_size from all results files.
    max_sample_sizes_seen.discard(None)
    if len(max_sample_sizes_seen) == 1:
        global_max_sample_size = max_sample_sizes_seen.pop()
    elif len(max_sample_sizes_seen) == 0:
        global_max_sample_size = None
    else:
        global_max_sample_size = None
        logger.warning(
            "Conflicting max_sample_size_per_model values across directories: %s. Statistical tests will not auto-run.",
            max_sample_sizes_seen,
        )

    return episodes, dict(pending_by), dict(crashed_by), stale_info, global_max_sample_size


def aggregate_episodes(
    episodes: list[dict],
    ci_fn=None,
    pending_by: dict | None = None,
    crashed_by: dict | None = None,
) -> list[dict]:
    """Per-(task, model) summary statistics from a flat episode list.

    Success rate = successes / total episodes for each (task, model) pair.
    *ci_fn*, if provided, is called as ``ci_fn(successes, total)`` and must
    return ``(ci_low, ci_high)``.  Defaults to ``(0.0, 1.0)`` when omitted.
    """

    def _no_ci(s, n):
        return 0.0, 1.0

    if ci_fn is None:
        ci_fn = _no_ci
    pending_by = pending_by or {}
    crashed_by = crashed_by or {}
    totals: dict[tuple[str, str], int] = Counter()
    successes: dict[tuple[str, str], int] = Counter()
    durations: dict[tuple[str, str], list[float]] = {}
    for ep in episodes:
        key = (ep["task"], ep["model"])
        totals[key] += 1
        successes[key] += ep["success"]
        durations.setdefault(key, []).append(ep["duration"])

    # Include all keys — even tasks with only pending/crashed episodes
    all_keys = sorted(set(totals) | set(pending_by) | set(crashed_by))
    stats: list[dict] = []
    for key in all_keys:
        task, model = key
        n, s = totals.get(key, 0), successes.get(key, 0)
        lo, hi = ci_fn(s, n)
        stats.append(
            {
                "task": task,
                "model": model,
                "total": n,
                "successes": s,
                "pct": 100.0 * s / n if n else 0,
                "avg_dur": sum(durations.get(key, [0])) / max(n, 1),
                "ci_low": lo,
                "ci_high": hi,
                "pending": pending_by.get(key, 0),
                "crashed": crashed_by.get(key, 0),
            }
        )
    return stats
