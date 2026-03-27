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
    """Infer model name from directory structure."""
    rel = results_path.relative_to(root)
    # ("Task", "model", "results-xxx.json") or ("Task", "results-xxx.json")
    if len(rel.parts) >= 3:
        return rel.parts[-2]
    return "default"


def load_episodes(root: Path) -> list[dict]:
    """Parse all ``results-*.json`` under *root* into a flat episode list.

    Each episode dict has keys: ``task``, ``model``, ``demo_id``, ``success``,
    ``duration``, ``recording``, ``failure``.
    """
    all_result_files = sorted(root.rglob("results-*.json"))
    if not all_result_files:
        logger.warning("NO RESULTS FILES FOUND under %s — is the path correct?", root)
        return []

    # Keep only the most recent results file per directory. Re-running
    # evaluation creates a new timestamped file; older ones are stale.
    files_by_dir: dict[Path, list[Path]] = {}
    for rj in all_result_files:
        files_by_dir.setdefault(rj.parent, []).append(rj)

    result_files = []
    stale_info: list[dict] = []  # [{dir, kept, skipped}]
    for d, files in files_by_dir.items():
        files_sorted = sorted(files, key=lambda p: p.name)
        result_files.append(files_sorted[-1])
        if len(files_sorted) > 1:
            stale_info.append(
                {
                    "dir": str(d.relative_to(root)),
                    "kept": files_sorted[-1].name,
                    "skipped": [f.name for f in files_sorted[:-1]],
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
    return episodes, dict(pending_by), dict(crashed_by), stale_info


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
