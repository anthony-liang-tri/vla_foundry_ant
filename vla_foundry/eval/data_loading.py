"""Filesystem scanning and aggregation for rollout results.

Expected directory layout::

    root/
      {model}/
        {Task}/
          rollouts/
            {timestamp}/
              results.json
              {skill_name}/
                demonstration_{N}/
                  ...

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
    """Infer model name from ``{model}/{Task}/rollouts/{timestamp}/results.json``."""
    rel = results_path.relative_to(root)
    if len(rel.parts) != 5 or rel.name != "results.json" or rel.parts[2] != "rollouts":
        raise ValueError(
            f"Unexpected results path layout: {rel}. Expected {{model}}/{{Task}}/rollouts/{{timestamp}}/results.json"
        )
    return rel.parts[0]


def _try_combine_files(
    directory: Path,
    files: list[Path],
    root: Path,
) -> tuple[str, bool]:
    """Check if multiple result files under a model directory can be combined.

    Validates that all files share the same ``max_sample_size_per_model``
    and that no ``(skill_type, scenario_index)`` pairs overlap.

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
    """Parse all ``results.json`` under *root* into a flat episode list.

    Expected layout: ``{model}/{Task}/rollouts/{timestamp}/results.json``.

    When a rollouts directory contains multiple timestamped result files,
    they are combined if they share the same ``max_sample_size_per_model``
    and have no overlapping ``(skill_type, scenario_index)`` pairs.  Otherwise
    a ``ValueError`` is raised — remove stale results files before loading.

    Returns ``(episodes, pending_by, crashed_by, max_sample_size_per_model)``.
    """
    all_result_files = sorted(root.rglob("results.json"))
    if not all_result_files:
        logger.warning("NO RESULTS FILES FOUND under %s — is the path correct?", root)
        return [], {}, {}, None

    # Validate layout and reject old-style results-TIMESTAMP.json files.
    old_style = sorted(root.rglob("results-*.json"))
    if old_style:
        raise ValueError(
            f"Found {len(old_style)} old-style results-*.json file(s) under {root}. "
            f"Only the {{model}}/{{Task}}/rollouts/{{timestamp}}/results.json layout is supported. "
            f"First offender: {old_style[0]}"
        )

    for rj in all_result_files:
        rel = rj.relative_to(root)
        if len(rel.parts) != 5 or rel.parts[2] != "rollouts":
            raise ValueError(
                f"Unexpected results path layout: {rel}. "
                f"Expected {{model}}/{{Task}}/rollouts/{{timestamp}}/results.json"
            )

    # Validate that every results.json contains max_sample_size_per_model.
    for rj in all_result_files:
        data = json.loads(rj.read_text())
        if "max_sample_size_per_model" not in data:
            raise ValueError(
                f"Missing required field 'max_sample_size_per_model' in {rj.relative_to(root)}. "
                f"Every results.json must record the evaluation budget."
            )

    # Group result files by rollouts directory (grandparent of results.json)
    # so that multiple timestamped runs for the same model+task are compared
    # for overlapping episodes.
    files_by_dir: dict[Path, list[Path]] = {}
    for rj in all_result_files:
        rollouts_dir = rj.parent.parent  # {model}/{Task}/rollouts
        files_by_dir.setdefault(rollouts_dir, []).append(rj)

    result_files: list[Path] = []
    max_sample_sizes_seen: set[int] = set()

    for d, files in files_by_dir.items():
        files_sorted = sorted(files)

        if len(files_sorted) == 1:
            result_files.append(files_sorted[0])
            data = json.loads(files_sorted[0].read_text())
            max_sample_sizes_seen.add(data["max_sample_size_per_model"])
            continue

        # Multiple files — attempt to combine
        reason, can_combine = _try_combine_files(d, files_sorted, root)
        if can_combine:
            result_files.extend(files_sorted)
            for f in files_sorted:
                data = json.loads(f.read_text())
                max_sample_sizes_seen.add(data["max_sample_size_per_model"])
            logger.info(
                "Combined %d results files in %s",
                len(files_sorted),
                d.relative_to(root),
            )
        else:
            raise ValueError(
                f"Cannot combine results files in {d.relative_to(root)}: {reason}. "
                f"Video recordings for older runs may have been overwritten. "
                f"Remove stale results files before viewing."
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
    # Sanity check: no duplicate (task, model, demo_id) triples.
    seen_keys: set[tuple[str, str, int]] = set()
    for ep in episodes:
        key = (ep["task"], ep["model"], ep["demo_id"])
        if key in seen_keys:
            logger.error(
                "Duplicate episode detected: task=%s, model=%s, demo_id=%d. "
                "This likely means overlapping result files were loaded from "
                "different timestamped subdirectories. Check your results directory "
                "for redundant runs.",
                *key,
            )
            raise ValueError(
                f"Duplicate episode (task={key[0]}, model={key[1]}, demo_id={key[2]}). "
                f"Remove or rename stale result files under {root}."
            )
        seen_keys.add(key)

    logger.info("Loaded %d episodes from %d results file(s)", len(episodes), len(result_files))

    # All files are required to have max_sample_size_per_model (validated above).
    if len(max_sample_sizes_seen) == 1:
        global_max_sample_size = max_sample_sizes_seen.pop()
    else:
        raise ValueError(
            f"Conflicting max_sample_size_per_model values across results files: {max_sample_sizes_seen}. "
            f"All results must use the same evaluation budget."
        )

    # Verify no (task, model) pair exceeds the declared budget.
    ep_counts: Counter[tuple[str, str]] = Counter()
    for ep in episodes:
        ep_counts[(ep["task"], ep["model"])] += 1
    over_budget = {k: v for k, v in ep_counts.items() if v > global_max_sample_size}
    if over_budget:
        details = ", ".join(f"{t}/{m}: {n}" for (t, m), n in sorted(over_budget.items()))
        raise ValueError(
            f"Episode count exceeds max_sample_size_per_model ({global_max_sample_size}): {details}. "
            f"The sequential test budget must not be exceeded."
        )

    return episodes, dict(pending_by), dict(crashed_by), global_max_sample_size


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
