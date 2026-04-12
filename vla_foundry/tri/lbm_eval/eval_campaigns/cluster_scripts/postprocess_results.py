#!/usr/bin/env python3
"""Post-process evaluation results before S3 upload.

Combines two operations that run after episodes complete:

1. convert  — Generate summary.yaml from OSS results-*.json (idempotent).
2. stamp    — Append eval provenance metadata to summary.yaml and results-*.json.

Usage (from run_inference_bundle.sh):
    python3 postprocess_results.py <save_dir>

Both operations run in sequence. Convert is a no-op if no results-*.json exist
(standard sim). Stamp reads eval config from environment variables set by the
campaign runner.
"""

import json
import os
import pathlib
import sys
from datetime import UTC, datetime


def convert_oss_results(save_dir: pathlib.Path) -> int:
    """Generate summary.yaml from OSS results-*.json files.

    The OSS simulator writes per-worker results-*.json with episode outcomes.
    This creates demonstration_*/summary.yaml files that existing tooling
    (gather_results.py, compute_success_rates, etc.) expects.

    Only creates summary.yaml if it doesn't already exist (idempotent).
    """
    created = 0
    for rj in sorted(save_dir.rglob("results-*.json")):
        with open(rj) as fh:
            data = json.load(fh)
        for ep in data.get("evaluations", []):
            skill = ep.get("skill_type", "unknown")
            idx = ep.get("scenario_index", 0)
            demo_dir = rj.parent / skill / f"demonstration_{idx}"
            summary_path = demo_dir / "summary.yaml"
            if demo_dir.exists() and not summary_path.exists():
                summary = {
                    "success": bool(ep.get("is_success", False)),
                    "last_t": float(ep.get("total_time", 0)),
                    "index": idx,
                    "episode_type": "Rollout",
                }
                with open(summary_path, "w") as f:
                    for k, v in summary.items():
                        if isinstance(v, bool):
                            f.write(f"{k}: {str(v).lower()}\n")
                        else:
                            f.write(f"{k}: {v}\n")
                created += 1
    return created


def _get_eval_metadata() -> dict[str, str]:
    """Collect eval provenance from environment variables."""
    return {
        "vla_foundry_git_sha": os.environ.get("VLA_FOUNDRY_GIT_SHA", "unknown"),
        "vla_foundry_git_branch": os.environ.get("VLA_FOUNDRY_GIT_BRANCH", "unknown"),
        "checkpoint_s3": os.environ.get("CHECKPOINT_DIR", "unknown"),
        "docker_image": os.environ.get("DOCKER_IMAGE", "unknown"),
        "job_name": os.environ.get("JOB_NAME", "unknown"),
        "num_flow_steps": os.environ.get("NUM_FLOW_STEPS", ""),
        "open_loop_steps": os.environ.get("OPEN_LOOP_STEPS", ""),
        "evaluation_subfolder": os.environ.get("LAUNCH_EVALUATION_SUBFOLDER", ""),
        "task_name": os.environ.get("LAUNCH_TASK_NAME", ""),
        "eval_timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def stamp_provenance(save_dir: pathlib.Path) -> tuple[int, int]:
    """Stamp eval metadata into summary.yaml and results-*.json files.

    Appends an eval_metadata block to each summary.yaml and adds an
    eval_metadata key to each results-*.json. Idempotent — skips files
    that already contain eval_metadata.

    Returns (summary_count, json_count) of files stamped.
    """
    metadata = _get_eval_metadata()

    summary_count = 0
    for f in save_dir.rglob("summary.yaml"):
        content = f.read_text()
        if "eval_metadata:" in content:
            continue
        with open(f, "a") as fh:
            fh.write("eval_metadata:\n")
            for k, v in metadata.items():
                fh.write(f"  {k}: {v}\n")
        summary_count += 1

    json_count = 0
    for f in save_dir.rglob("results-*.json"):
        with open(f) as fh:
            data = json.load(fh)
        if "eval_metadata" in data:
            continue
        data["eval_metadata"] = metadata
        with open(f, "w") as fh:
            json.dump(data, fh, indent=2)
        json_count += 1

    return summary_count, json_count


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <save_dir>", file=sys.stderr)
        sys.exit(1)

    save_dir = pathlib.Path(sys.argv[1])
    if not save_dir.exists():
        print(f"Directory not found: {save_dir}", file=sys.stderr)
        sys.exit(1)

    # Convert OSS results (no-op if no results-*.json exist)
    n_converted = convert_oss_results(save_dir)
    if n_converted:
        print(f"  Converted {n_converted} OSS episodes to summary.yaml")

    # Stamp eval provenance into all result files
    n_summary, n_json = stamp_provenance(save_dir)
    print(f"  Stamped {n_summary} summary.yaml, {n_json} results-*.json")


if __name__ == "__main__":
    main()
