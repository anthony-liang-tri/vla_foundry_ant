#!/usr/bin/env python3
"""
Leaderboard CLI for DynamoDB Backend

Upload evaluation results directly to the VLA Foundry leaderboard (DynamoDB).

Usage:
    # Upload results
    python leaderboard_cli.py upload task_list.txt --campaign-name my_campaign

    # Upload with overwrite (update existing entries)
    python leaderboard_cli.py upload task_list.txt --campaign-name my_campaign --overwrite

    # Quick upload without S3 downloads (no configs/success rates)
    python leaderboard_cli.py upload task_list.txt --campaign-name my_campaign --skip-s3

    # Upload without requiring config/wandb links
    python leaderboard_cli.py upload task_list.txt --campaign-name my_campaign --no-link

    # Delete entries by filter
    python leaderboard_cli.py delete --ablation continuous_flow

    # Delete entries by campaign and ablation
    python leaderboard_cli.py delete --campaign-name my_campaign --ablation continuous_flow
"""

import argparse
import ast
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import boto3
import yaml

# DynamoDB configuration
REGION = "us-west-2"
LEADERBOARD_RUNS_TABLE = "vla_foundry_leaderboard_runs"
LEADERBOARD_EVALUATIONS_TABLE = "vla_foundry_leaderboard_evaluations"
SAMPLE_RESULTS_TABLE = "vla_foundry_sample_results"


def parse_tasks_file(tasks_file: Path) -> list[tuple]:
    """Parse the tasks file and extract task information.

    Returns tuples of (scenario_name, task_name, s3_path) or
    (scenario_name, task_name, s3_path, vla_ref) if vla_ref is specified.
    """
    with open(tasks_file) as f:
        content = f.read()

    content = content.replace('f"', '"')
    wrapped_content = f"[{content}]"

    try:
        tasks = ast.literal_eval(wrapped_content)
    except Exception as e:
        print(f"ERROR: Failed to parse tasks file: {e}", file=sys.stderr)
        return []

    return tasks


def extract_metadata_from_s3_path(s3_path: str) -> dict:
    """Extract metadata from S3 checkpoint path."""
    metadata = {
        "model_type": None,
        "ablation": None,
        "learning_rate": None,
        "batch_size": None,
        "run_timestamp": None,
    }

    path_parts = s3_path.rstrip("/").split("/")

    if "diffusion_policy" in path_parts:
        metadata["model_type"] = "diffusion_policy"

    if "ablations" in path_parts:
        ablations_idx = path_parts.index("ablations")
        if ablations_idx + 2 < len(path_parts):
            metadata["ablation"] = path_parts[ablations_idx + 2]

    run_dir = path_parts[-1]
    timestamp_match = re.match(r"(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})", run_dir)
    if timestamp_match:
        metadata["run_timestamp"] = timestamp_match.group(1)

    lr_match = re.search(r"-lr_([0-9e.\-]+)", run_dir)
    if lr_match:
        metadata["learning_rate"] = lr_match.group(1)

    bsz_match = re.search(r"-bsz_(\d+)", run_dir)
    if bsz_match:
        metadata["batch_size"] = int(bsz_match.group(1))

    # Auto-generate ablation from timestamp if not found in path
    if metadata["ablation"] is None and metadata["run_timestamp"]:
        metadata["ablation"] = f"run_{metadata['run_timestamp']}"

    return metadata


def _candidate_checkpoint_bases(s3_path: str) -> list[str]:
    """Return candidate S3 base paths to search for evaluation artifacts."""
    raw = s3_path.rstrip("/")
    candidates: list[str] = []

    def add(path: str):
        path = path.rstrip("/")
        if path and path not in candidates:
            candidates.append(path)

    if raw.endswith(".pt") or raw.endswith(".ckpt"):
        parent = raw.rsplit("/", 1)[0]
        add(parent)
        if "/checkpoints/" in raw:
            add(raw.split("/checkpoints/", 1)[0])
        return candidates

    add(raw)
    if raw.endswith("/checkpoints"):
        add(raw[: -len("/checkpoints")])
    if "/checkpoints/" in raw:
        add(raw.split("/checkpoints/", 1)[0])
    return candidates


def download_config(s3_path: str) -> dict | None:
    """Download config.yaml from an S3 checkpoint path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "config.yaml"

        for checkpoint_base in _candidate_checkpoint_bases(s3_path):
            config_s3_path = f"{checkpoint_base}/config.yaml"
            result = subprocess.run(
                ["aws", "s3", "cp", config_s3_path, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                continue

            with open(tmp_path) as f:
                return yaml.safe_load(f)

    return None


def find_wandb_link(config: dict) -> str | None:
    """Find wandb run URL from config using run name."""
    try:
        import wandb

        run_name = config.get("name")
        project = config.get("wandb_project_name")

        if not run_name or not project:
            return None

        api = wandb.Api()
        # Use 'tri' as default entity since that's where the runs are stored
        entity = config.get("wandb_entity") or "tri"

        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})

        for run in runs:
            return run.url

        return None

    except Exception as e:
        print(f"  WARNING: Failed to find wandb link: {e}")
        return None


def resolve_task_name_on_s3(checkpoint_base: str, task_name: str) -> str:
    """Resolve the actual task name used on S3."""
    if not task_name:
        return task_name

    for base in _candidate_checkpoint_bases(checkpoint_base):
        result = subprocess.run(
            ["aws", "s3", "ls", f"{base}/evaluation/"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            continue

        prefixes = []
        for line in result.stdout.splitlines():
            if " PRE " in line:
                token = line.split(" PRE ", 1)[1].strip()
                if token.endswith("/"):
                    token = token[:-1]
                prefixes.append(token)

        if task_name in prefixes:
            return task_name

        startswith_matches = [p for p in prefixes if p.startswith(task_name)]
        if len(startswith_matches) == 1:
            return startswith_matches[0]

    return task_name


def _extract_sample_data(summary_data: dict, summary_file: Path) -> dict:
    """Extract relevant fields from a summary.yaml for storage."""
    # Extract demonstration index from path (e.g., demonstration_100)
    demo_dir = summary_file.parent.name
    demo_index = int(demo_dir.split("_")[1]) if "_" in demo_dir else 0

    # Extract nested fields safely
    env_metadata = summary_data.get("env_metadata", {}) or {}
    policy_metadata = summary_data.get("policy_metadata", {}) or {}
    timing_info = summary_data.get("timing_info", {}) or {}
    timed_instructions = summary_data.get("timed_language_instructions", []) or []

    # Get language instruction from timed instructions if available
    language_instruction = None
    if timed_instructions and len(timed_instructions) > 0:
        language_instruction = timed_instructions[0].get("language_instruction")

    return {
        "demonstration_index": demo_index,
        "success": summary_data.get("success", False),
        "seed": summary_data.get("seed"),
        "index": summary_data.get("index"),
        "last_step_reward": summary_data.get("last_step_reward"),
        "last_t": summary_data.get("last_t"),
        "language_instruction": language_instruction,
        # Environment metadata
        "skill": env_metadata.get("skill") if isinstance(env_metadata, dict) else None,
        "domain": env_metadata.get("domain") if isinstance(env_metadata, dict) else None,
        "hardware_platform_type": env_metadata.get("hardware_platform_type")
        if isinstance(env_metadata, dict)
        else None,
        "station_name": summary_data.get("station_name"),
        # Policy metadata
        "checkpoint_path": policy_metadata.get("checkpoint_path") if isinstance(policy_metadata, dict) else None,
        "policy_name": policy_metadata.get("name") if isinstance(policy_metadata, dict) else None,
        "skill_type": policy_metadata.get("skill_type") if isinstance(policy_metadata, dict) else None,
        # Timing
        "episode_start_time": timing_info.get("episode_start_time") if isinstance(timing_info, dict) else None,
        "episode_end_time": timing_info.get("episode_end_time") if isinstance(timing_info, dict) else None,
        # Git info
        "git_sha": summary_data.get("git_sha"),
        "git_branch": summary_data.get("git_branch"),
        "lbm_release": summary_data.get("lbm_release"),
    }


def _discover_eval_id_paths_for_leaderboard(
    checkpoint_base: str,
    task_name: str | None,
    evaluation_subfolder: str | None,
) -> list[str]:
    """Discover eval_id subdirectories under evaluation/ on S3.

    Eval campaigns write results to paths like:
        {checkpoint}/evaluation/{subfolder?}/{eval_id}/{task}/rollouts/
    where eval_id looks like "2026-02-19_a1b2c3d4".

    Returns candidate paths sorted most-recent-first.
    """
    eval_id_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_[0-9a-f]{8}/$")
    paths: list[str] = []

    def _list_s3_prefixes(s3_prefix: str) -> list[str]:
        try:
            result = subprocess.run(
                ["aws", "s3", "ls", s3_prefix],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return []
            prefixes = []
            for line in result.stdout.splitlines():
                if " PRE " in line:
                    token = line.split(" PRE ", 1)[1].strip()
                    prefixes.append(token)
            return prefixes
        except subprocess.TimeoutExpired as exc:
            sys.stderr.write(f"[leaderboard_cli] Timed out listing S3 prefix '{s3_prefix}': {exc}\n")
            return []
        except OSError as exc:
            sys.stderr.write(f"[leaderboard_cli] Failed to invoke AWS CLI for prefix '{s3_prefix}': {exc}\n")
            return []

    def _add_paths_for_eval_ids(base_prefix: str, eval_ids: list[str]):
        for eid in sorted(eval_ids, reverse=True):
            if task_name:
                paths.append(f"{base_prefix}{eid}/{task_name}/rollouts/")
                paths.append(f"{base_prefix}{eid}/{task_name}/summary/")
            paths.append(f"{base_prefix}{eid}/rollouts/")
            paths.append(f"{base_prefix}{eid}/summary/")

    eval_root = f"{checkpoint_base}/evaluation/"

    # If subfolder is specified, look for eval_ids inside it first
    if evaluation_subfolder:
        sub_prefix = f"{eval_root}{evaluation_subfolder}/"
        sub_prefixes = _list_s3_prefixes(sub_prefix)
        sub_eval_ids = [p.rstrip("/") for p in sub_prefixes if eval_id_pattern.match(p)]
        _add_paths_for_eval_ids(sub_prefix, sub_eval_ids)

    # Also look for eval_ids directly under evaluation/
    top_prefixes = _list_s3_prefixes(eval_root)
    top_eval_ids = [p.rstrip("/") for p in top_prefixes if eval_id_pattern.match(p)]
    _add_paths_for_eval_ids(eval_root, top_eval_ids)

    # Check other subfolders for eval_ids too.
    # Cap to avoid unbounded S3 list calls on checkpoints with many legacy task directories.
    subfolder_names = [
        p.rstrip("/") for p in top_prefixes if not eval_id_pattern.match(p) and p.rstrip("/") != evaluation_subfolder
    ]
    for subfolder in sorted(subfolder_names)[:50]:
        sub_prefix = f"{eval_root}{subfolder}/"
        sub_prefixes = _list_s3_prefixes(sub_prefix)
        sub_eval_ids = [p.rstrip("/") for p in sub_prefixes if eval_id_pattern.match(p)]
        _add_paths_for_eval_ids(sub_prefix, sub_eval_ids)

    return paths


def download_summaries_and_compute_rate(
    checkpoint_s3_path: str,
    campaign_name: str,
    task_name: str | None = None,
    evaluation_subfolder: str | None = None,
) -> dict:
    """Download summaries and compute success rate for a checkpoint.

    Returns a dict with:
        - samples: list of individual sample data dicts
        - total, successes, failures, success_rate: aggregated stats
        - source_path: S3 path where summaries were found
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        task_dir = Path(tmp_dir)

        for checkpoint_base in _candidate_checkpoint_bases(checkpoint_s3_path):
            resolved_task_name = resolve_task_name_on_s3(checkpoint_base, task_name) if task_name else None

            paths_to_check: list[str] = []

            # Auto-discover eval_id subdirectories (most recent first)
            discovered = _discover_eval_id_paths_for_leaderboard(
                checkpoint_base, resolved_task_name, evaluation_subfolder
            )
            paths_to_check.extend(discovered)

            # Legacy paths (no eval_id)
            if evaluation_subfolder and resolved_task_name:
                paths_to_check.extend(
                    [
                        f"{checkpoint_base}/evaluation/{evaluation_subfolder}/{resolved_task_name}/rollouts/",
                        f"{checkpoint_base}/evaluation/{evaluation_subfolder}/{resolved_task_name}/summary/",
                    ]
                )
            if resolved_task_name:
                paths_to_check.extend(
                    [
                        f"{checkpoint_base}/evaluation/{resolved_task_name}/rollouts/",
                        f"{checkpoint_base}/evaluation/{resolved_task_name}/summary/",
                    ]
                )
            if evaluation_subfolder:
                paths_to_check.extend(
                    [
                        f"{checkpoint_base}/evaluation/{evaluation_subfolder}/rollouts/",
                        f"{checkpoint_base}/evaluation/{evaluation_subfolder}/summary/",
                    ]
                )
            paths_to_check.extend(
                [
                    f"{checkpoint_base}/evaluation/rollouts/",
                    f"{checkpoint_base}/evaluation/summary/",
                    f"{checkpoint_base}/evaluation/{campaign_name}/artifacts/rollouts/",
                ]
            )

            for s3_base in paths_to_check:
                for item in task_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

                cmd = [
                    "aws",
                    "s3",
                    "sync",
                    s3_base,
                    str(task_dir),
                    "--exclude",
                    "*",
                    "--include",
                    "*/summary.yaml",
                ]

                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                    if result.returncode != 0:
                        continue

                    summary_files = list(task_dir.glob("demonstration_*/summary.yaml"))
                    if not summary_files:
                        continue

                    samples = []
                    successes = 0
                    total = 0

                    for summary_file in summary_files:
                        try:

                            def ignore_unknown_tags(loader, tag_suffix, node):
                                if isinstance(node, yaml.MappingNode):
                                    return loader.construct_mapping(node)
                                elif isinstance(node, yaml.SequenceNode):
                                    return loader.construct_sequence(node)
                                else:
                                    return loader.construct_scalar(node)

                            yaml.add_multi_constructor("!", ignore_unknown_tags, Loader=yaml.SafeLoader)

                            with open(summary_file) as f:
                                data = yaml.safe_load(f)

                            sample_data = _extract_sample_data(data, summary_file)
                            samples.append(sample_data)

                            total += 1
                            if data.get("success", False):
                                successes += 1

                        except Exception:
                            continue

                    if total > 0:
                        return {
                            "samples": samples,
                            "total": total,
                            "successes": successes,
                            "failures": total - successes,
                            "success_rate": successes / total,
                            "source_path": s3_base,
                        }

                except Exception:
                    continue

    return {
        "samples": [],
        "total": 0,
        "successes": 0,
        "failures": 0,
        "success_rate": 0.0,
        "source_path": None,
    }


class DynamoDBClient:
    """Client that writes directly to DynamoDB."""

    def __init__(self):
        self.dynamodb = boto3.resource("dynamodb", region_name=REGION)
        self.runs_table = self.dynamodb.Table(LEADERBOARD_RUNS_TABLE)
        self.evals_table = self.dynamodb.Table(LEADERBOARD_EVALUATIONS_TABLE)
        self.samples_table = self.dynamodb.Table(SAMPLE_RESULTS_TABLE)

    def create_run(self, overwrite: bool = False, **kwargs) -> int:
        """Create a run. Raises if entry exists unless overwrite=True."""
        from datetime import datetime

        # Generate run_id from scenario_name and s3_path using deterministic hash
        # (Python's hash() is randomized between processes)
        key = f"{kwargs['scenario_name']}_{kwargs['s3_path']}"
        run_id = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)

        # Check if exists
        if not overwrite:
            response = self.runs_table.get_item(Key={"run_id": run_id})
            if "Item" in response:
                raise ValueError(f"Run {run_id} already exists. Use --overwrite to update.")

        item = {
            "run_id": run_id,
            "scenario_name": kwargs.get("scenario_name", ""),
            "task_name": kwargs.get("task_name", ""),
            "s3_path": kwargs.get("s3_path", ""),
            "model_type": kwargs.get("model_type") or "",
            "ablation": kwargs.get("ablation") or "",
            "learning_rate": kwargs.get("learning_rate") or "",
            "batch_size": kwargs.get("batch_size") or 0,
            "run_timestamp": kwargs.get("run_timestamp") or "",
            "wandb_link": kwargs.get("wandb_link") or "",
            "has_config": kwargs.get("has_config", False),
            "has_wandb": kwargs.get("has_wandb", False),
            "config_json": "",
            "updated_at": datetime.now().isoformat(),
        }

        self.runs_table.put_item(Item=item)
        return run_id

    def delete_samples_for_eval(self, eval_id: str) -> int:
        """Delete all sample results for a given eval_id. Returns count deleted."""
        samples = self.get_samples_for_eval(eval_id)
        deleted = 0
        for sample in samples:
            sample_id = sample.get("sample_id")
            if sample_id:
                self.samples_table.delete_item(Key={"sample_id": sample_id})
                deleted += 1
        return deleted

    def create_evaluation(self, overwrite: bool = False, **kwargs) -> str:
        """Create an evaluation result. Raises if entry exists unless overwrite=True.

        Args:
            overwrite: If False, raises ValueError if eval already exists
            run_id: The run ID
            campaign_name: Campaign name
            total_rollouts: Total number of rollouts
            successes: Number of successful rollouts
            failures: Number of failed rollouts (optional, computed if not provided)
            success_rate: Success rate (optional, computed if not provided)
            source_path: S3 path where results were found
            aggregation_source: "samples" if computed from individual sample results,
                               "direct" if uploaded directly as aggregated stats
        """
        from datetime import datetime

        run_id = kwargs["run_id"]
        campaign_name = kwargs["campaign_name"]
        total = kwargs["total_rollouts"]
        successes = kwargs["successes"]
        failures = kwargs.get("failures", total - successes)
        success_rate = kwargs.get("success_rate", successes / total if total > 0 else 0)
        aggregation_source = kwargs.get("aggregation_source", "samples")

        eval_id = f"{run_id}_{campaign_name}"

        # Check if exists
        response = self.evals_table.get_item(Key={"eval_id": eval_id})
        if "Item" in response:
            if not overwrite:
                raise ValueError(f"Evaluation {eval_id} already exists. Use --overwrite to update.")
            # Delete old samples when overwriting
            self.delete_samples_for_eval(eval_id)

        item = {
            "eval_id": eval_id,
            "run_id": run_id,
            "campaign_name": campaign_name,
            "total_rollouts": total,
            "successes": successes,
            "failures": failures,
            "success_rate": Decimal(str(success_rate)),
            "source_path": kwargs.get("source_path") or "",
            "aggregation_source": aggregation_source,
            "evaluated_at": datetime.now().isoformat(),
        }

        self.evals_table.put_item(Item=item)
        return eval_id

    def create_sample_result(
        self, eval_id: str, run_id: int, campaign_name: str, sample_data: dict, overwrite: bool = False
    ) -> str:
        """Create a sample result entry. Raises if entry exists unless overwrite=True."""

        demo_index = sample_data.get("demonstration_index", 0)
        sample_id = f"{eval_id}_{demo_index}"

        # Check if exists
        if not overwrite:
            response = self.samples_table.get_item(Key={"sample_id": sample_id})
            if "Item" in response:
                raise ValueError(f"Sample {sample_id} already exists. Use --overwrite to update.")

        item = self._build_sample_item(eval_id, run_id, campaign_name, sample_data)
        self.samples_table.put_item(Item=item)
        return sample_id

    def _build_sample_item(self, eval_id: str, run_id: int, campaign_name: str, sample_data: dict) -> dict:
        """Build a sample item dict for DynamoDB."""
        from datetime import datetime

        def to_decimal(val):
            if val is None:
                return None
            if isinstance(val, float):
                return Decimal(str(val))
            return val

        demo_index = sample_data.get("demonstration_index", 0)
        sample_id = f"{eval_id}_{demo_index}"

        return {
            "sample_id": sample_id,
            "eval_id": eval_id,
            "run_id": run_id,
            "campaign_name": campaign_name,
            "demonstration_index": demo_index,
            "success": sample_data.get("success", False),
            "seed": sample_data.get("seed"),
            "index": sample_data.get("index"),
            "last_step_reward": to_decimal(sample_data.get("last_step_reward")),
            "last_t": to_decimal(sample_data.get("last_t")),
            "language_instruction": sample_data.get("language_instruction") or "",
            "skill": sample_data.get("skill") or "",
            "domain": sample_data.get("domain") or "",
            "hardware_platform_type": sample_data.get("hardware_platform_type") or "",
            "station_name": sample_data.get("station_name") or "",
            "checkpoint_path": sample_data.get("checkpoint_path") or "",
            "policy_name": sample_data.get("policy_name") or "",
            "skill_type": sample_data.get("skill_type") or "",
            "episode_start_time": sample_data.get("episode_start_time") or "",
            "episode_end_time": sample_data.get("episode_end_time") or "",
            "git_sha": sample_data.get("git_sha") or "",
            "git_branch": sample_data.get("git_branch") or "",
            "lbm_release": sample_data.get("lbm_release") or "",
            "created_at": datetime.now().isoformat(),
        }

    def batch_create_sample_results(
        self, eval_id: str, run_id: int, campaign_name: str, samples: list[dict], overwrite: bool = False
    ) -> int:
        """Batch create sample results using DynamoDB batch_write_item (up to 25 items per batch).

        Returns the number of items written.
        """
        if not samples:
            return 0

        items = [self._build_sample_item(eval_id, run_id, campaign_name, s) for s in samples]

        # DynamoDB batch_write_item supports max 25 items per request
        batch_size = 25
        written = 0

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            request_items = {SAMPLE_RESULTS_TABLE: [{"PutRequest": {"Item": item}} for item in batch]}

            # Handle unprocessed items with retry
            while request_items:
                response = self.dynamodb.meta.client.batch_write_item(RequestItems=request_items)
                written += len(batch) - len(response.get("UnprocessedItems", {}).get(SAMPLE_RESULTS_TABLE, []))
                request_items = response.get("UnprocessedItems", {})

        return written

    def get_samples_for_eval(self, eval_id: str) -> list[dict]:
        """Get all sample results for a given eval_id."""
        response = self.samples_table.query(
            IndexName="eval_id-index",
            KeyConditionExpression="eval_id = :eval_id",
            ExpressionAttributeValues={":eval_id": eval_id},
        )
        items = response.get("Items", [])
        while "LastEvaluatedKey" in response:
            response = self.samples_table.query(
                IndexName="eval_id-index",
                KeyConditionExpression="eval_id = :eval_id",
                ExpressionAttributeValues={":eval_id": eval_id},
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            items.extend(response.get("Items", []))
        return items

    def compute_aggregates_from_samples(self, eval_id: str) -> dict:
        """Compute aggregated stats from individual sample results."""
        samples = self.get_samples_for_eval(eval_id)
        if not samples:
            return {"total": 0, "successes": 0, "failures": 0, "success_rate": 0.0}

        total = len(samples)
        successes = sum(1 for s in samples if s.get("success", False))
        return {
            "total": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": successes / total if total > 0 else 0.0,
        }

    def validate_evaluation_coherence(self, eval_id: str) -> dict:
        """Validate that aggregated evaluation stats match the individual samples.

        Returns a dict with:
            - is_valid: bool indicating if stats are coherent
            - evaluation: the evaluation record
            - computed: stats computed from samples
            - message: description of any mismatch
        """
        # Get the evaluation record
        response = self.evals_table.get_item(Key={"eval_id": eval_id})
        evaluation = response.get("Item")

        if not evaluation:
            return {"is_valid": False, "message": f"Evaluation {eval_id} not found"}

        aggregation_source = evaluation.get("aggregation_source", "unknown")

        # If it's a direct upload, samples may not exist - that's fine
        if aggregation_source == "direct":
            return {
                "is_valid": True,
                "evaluation": evaluation,
                "computed": None,
                "message": "Direct upload - no sample validation needed",
            }

        # Compute stats from samples
        computed = self.compute_aggregates_from_samples(eval_id)

        # Compare
        eval_total = int(evaluation.get("total_rollouts", 0))
        eval_successes = int(evaluation.get("successes", 0))

        if computed["total"] == 0:
            return {
                "is_valid": False,
                "evaluation": evaluation,
                "computed": computed,
                "message": "No samples found but aggregation_source is 'samples'",
            }

        if eval_total != computed["total"] or eval_successes != computed["successes"]:
            return {
                "is_valid": False,
                "evaluation": evaluation,
                "computed": computed,
                "message": (
                    f"Mismatch: evaluation has {eval_successes}/{eval_total}, "
                    f"but samples show {computed['successes']}/{computed['total']}"
                ),
            }

        return {
            "is_valid": True,
            "evaluation": evaluation,
            "computed": computed,
            "message": "Stats are coherent with samples",
        }

    def refresh_evaluation_from_samples(self, eval_id: str) -> dict | None:
        """Recompute and update evaluation stats from its sample results.

        Only works for evaluations with aggregation_source='samples'.
        Returns the updated evaluation or None if not applicable.
        """
        response = self.evals_table.get_item(Key={"eval_id": eval_id})
        evaluation = response.get("Item")

        if not evaluation:
            return None

        if evaluation.get("aggregation_source") == "direct":
            return None  # Don't overwrite direct uploads

        computed = self.compute_aggregates_from_samples(eval_id)
        if computed["total"] == 0:
            return None

        # Update the evaluation
        self.evals_table.update_item(
            Key={"eval_id": eval_id},
            UpdateExpression=(
                "SET total_rollouts = :total, successes = :successes, failures = :failures, success_rate = :rate",
            ),
            ExpressionAttributeValues={
                ":total": computed["total"],
                ":successes": computed["successes"],
                ":failures": computed["failures"],
                ":rate": Decimal(str(computed["success_rate"])),
            },
        )

        return {**evaluation, **computed}

    def create_config(self, run_id: int, config: dict):
        """Update config for a run."""
        import json

        self.runs_table.update_item(
            Key={"run_id": run_id},
            UpdateExpression="SET config_json = :config",
            ExpressionAttributeValues={":config": json.dumps(config)},
        )

    def get_evaluations_by_campaign(self, campaign_name: str) -> list[dict]:
        """Get all evaluations for a given campaign using GSI."""
        try:
            response = self.evals_table.query(
                IndexName="campaign_name-index",
                KeyConditionExpression="campaign_name = :cn",
                ExpressionAttributeValues={":cn": campaign_name},
            )
            items = response.get("Items", [])
            while "LastEvaluatedKey" in response:
                response = self.evals_table.query(
                    IndexName="campaign_name-index",
                    KeyConditionExpression="campaign_name = :cn",
                    ExpressionAttributeValues={":cn": campaign_name},
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                items.extend(response.get("Items", []))
            return items
        except Exception:
            # Fall back to scan if GSI doesn't exist yet
            response = self.evals_table.scan(
                FilterExpression="campaign_name = :cn",
                ExpressionAttributeValues={":cn": campaign_name},
            )
            return response.get("Items", [])

    def get_leaderboard(self, campaign_name: str | None = None) -> list:
        """Get leaderboard entries, optionally filtered by campaign."""
        runs = self._scan_table(self.runs_table)
        runs_by_id = {r["run_id"]: r for r in runs}

        evals = self.get_evaluations_by_campaign(campaign_name) if campaign_name else self._scan_table(self.evals_table)

        results = []
        for eval_item in evals:
            run_id = eval_item.get("run_id")
            run = runs_by_id.get(run_id, {})
            results.append(
                {
                    "run_id": run_id,
                    "scenario_name": run.get("scenario_name", ""),
                    "task_name": run.get("task_name", ""),
                    "ablation": run.get("ablation"),
                    "campaign_name": eval_item.get("campaign_name"),
                    "success_rate": float(eval_item.get("success_rate", 0)),
                }
            )
        return results

    def _scan_table(self, table) -> list:
        """Scan a DynamoDB table."""
        result = table.scan()
        items = result.get("Items", [])
        while "LastEvaluatedKey" in result:
            result = table.scan(ExclusiveStartKey=result["LastEvaluatedKey"])
            items.extend(result.get("Items", []))
        return items


def expand_tasks_paths(paths: list[Path]) -> list[Path]:
    """Expand paths to include all .txt files, recursively searching directories."""
    result = []
    for path in paths:
        if path.is_dir():
            txt_files = sorted(path.rglob("*.txt"))
            if txt_files:
                print(f"Found {len(txt_files)} task file(s) in {path}")
                result.extend(txt_files)
            else:
                print(f"WARNING: No .txt files found in {path}")
        elif path.is_file():
            result.append(path)
        else:
            print(f"WARNING: Path does not exist: {path}")
    return result


def _process_single_task(
    task: tuple,
    task_index: int,
    total_tasks: int,
    campaign_name: str,
    db_client: DynamoDBClient,
    skip_s3: bool,
    evaluation_subfolder: str | None,
    no_link: bool,
    overwrite: bool,
) -> dict:
    """Process a single task and return result dict (for parallel processing)."""
    scenario_name, task_name, s3_path = task[0], task[1], task[2]
    result = {
        "scenario_name": scenario_name,
        "task_index": task_index,
        "success": False,
        "missing": [],
        "message": "",
        "success_rate": None,
    }

    checkpoint_base = s3_path.rstrip("/")
    resolved_task_name = resolve_task_name_on_s3(checkpoint_base, task_name)
    metadata = extract_metadata_from_s3_path(s3_path)

    config = None
    wandb_link = None
    has_config = False
    has_wandb = False

    if not skip_s3:
        config = download_config(s3_path)
        if config:
            has_config = True
            wandb_link = find_wandb_link(config)
            if wandb_link:
                has_wandb = True

        task_missing = []
        if not has_config:
            task_missing.append("config")
        if not has_wandb:
            task_missing.append("wandb")

        if task_missing and not no_link:
            result["missing"] = task_missing
            result["message"] = f"Missing {', '.join(task_missing)}"
            return result

    run_id = db_client.create_run(
        overwrite=overwrite,
        scenario_name=scenario_name,
        task_name=resolved_task_name,
        s3_path=s3_path,
        model_type=metadata["model_type"],
        ablation=metadata["ablation"],
        learning_rate=metadata["learning_rate"],
        batch_size=metadata["batch_size"],
        run_timestamp=metadata["run_timestamp"],
        wandb_link=wandb_link,
        has_config=has_config,
        has_wandb=has_wandb,
    )

    if not skip_s3:
        if config:
            db_client.create_config(run_id, config)

        stats = download_summaries_and_compute_rate(
            s3_path,
            campaign_name,
            resolved_task_name,
            evaluation_subfolder=evaluation_subfolder,
        )

        if stats["total"] > 0:
            eval_id = db_client.create_evaluation(
                overwrite=overwrite,
                run_id=run_id,
                campaign_name=campaign_name,
                total_rollouts=stats["total"],
                successes=stats["successes"],
                failures=stats["failures"],
                success_rate=stats["success_rate"],
                source_path=stats["source_path"],
                aggregation_source="samples",
            )

            samples = stats.get("samples", [])
            if samples:
                db_client.batch_create_sample_results(eval_id, run_id, campaign_name, samples, overwrite=overwrite)

            result["success_rate"] = stats["success_rate"]
            result["successes"] = stats["successes"]
            result["total"] = stats["total"]
            result["message"] = f"{stats['successes']}/{stats['total']} = {stats['success_rate']:.1%}"
        else:
            result["message"] = "No rollouts found"

    result["success"] = True
    result["wandb_link"] = wandb_link
    return result


def upload_tasks(
    tasks_file: Path,
    campaign_name: str,
    db_client: DynamoDBClient,
    skip_s3: bool = False,
    evaluation_subfolder: str | None = None,
    no_link: bool = False,
    overwrite: bool = False,
):
    """Process tasks and upload directly to DynamoDB.

    Args:
        overwrite: If False (default), raises error if entries already exist.
                   If True, overwrites existing entries.
    """
    tasks = parse_tasks_file(tasks_file)
    print(f"Found {len(tasks)} tasks in {tasks_file}")

    # Always use parallel processing (8 workers by default)
    max_workers = min(8, len(tasks)) if len(tasks) > 1 else 1
    print(f"Processing with {max_workers} workers...")
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_task,
                task,
                i,
                len(tasks),
                campaign_name,
                db_client,
                skip_s3,
                evaluation_subfolder,
                no_link,
                overwrite,
            ): i
            for i, task in enumerate(tasks, 1)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            idx = result["task_index"]
            scenario = result["scenario_name"]
            if result["success"]:
                print(f"[{idx}/{len(tasks)}] {scenario}: {result['message']}")
            else:
                print(f"[{idx}/{len(tasks)}] {scenario}: ERROR - {result['message']}")

    # Check for missing links
    missing_links = [(r["scenario_name"], r["missing"]) for r in results if r["missing"]]

    if missing_links:
        print("\n" + "=" * 60)
        print("ERROR: Upload aborted due to missing links!")
        print("The following tasks are missing required links:")
        for scenario, missing in missing_links:
            print(f"  - {scenario}: missing {', '.join(missing)}")
        print("\nUse --no-link flag to upload without links")
        print("=" * 60)
        raise SystemExit(1)


def cmd_upload(args):
    """Handle the upload command."""
    if not args.tasks_files:
        print("ERROR: At least one tasks file or directory is required")
        return 1

    # Backward-compat: allow positional campaign name
    if args.campaign_name == "default_campaign" and len(args.tasks_files) >= 2 and not args.tasks_files[-1].exists():
        args.campaign_name = str(args.tasks_files[-1])
        args.tasks_files = args.tasks_files[:-1]

    expanded_files = expand_tasks_paths(args.tasks_files)
    if not expanded_files:
        print("ERROR: No task files found")
        return 1

    print(f"Processing {len(expanded_files)} task file(s)")
    print(f"Writing directly to DynamoDB ({REGION})")

    db_client = DynamoDBClient()

    for tasks_file in expanded_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {tasks_file}")
        print(f"{'=' * 60}")
        upload_tasks(
            tasks_file,
            args.campaign_name,
            db_client,
            args.skip_s3,
            evaluation_subfolder=args.evaluation_subfolder,
            no_link=args.no_link,
            overwrite=args.overwrite,
        )

    print("\nDone!")
    return 0


def cmd_delete(args):
    """Handle the delete command."""
    if not any([args.campaign_name, args.ablation, args.scenario_pattern]):
        print("ERROR: At least one filter required: --campaign-name, --ablation, or --scenario-pattern")
        return 1

    db_client = DynamoDBClient()

    print(f"Writing directly to DynamoDB ({REGION})")
    print(f"  Campaign: {args.campaign_name or 'any'}")
    print(f"  Ablation: {args.ablation or 'any'}")
    print(f"  Scenario pattern: {args.scenario_pattern or 'any'}")

    # Get entries matching filter
    entries = db_client.get_leaderboard()
    to_delete = []

    for entry in entries:
        if args.campaign_name and entry.get("campaign_name") != args.campaign_name:
            continue
        if args.ablation and entry.get("ablation") != args.ablation:
            continue
        if args.scenario_pattern and args.scenario_pattern not in entry.get("scenario_name", ""):
            continue
        to_delete.append(entry)

    if not to_delete:
        print("\nNo entries match the filter criteria")
        return 0

    print(f"\nFound {len(to_delete)} entries to delete:")
    for entry in to_delete[:10]:
        print(f"  - {entry.get('scenario_name')} ({entry.get('campaign_name')})")
    if len(to_delete) > 10:
        print(f"  ... and {len(to_delete) - 10} more")

    if not args.force:
        confirm = input("\nAre you sure you want to delete? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return 0

    # Delete from DynamoDB
    deleted_runs = set()
    deleted_evals = 0
    deleted_samples = 0

    for entry in to_delete:
        run_id = entry.get("run_id")
        campaign_name = entry.get("campaign_name")

        # Delete evaluation
        eval_id = f"{run_id}_{campaign_name}"

        # Delete associated sample results first
        try:
            samples = db_client.get_samples_for_eval(eval_id)
            for sample in samples:
                sample_id = sample.get("sample_id")
                if sample_id:
                    db_client.samples_table.delete_item(Key={"sample_id": sample_id})
                    deleted_samples += 1
        except Exception as e:
            print(f"  WARNING: Failed to delete samples for {eval_id}: {e}")

        try:
            db_client.evals_table.delete_item(Key={"eval_id": eval_id})
            deleted_evals += 1
        except Exception as e:
            print(f"  WARNING: Failed to delete evaluation {eval_id}: {e}")

        # Track runs to delete
        deleted_runs.add(run_id)

    # Delete runs
    for run_id in deleted_runs:
        try:
            db_client.runs_table.delete_item(Key={"run_id": run_id})
        except Exception as e:
            print(f"  WARNING: Failed to delete run {run_id}: {e}")

    print(f"\nDeleted {deleted_evals} evaluations, {deleted_samples} samples, and {len(deleted_runs)} runs")
    return 0


def cmd_list(args):
    """Handle the list command."""
    db_client = DynamoDBClient()

    print(f"Reading from DynamoDB ({REGION})")

    entries = db_client.get_leaderboard(campaign_name=args.campaign_name)

    if not entries:
        print("No entries found")
        return 0

    if args.campaign_name:
        # Show detailed view for specific campaign
        print(f"\nFound {len(entries)} entries in campaign '{args.campaign_name}':\n")
        for entry in sorted(entries, key=lambda e: e.get("scenario_name", "")):
            print(f"  {entry.get('scenario_name')}")
            print(f"    ablation: {entry.get('ablation') or '(none)'}")
            print(f"    success_rate: {entry.get('success_rate', 0):.1%}")
    else:
        # Group by campaign
        campaigns = {}
        for entry in entries:
            campaign = entry.get("campaign_name", "unknown")
            if campaign not in campaigns:
                campaigns[campaign] = []
            campaigns[campaign].append(entry)

        print(f"\nFound {len(entries)} entries in {len(campaigns)} campaign(s):\n")

        for campaign, camp_entries in sorted(campaigns.items()):
            avg_rate = sum(e.get("success_rate", 0) for e in camp_entries) / len(camp_entries)
            print(f"  {campaign}: {len(camp_entries)} entries, avg success rate: {avg_rate:.1%}")

    return 0


def cmd_validate(args):
    """Handle the validate command."""
    db_client = DynamoDBClient()

    print(f"Reading from DynamoDB ({REGION})")

    # Get all evaluations
    evals = db_client._scan_table(db_client.evals_table)

    if not evals:
        print("No evaluations found")
        return 0

    # Filter if specified
    if args.campaign_name:
        evals = [e for e in evals if e.get("campaign_name") == args.campaign_name]
    if args.eval_id:
        evals = [e for e in evals if e.get("eval_id") == args.eval_id]

    print(f"\nValidating {len(evals)} evaluation(s)...")

    valid_count = 0
    invalid_count = 0
    direct_count = 0

    for eval_item in evals:
        eval_id = eval_item.get("eval_id")
        result = db_client.validate_evaluation_coherence(eval_id)

        if result.get("evaluation", {}).get("aggregation_source") == "direct":
            direct_count += 1
            if args.verbose:
                print(f"  {eval_id}: DIRECT (no validation needed)")
        elif result["is_valid"]:
            valid_count += 1
            if args.verbose:
                print(f"  {eval_id}: OK")
        else:
            invalid_count += 1
            print(f"  {eval_id}: INVALID - {result['message']}")

    print(f"\nSummary: {valid_count} valid, {invalid_count} invalid, {direct_count} direct uploads")
    return 1 if invalid_count > 0 else 0


def cmd_refresh(args):
    """Handle the refresh command."""
    db_client = DynamoDBClient()

    print(f"Reading from DynamoDB ({REGION})")

    # Get all evaluations
    evals = db_client._scan_table(db_client.evals_table)

    if not evals:
        print("No evaluations found")
        return 0

    # Filter if specified
    if args.campaign_name:
        evals = [e for e in evals if e.get("campaign_name") == args.campaign_name]
    if args.eval_id:
        evals = [e for e in evals if e.get("eval_id") == args.eval_id]

    # Only refresh sample-based evaluations
    sample_evals = [e for e in evals if e.get("aggregation_source") != "direct"]

    if not sample_evals:
        print("No sample-based evaluations to refresh")
        return 0

    print(f"\nRefreshing {len(sample_evals)} evaluation(s) from samples...")

    refreshed = 0
    skipped = 0

    for eval_item in sample_evals:
        eval_id = eval_item.get("eval_id")
        result = db_client.refresh_evaluation_from_samples(eval_id)

        if result:
            old_rate = float(eval_item.get("success_rate", 0))
            new_rate = result["success_rate"]
            if abs(old_rate - new_rate) > 0.001:
                print(f"  {eval_id}: {old_rate:.1%} -> {new_rate:.1%}")
            else:
                print(f"  {eval_id}: unchanged ({new_rate:.1%})")
            refreshed += 1
        else:
            print(f"  {eval_id}: skipped (no samples or direct upload)")
            skipped += 1

    print(f"\nRefreshed {refreshed} evaluations, skipped {skipped}")
    return 0


def cmd_fix_flags(args):
    """Fix has_config and has_wandb flags based on actual data presence."""
    db_client = DynamoDBClient()

    print(f"Scanning runs in DynamoDB ({REGION})")

    # Get all runs
    runs = db_client._scan_table(db_client.runs_table)

    if args.campaign_name:
        # Filter by campaign - need to check evaluations
        evals = db_client._scan_table(db_client.evals_table)
        run_ids_in_campaign = {e.get("run_id") for e in evals if e.get("campaign_name") == args.campaign_name}
        runs = [r for r in runs if r.get("run_id") in run_ids_in_campaign]

    if not runs:
        print("No runs found")
        return 0

    print(f"Found {len(runs)} runs to check")

    fixed_config = 0
    fixed_wandb = 0
    fixed_s3 = 0

    for run in runs:
        run_id = run.get("run_id")
        updates = {}

        # Check has_config: should be False if config_json is empty/missing
        config_json = run.get("config_json", "")
        current_has_config = run.get("has_config", True)
        actual_has_config = bool(config_json and config_json.strip() and config_json != "{}")

        if current_has_config != actual_has_config:
            updates["has_config"] = actual_has_config
            fixed_config += 1

        # Check has_wandb: should be False if wandb_link is empty/missing
        wandb_link = run.get("wandb_link", "")
        current_has_wandb = run.get("has_wandb", True)
        actual_has_wandb = bool(wandb_link and wandb_link.strip())

        if current_has_wandb != actual_has_wandb:
            updates["has_wandb"] = actual_has_wandb
            fixed_wandb += 1

        # Check s3_path: set to empty string if it's None, empty, or invalid
        s3_path = run.get("s3_path", "")
        if (not s3_path or not isinstance(s3_path, str) or not s3_path.startswith("s3://")) and s3_path != "":
            updates["s3_path"] = ""
            fixed_s3 += 1

        if updates:
            # Update the run in DynamoDB
            update_expr = "SET " + ", ".join(f"#{k} = :{k}" for k in updates)
            expr_names = {f"#{k}": k for k in updates}
            expr_values = {f":{k}": v for k, v in updates.items()}

            db_client.runs_table.update_item(
                Key={"run_id": run_id},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
            )

            scenario = run.get("scenario_name", run_id)
            print(f"  Fixed {scenario}: {updates}")

    print(f"\nFixed {fixed_config} has_config flags, {fixed_wandb} has_wandb flags, {fixed_s3} s3_path values")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Leaderboard CLI - Upload results to VLA Foundry leaderboard (DynamoDB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload results
  %(prog)s upload task_list.txt --campaign-name my_campaign

  # Upload multiple files
  %(prog)s upload file1.txt file2.txt --campaign-name my_campaign

  # List all entries
  %(prog)s list

  # Delete entries
  %(prog)s delete --ablation continuous_flow
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Upload subcommand
    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload evaluation results",
    )
    upload_parser.add_argument(
        "tasks_files",
        type=Path,
        nargs="+",
        help="Path(s) to task files or directories",
    )
    upload_parser.add_argument(
        "--campaign-name",
        type=str,
        default="default_campaign",
        help="Campaign name for organizing results",
    )
    upload_parser.add_argument(
        "--skip-s3",
        action="store_true",
        help="Skip S3 downloads (faster, but no configs/success rates)",
    )
    upload_parser.add_argument(
        "--evaluation-subfolder",
        type=str,
        default=None,
        help="Optional evaluation subfolder",
    )
    upload_parser.add_argument(
        "--no-link",
        action="store_true",
        help="Allow uploading without config/wandb links",
    )
    upload_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing entries (default: refuse to overwrite)",
    )
    upload_parser.set_defaults(func=cmd_upload)

    # Delete subcommand
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete entries by filter",
    )
    delete_parser.add_argument("--campaign-name", type=str, help="Filter by campaign name")
    delete_parser.add_argument("--ablation", type=str, help="Filter by ablation name")
    delete_parser.add_argument("--scenario-pattern", type=str, help="Filter by scenario name pattern")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")
    delete_parser.set_defaults(func=cmd_delete)

    # List subcommand
    list_parser = subparsers.add_parser(
        "list",
        help="List leaderboard entries",
    )
    list_parser.add_argument("--campaign-name", type=str, help="Filter by campaign name (shows detailed view)")
    list_parser.set_defaults(func=cmd_list)

    # Validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate that aggregated stats match sample results",
    )
    validate_parser.add_argument("--campaign-name", type=str, help="Filter by campaign name")
    validate_parser.add_argument("--eval-id", type=str, help="Validate specific evaluation")
    validate_parser.add_argument("--verbose", "-v", action="store_true", help="Show all results, not just invalid")
    validate_parser.set_defaults(func=cmd_validate)

    # Refresh subcommand
    refresh_parser = subparsers.add_parser(
        "refresh",
        help="Recompute aggregated stats from sample results",
    )
    refresh_parser.add_argument("--campaign-name", type=str, help="Filter by campaign name")
    refresh_parser.add_argument("--eval-id", type=str, help="Refresh specific evaluation")
    refresh_parser.set_defaults(func=cmd_refresh)

    # Fix-flags subcommand
    fix_flags_parser = subparsers.add_parser(
        "fix-flags",
        help="Fix has_config/has_wandb flags based on actual data presence",
    )
    fix_flags_parser.add_argument("--campaign-name", type=str, help="Filter by campaign name")
    fix_flags_parser.set_defaults(func=cmd_fix_flags)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
