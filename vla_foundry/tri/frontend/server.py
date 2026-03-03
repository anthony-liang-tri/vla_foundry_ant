#!/usr/bin/env python3
"""
VLA Foundry Dashboard & Leaderboard Server

A read-only FastAPI server that:
- Serves the static dashboard and leaderboard frontends
- Provides read-only API endpoints to query DynamoDB for training runs, datasets, and evaluation results

Note: Write operations (uploading results) are done directly via the CLI (leaderboard_cli.py)
which writes to DynamoDB using boto3.

Run locally: python server.py
Run on EC2: python server.py --host 0.0.0.0 --port 8080
"""

import argparse
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import uvicorn
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from scipy import stats

app = FastAPI(title="VLA Foundry Dashboard & Leaderboard")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# DynamoDB config
REGION = "us-west-2"
MODELS_TABLE = "vla_foundry_models"
DATASETS_TABLE = "vla_foundry_datasets"
LEADERBOARD_RUNS_TABLE = "vla_foundry_leaderboard_runs"
LEADERBOARD_EVALUATIONS_TABLE = "vla_foundry_leaderboard_evaluations"
SAMPLE_RESULTS_TABLE = "vla_foundry_sample_results"

dynamodb = boto3.resource("dynamodb", region_name=REGION)


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def decimal_to_python(obj):
    """Recursively convert Decimal objects to Python native types."""
    if isinstance(obj, Decimal):
        if obj % 1 == 0:
            return int(obj)
        return float(obj)
    elif isinstance(obj, dict):
        return {k: decimal_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_python(v) for v in obj]
    return obj


def scan_table(table_name: str) -> list:
    """Scan a DynamoDB table and return all items."""
    table = dynamodb.Table(table_name)
    result = table.scan()
    items = result.get("Items", [])

    # Handle pagination
    while "LastEvaluatedKey" in result:
        result = table.scan(ExclusiveStartKey=result["LastEvaluatedKey"])
        items.extend(result.get("Items", []))

    # Convert Decimals to Python types
    return [decimal_to_python(item) for item in items]


def query_table_by_index(
    table_name: str,
    index_name: str,
    key_condition: dict,
) -> list:
    """Query a DynamoDB table using a GSI."""
    table = dynamodb.Table(table_name)
    items = []

    try:
        response = table.query(
            IndexName=index_name,
            KeyConditionExpression=key_condition,
        )
        items.extend(response.get("Items", []))

        while "LastEvaluatedKey" in response:
            response = table.query(
                IndexName=index_name,
                KeyConditionExpression=key_condition,
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            items.extend(response.get("Items", []))
    except ClientError:
        # Index might not exist, fall back to scan
        pass

    return [decimal_to_python(item) for item in items]


# ============================================================================
# Dashboard API Endpoints
# ============================================================================


@app.get("/api/models")
async def get_models():
    """Get all model training runs."""
    return scan_table(MODELS_TABLE)


@app.get("/api/datasets")
async def get_datasets():
    """Get all datasets."""
    return scan_table(DATASETS_TABLE)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ============================================================================
# Leaderboard API Endpoints
# ============================================================================


@app.get("/api/leaderboard")
async def get_leaderboard(
    request: Request,
    min_rollouts: int = 0,
    order_by: str = "success_rate",
    descending: bool = True,
    limit: int | None = None,
    config_filters: str | None = None,
):
    """Get leaderboard data with optional filters."""
    # Get filter params - support multiple values
    campaigns = request.query_params.getlist("campaign")
    tasks = request.query_params.getlist("task")
    ablations = request.query_params.getlist("ablation")

    # Scan runs and evaluations
    runs = scan_table(LEADERBOARD_RUNS_TABLE)
    evaluations = scan_table(LEADERBOARD_EVALUATIONS_TABLE)

    # Build run lookup by run_id
    runs_by_id = {r["run_id"]: r for r in runs}

    # Join evaluations with runs
    results = []
    for eval_item in evaluations:
        run_id = eval_item.get("run_id")
        run = runs_by_id.get(run_id, {})

        # Apply filters
        if campaigns and eval_item.get("campaign_name") not in campaigns:
            continue
        if tasks and run.get("task_name") not in tasks:
            continue
        if ablations and run.get("ablation") not in ablations:
            continue
        if min_rollouts > 0 and eval_item.get("total_rollouts", 0) < min_rollouts:
            continue

        # Combine run and evaluation data
        entry = {
            "run_id": run_id,
            "scenario_name": run.get("scenario_name", ""),
            "task_name": run.get("task_name", ""),
            "s3_path": run.get("s3_path", ""),
            "model_type": run.get("model_type"),
            "ablation": run.get("ablation"),
            "learning_rate": run.get("learning_rate"),
            "batch_size": run.get("batch_size"),
            "run_timestamp": run.get("run_timestamp"),
            "wandb_link": run.get("wandb_link"),
            "has_config": run.get("has_config", False),
            "has_wandb": run.get("has_wandb", False),
            "campaign_name": eval_item.get("campaign_name"),
            "total_rollouts": eval_item.get("total_rollouts", 0),
            "successes": eval_item.get("successes", 0),
            "failures": eval_item.get("failures", 0),
            "success_rate": eval_item.get("success_rate", 0),
            "evaluated_at": eval_item.get("evaluated_at"),
        }
        results.append(entry)

    # Sort results
    reverse = descending
    if order_by == "success_rate":
        results.sort(key=lambda x: x.get("success_rate", 0), reverse=reverse)
    elif order_by == "total_rollouts":
        results.sort(key=lambda x: x.get("total_rollouts", 0), reverse=reverse)
    elif order_by == "task_name":
        results.sort(key=lambda x: x.get("task_name", ""), reverse=reverse)
    elif order_by == "campaign_name":
        results.sort(key=lambda x: x.get("campaign_name", ""), reverse=reverse)
    elif order_by == "evaluated_at":
        results.sort(key=lambda x: x.get("evaluated_at", ""), reverse=reverse)

    # Apply limit
    if limit:
        results = results[:limit]

    return {"success": True, "data": results, "count": len(results)}


@app.get("/api/campaigns")
async def get_campaigns():
    """Get all unique campaigns from evaluations."""
    evaluations = scan_table(LEADERBOARD_EVALUATIONS_TABLE)
    campaigns = {}
    for e in evaluations:
        name = e.get("campaign_name")
        if name and name not in campaigns:
            campaigns[name] = {
                "name": name,
                "id": name,
            }
    return {"success": True, "data": list(campaigns.values())}


@app.get("/api/tasks")
async def get_tasks():
    """Get all unique task names from runs."""
    runs = scan_table(LEADERBOARD_RUNS_TABLE)
    tasks = sorted(set(r.get("task_name") for r in runs if r.get("task_name")))
    return {"success": True, "data": tasks}


@app.get("/api/ablations")
async def get_ablations():
    """Get all unique ablation types from runs."""
    runs = scan_table(LEADERBOARD_RUNS_TABLE)
    ablations = sorted(set(r.get("ablation") for r in runs if r.get("ablation")))
    return {"success": True, "data": ablations}


@app.get("/api/config/{run_id}")
async def get_config(run_id: int):
    """Get config for a specific run."""
    runs = scan_table(LEADERBOARD_RUNS_TABLE)
    run = next((r for r in runs if r.get("run_id") == run_id), None)

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    config_json = run.get("config_json")
    if config_json:
        try:
            return {"success": True, "data": json.loads(config_json)}
        except json.JSONDecodeError:
            return {"success": True, "data": config_json}

    return {"success": False, "error": "Config not found"}


class CompareRequest(BaseModel):
    selections: list[dict]


@app.post("/api/compare")
async def compare_runs(req: CompareRequest):
    """Get comparison data for selected runs."""
    if not req.selections:
        raise HTTPException(status_code=400, detail="No selections provided")

    runs = scan_table(LEADERBOARD_RUNS_TABLE)
    evaluations = scan_table(LEADERBOARD_EVALUATIONS_TABLE)
    runs_by_id = {r["run_id"]: r for r in runs}

    results = []
    for sel in req.selections:
        run_id = sel.get("runId")
        campaign_name = sel.get("campaignName")

        run = runs_by_id.get(run_id, {})
        eval_item = next(
            (e for e in evaluations if e.get("run_id") == run_id and e.get("campaign_name") == campaign_name),
            None,
        )

        if eval_item:
            results.append(
                {
                    "run_id": run_id,
                    "scenario_name": run.get("scenario_name", ""),
                    "task_name": run.get("task_name", ""),
                    "ablation": run.get("ablation"),
                    "learning_rate": run.get("learning_rate"),
                    "batch_size": run.get("batch_size"),
                    "wandb_link": run.get("wandb_link"),
                    "campaign_name": campaign_name,
                    "total_rollouts": eval_item.get("total_rollouts", 0),
                    "successes": eval_item.get("successes", 0),
                    "success_rate": eval_item.get("success_rate", 0),
                }
            )

    return {"success": True, "data": results}


# ============================================================================
# Statistical Comparison API
# ============================================================================


def get_samples_for_eval(eval_id: str) -> list[dict]:
    """Get all sample results for a given eval_id."""
    table = dynamodb.Table(SAMPLE_RESULTS_TABLE)
    items = []
    try:
        response = table.query(
            IndexName="eval_id-index",
            KeyConditionExpression="eval_id = :eval_id",
            ExpressionAttributeValues={":eval_id": eval_id},
        )
        items.extend(response.get("Items", []))
        while "LastEvaluatedKey" in response:
            response = table.query(
                IndexName="eval_id-index",
                KeyConditionExpression="eval_id = :eval_id",
                ExpressionAttributeValues={":eval_id": eval_id},
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            items.extend(response.get("Items", []))
    except ClientError:
        pass
    return [decimal_to_python(item) for item in items]


def compute_beta_params(successes: int, total: int, alpha_prior: float = 1, beta_prior: float = 1):
    """Compute beta distribution parameters from success counts."""
    alpha = alpha_prior + successes
    beta = beta_prior + (total - successes)
    return alpha, beta


def compute_beta_quantiles(alpha: float, beta: float, quantiles: list[float]) -> list[float]:
    """Compute quantiles of beta distribution."""
    dist = stats.beta(alpha, beta)
    return [float(dist.ppf(q)) for q in quantiles]


def draw_beta_samples(alpha: float, beta: float, n_samples: int = 1000, seed: int = 42) -> list[float]:
    """Draw samples from beta distribution for violin plot."""
    rng = np.random.default_rng(seed)
    dist = stats.beta(alpha, beta)
    return dist.rvs(n_samples, random_state=rng).tolist()


def compare_two_proportions(n_a: int, k_a: int, n_b: int, k_b: int, alpha: float = 0.05) -> int:
    """Compare two proportions using z-test. Returns 1 if a>b, -1 if a<b, 0 if not significant."""
    if n_a == 0 or n_b == 0:
        return 0

    p_a = k_a / n_a
    p_b = k_b / n_b
    p_pooled = (k_a + k_b) / (n_a + n_b)

    if p_pooled == 0 or p_pooled == 1:
        return 0

    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0

    z = (p_a - p_b) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    if p_value < alpha:
        return 1 if p_a > p_b else -1
    return 0


def compute_cld(results_list: list[dict], alpha: float = 0.05) -> dict[str, str]:
    """
    Compute Compact Letter Display for statistical groupings.
    results_list: list of {label, successes, total}
    Returns dict: label -> letter(s)
    """
    labels = [r["label"] for r in results_list]
    num_models = len(results_list)

    if num_models < 2:
        return {labels[0]: "a"} if num_models == 1 else {}

    # Pairwise comparisons to find significant differences
    sig_pairs = []
    for i in range(num_models):
        for j in range(i + 1, num_models):
            r_i, r_j = results_list[i], results_list[j]
            if r_i["total"] == 0 or r_j["total"] == 0:
                continue
            cmp = compare_two_proportions(r_i["total"], r_i["successes"], r_j["total"], r_j["successes"], alpha)
            if cmp != 0:
                sig_pairs.append((labels[i], labels[j]))

    # Sort labels by success rate (descending)
    sr_pairs = []
    for r in results_list:
        sr = r["successes"] / r["total"] if r["total"] > 0 else 0
        sr_pairs.append((r["label"], sr))
    sr_pairs.sort(key=lambda x: x[1], reverse=True)
    sorted_labels = [p[0] for p in sr_pairs]

    # Assign CLD letters
    cld = {}
    current_letter = ord("a")
    assigned = set()

    for label in sorted_labels:
        if label in assigned:
            continue

        # Find all labels not significantly different from this one
        group = [label]
        for other in sorted_labels:
            if other == label or other in assigned:
                continue
            is_sig = (label, other) in sig_pairs or (other, label) in sig_pairs
            if not is_sig:
                group.append(other)

        letter = chr(current_letter)
        for g in group:
            if g not in cld:
                cld[g] = letter
            else:
                cld[g] += letter
        assigned.update(group)
        current_letter += 1

    return cld


class StatCompareRequest(BaseModel):
    selections: list[dict]  # [{runId, campaignName}, ...]
    n_samples: int = 500  # Number of samples for violin plots
    max_per_task: int = 5  # Maximum runs per task for violin plots


@app.post("/api/compare/stats")
async def compare_stats(req: StatCompareRequest):
    """
    Compute statistical comparison data for selected runs.
    Returns data for violin plots, scatter plots, and CLD.
    """
    if not req.selections:
        raise HTTPException(status_code=400, detail="No selections provided")

    runs = scan_table(LEADERBOARD_RUNS_TABLE)
    evaluations = scan_table(LEADERBOARD_EVALUATIONS_TABLE)
    runs_by_id = {r["run_id"]: r for r in runs}

    # Collect data for each selection
    results_by_task = {}  # task_name -> [{label, successes, total, alpha, beta, samples, ...}]
    all_results = []  # flat list for aggregate

    for sel in req.selections:
        run_id = sel.get("runId")
        campaign_name = sel.get("campaignName")
        eval_id = f"{run_id}_{campaign_name}"

        run = runs_by_id.get(run_id, {})
        eval_item = next(
            (e for e in evaluations if e.get("run_id") == run_id and e.get("campaign_name") == campaign_name),
            None,
        )

        if not eval_item:
            continue

        task_name = run.get("task_name", "unknown")
        ablation = run.get("ablation", "default")
        label = f"{ablation}"

        successes = eval_item.get("successes", 0)
        total = eval_item.get("total_rollouts", 0)

        # Try to get sample-level data for more accurate stats
        samples_data = get_samples_for_eval(eval_id)
        if samples_data:
            # Use actual sample data
            successes = sum(1 for s in samples_data if s.get("success", False))
            total = len(samples_data)

        # Compute beta parameters
        alpha_param, beta_param = compute_beta_params(successes, total)

        # Draw samples for violin plot
        violin_samples = draw_beta_samples(alpha_param, beta_param, req.n_samples)

        # Compute quantiles for box plot overlay
        quantiles = compute_beta_quantiles(alpha_param, beta_param, [0.025, 0.25, 0.5, 0.75, 0.975])

        result_entry = {
            "run_id": run_id,
            "campaign_name": campaign_name,
            "task_name": task_name,
            "label": label,
            "ablation": ablation,
            "successes": successes,
            "total": total,
            "success_rate": successes / total if total > 0 else 0,
            "alpha": alpha_param,
            "beta": beta_param,
            "violin_samples": violin_samples,
            "quantiles": {
                "q025": quantiles[0],
                "q25": quantiles[1],
                "median": quantiles[2],
                "q75": quantiles[3],
                "q975": quantiles[4],
            },
        }

        # Group by task
        if task_name not in results_by_task:
            results_by_task[task_name] = []
        results_by_task[task_name].append(result_entry)
        all_results.append(result_entry)

    # Check if any task has too many runs
    max_runs_per_task = max(len(results) for results in results_by_task.values()) if results_by_task else 0
    if max_runs_per_task > req.max_per_task:
        raise HTTPException(
            status_code=400,
            detail=f"Too many runs per task ({max_runs_per_task}). Maximum allowed is {req.max_per_task}. "
            f"Please select fewer runs or filter by task/ablation.",
        )

    # Compute CLD for each task
    cld_by_task = {}
    for task_name, task_results in results_by_task.items():
        cld = compute_cld(task_results)
        cld_by_task[task_name] = cld
        # Add CLD letter to each result
        for r in task_results:
            r["cld_letter"] = cld.get(r["label"], "")

    # Compute aggregate stats (all tasks combined per ablation)
    aggregate_by_ablation = {}
    for r in all_results:
        abl = r["ablation"]
        if abl not in aggregate_by_ablation:
            aggregate_by_ablation[abl] = {"successes": 0, "total": 0, "label": abl}
        aggregate_by_ablation[abl]["successes"] += r["successes"]
        aggregate_by_ablation[abl]["total"] += r["total"]

    # Compute beta params and samples for aggregates
    aggregate_results = []
    for abl, agg in aggregate_by_ablation.items():
        alpha_param, beta_param = compute_beta_params(agg["successes"], agg["total"])
        violin_samples = draw_beta_samples(alpha_param, beta_param, req.n_samples)
        quantiles = compute_beta_quantiles(alpha_param, beta_param, [0.025, 0.25, 0.5, 0.75, 0.975])

        aggregate_results.append(
            {
                "label": abl,
                "ablation": abl,
                "successes": agg["successes"],
                "total": agg["total"],
                "success_rate": agg["successes"] / agg["total"] if agg["total"] > 0 else 0,
                "alpha": alpha_param,
                "beta": beta_param,
                "violin_samples": violin_samples,
                "quantiles": {
                    "q025": quantiles[0],
                    "q25": quantiles[1],
                    "median": quantiles[2],
                    "q75": quantiles[3],
                    "q975": quantiles[4],
                },
            }
        )

    # Compute CLD for aggregates
    aggregate_cld = compute_cld(aggregate_results)
    for r in aggregate_results:
        r["cld_letter"] = aggregate_cld.get(r["label"], "")

    # Sort tasks by average success rate
    task_order = sorted(
        results_by_task.keys(),
        key=lambda t: np.mean([r["success_rate"] for r in results_by_task[t]]),
        reverse=True,
    )

    return {
        "success": True,
        "data": {
            "by_task": results_by_task,
            "task_order": task_order,
            "cld_by_task": cld_by_task,
            "aggregate": aggregate_results,
            "aggregate_cld": aggregate_cld,
        },
    }


@app.get("/api/config-schema")
async def get_config_schema():
    """Get config field paths with their types and value distributions."""
    runs = scan_table(LEADERBOARD_RUNS_TABLE)
    schema: dict[str, Any] = {}

    for run in runs:
        config_json = run.get("config_json")
        if config_json:
            try:
                config = json.loads(config_json) if isinstance(config_json, str) else config_json
                fields = flatten_config(config)
                for path, value, field_type in fields:
                    if path not in schema:
                        schema[path] = {"type": field_type, "values": []}
                    # Track unique values (limit to avoid huge lists)
                    existing = [v["value"] for v in schema[path]["values"]]
                    if value not in existing and len(existing) < 50:
                        schema[path]["values"].append({"value": value, "count": 1})
                    elif value in existing:
                        for v in schema[path]["values"]:
                            if v["value"] == value:
                                v["count"] += 1
                                break
            except (json.JSONDecodeError, TypeError):
                pass

    return {"success": True, "data": schema}


def flatten_config(config: dict, prefix: str = "") -> list[tuple[str, str, str]]:
    """Flatten a nested config into (path, value, type) tuples."""
    results = []

    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            results.extend(flatten_config(value, path))
        elif isinstance(value, list):
            results.append((path, json.dumps(value), "list"))
        elif isinstance(value, bool):
            results.append((path, str(value).lower(), "boolean"))
        elif isinstance(value, (int, float)):
            results.append((path, str(value), "number"))
        elif value is None:
            results.append((path, None, "null"))
        else:
            results.append((path, str(value), "string"))

    return results


class ConfigDiffRequest(BaseModel):
    run_ids: list[int]


@app.post("/api/config-diff")
async def get_config_diff(req: ConfigDiffRequest):
    """Get config diff for multiple runs."""
    if len(req.run_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 run IDs")

    runs = scan_table(LEADERBOARD_RUNS_TABLE)
    evaluations = scan_table(LEADERBOARD_EVALUATIONS_TABLE)
    runs_by_id = {r["run_id"]: r for r in runs}

    # Get configs and flatten
    fields_by_path: dict[str, dict[int, str]] = {}
    runs_info = []

    for run_id in req.run_ids:
        run = runs_by_id.get(run_id, {})
        eval_item = next((e for e in evaluations if e.get("run_id") == run_id), None)

        runs_info.append(
            {
                "run_id": run_id,
                "scenario_name": run.get("scenario_name", ""),
                "success_rate": eval_item.get("success_rate") if eval_item else None,
            }
        )

        config_json = run.get("config_json")
        if config_json:
            try:
                config = json.loads(config_json) if isinstance(config_json, str) else config_json
                fields = flatten_config(config)
                for path, value, _ in fields:
                    if path not in fields_by_path:
                        fields_by_path[path] = {}
                    fields_by_path[path][run_id] = value
            except (json.JSONDecodeError, TypeError):
                pass

    # Find differing fields
    differing_fields = []
    for path, values_by_run in fields_by_path.items():
        unique_values = set(values_by_run.values())
        if len(unique_values) > 1:
            differing_fields.append(
                {
                    "field_path": path,
                    "values": {str(k): v for k, v in values_by_run.items()},
                }
            )

    differing_fields.sort(key=lambda x: x["field_path"])

    return {
        "success": True,
        "data": {
            "fields": differing_fields,
            "runs": runs_info,
        },
    }


@app.get("/api/stats")
async def get_stats():
    """Get overall statistics."""
    runs = scan_table(LEADERBOARD_RUNS_TABLE)
    evaluations = scan_table(LEADERBOARD_EVALUATIONS_TABLE)

    if not evaluations:
        return {
            "success": True,
            "data": {
                "total_runs": 0,
                "total_rollouts": 0,
                "avg_success_rate": 0,
                "tasks": 0,
                "campaigns": 0,
            },
        }

    total_rollouts = sum(e.get("total_rollouts", 0) for e in evaluations)
    avg_success = sum(e.get("success_rate", 0) for e in evaluations) / len(evaluations) if evaluations else 0
    tasks = set(r.get("task_name") for r in runs if r.get("task_name"))
    campaigns = set(e.get("campaign_name") for e in evaluations if e.get("campaign_name"))

    return {
        "success": True,
        "data": {
            "total_runs": len(evaluations),
            "total_rollouts": total_rollouts,
            "avg_success_rate": avg_success,
            "tasks": len(tasks),
            "campaigns": len(campaigns),
        },
    }


@app.get("/api/export")
async def export_data():
    """Export leaderboard to JSON."""
    runs = scan_table(LEADERBOARD_RUNS_TABLE)
    evaluations = scan_table(LEADERBOARD_EVALUATIONS_TABLE)

    tasks = sorted(set(r.get("task_name") for r in runs if r.get("task_name")))
    ablations = sorted(set(r.get("ablation") for r in runs if r.get("ablation")))
    campaigns = list(set(e.get("campaign_name") for e in evaluations if e.get("campaign_name")))

    return {
        "exported_at": datetime.now().isoformat(),
        "campaigns": [{"name": c} for c in campaigns],
        "tasks": tasks,
        "ablations": ablations,
        "entries": evaluations,
    }


# ============================================================================
# Static File Serving
# ============================================================================

FRONTEND_DIR = Path(__file__).parent


@app.get("/")
async def root():
    """Serve the main dashboard page."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/leaderboard")
async def leaderboard_page():
    """Serve the leaderboard page."""
    return FileResponse(FRONTEND_DIR / "leaderboard" / "index.html")


@app.get("/leaderboard/{path:path}")
async def serve_leaderboard_static(path: str):
    """Serve leaderboard static files."""
    file_path = FRONTEND_DIR / "leaderboard" / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")


def main():
    parser = argparse.ArgumentParser(description="VLA Foundry Dashboard & Leaderboard Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"  Dashboard: http://{args.host}:{args.port}/")
    print(f"  Leaderboard: http://{args.host}:{args.port}/leaderboard")
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
