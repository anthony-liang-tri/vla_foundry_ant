"""
Standalone evaluation campaign orchestrator.

This package provides tools to run distributed evaluation campaigns on Ray clusters
without dependencies on the Anzu monorepo.
"""

from .ray_policy_runner import (
    EVAL_DOCKER_LABEL,
    TaskSpec,
    load_tasks_from_file,
)
from .success_stats import (
    calculate_confidence_interval,
    collect_multi_rollout_success_stats,
)

__all__ = [
    "EVAL_DOCKER_LABEL",
    "TaskSpec",
    "load_tasks_from_file",
    "calculate_confidence_interval",
    "collect_multi_rollout_success_stats",
]
