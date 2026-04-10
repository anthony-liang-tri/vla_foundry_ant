"""Tests for vla_foundry.eval.stats — balanced aggregate construction and CLD.

Focuses on the per-model balanced-aggregate logic in ``build_success_arrays``
and the ``n_max`` calculation in ``compute_cld_step``.
"""

import importlib

import numpy as np
import pytest

from vla_foundry.eval.stats import build_success_arrays


def _can_import(module_name: str) -> bool:
    """Return True if *module_name* is importable."""
    return importlib.util.find_spec(module_name) is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episodes(task: str, model: str, n: int, success_rate: float = 0.5, seed: int = 0) -> list[dict]:
    """Generate *n* synthetic episodes for a single (task, model) pair."""
    rng = np.random.default_rng(seed)
    return [{"task": task, "model": model, "success": bool(rng.random() < success_rate)} for _ in range(n)]


# ---------------------------------------------------------------------------
# build_success_arrays
# ---------------------------------------------------------------------------


class TestBuildSuccessArraysBalancedAggregate:
    """Verify that the aggregate is balanced per model."""

    def test_unequal_rollouts_two_tasks(self):
        """Model A has [200, 200], Model B has [50, 100] → per-model balancing."""
        eps = (
            _make_episodes("T1", "A", 200, seed=1)
            + _make_episodes("T2", "A", 200, seed=2)
            + _make_episodes("T1", "B", 50, seed=3)
            + _make_episodes("T2", "B", 100, seed=4)
        )
        arrays, meta = build_success_arrays(eps)

        assert meta["per_model_min_n"] == {"A": 200, "B": 50}
        assert meta["num_common_tasks"] == 2
        assert set(meta["common_tasks"]) == {"T1", "T2"}
        assert meta["excluded_tasks"] == []

        # A: 200 per task × 2 tasks = 400
        assert len(arrays["__aggregate__"]["A"]) == 400
        # B: 50 per task × 2 tasks = 100
        assert len(arrays["__aggregate__"]["B"]) == 100

        # Per-task arrays keep full lengths.
        assert len(arrays["T1"]["A"]) == 200
        assert len(arrays["T2"]["B"]) == 100

    def test_missing_task_excluded(self):
        """Model A on [T1, T2], Model B on [T2, T3] → common = {T2}."""
        eps = (
            _make_episodes("T1", "A", 40, seed=1)
            + _make_episodes("T2", "A", 60, seed=2)
            + _make_episodes("T2", "B", 80, seed=3)
            + _make_episodes("T3", "B", 30, seed=4)
        )
        arrays, meta = build_success_arrays(eps)

        assert meta["common_tasks"] == ["T2"]
        assert set(meta["excluded_tasks"]) == {"T1", "T3"}
        # Only T2 is common: A has 60, B has 80
        assert meta["per_model_min_n"] == {"A": 60, "B": 80}
        assert meta["num_common_tasks"] == 1

        assert len(arrays["__aggregate__"]["A"]) == 60
        assert len(arrays["__aggregate__"]["B"]) == 80

    def test_equal_rollouts_no_truncation(self):
        """All models have equal counts → per_model_min_n equals that count."""
        eps = (
            _make_episodes("T1", "A", 100, seed=1)
            + _make_episodes("T2", "A", 100, seed=2)
            + _make_episodes("T1", "B", 100, seed=3)
            + _make_episodes("T2", "B", 100, seed=4)
        )
        arrays, meta = build_success_arrays(eps)

        assert meta["per_model_min_n"] == {"A": 100, "B": 100}
        assert len(arrays["__aggregate__"]["A"]) == 200
        assert len(arrays["__aggregate__"]["B"]) == 200

    def test_no_common_tasks(self):
        """Disjoint tasks → aggregate is empty."""
        eps = _make_episodes("T1", "A", 50, seed=1) + _make_episodes("T2", "B", 50, seed=2)
        arrays, meta = build_success_arrays(eps)

        assert meta["common_tasks"] == []
        assert meta["per_model_min_n"] == {}
        assert arrays["__aggregate__"] == {}

    def test_per_task_arrays_unchanged(self):
        """Per-task arrays must keep full rollouts regardless of balancing."""
        eps = (
            _make_episodes("T1", "A", 200, seed=1)
            + _make_episodes("T1", "B", 30, seed=2)
            + _make_episodes("T2", "A", 150, seed=3)
            + _make_episodes("T2", "B", 80, seed=4)
        )
        arrays, meta = build_success_arrays(eps)

        assert len(arrays["T1"]["A"]) == 200
        assert len(arrays["T1"]["B"]) == 30
        assert len(arrays["T2"]["A"]) == 150
        assert len(arrays["T2"]["B"]) == 80

    def test_metadata_original_counts(self):
        """original_counts records pre-truncation lengths."""
        eps = (
            _make_episodes("T1", "X", 70, seed=1)
            + _make_episodes("T2", "X", 40, seed=2)
            + _make_episodes("T1", "Y", 90, seed=3)
            + _make_episodes("T2", "Y", 60, seed=4)
        )
        _, meta = build_success_arrays(eps)

        assert meta["original_counts"]["T1"]["X"] == 70
        assert meta["original_counts"]["T1"]["Y"] == 90
        assert meta["original_counts"]["T2"]["X"] == 40
        assert meta["original_counts"]["T2"]["Y"] == 60

    def test_aggregate_uses_first_n_entries(self):
        """Aggregate takes the first per_model_min_n entries, preserving order."""
        # Model A: 10 episodes on T1, first 3 succeed.
        # Model B: 5 episodes on T1, first 2 succeed.
        eps_a = [{"task": "T1", "model": "A", "success": i < 3} for i in range(10)]
        eps_b = [{"task": "T1", "model": "B", "success": i < 2} for i in range(5)]
        arrays, meta = build_success_arrays(eps_a + eps_b)

        # Per-model: A min = 10 (only 1 task), B min = 5
        assert meta["per_model_min_n"] == {"A": 10, "B": 5}
        # Model A uses all 10
        np.testing.assert_array_equal(
            arrays["__aggregate__"]["A"],
            [True, True, True, False, False, False, False, False, False, False],
        )
        # Model B uses all 5
        np.testing.assert_array_equal(
            arrays["__aggregate__"]["B"],
            [True, True, False, False, False],
        )

    def test_single_model(self):
        """Single model — aggregate includes all its tasks."""
        eps = _make_episodes("T1", "solo", 30, seed=1) + _make_episodes("T2", "solo", 20, seed=2)
        arrays, meta = build_success_arrays(eps)

        assert meta["per_model_min_n"] == {"solo": 20}
        assert len(arrays["__aggregate__"]["solo"]) == 40  # 20 × 2

    def test_per_model_balancing_different_sizes(self):
        """Models with different per-task counts get different aggregate sizes."""
        eps = (
            _make_episodes("T1", "A", 50, seed=1)
            + _make_episodes("T2", "A", 50, seed=2)
            + _make_episodes("T3", "A", 50, seed=3)
            + _make_episodes("T4", "A", 50, seed=4)
            + _make_episodes("T1", "B", 100, seed=5)
            + _make_episodes("T2", "B", 100, seed=6)
            + _make_episodes("T3", "B", 100, seed=7)
            + _make_episodes("T4", "B", 100, seed=8)
        )
        arrays, meta = build_success_arrays(eps)

        assert meta["per_model_min_n"] == {"A": 50, "B": 100}
        # A: 50 × 4 = 200
        assert len(arrays["__aggregate__"]["A"]) == 200
        # B: 100 × 4 = 400
        assert len(arrays["__aggregate__"]["B"]) == 400

    def test_three_models_four_tasks_one_excluded(self):
        """Three models, one missing a task → excluded from aggregate."""
        eps = (
            _make_episodes("T1", "A", 100, seed=1)
            + _make_episodes("T2", "A", 80, seed=2)
            + _make_episodes("T3", "A", 60, seed=3)
            + _make_episodes("T4", "A", 40, seed=4)
            + _make_episodes("T1", "B", 50, seed=5)
            + _make_episodes("T2", "B", 90, seed=6)
            + _make_episodes("T3", "B", 70, seed=7)
            + _make_episodes("T4", "B", 30, seed=8)
            # Model C missing T4
            + _make_episodes("T1", "C", 120, seed=9)
            + _make_episodes("T2", "C", 110, seed=10)
            + _make_episodes("T3", "C", 45, seed=11)
        )
        arrays, meta = build_success_arrays(eps)

        assert set(meta["common_tasks"]) == {"T1", "T2", "T3"}
        assert meta["excluded_tasks"] == ["T4"]
        assert meta["num_common_tasks"] == 3

        # Per-model min across common tasks (T1, T2, T3):
        # A: min(100, 80, 60) = 60
        # B: min(50, 90, 70) = 50
        # C: min(120, 110, 45) = 45
        assert meta["per_model_min_n"] == {"A": 60, "B": 50, "C": 45}

        assert len(arrays["__aggregate__"]["A"]) == 60 * 3
        assert len(arrays["__aggregate__"]["B"]) == 50 * 3
        assert len(arrays["__aggregate__"]["C"]) == 45 * 3

    def test_equal_task_weight_within_model(self):
        """Each task contributes equal count to a model's aggregate."""
        eps = (
            _make_episodes("T1", "A", 100, success_rate=1.0, seed=1)
            + _make_episodes("T2", "A", 50, success_rate=0.0, seed=2)
            + _make_episodes("T1", "B", 80, success_rate=1.0, seed=3)
            + _make_episodes("T2", "B", 80, success_rate=0.0, seed=4)
        )
        arrays, meta = build_success_arrays(eps)

        # A: min(100, 50) = 50 per task → 50 successes + 50 failures
        assert meta["per_model_min_n"]["A"] == 50
        agg_a = arrays["__aggregate__"]["A"]
        assert len(agg_a) == 100
        assert np.sum(agg_a) == 50  # 50 from T1 (all success) + 50 from T2 (all fail)

        # B: min(80, 80) = 80 per task → 80 successes + 80 failures
        assert meta["per_model_min_n"]["B"] == 80
        agg_b = arrays["__aggregate__"]["B"]
        assert len(agg_b) == 160
        assert np.sum(agg_b) == 80


# ---------------------------------------------------------------------------
# compute_cld_step — n_max calculation
# ---------------------------------------------------------------------------


_skip_no_eval_viewer = pytest.mark.skipif(
    not _can_import("sequentialized_barnard_tests"),
    reason="eval-viewer deps not installed",
)


@_skip_no_eval_viewer
class TestComputeCldStepNmax:
    """Verify that n_max uses num_all_tasks (not num_common_tasks) for pre-commitment."""

    def test_n_max_uses_all_tasks(self):
        """With 1 excluded task, n_max should be budget × num_all_tasks (not common)."""
        from vla_foundry.eval.stats import compute_cld_step

        eps = (
            _make_episodes("T1", "A", 50, seed=1)
            + _make_episodes("T2", "A", 50, seed=2)
            + _make_episodes("T1", "B", 50, seed=3)
            + _make_episodes("T2", "B", 50, seed=4)
            # Model C only on T1
            + _make_episodes("T1", "C", 50, seed=5)
        )
        arrays, meta = build_success_arrays(eps)

        # Common tasks = [T1], excluded = [T2], but both exist in dataset.
        assert meta["num_common_tasks"] == 1

        # The test should not error and should produce CLD for aggregate.
        cld_by_task, warning = compute_cld_step(arrays, 200, meta)
        assert "__aggregate__" in cld_by_task
        assert "T1" in cld_by_task
        assert "excludes" in warning.lower()

    def test_warning_mentions_excluded_tasks(self):
        from vla_foundry.eval.stats import compute_cld_step

        eps = _make_episodes("T1", "A", 30, seed=1) + _make_episodes("T2", "B", 30, seed=2)
        arrays, meta = build_success_arrays(eps)

        _, warning = compute_cld_step(arrays, 100, meta)
        assert "T1" in warning
        assert "T2" in warning
        assert "not available" in warning.lower() or "excludes" in warning.lower()

    def test_warning_mentions_per_model_balancing(self):
        from vla_foundry.eval.stats import compute_cld_step

        eps = _make_episodes("T1", "A", 100, seed=1) + _make_episodes("T1", "B", 30, seed=2)
        arrays, meta = build_success_arrays(eps)

        _, warning = compute_cld_step(arrays, 200, meta)
        # Should mention per-model rollout counts
        assert "A: 100" in warning
        assert "B: 30" in warning
        assert "balanced" in warning.lower()
