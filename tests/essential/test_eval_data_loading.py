"""Tests for vla_foundry.eval.data_loading using dashboard_fixtures/.

Each subdirectory under dashboard_fixtures/ is a self-contained rollout root
with the canonical layout: {model}/{Task}/rollouts/{timestamp}/results.json

load_episodes enforces the following rules:
- Layout: Only accepts {model}/{Task}/rollouts/{timestamp}/results.json.
  Anything else raises ValueError.
- max_sample_size_per_model: Required in every results.json — missing field
  raises ValueError.
- Conflicting budgets: Raises ValueError if different max_sample_size_per_model
  values appear across files (globally) or within the same rollouts directory.
- Overlapping episodes: When multiple timestamped runs exist under the same
  model/Task/rollouts/ with overlapping (skill_type, scenario_index) pairs,
  raises ValueError. Non-overlapping episodes across files are combined.
- Duplicate episodes: Hard error if the same (task, model, demo_id) appears
  after deduplication.
- Budget exceeded: Raises ValueError if any (task, model) pair has more
  completed episodes than max_sample_size_per_model.
"""

from pathlib import Path

import pytest

from vla_foundry.eval.data_loading import aggregate_episodes, load_episodes

FIXTURES = Path(__file__).resolve().parent / "test_assets" / "dashboard_fixtures"


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestMixedPendingCrashedSuccess:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.eps, self.pending, self.crashed, self.mss = load_episodes(FIXTURES / "mixed_pending_crashed_success")

    def test_completed_episodes(self):
        # 5 evaluations: 2 success, 1 failure, 1 pending, 1 crashed
        # load_episodes returns only completed (success/failure) episodes
        assert len(self.eps) == 3

    def test_pending_tracked(self):
        assert sum(v for v in self.pending.values()) > 0

    def test_crashed_tracked(self):
        assert sum(v for v in self.crashed.values()) > 0

    def test_max_sample_size(self):
        assert self.mss == 50

    def test_aggregate(self):
        stats = aggregate_episodes(self.eps, pending_by=self.pending, crashed_by=self.crashed)
        assert len(stats) == 1
        s = stats[0]
        assert s["successes"] == 2
        assert s["total"] == 3
        assert s["pending"] == 1
        assert s["crashed"] == 1


class TestCrashedEpisodes:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.eps, self.pending, self.crashed, self.mss = load_episodes(FIXTURES / "crashed_episodes")

    def test_only_completed_returned(self):
        # 3 evaluations: 2 crashed, 1 success
        assert len(self.eps) == 1

    def test_crashed_count(self):
        assert self.crashed[("task_a", "model_a")] == 2


class TestPendingEpisodes:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.eps, self.pending, self.crashed, self.mss = load_episodes(FIXTURES / "pending_episodes")

    def test_no_completed_episodes(self):
        assert len(self.eps) == 0

    def test_pending_count(self):
        assert self.pending[("task_a", "model_a")] == 3


class TestEmptyEvaluations:
    def test_loads_with_no_episodes(self):
        eps, pending, crashed, mss = load_episodes(FIXTURES / "empty_evaluations")
        assert len(eps) == 0
        assert mss == 50


# ---------------------------------------------------------------------------
# Error tests
# ---------------------------------------------------------------------------


class TestNoMaxSampleSize:
    def test_raises(self):
        with pytest.raises(ValueError, match="max_sample_size_per_model"):
            load_episodes(FIXTURES / "no_max_sample_size")


class TestConflictingMaxSampleSize:
    def test_raises(self):
        with pytest.raises(ValueError, match="Conflicting max_sample_size_per_model"):
            load_episodes(FIXTURES / "conflicting_max_sample_size_across_tasks")


class TestMismatchedMaxSampleSize:
    """Two runs for the same model+task with different max_sample_size values."""

    def test_raises(self):
        with pytest.raises(ValueError, match="Mismatched max_sample_size_per_model"):
            load_episodes(FIXTURES / "mismatched_max_sample_size")


class TestOverlappingEpisodes:
    """Two timestamped runs with overlapping (skill_type, scenario_index) pairs."""

    def test_raises(self):
        with pytest.raises(ValueError, match="Overlapping episode"):
            load_episodes(FIXTURES / "stale_overlapping_runs")


class TestExceedsMaxSampleSize:
    """More completed episodes than the declared max_sample_size_per_model budget."""

    def test_raises(self):
        with pytest.raises(ValueError, match="exceeds max_sample_size_per_model"):
            load_episodes(FIXTURES / "exceeds_max_sample_size")


class TestNoResultsFiles:
    def test_empty_dir(self, tmp_path):
        eps, pending, crashed, mss = load_episodes(tmp_path)
        assert len(eps) == 0
