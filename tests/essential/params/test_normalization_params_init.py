"""
Tests for NormalizationParams.init_shared_attributes and resolve_normalization_timesteps.

init_shared_attributes is pure (no filesystem I/O). resolve_normalization_timesteps
reads the dataset-side preprocessing_config.yaml and validates the requested window
against what the data supports. Pre-set timesteps are preserved.
"""

import os
from types import SimpleNamespace

import pytest

from vla_foundry.params.resolve import resolve_normalization_timesteps
from vla_foundry.params.robotics.normalization_params import NormalizationParams


@pytest.fixture
def dataset_dir():
    return os.path.join(os.path.dirname(__file__), "..", "test_assets", "small_lbm_dataset")


@pytest.fixture
def stats_path(dataset_dir):
    return os.path.join(dataset_dir, "stats.json")


def _make_cfg(*, dataset_statistics, past, future):
    return SimpleNamespace(
        data=SimpleNamespace(
            proprioception_fields=["robot__action__poses__right::panda__xyz"],
            action_fields=["robot__action__poses__right::panda__xyz"],
            use_point_cloud=False,
            dataset_statistics=dataset_statistics,
            lowdim_past_timesteps=past,
            lowdim_future_timesteps=future,
        ),
    )


def test_init_shared_attributes_is_pure():
    """init_shared_attributes must not read the dataset; pre-set timesteps stay as-is."""
    norm_params = NormalizationParams(
        enabled=True,
        method="std",
        scope="global",
        lowdim_past_timesteps=1,
        lowdim_future_timesteps=14,
    )
    # Unreachable path proves no I/O happens in init_shared_attributes.
    cfg = _make_cfg(dataset_statistics=["/nonexistent/path/stats.json"], past=1, future=14)

    norm_params.init_shared_attributes(cfg)

    assert norm_params.lowdim_past_timesteps == 1
    assert norm_params.lowdim_future_timesteps == 14


def test_resolve_preserves_preset_timesteps(stats_path):
    """Pre-set timesteps survive resolve_normalization_timesteps; the dataset-side window is still validated."""
    norm_params = NormalizationParams(
        enabled=True,
        method="std",
        scope="global",
        lowdim_past_timesteps=1,
        lowdim_future_timesteps=14,
    )
    cfg = _make_cfg(dataset_statistics=[stats_path], past=1, future=14)

    norm_params.init_shared_attributes(cfg)
    resolve_normalization_timesteps(norm_params, cfg.data)

    assert norm_params.lowdim_past_timesteps == 1
    assert norm_params.lowdim_future_timesteps == 14


def test_resolve_fills_timesteps_when_unset(stats_path):
    """Unset timesteps must be populated from preprocessing_config.yaml."""
    norm_params = NormalizationParams(
        enabled=True,
        method="std",
        scope="global",
        lowdim_past_timesteps=None,
        lowdim_future_timesteps=None,
    )
    cfg = _make_cfg(dataset_statistics=[stats_path], past=None, future=None)

    norm_params.init_shared_attributes(cfg)
    resolve_normalization_timesteps(norm_params, cfg.data)

    # Values come from tests/essential/test_assets/small_lbm_dataset/preprocessing_config.yaml
    assert norm_params.lowdim_past_timesteps == 1
    assert norm_params.lowdim_future_timesteps == 14


def test_resolve_rejects_timesteps_exceeding_available(stats_path):
    """Requesting a window wider than the dataset supports must raise — no silent degradation."""
    norm_params = NormalizationParams(
        enabled=True,
        method="std",
        scope="global",
        lowdim_past_timesteps=999,
        lowdim_future_timesteps=14,
    )
    cfg = _make_cfg(dataset_statistics=[stats_path], past=999, future=14)

    norm_params.init_shared_attributes(cfg)

    with pytest.raises(ValueError, match="lowdim_past_timesteps"):
        resolve_normalization_timesteps(norm_params, cfg.data)


def test_resolve_rejects_data_timesteps_exceeding_available(stats_path):
    """The data-side loading window must be validated even when normalization uses a smaller window."""
    norm_params = NormalizationParams(
        enabled=True,
        method="std",
        scope="global",
        lowdim_past_timesteps=1,
        lowdim_future_timesteps=14,
    )
    cfg = _make_cfg(dataset_statistics=[stats_path], past=999, future=14)

    norm_params.init_shared_attributes(cfg)

    with pytest.raises(ValueError, match="lowdim_past_timesteps"):
        resolve_normalization_timesteps(norm_params, cfg.data)
