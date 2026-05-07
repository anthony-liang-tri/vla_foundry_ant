"""
Tests for RoboticsDataParams initialization vs. dataset-derived field resolution.

The pure-config-load refactor splits RoboticsDataParams setup into two phases:

  1. ``__post_init__`` (and its ``_post_init_impl``) is pure — no filesystem
     I/O. It runs on every instantiation, including when loading a saved
     checkpoint config on a machine that doesn't have access to the original
     training dataset paths.

  2. ``resolve_robotics_data_fields(data)`` (free function in
     ``vla_foundry.params.resolve``) performs the dataset-side I/O — reading
     ``preprocessing_config.yaml`` and the stats file — to fill in derived
     fields that were not pre-populated. It is called explicitly by any caller
     that holds a partially-specified config and has the datasets on disk.

These tests pin both halves.
"""

import os

import pytest

from vla_foundry.params.data_params import RoboticsDataParams
from vla_foundry.params.distributed_params import DistributedParams
from vla_foundry.params.hyper_params import HyperParams
from vla_foundry.params.model_params import ManiFlowParams
from vla_foundry.params.resolve import resolve_robotics_data_fields
from vla_foundry.params.robotics.normalization_params import FieldNormalizationParams, NormalizationParams
from vla_foundry.params.train_experiment_params import TrainExperimentParams

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "test_assets", "small_lbm_dataset")
STATS_PATH = os.path.join(DATASET_DIR, "stats.json")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.jsonl")

# These fields exist in tests/essential/test_assets/small_lbm_dataset/stats.json
ACTION_FIELDS = ["robot__action__poses__right::panda__xyz"]
PROPRIO_FIELDS = ["robot__actual__poses__right::panda__xyz"]
EXPECTED_ACTION_DIM = 3
EXPECTED_PROPRIO_DIM = 3

# Values authored in small_lbm_dataset/preprocessing_config.yaml
EXPECTED_CAMERA_NAMES = [
    "scene_right_0",
    "scene_left_0",
    "wrist_left_minus",
    "wrist_left_plus",
    "wrist_right_minus",
    "wrist_right_plus",
]
EXPECTED_IMAGE_INDICES = [-1, 0]
EXPECTED_POINT_CLOUD_NUM_POINTS = 4096


def _make_normalization(*, past=1, future=14):
    return NormalizationParams(
        enabled=True,
        method="std",
        scope="global",
        lowdim_past_timesteps=past,
        lowdim_future_timesteps=future,
        field_configs={
            PROPRIO_FIELDS[0]: FieldNormalizationParams(method="std", scope="global"),
            ACTION_FIELDS[0]: FieldNormalizationParams(method="std", scope="global"),
        },
    )


def _make_train_cfg_for_resolution():
    data = RoboticsDataParams(
        dataset_manifest=[MANIFEST_PATH],
        dataset_statistics=[STATS_PATH],
        dataset_modality=["robotics"],
        dataset_weighting=[1.0],
        proprioception_fields=PROPRIO_FIELDS,
        action_fields=ACTION_FIELDS,
        camera_names=["rgb"],
        image_indices=[-1, 0],
        image_names=["rgb_t-1", "rgb_t0"],
        action_dim=EXPECTED_ACTION_DIM,
        proprioception_dim=EXPECTED_PROPRIO_DIM,
        lowdim_past_timesteps=1,
        lowdim_future_timesteps=14,
        normalization=_make_normalization(),
    )
    return TrainExperimentParams(
        total_train_samples=1,
        data=data,
        distributed=DistributedParams(world_size=1),
        hparams=HyperParams(global_batch_size=1, per_gpu_batch_size=1, seed=42),
        model=ManiFlowParams(),
    )


def test_post_init_does_no_io_when_derived_fields_set(monkeypatch):
    """_post_init_impl must not touch disk when every derived field is pre-populated."""

    def _fail(*args, **kwargs):
        raise AssertionError(f"should not be called: args={args!r} kwargs={kwargs!r}")

    # Any filesystem helper params/ might reach for should stay unreachable.
    monkeypatch.setattr("vla_foundry.params.resolve.yaml_load", _fail)
    monkeypatch.setattr("vla_foundry.data.robotics.normalization.RoboticsNormalizer", _fail)

    params = RoboticsDataParams(
        dataset_manifest=["/nonexistent/path/manifest.jsonl"],
        dataset_statistics=["/nonexistent/path/stats.json"],
        dataset_modality=["robotics"],
        dataset_weighting=[1.0],
        proprioception_fields=PROPRIO_FIELDS,
        action_fields=ACTION_FIELDS,
        camera_names=["rgb"],
        image_indices=[-1, 0],
        image_names=["rgb_t-1", "rgb_t0"],
        action_dim=EXPECTED_ACTION_DIM,
        proprioception_dim=EXPECTED_PROPRIO_DIM,
        lowdim_past_timesteps=1,
        lowdim_future_timesteps=14,
        normalization=_make_normalization(),
    )

    assert params.action_dim == EXPECTED_ACTION_DIM
    assert params.proprioception_dim == EXPECTED_PROPRIO_DIM
    assert params.camera_names == ["rgb"]
    assert params.image_indices == [-1, 0]
    assert params.image_names == ["rgb_t-1", "rgb_t0"]


def test_resolve_fills_camera_names_when_unset():
    """resolve_robotics_data_fields populates camera_names/image_indices/image_names
    from the dataset-side preprocessing_config.yaml when they are unset."""
    params = RoboticsDataParams(
        dataset_manifest=[MANIFEST_PATH],
        dataset_statistics=[STATS_PATH],
        dataset_modality=["robotics"],
        dataset_weighting=[1.0],
        proprioception_fields=PROPRIO_FIELDS,
        action_fields=ACTION_FIELDS,
        camera_names=[],
        image_indices=[],
        image_names=[],
        action_dim=EXPECTED_ACTION_DIM,
        proprioception_dim=EXPECTED_PROPRIO_DIM,
        lowdim_past_timesteps=1,
        lowdim_future_timesteps=14,
        normalization=_make_normalization(),
    )

    resolve_robotics_data_fields(params)

    assert params.camera_names == EXPECTED_CAMERA_NAMES
    assert params.image_indices == EXPECTED_IMAGE_INDICES
    expected_image_names = [f"{c}_t{i}" for i in EXPECTED_IMAGE_INDICES for c in EXPECTED_CAMERA_NAMES]
    assert params.image_names == expected_image_names


def test_resolve_fills_action_dim_when_unset():
    """resolve_robotics_data_fields computes action_dim/proprioception_dim from the
    stats file when they are left unset at construction time."""
    params = RoboticsDataParams(
        dataset_manifest=[MANIFEST_PATH],
        dataset_statistics=[STATS_PATH],
        dataset_modality=["robotics"],
        dataset_weighting=[1.0],
        proprioception_fields=PROPRIO_FIELDS,
        action_fields=ACTION_FIELDS,
        camera_names=["rgb"],
        image_indices=[-1, 0],
        image_names=["rgb_t-1", "rgb_t0"],
        action_dim=None,
        proprioception_dim=None,
        lowdim_past_timesteps=1,
        lowdim_future_timesteps=14,
        normalization=_make_normalization(),
    )

    resolve_robotics_data_fields(params)

    assert params.action_dim == EXPECTED_ACTION_DIM
    assert params.proprioception_dim == EXPECTED_PROPRIO_DIM


def test_resolve_derives_image_names_without_preprocessing_io(monkeypatch):
    """image_names can be derived from explicit camera_names/image_indices without reading dataset files."""

    def _fail(*args, **kwargs):
        raise AssertionError(f"should not be called: args={args!r} kwargs={kwargs!r}")

    monkeypatch.setattr("vla_foundry.params.resolve.yaml_load", _fail)

    params = RoboticsDataParams(
        dataset_manifest=["/nonexistent/path/manifest.jsonl"],
        dataset_statistics=["/nonexistent/path/stats.json"],
        dataset_modality=["robotics"],
        dataset_weighting=[1.0],
        proprioception_fields=PROPRIO_FIELDS,
        action_fields=ACTION_FIELDS,
        camera_names=["rgb"],
        image_indices=[-1, 0],
        image_names=[],
        action_dim=EXPECTED_ACTION_DIM,
        proprioception_dim=EXPECTED_PROPRIO_DIM,
        lowdim_past_timesteps=1,
        lowdim_future_timesteps=14,
        normalization=_make_normalization(),
    )

    resolve_robotics_data_fields(params)

    assert params.image_names == ["rgb_t-1", "rgb_t0"]


def test_resolve_fills_point_cloud_num_points_when_use_point_cloud():
    """resolve_robotics_data_fields fills point_cloud_num_points from the
    dataset-side preprocessing_config.yaml when point clouds are enabled and
    the field is unset."""
    params = RoboticsDataParams(
        dataset_manifest=[MANIFEST_PATH],
        dataset_statistics=[STATS_PATH],
        dataset_modality=["robotics"],
        dataset_weighting=[1.0],
        proprioception_fields=PROPRIO_FIELDS,
        action_fields=ACTION_FIELDS,
        camera_names=["rgb"],
        image_indices=[-1, 0],
        image_names=["rgb_t-1", "rgb_t0"],
        use_point_cloud=True,
        point_cloud_num_points=None,
        action_dim=EXPECTED_ACTION_DIM,
        proprioception_dim=EXPECTED_PROPRIO_DIM,
        lowdim_past_timesteps=1,
        lowdim_future_timesteps=14,
        normalization=_make_normalization(),
    )

    resolve_robotics_data_fields(params)

    assert params.point_cloud_num_points == EXPECTED_POINT_CLOUD_NUM_POINTS


def test_resolve_validates_point_cloud_num_points_mismatch():
    """User-provided point_cloud_num_points that disagrees with the dataset must raise."""
    params = RoboticsDataParams(
        dataset_manifest=[MANIFEST_PATH],
        dataset_statistics=[STATS_PATH],
        dataset_modality=["robotics"],
        dataset_weighting=[1.0],
        proprioception_fields=PROPRIO_FIELDS,
        action_fields=ACTION_FIELDS,
        camera_names=["rgb"],
        image_indices=[-1, 0],
        image_names=["rgb_t-1", "rgb_t0"],
        use_point_cloud=True,
        point_cloud_num_points=1,  # disagrees with dataset's 4096
        action_dim=EXPECTED_ACTION_DIM,
        proprioception_dim=EXPECTED_PROPRIO_DIM,
        lowdim_past_timesteps=1,
        lowdim_future_timesteps=14,
        normalization=_make_normalization(),
    )

    with pytest.raises(AssertionError, match="point_cloud_num_points mismatch"):
        resolve_robotics_data_fields(params)


def test_resolve_validates_dimension_mismatch():
    """User-provided action_dim that disagrees with the stats file must raise loudly."""
    params = RoboticsDataParams(
        dataset_manifest=[MANIFEST_PATH],
        dataset_statistics=[STATS_PATH],
        dataset_modality=["robotics"],
        dataset_weighting=[1.0],
        proprioception_fields=PROPRIO_FIELDS,
        action_fields=ACTION_FIELDS,
        camera_names=["rgb"],
        image_indices=[-1, 0],
        image_names=["rgb_t-1", "rgb_t0"],
        action_dim=999,
        proprioception_dim=None,
        lowdim_past_timesteps=1,
        lowdim_future_timesteps=14,
        normalization=_make_normalization(),
    )

    with pytest.raises(AssertionError, match="Action dimension mismatch"):
        resolve_robotics_data_fields(params)


def test_resolve_derived_fields_propagates_lowdim_to_data_and_model():
    """resolve_derived_fields fills norm.lowdim_* from preprocessing_config.yaml and
    propagates them into data.lowdim_* and model.lowdim_* via init_shared_attributes.

    This pins the high-priority finding: without re-running the propagation chain
    after fields are resolved, data.lowdim_* would stay None when the YAML omits them.
    """
    cfg = _make_train_cfg_for_resolution()
    object.__setattr__(cfg.data, "lowdim_past_timesteps", None)
    object.__setattr__(cfg.data, "lowdim_future_timesteps", None)
    object.__setattr__(cfg.data.normalization, "lowdim_past_timesteps", None)
    object.__setattr__(cfg.data.normalization, "lowdim_future_timesteps", None)
    object.__setattr__(cfg.model, "lowdim_past_timesteps", None)
    object.__setattr__(cfg.model, "lowdim_future_timesteps", None)

    cfg.resolve_derived_fields()

    assert cfg.data.normalization.lowdim_past_timesteps == 1
    assert cfg.data.normalization.lowdim_future_timesteps == 14
    assert cfg.data.lowdim_past_timesteps == 1
    assert cfg.data.lowdim_future_timesteps == 14
    assert cfg.model.lowdim_past_timesteps == 1
    assert cfg.model.lowdim_future_timesteps == 14
    assert cfg.data._resolved is True


def test_resolve_derived_fields_is_idempotent(monkeypatch):
    """Calling resolve_derived_fields twice on the same cfg must be safe — second call
    is a no-op and does not flip persisted values."""
    cfg = _make_train_cfg_for_resolution()
    cfg.resolve_derived_fields()
    snapshot = (
        cfg.data.action_dim,
        cfg.data.proprioception_dim,
        cfg.data.lowdim_past_timesteps,
        cfg.data.lowdim_future_timesteps,
        cfg.data.point_cloud_num_points,
        tuple(cfg.data.camera_names),
    )

    def _fail(*args, **kwargs):
        raise AssertionError(f"should not be called: args={args!r} kwargs={kwargs!r}")

    monkeypatch.setattr("vla_foundry.params.resolve.yaml_load", _fail)
    monkeypatch.setattr("vla_foundry.data.robotics.normalization.RoboticsNormalizer", _fail)

    cfg.resolve_derived_fields()
    assert cfg.data._resolved is True
    assert (
        cfg.data.action_dim,
        cfg.data.proprioception_dim,
        cfg.data.lowdim_past_timesteps,
        cfg.data.lowdim_future_timesteps,
        cfg.data.point_cloud_num_points,
        tuple(cfg.data.camera_names),
    ) == snapshot
