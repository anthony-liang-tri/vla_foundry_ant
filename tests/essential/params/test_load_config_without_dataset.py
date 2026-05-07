"""
Tests for loading experiment configs when dataset paths are inaccessible.

When running inference on a robot, the checkpoint's config.yaml references
dataset paths (dataset_statistics, dataset_manifest) from the training machine.
These paths may not exist on the inference machine. Loading must still succeed
as long as lowdim_past/future_timesteps and action_dim/proprioception_dim are
already saved in the config.
"""

import os
import tempfile

import yaml

from vla_foundry.data.dataloader import _missing_robotics_resolution_fields
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml


def _make_inference_config(
    dataset_stats_path: str,
    dataset_manifest_path: str,
    tmp_dir: str,
    *,
    action_dim: int = 7,
    proprioception_dim: int = 7,
) -> str:
    """Create a minimal config.yaml that mimics a saved training checkpoint.

    A saved checkpoint config has all derived fields already resolved
    (action_dim, proprioception_dim, camera_names, image_indices, image_names,
    lowdim_past/future_timesteps), so loading should not need to access
    the original dataset paths.
    """
    config = {
        "data": {
            "type": "robotics",
            "action_dim": action_dim,
            "action_fields": ["chest_T_left_gripper_tip_pose"],
            "proprioception_dim": proprioception_dim,
            "proprioception_fields": ["chassis_T_left_eef_pose"],
            "lowdim_past_timesteps": 1,
            "lowdim_future_timesteps": 14,
            "dataset_manifest": [dataset_manifest_path],
            "dataset_statistics": [dataset_stats_path],
            "dataset_modality": ["robotics"],
            "dataset_weighting": [1.0],
            "camera_names": ["rgb"],
            "image_indices": [-1, 0],
            "image_names": ["rgb_t-1", "rgb_t0"],
            "image_size": 224,
            "img_num_tokens": 64,
            "seq_len": 2048,
            "processor": "openai/clip-vit-base-patch32",
            "normalization": {
                "enabled": True,
                "method": "std",
                "scope": "global",
                "lowdim_past_timesteps": 1,
                "lowdim_future_timesteps": 14,
                "field_configs": {
                    "chassis_T_left_eef_pose": {
                        "enabled": True,
                        "method": "std",
                        "scope": "global",
                        "epsilon": 1e-8,
                    },
                    "chest_T_left_gripper_tip_pose": {
                        "enabled": True,
                        "method": "std",
                        "scope": "global",
                        "epsilon": 1e-8,
                    },
                },
            },
        },
        "model": {
            "type": "diffusion_policy",
            "action_dim": 7,
            "proprioception_dim": 7,
            "vision_language_backbone": {
                "type": "clip_backbone",
                "hf_pretrained": "openai/clip-vit-base-patch32",
            },
            "transformer": {"type": "transformer"},
            "noise_scheduler": {"type": "noise_scheduler"},
        },
        "hparams": {"global_batch_size": 256, "per_gpu_batch_size": 128},
        "total_train_samples": 5000000,
    }

    config_path = os.path.join(tmp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def test_load_config_with_inaccessible_dataset_path():
    """Persisted derived fields must survive a YAML round-trip even when the stats /
    manifest paths are unreachable. We use deliberately weird values (42) that no
    `resolve_robotics_data_fields` call could plausibly produce, so any silent
    re-derivation would fail the assertion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = _make_inference_config(
            "/nonexistent/path/stats.json",
            "/nonexistent/path/manifest.jsonl",
            tmp_dir,
            action_dim=42,
            proprioception_dim=42,
        )
        cfg = load_experiment_params_from_yaml(config_path)

        assert cfg.data.normalization.lowdim_past_timesteps == 1
        assert cfg.data.normalization.lowdim_future_timesteps == 14
        assert cfg.data.action_dim == 42
        assert cfg.data.proprioception_dim == 42
        # _resolved is False because we never called resolve_derived_fields — load alone
        # does not flip it. Inference paths that bypass the dataloader can still operate
        # on this config; the dataloader boundary allows already-materialized fields.
        assert cfg.data._resolved is False
        assert _missing_robotics_resolution_fields(cfg.data) == []


def test_load_config_with_accessible_dataset_path():
    """Loading a config that points to real on-disk data still works and still does no I/O —
    persisted dims survive even if they disagree with what the stats file would compute."""
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_assets", "small_lbm_dataset")
    stats_path = os.path.join(dataset_dir, "stats.json")
    manifest_path = os.path.join(dataset_dir, "manifest.jsonl")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Persist 7 even though the stats file would compute 3 from the xyz field —
        # load alone must not re-derive.
        config_path = _make_inference_config(stats_path, manifest_path, tmp_dir, action_dim=7, proprioception_dim=7)
        cfg = load_experiment_params_from_yaml(config_path)

        assert cfg.data.normalization.lowdim_past_timesteps == 1
        assert cfg.data.normalization.lowdim_future_timesteps == 14
        assert cfg.data.action_dim == 7
        assert cfg.data.proprioception_dim == 7


def test_load_config_ignores_persisted_resolved_flag():
    """_resolved is runtime-only; persisted configs cannot mark themselves resolved."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = _make_inference_config(
            "/nonexistent/path/stats.json",
            "/nonexistent/path/manifest.jsonl",
            tmp_dir,
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)
        config["data"]["_resolved"] = True
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        cfg = load_experiment_params_from_yaml(config_path)

        assert cfg.data._resolved is False
