import os
import sys
import tempfile
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import draccus
import pytest
import yaml

from vla_foundry.params.model_params import ModelParams
from vla_foundry.params.train_experiment_params import (
    TrainExperimentParams,
    load_experiment_params_from_yaml,
    load_params_from_yaml,
    localize_paths,
)


def get_args_text():
    test_args = [
        "--name",
        "test_experiment",
        "--total_train_samples",
        "1000000",
        "--model.type",
        "transformer",
        "--data.type",
        "text",
        "--data.dataset_manifest",
        ["s3://test-bucket/manifest.jsonl"],
        "--data.dataset_modality",
        ["text"],
        "--data.dataset_weighting",
        ["1.0"],
        "--model.hidden_dim",
        "999",
    ]
    with patch.object(sys, "argv", ["test"] + test_args):
        args = draccus.parse(config_class=TrainExperimentParams)
    return args


def get_args_vlm():
    test_args = [
        "--name",
        "test_experiment_vlm",
        "--total_train_samples",
        "1000000",
        "--model.type",
        "vlm",
        "--data.type",
        "image_caption",
        "--data.dataset_manifest",
        ["s3://test-bucket/manifest.jsonl", "s3://test-bucket-2/manifest.jsonl"],
        "--data.dataset_modality",
        ["image", "text"],
        "--data.dataset_weighting",
        ["1.0", "1.0"],
        "--data.processor",
        "debug",
        "--model.vit.hidden_dim",
        "999",
    ]
    with patch.object(sys, "argv", ["test"] + test_args):
        args = draccus.parse(config_class=TrainExperimentParams)
    return args


def get_args_vlm_from_load_path(**kwargs):
    if kwargs is None:
        kwargs = {}
    test_args = [
        "--model.type",
        "vlm",
        "--model.transformer",
        "include tests/params/dummy_configs/dummy_transformer_config.yaml",
        "--model.vit",
        "include tests/params/dummy_configs/dummy_vit_config.yaml",
        "--model.vit.hidden_dim",
        str(kwargs.get("hidden_dim", 999)),
        "--distributed.fsdp",
        "False",
        "--data.type",
        "image_caption",
        "--data.processor",
        "debug",
        "--data.dataset_manifest",
        ["s3://tri-ml-datasets/datasets/datacompdr_1b/manifest.jsonl"],
        "--data.dataset_modality",
        ["image_caption"],
        "--data.dataset_weighting",
        ["1.0"],
        "--data.seq_len",
        str(kwargs.get("seq_len", 2048)),
        "--data.img_num_tokens",
        str(kwargs.get("img_num_tokens", 256)),
        "--total_train_samples",
        str(kwargs.get("total_train_samples", 14_000_000)),
        "--num_checkpoints",
        str(kwargs.get("num_checkpoints", 5)),
    ]
    with patch.object(sys, "argv", ["test"] + test_args):
        args = draccus.parse(config_class=TrainExperimentParams)
    return args


def test_get_text_args():
    args = get_args_text()
    assert args.name == "test_experiment"
    assert args.model.type == "transformer"
    assert args.data.dataset_manifest == ["s3://test-bucket/manifest.jsonl"]
    assert args.data.dataset_modality == ["text"]
    assert args.model.hidden_dim == 999
    # Some random subset of args
    assert args.wandb_project_name == "vla_foundry"
    assert args.hparams.lr == 0.0001
    assert args.hparams.eps == 1e-08
    assert args.save_path is None
    assert not args.model.qk_norm


def test_get_args_vlm():
    args = get_args_vlm()
    assert args.name == "test_experiment_vlm"
    assert args.model.type == "vlm"
    assert args.data.dataset_manifest == ["s3://test-bucket/manifest.jsonl", "s3://test-bucket-2/manifest.jsonl"]
    assert args.data.dataset_modality == ["image", "text"]
    assert args.data.dataset_weighting == [1.0, 1.0]
    assert args.model.vit.hidden_dim == 999


def test_load_path_flag():
    hidden_dim = 999
    args = get_args_vlm_from_load_path(hidden_dim=hidden_dim)
    assert args.model.type == "vlm"
    assert args.model.vit.type == "vit"
    assert not args.distributed.fsdp
    assert args.data.type == "image_caption"
    assert args.data.processor == "debug"
    assert args.data.dataset_manifest == ["s3://tri-ml-datasets/datasets/datacompdr_1b/manifest.jsonl"]
    assert args.data.dataset_modality == ["image_caption"]
    assert args.data.dataset_weighting == [1.0]
    assert args.data.seq_len == 2048
    assert args.data.img_num_tokens == 256
    assert args.total_train_samples == 14_000_000
    assert args.num_checkpoints == 5
    assert args.model.transformer.hidden_dim == 128
    assert args.model.transformer.n_layers == 2
    assert args.model.transformer.n_heads == 2
    assert args.model.transformer.max_seq_len == 16
    assert args.model.transformer.vocab_size == 1000
    assert not args.model.transformer.post_embed_norm
    assert not args.model.transformer.weight_tying
    assert args.model.vit.img_size == 32
    assert args.model.vit.hidden_dim == hidden_dim  # overridden by hidden_dim flag
    assert args.model.vit.inter_dim == 4300
    assert args.model.vit.n_heads == 2
    assert args.model.vit.n_layers == 2
    assert args.model.vit.patch_size == 10
    assert args.model.vit.projector_pixel_shuffle_factor == 2


@pytest.mark.parametrize("params_yaml", ["tests/params/dummy_configs/dummy_vlm_config.yaml"])
def test_load_experiment_params_from_yaml(params_yaml):
    params = load_experiment_params_from_yaml(params_yaml)
    assert params.model.vit.hidden_dim == 128


@pytest.mark.parametrize(
    "params_yaml",
    [
        "tests/params/dummy_configs/dummy_vlm_config_include_vit.yaml",
        "tests/params/dummy_configs/dummy_vlm_config_include_model.yaml",
    ],
)
def test_load_experiment_params_from_yaml_include(params_yaml):
    params = load_experiment_params_from_yaml(params_yaml)
    assert params.model.vit.hidden_dim == 100


@pytest.mark.parametrize(
    "params_yaml, hidden_dim",
    [
        ("tests/params/dummy_configs/dummy_vlm_config_include_vit.yaml", 900),
        ("tests/params/dummy_configs/dummy_vlm_config_include_model.yaml", 1000),
    ],
)
def get_args_vlm_from_load_path_modify(params_yaml, hidden_dim):
    if hidden_dim is not None:
        params = get_args_vlm_from_load_path(params_yaml, hidden_dim=hidden_dim)
        assert params.model.vit.hidden_dim == hidden_dim
    else:
        params = get_args_vlm_from_load_path(params_yaml)
        assert params.model.vit.hidden_dim == 100


def test_immutable_params():
    params = get_args_text()
    assert params.model.hidden_dim == 999
    with pytest.raises(FrozenInstanceError):
        params.model.hidden_dim = 1000
    assert params.model.hidden_dim == 999
    object.__setattr__(params.model, "hidden_dim", 1000)
    assert params.model.hidden_dim == 1000


def test_localize_paths_string_s3_to_local():
    """Test that s3 paths are converted to local paths with new base path."""
    s3_path = "s3://bucket/some/dir/file.yaml"
    base_path = "/local/base"
    result = localize_paths(s3_path, base_path)
    assert result == "/local/base/file.yaml"


def test_localize_paths_string_non_s3():
    """Test that non-s3 paths are left unchanged."""
    local_path = "/local/path/to/file.yaml"
    base_path = "/local/base"
    result = localize_paths(local_path, base_path)
    assert result == local_path


def test_localize_paths_list():
    """Test that s3 paths in lists are converted."""
    data = [
        "s3://bucket/dir1/file1.yaml",
        "/local/file.yaml",
        "s3://bucket/dir2/file2.yaml",
    ]
    base_path = "/local/base"
    result = localize_paths(data, base_path)
    assert result == [
        "/local/base/file1.yaml",
        "/local/file.yaml",
        "/local/base/file2.yaml",
    ]


def test_localize_paths_nested_dict():
    """Test that s3 paths in nested dictionaries are converted."""
    data = {
        "model": {
            "checkpoint": "s3://bucket/models/checkpoint.pt",
            "config": "/local/config.yaml",
        },
        "data": {
            "manifest": "s3://bucket/data/manifest.jsonl",
            "nested": {
                "path": "s3://bucket/nested/file.yaml",
            },
        },
        "other": "value",
    }
    base_path = "/local/base"
    result = localize_paths(data, base_path)
    assert result["model"]["checkpoint"] == "/local/base/checkpoint.pt"
    assert result["model"]["config"] == "/local/config.yaml"
    assert result["data"]["manifest"] == "/local/base/manifest.jsonl"
    assert result["data"]["nested"]["path"] == "/local/base/file.yaml"
    assert result["other"] == "value"


def test_localize_paths_mixed_list_and_dict():
    """Test that s3 paths in mixed structures are converted."""
    data = {
        "datasets": [
            "s3://bucket/dataset1/manifest.jsonl",
            "s3://bucket/dataset2/manifest.jsonl",
        ],
        "weights": [0.5, 0.5],
    }
    base_path = "/local/base"
    result = localize_paths(data, base_path)
    assert result["datasets"] == [
        "/local/base/manifest.jsonl",
        "/local/base/manifest.jsonl",
    ]
    assert result["weights"] == [0.5, 0.5]


def test_load_params_from_yaml_without_localize():
    """Test loading params from yaml without localization."""
    params = load_params_from_yaml(
        ModelParams, "tests/params/dummy_configs/dummy_transformer_config.yaml", localize_params=False
    )
    assert params.type == "transformer"
    assert params.hidden_dim == 128
    assert params.n_layers == 2


def test_load_params_from_yaml_with_localize():
    """Test loading params from yaml with path localization."""
    # Create a temporary yaml file with s3 paths
    config_data = {
        "type": "transformer",
        "hidden_dim": 256,
        "n_layers": 4,
        "n_heads": 4,
        "max_seq_len": 512,
        "vocab_size": 1000,
        "resume_from_checkpoint": "s3://bucket/models/checkpoint.pt",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_yaml_path = f.name

    try:
        params = load_params_from_yaml(ModelParams, temp_yaml_path, localize_params=True)
        assert params.type == "transformer"
        assert params.hidden_dim == 256
        assert params.n_layers == 4
        # The s3 path should be converted to use the same base directory as the config file
        base_path = os.path.dirname(temp_yaml_path)
        assert params.resume_from_checkpoint == f"{base_path}/checkpoint.pt"
    finally:
        # Clean up
        if os.path.exists(temp_yaml_path):
            os.unlink(temp_yaml_path)


def test_load_params_from_yaml_ignores_unknown_fields():
    """Extra keys should be quietly ignored by lenient BaseParams decoding."""
    params = load_params_from_yaml(
        ModelParams, "tests/params/dummy_configs/dummy_transformer_config_extra.yaml", localize_params=False
    )
    assert params.type == "transformer"
    assert params.hidden_dim == 128
    assert not hasattr(params, "totally_unused_flag")


def test_load_params_from_yaml_with_localize_complex():
    """Test loading params with nested s3 paths that need localization."""
    # Create a config with multiple s3 paths at different nesting levels
    base_dir = tempfile.mkdtemp()
    config_data = {
        "type": "vlm",
        "image_token_id": 257152,
        "transformer": {
            "type": "transformer",
            "hidden_dim": 128,
            "n_layers": 2,
            "n_heads": 2,
            "max_seq_len": 16,
            "vocab_size": 1000,
            "resume_from_checkpoint": "s3://bucket/transformer/checkpoint.pt",
        },
        "vit": {
            "type": "vit",
            "hidden_dim": 128,
            "img_size": 224,
            "patch_size": 14,
            "n_layers": 2,
            "n_heads": 4,
            "resume_from_checkpoint": "s3://bucket/vit/checkpoint.pt",
        },
    }

    yaml_path = os.path.join(base_dir, "config.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config_data, f)

    try:
        from vla_foundry.params.model_params import VLMParams

        params = load_params_from_yaml(VLMParams, yaml_path, localize_params=True)
        assert params.type == "vlm"
        assert params.transformer.resume_from_checkpoint == f"{base_dir}/checkpoint.pt"
        assert params.vit.resume_from_checkpoint == f"{base_dir}/checkpoint.pt"
    finally:
        # Clean up
        if os.path.exists(yaml_path):
            os.unlink(yaml_path)
        if os.path.exists(base_dir):
            os.rmdir(base_dir)
