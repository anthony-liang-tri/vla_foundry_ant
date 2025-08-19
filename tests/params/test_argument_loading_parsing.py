import sys
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import draccus
import pytest

from lbm2.params.train_experiment_params import TrainExperimentParams, load_experiment_params_from_yaml


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
        "True",
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
    assert args.wandb_project_name == "lbm2"
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
    assert args.distributed.fsdp
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
