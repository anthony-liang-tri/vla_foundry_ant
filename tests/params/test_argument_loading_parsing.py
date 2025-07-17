import sys
import pytest
from unittest.mock import patch
from dataclasses import FrozenInstanceError
from lbm2.params.train_experiment_params import TrainExperimentParams, load_params_from_yaml
import draccus


def get_args_text():
    test_args = [
        "--name", "test_experiment",
        "--total_train_samples", "1000000",
        "--model.type", "transformer",
        "--data.type", "text",
        "--data.dataset_manifest", ["s3://test-bucket/manifest.jsonl"],
        "--data.dataset_modality", ["text"],
        "--data.dataset_weighting", ["1.0"],
        "--model.hidden_dim", "999",
    ]
    with patch.object(sys, 'argv', ['test'] + test_args):
        args = draccus.parse(config_class=TrainExperimentParams)
    return args

def get_args_vlm():
    test_args = [
        "--name", "test_experiment_vlm",
        "--total_train_samples", "1000000",
        "--model.type", "vlm",
        "--data.type", "image_caption",
        "--data.dataset_manifest", ["s3://test-bucket/manifest.jsonl", "s3://test-bucket-2/manifest.jsonl"],
        "--data.dataset_modality", ["image", "text"],
        "--data.dataset_weighting", ["1.0", "1.0"],
        "--data.processor", "debug",
        "--model.vit.vit_hidden_dim", "999",
    ]
    with patch.object(sys, 'argv', ['test'] + test_args):
        args = draccus.parse(config_class=TrainExperimentParams)
    return args

def get_args_vlm_from_load_path():
    test_args = [
        "--model.type", "vlm",
        "--model.transformer.load_path", "lbm2/config_presets/models/vlm_3b.yaml",
        "--model.vit.type", "vit",
        "--model.vit.load_path", "lbm2/config_presets/models/vit_paligemma.yaml",
        "--distributed.fsdp", "True",
        "--distributed.fsdp_use_orig_params", "True",
        "--distributed.fsdp_limit_all_gathers", "True",
        "--data.type", "image_caption",
        "--data.processor", "debug",
        "--data.dataset_manifest", ["s3://tri-ml-datasets/datasets/datacompdr_1b/manifest.jsonl"],
        "--data.dataset_modality", ["image_caption"],
        "--data.dataset_weighting", ["1.0"],
        "--data.seq_len", "2048",
        "--data.img_num_tokens", "256",
        "--total_train_samples", "14_000_000",
        "--num_checkpoints", "5",
    ]
    with patch.object(sys, 'argv', ['test'] + test_args):
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
    assert args.save_path == None
    assert args.model.qk_norm == False

def test_get_args_vlm():
    args = get_args_vlm()
    assert args.name == "test_experiment_vlm"
    assert args.model.type == "vlm"
    assert args.data.dataset_manifest == ["s3://test-bucket/manifest.jsonl", "s3://test-bucket-2/manifest.jsonl"]
    assert args.data.dataset_modality == ["image", "text"]
    assert args.data.dataset_weighting == [1.0, 1.0]
    assert args.model.vit.vit_hidden_dim == 999
    
def test_load_path_flag():
    args = get_args_vlm_from_load_path()
    assert args.model.type == "vlm"
    assert args.model.transformer.load_path == "lbm2/config_presets/models/vlm_3b.yaml"
    assert args.model.vit.type == "vit"
    assert args.model.vit.load_path == "lbm2/config_presets/models/vit_paligemma.yaml"
    assert args.distributed.fsdp == True
    assert args.distributed.fsdp_use_orig_params == True
    assert args.data.type == "image_caption"
    assert args.data.processor == "debug"
    assert args.data.dataset_manifest == ["s3://tri-ml-datasets/datasets/datacompdr_1b/manifest.jsonl"]
    assert args.data.dataset_modality == ["image_caption"]
    assert args.data.dataset_weighting == [1.0]
    assert args.data.seq_len == 2048
    assert args.data.img_num_tokens == 256
    assert args.total_train_samples == 14_000_000
    assert args.num_checkpoints == 5
    assert args.model.transformer.hidden_dim == 2048
    assert args.model.transformer.n_layers == 18
    assert args.model.transformer.n_heads == 8
    assert args.model.transformer.max_seq_len == 2048
    assert args.model.transformer.vocab_size == 257216
    assert args.model.transformer.post_embed_norm == False
    assert args.model.transformer.weight_tying == False
    assert args.model.vit.vit_img_size == 224
    assert args.model.vit.vit_hidden_dim == 1152
    assert args.model.vit.vit_inter_dim == 4304
    assert args.model.vit.vit_n_heads == 16
    assert args.model.vit.vit_n_layers == 27
    assert args.model.vit.vit_patch_size == 14
    assert args.model.vit.projector_pixel_shuffle_factor == 1



@pytest.mark.parametrize("params_yaml", ["tests/shared/dummy_vlm_config.yaml"])
def test_load_params_from_yaml(params_yaml):
    params = load_params_from_yaml(params_yaml)
    assert params.model.vit.vit_hidden_dim == 999


def test_immutable_params():
    params = get_args_text()
    assert params.model.hidden_dim == 999
    with pytest.raises(FrozenInstanceError):
        params.model.hidden_dim = 1000
    assert params.model.hidden_dim == 999
    object.__setattr__(params.model, 'hidden_dim', 1000)
    assert params.model.hidden_dim == 1000
