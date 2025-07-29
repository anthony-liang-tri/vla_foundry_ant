from copy import deepcopy

import torch

from lbm2.file_utils import load_model_checkpoint
from lbm2.models import create_model
from lbm2.params.train_experiment_params import load_experiment_params_from_yaml


def test_model_loading():
    cfg = load_experiment_params_from_yaml("tests/shared/tiny_model/config.yaml")
    model = create_model(cfg.model)
    initial_keys = set(model.state_dict().keys())
    initial_state_dict = deepcopy(model.state_dict())

    ckpt = "tests/shared/tiny_model/checkpoint.pt"
    load_model_checkpoint(model, ckpt, cfg.distributed)
    loaded_keys = set(model.state_dict().keys())
    loaded_state_dict = model.state_dict()
    assert initial_keys == loaded_keys, "State dict keys changed after loading checkpoint"

    # Check all weights changed
    unchanged = [
        name
        for name, initial in initial_state_dict.items()
        if torch.equal(initial, loaded_state_dict[name]) and "pos_embed.inv_freq" not in name
    ]
    assert len(unchanged) == 0, f"These weights didn't change: {unchanged}"

    # Check significant change
    total_change = sum(
        (initial - loaded_state_dict[name]).abs().sum().item() for name, initial in initial_state_dict.items()
    )
    total_params = sum(p.numel() for p in initial_state_dict.values())
    assert total_change / total_params > 1e-4, "Overall parameter change too small"


def test_model_same_seed_same_initialization():
    """Test that model initializes same parameters with same seed."""
    cfg = load_experiment_params_from_yaml("tests/shared/tiny_model/config.yaml")

    # Set seed and create model
    torch.manual_seed(42)
    model1 = create_model(cfg.model)

    # Set same seed and create another model
    torch.manual_seed(42)
    model2 = create_model(cfg.model)

    # Both models should have identical parameters
    for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=False):
        assert torch.equal(p1, p2), "Models with same seed should have identical parameters"


def test_model_deterministic_loading():
    """Test that model loads deterministically regardless of seed."""
    cfg = load_experiment_params_from_yaml("tests/shared/tiny_model/config.yaml")

    # Set seed and create model
    torch.manual_seed(42)
    model1 = create_model(cfg.model)

    # Try different seed. This shouldn't matter because we're loading the same checkpoint.
    torch.manual_seed(43)
    model2 = create_model(cfg.model)

    ckpt = "tests/shared/tiny_model/checkpoint.pt"
    load_model_checkpoint(model1, ckpt, cfg.distributed)
    load_model_checkpoint(model2, ckpt, cfg.distributed)

    for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=False):
        assert torch.equal(p1, p2), "Models with same seed should have identical parameters"
