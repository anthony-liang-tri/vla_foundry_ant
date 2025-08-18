#!/usr/bin/env python3

from lbm2.models.transformer import Transformer
from lbm2.params.model_params import TransformerParams


def test_freeze_functionality():
    print("Testing freeze functionality...")

    # Test with freeze=False (default)
    print("\n1. Testing with freeze=False (default):")
    params = TransformerParams()
    print(f"freeze parameter: {params.freeze}")

    model = Transformer(params)
    print(f"Model created successfully: {type(model).__name__}")
    print(f"Model inherits from BaseModel: {isinstance(model, type(model).__bases__[0])}")

    # Check if parameters are trainable
    trainable_params = sum(p.requires_grad for p in model.parameters())
    total_params = sum(1 for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}/{total_params}")

    # Test with freeze=True
    print("\n2. Testing with freeze=True:")
    params_frozen = TransformerParams(freeze=True)
    print(f"freeze parameter: {params_frozen.freeze}")

    model_frozen = Transformer(params_frozen)
    print(f"Model created successfully: {type(model_frozen).__name__}")

    # Check if parameters are frozen
    trainable_params_frozen = sum(p.requires_grad for p in model_frozen.parameters())
    total_params_frozen = sum(1 for p in model_frozen.parameters())
    print(f"Trainable parameters: {trainable_params_frozen}/{total_params_frozen}")

    # Verify that frozen model has no trainable parameters
    assert trainable_params_frozen == 0, f"Expected 0 trainable parameters, got {trainable_params_frozen}"
    print("✓ Freeze functionality working correctly!")

    # Test that we can still access model_params
    print("\n3. Testing model_params access:")
    print(f"Model params type: {type(model_frozen.model_params).__name__}")
    print(f"Model params freeze value: {model_frozen.model_params.freeze}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_freeze_functionality()
