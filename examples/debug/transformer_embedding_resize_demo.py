#!/usr/bin/env python3
"""
Demonstration script showing transformer model creation and embedding resizing.

This script demonstrates:
1. Creating a transformer model with custom parameters
2. Testing the model with different input sizes
3. Resizing the token embeddings to extend vocabulary
4. Verifying that the model works correctly after resizing
"""

import torch

from vla_foundry.models.transformer import Transformer
from vla_foundry.params.model_params import TransformerParams


def create_demo_transformer():
    """Create a small transformer model for demonstration purposes."""
    # Create a small transformer configuration
    config = TransformerParams(
        hidden_dim=64,  # Small hidden dimension for demo
        n_layers=2,  # Few layers for quick demo
        n_heads=2,  # Small number of attention heads
        vocab_size=100,  # Small vocabulary size
        max_seq_len=32,  # Short sequence length
        ffn_type="swiglu",  # Use SwiGLU activation
        norm_type="default_layer_norm",
        positional_embedding_type="rotary",
        attn_name="torch_attn",
        weight_tying=False,  # Disable weight tying for demo
        post_embed_norm=False,
        qk_norm=False,
        norm_eps=1e-5,
        cast_output_to_float32=False,
    )

    # Create the transformer model
    model = Transformer(config)
    return model, config


def test_model_before_resize(model, config):
    """Test the model with original vocabulary size."""
    print("\n=== Testing model BEFORE embedding resize ===")
    print(f"Original vocabulary size: {model.vocab_size}")
    print(f"Hidden dimension: {model.hidden_dim}")
    print(f"Number of layers: {model.n_layers}")

    # Create test input
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Test forward pass
    with torch.no_grad():
        logits, past_key_values, hidden_states = model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False
        )

    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected output shape: (batch_size, seq_len, vocab_size) = ({batch_size}, {seq_len}, {config.vocab_size})")
    print("✓ Forward pass successful!")

    return input_ids, attention_mask


def test_embedding_resize(model, new_vocab_size):
    """Test embedding resizing functionality."""
    print("\n=== Testing embedding resize ===")
    print(f"Resizing vocabulary from {model.vocab_size} to {new_vocab_size}")

    # Store original weights for comparison
    original_embedding_weight = model.embeddings.weight.data.clone()
    original_output_weight = model.output.weight.data.clone()

    # Resize embeddings
    result_token_id = model.resize_token_embeddings(new_vocab_size)

    print("✓ Resize completed successfully!")
    print(f"New vocabulary size: {model.vocab_size}")
    print(f"Result token ID: {result_token_id}")
    print(f"Embedding layer size: {model.embeddings.weight.shape}")
    print(f"Output layer size: {model.output.weight.shape}")

    # Verify that original weights were preserved
    original_vocab_size = original_embedding_weight.shape[0]
    torch.testing.assert_close(model.embeddings.weight.data[:original_vocab_size], original_embedding_weight)
    torch.testing.assert_close(model.output.weight.data[:original_vocab_size], original_output_weight)
    print("✓ Original weights preserved!")

    # Verify that new weights were initialized
    assert model.embeddings.weight.data[original_vocab_size:].abs().sum() > 0
    assert model.output.weight.data[original_vocab_size:].abs().sum() > 0
    print("✓ New weights initialized!")


def test_model_after_resize(model, input_ids, attention_mask):
    """Test the model after embedding resize."""
    print("\n=== Testing model AFTER embedding resize ===")
    print(f"Current vocabulary size: {model.vocab_size}")

    # Test forward pass with original input (should still work)
    with torch.no_grad():
        logits, past_key_values, hidden_states = model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False
        )

    print(f"Original input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print("✓ Forward pass with original input successful!")

    # Test forward pass with new vocabulary tokens
    batch_size, seq_len = input_ids.shape
    new_input_ids = torch.randint(model.vocab_size - 10, model.vocab_size, (batch_size, seq_len))
    new_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    with torch.no_grad():
        new_logits, new_past_key_values, new_hidden_states = model(
            input_ids=new_input_ids, attention_mask=new_attention_mask, output_hidden_states=False
        )

    print(f"New input shape: {new_input_ids.shape}")
    print(f"New output logits shape: {new_logits.shape}")
    print("✓ Forward pass with new vocabulary tokens successful!")


def test_multiple_resizes(model):
    """Test multiple consecutive embedding resizes."""
    print("\n=== Testing multiple consecutive resizes ===")

    original_vocab_size = model.vocab_size
    print(f"Starting vocabulary size: {original_vocab_size}")

    # First resize
    first_new_size = original_vocab_size + 50
    model.resize_token_embeddings(first_new_size)
    print(f"After first resize: {model.vocab_size}")

    # Second resize
    second_new_size = first_new_size + 75
    model.resize_token_embeddings(second_new_size)
    print(f"After second resize: {model.vocab_size}")

    # Third resize
    third_new_size = second_new_size + 25
    model.resize_token_embeddings(third_new_size)
    print(f"After third resize: {model.vocab_size}")

    # Test forward pass with final vocabulary size
    batch_size, seq_len = 1, 5
    final_input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    final_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    with torch.no_grad():
        final_logits, _, _ = model(input_ids=final_input_ids, attention_mask=final_attention_mask)

    print(f"Final vocabulary size: {model.vocab_size}")
    print(f"Final output shape: {final_logits.shape}")
    print("✓ Multiple resizes successful!")


def main():
    """Main demonstration function."""
    print("🚀 Transformer Model and Embedding Resize Demonstration")
    print("=" * 60)

    # Create the transformer model
    print("Creating transformer model...")
    model, config = create_demo_transformer()
    print("✓ Model created successfully!")

    # Test the model before resizing
    input_ids, attention_mask = test_model_before_resize(model, config)

    # Test embedding resizing
    new_vocab_size = 150  # Extend vocabulary by 50 tokens
    test_embedding_resize(model, new_vocab_size)

    # Test the model after resizing
    test_model_after_resize(model, input_ids, attention_mask)

    # Test multiple consecutive resizes
    test_multiple_resizes(model)

    print("\n" + "=" * 60)
    print("🎉 All tests completed successfully!")
    print("\nSummary:")
    print(f"- Original vocabulary size: {config.vocab_size}")
    print(f"- Final vocabulary size: {model.vocab_size}")
    print(f"- Total vocabulary extension: {model.vocab_size - config.vocab_size} tokens")
    print("- Model architecture preserved: ✓")
    print("- Forward pass functionality maintained: ✓")
    print("- Weight preservation verified: ✓")


if __name__ == "__main__":
    main()
