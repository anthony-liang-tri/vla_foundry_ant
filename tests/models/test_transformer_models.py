from unittest.mock import Mock, patch

import pytest
import torch

from lbm2.models import create_model
from lbm2.params.model_params import TransformerParams
from lbm2.params.train_experiment_params import load_params_from_yaml


class TestTransformer:
    @pytest.fixture
    def transformer_config(self):
        return load_params_from_yaml(TransformerParams, "tests/params/dummy_configs/dummy_transformer_config.yaml")

    @pytest.fixture
    def transformer(self, transformer_config):
        return create_model(transformer_config)

    def test_transformer_forward_basic(self, transformer):
        """Test basic forward pass without hidden states"""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        logits, past_key_values, hidden_states = transformer(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False
        )

        assert logits.shape == (batch_size, seq_len, 1000)
        assert past_key_values is None
        assert hidden_states is None

    def test_transformer_forward_with_hidden_states(self, transformer):
        """Test forward pass with hidden states returned"""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        logits, past_key_values, hidden_states = transformer(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )

        assert logits.shape == (batch_size, seq_len, 1000)
        assert past_key_values is None
        assert isinstance(hidden_states, list)
        assert len(hidden_states) == 2  # n_layers = 2
        assert hidden_states[0].shape == (batch_size, seq_len, 128)  # hidden_dim = 128
        assert hidden_states[1].shape == (batch_size, seq_len, 128)

    def test_transformer_forward_with_input_embeds(self, transformer):
        """Test forward pass with input embeddings instead of input_ids"""
        batch_size, seq_len = 2, 10
        input_embeds = torch.randn(batch_size, seq_len, 128)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        logits, past_key_values, hidden_states = transformer(
            input_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=False
        )

        assert logits.shape == (batch_size, seq_len, 1000)
        assert past_key_values is None
        assert hidden_states is None

    def test_transformer_forward_with_past_key_values(self, transformer):
        """Test forward pass with past key values for caching"""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Create mock past key values
        n_heads = transformer.model_params.n_heads
        head_dim = transformer.model_params.hidden_dim // n_heads
        past_key_values = [
            (
                torch.randn(batch_size, 5, n_heads, head_dim),
                torch.randn(batch_size, 5, n_heads, head_dim),
            )
            for _ in range(transformer.n_layers)
        ]

        logits, new_past_key_values, hidden_states = transformer(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=False,
        )

        assert logits.shape == (batch_size, seq_len, 1000)
        assert new_past_key_values is not None
        assert len(new_past_key_values) == 2
        assert hidden_states is None

    def test_transformer_forward_error_no_input(self, transformer):
        """Test that error is raised when neither input_ids nor input_embeds provided"""
        attention_mask = torch.ones(2, 10, dtype=torch.bool)

        with pytest.raises(ValueError, match="Either input_ids or input_embeds must be provided"):
            transformer(attention_mask=attention_mask)

    def test_transformer_properties(self, transformer):
        """Test transformer properties"""
        assert transformer.hidden_dim == 128
        assert transformer.n_layers == 2
        assert transformer.vocab_size == 1000

    def test_transformer_grad_checkpointing(self, transformer):
        """Test gradient checkpointing setting"""
        assert transformer.grad_checkpointing is False

        transformer.set_grad_checkpointing(True)
        assert transformer.grad_checkpointing is True

        transformer.set_grad_checkpointing(False)
        assert transformer.grad_checkpointing is False


class TestTransformerHF:
    @pytest.fixture
    def transformer_hf_config(self):
        # Create a new config with transformer_hf type
        from lbm2.params.model_params import TransformerHFParams

        return TransformerHFParams(
            hf_pretrained="microsoft/DialoGPT-small", resume_from_checkpoint=None, resume_weights_only=False
        )

    @patch("lbm2.models.transformer_hf.AutoModelForCausalLM.from_pretrained")
    def test_transformer_hf_forward_basic(self, mock_from_pretrained, transformer_hf_config):
        """Test basic forward pass without hidden states"""
        # Mock the HF model
        mock_model = Mock()
        mock_model.logits = torch.randn(2, 10, 1000)
        mock_model.past_key_values = None

        # Mock the forward method to return the expected values
        def mock_forward(*args, **kwargs):
            mock_output = Mock()
            mock_output.logits = mock_model.logits
            mock_output.past_key_values = mock_model.past_key_values
            return mock_output

        mock_model.return_value = mock_forward()
        mock_from_pretrained.return_value = mock_model

        transformer = create_model(transformer_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        logits, past_key_values, hidden_states = transformer(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False
        )

        assert logits.shape == (batch_size, seq_len, 1000)
        assert past_key_values is None
        assert hidden_states is None

    @patch("lbm2.models.transformer_hf.AutoModelForCausalLM.from_pretrained")
    def test_transformer_hf_forward_with_hidden_states(self, mock_from_pretrained, transformer_hf_config):
        """Test forward pass with hidden states returned"""
        # Mock the HF model
        mock_model = Mock()
        mock_model.logits = torch.randn(2, 10, 1000)
        mock_model.past_key_values = None
        mock_model.hidden_states = [torch.randn(2, 10, 128) for _ in range(2)]

        # Mock the forward method to return the expected values
        def mock_forward(*args, **kwargs):
            mock_output = Mock()
            mock_output.logits = mock_model.logits
            mock_output.past_key_values = mock_model.past_key_values
            mock_output.hidden_states = mock_model.hidden_states
            return mock_output

        mock_model.return_value = mock_forward()
        mock_from_pretrained.return_value = mock_model

        transformer = create_model(transformer_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        logits, past_key_values, hidden_states = transformer(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )

        assert logits.shape == (batch_size, seq_len, 1000)
        assert past_key_values is None
        assert isinstance(hidden_states, list)
        assert len(hidden_states) == 2

    @patch("lbm2.models.transformer_hf.AutoModelForCausalLM.from_pretrained")
    def test_transformer_hf_properties(self, mock_from_pretrained, transformer_hf_config):
        """Test transformer HF properties"""
        # Mock the HF model with config
        mock_model = Mock()
        mock_config_obj = Mock()
        mock_config_obj.hidden_size = 128
        mock_config_obj.num_hidden_layers = 2
        mock_model.config = mock_config_obj
        mock_from_pretrained.return_value = mock_model

        transformer = create_model(transformer_hf_config)

        assert transformer.hidden_dim == 128
        assert transformer.num_hidden_layers == 2

    @patch("lbm2.models.transformer_hf.AutoModelForCausalLM.from_pretrained")
    def test_transformer_hf_set_grad_checkpointing(self, mock_from_pretrained, transformer_hf_config):
        """Test that set_grad_checkpointing raises NotImplementedError"""
        # Mock the HF model
        mock_model = Mock()
        mock_from_pretrained.return_value = mock_model

        transformer = create_model(transformer_hf_config)

        with pytest.raises(NotImplementedError):
            transformer.set_grad_checkpointing(True)
