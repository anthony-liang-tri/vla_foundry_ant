from unittest.mock import Mock, patch

import pytest
import torch

from lbm2.models import create_model
from lbm2.models.utils import compute_num_image_tokens
from lbm2.params.model_params import TransformerParams
from lbm2.params.train_experiment_params import load_experiment_params_from_yaml, load_params_from_yaml


class TestReturnHiddenStatesConsistency:
    """Test that all models consistently return 3 items when output_hidden_states=True"""

    @pytest.fixture
    def transformer_config(self):
        return load_params_from_yaml(TransformerParams, "tests/params/dummy_configs/dummy_transformer_config.yaml")

    @pytest.fixture
    def vlm_config(self):
        config = load_experiment_params_from_yaml("tests/params/dummy_configs/dummy_vlm_config.yaml")
        object.__setattr__(config.model, "image_token_id", 999)
        return config

    def test_transformer_return_format_consistency(self, transformer_config):
        """Test that Transformer consistently returns 3 items"""
        transformer = create_model(transformer_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Test without hidden states
        result = transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        assert len(result) == 3
        logits, past_key_values, hidden_states = result
        assert logits is not None
        assert past_key_values is None
        assert hidden_states is None

        # Test with hidden states
        result = transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        assert len(result) == 3
        logits, past_key_values, hidden_states = result
        assert logits is not None
        assert past_key_values is None
        assert hidden_states is not None
        assert isinstance(hidden_states, list)
        assert len(hidden_states) == 2  # n_layers = 2

    @patch("lbm2.models.transformer_hf.AutoModelForCausalLM.from_pretrained")
    def test_transformer_hf_return_format_consistency(self, mock_from_pretrained, transformer_config):
        """Test that TransformerHF consistently returns 3 items"""
        # Mock the HF model
        mock_model = Mock()
        mock_output = Mock()
        mock_output.logits = torch.randn(2, 10, 1000)
        mock_output.past_key_values = None
        mock_output.hidden_states = [torch.randn(2, 10, 128) for _ in range(2)]
        mock_model.return_value = mock_output
        mock_from_pretrained.return_value = mock_model

        # Create a new config with transformer_hf type
        from lbm2.params.model_params import TransformerHFParams

        transformer_hf_config = TransformerHFParams(
            hf_pretrained="microsoft/DialoGPT-small", resume_from_checkpoint=None, resume_weights_only=False
        )
        transformer = create_model(transformer_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Test without hidden states
        result = transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        assert len(result) == 3
        logits, past_key_values, hidden_states = result
        assert logits is not None
        assert past_key_values is None
        assert hidden_states is None

        # Test with hidden states
        result = transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        assert len(result) == 3
        logits, past_key_values, hidden_states = result
        assert logits is not None
        assert past_key_values is None
        assert hidden_states is not None
        assert isinstance(hidden_states, list)
        assert len(hidden_states) == 2

    def test_vlm_return_format_consistency(self, vlm_config):
        """Test that VLM consistently returns 3 items"""
        vlm = create_model(vlm_config.model)

        # Compute expected number of image tokens from ViT config
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        batch_size, seq_len = 2, num_image_tokens + 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # Set image tokens
        input_ids[0, 0:num_image_tokens] = vlm.model_params.image_token_id
        input_ids[1, 0:num_image_tokens] = vlm.model_params.image_token_id

        image = torch.randn(batch_size, 3, vit_cfg.img_size, vit_cfg.img_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Test without hidden states
        result = vlm(input_ids=input_ids, image=image, attention_mask=attention_mask, output_hidden_states=False)
        assert len(result) == 3
        logits, past_key_values, hidden_states = result
        assert logits is not None
        assert past_key_values is None
        assert hidden_states is None

        # Test with hidden states
        result = vlm(input_ids=input_ids, image=image, attention_mask=attention_mask, output_hidden_states=True)
        assert len(result) == 3
        logits, past_key_values, hidden_states = result
        assert logits is not None
        assert past_key_values is None
        assert hidden_states is not None
        assert isinstance(hidden_states, list)
        assert len(hidden_states) == 2  # n_layers = 2

    @patch("lbm2.models.vlm_hf.AutoModelForVision2Seq.from_pretrained")
    def test_vlm_hf_return_format_consistency(self, mock_from_pretrained, vlm_config):
        """Test that VLMHF consistently returns 3 items"""
        # Mock the HF model
        mock_model = Mock()
        mock_output = Mock()
        mock_output.logits = torch.randn(2, 10, 1000)
        mock_output.past_key_values = None
        mock_output.hidden_states = [torch.randn(2, 10, 128) for _ in range(2)]
        mock_model.return_value = mock_output
        mock_from_pretrained.return_value = mock_model

        # Create a new config with vlm_hf type
        from lbm2.params.model_params import VLMHFParams

        vlm_hf_config = VLMHFParams(
            hf_pretrained="microsoft/git-base", resume_from_checkpoint=None, resume_weights_only=False
        )
        vlm = create_model(vlm_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        image = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Test without hidden states
        result = vlm(input_ids=input_ids, image=image, attention_mask=attention_mask, output_hidden_states=False)
        assert len(result) == 3
        logits, past_key_values, hidden_states = result
        assert logits is not None
        assert past_key_values is None
        assert hidden_states is None

        # Test with hidden states
        result = vlm(input_ids=input_ids, image=image, attention_mask=attention_mask, output_hidden_states=True)
        assert len(result) == 3
        logits, past_key_values, hidden_states = result
        assert logits is not None
        assert past_key_values is None
        assert hidden_states is not None
        assert isinstance(hidden_states, list)
        assert len(hidden_states) == 2

    def test_all_models_return_same_format(self, transformer_config, vlm_config):
        """Test that all models return the same format: (logits, past_key_values, hidden_states)"""
        # This test ensures the interface is consistent across all model types

        # Create all model types
        transformer = create_model(transformer_config)

        # Mock HF models
        with patch("lbm2.models.transformer_hf.AutoModelForCausalLM.from_pretrained") as mock_transformer_hf:
            mock_model = Mock()
            mock_output = Mock()
            mock_output.logits = torch.randn(2, 10, 1000)
            mock_output.past_key_values = None
            mock_output.hidden_states = [torch.randn(2, 10, 128) for _ in range(2)]
            mock_model.return_value = mock_output
            mock_transformer_hf.return_value = mock_model

            # Create a new config with transformer_hf type
            from lbm2.params.model_params import TransformerHFParams

            transformer_hf_config = TransformerHFParams(
                hf_pretrained="microsoft/DialoGPT-small", resume_from_checkpoint=None, resume_weights_only=False
            )
            transformer_hf = create_model(transformer_hf_config)

        # Create VLM
        vlm = create_model(vlm_config.model)

        with patch("lbm2.models.vlm_hf.AutoModelForVision2Seq.from_pretrained") as mock_vlm_hf:
            mock_model = Mock()
            mock_output = Mock()
            mock_output.logits = torch.randn(2, 10, 1000)
            mock_output.past_key_values = None
            mock_output.hidden_states = [torch.randn(2, 10, 128) for _ in range(2)]
            mock_model.return_value = mock_output
            mock_vlm_hf.return_value = mock_model

            # Create a new config with vlm_hf type
            from lbm2.params.model_params import VLMHFParams

            vlm_hf_config = VLMHFParams(
                hf_pretrained="microsoft/git-base", resume_from_checkpoint=None, resume_weights_only=False
            )
            vlm_hf = create_model(vlm_hf_config)

        # Test inputs
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        batch_size, seq_len = 2, num_image_tokens + 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        image = torch.randn(batch_size, 3, vit_cfg.img_size, vit_cfg.img_size)

        # Set image tokens for VLM
        input_ids_vlm = input_ids.clone()
        input_ids_vlm[0, 0:num_image_tokens] = vlm.model_params.image_token_id
        input_ids_vlm[1, 0:num_image_tokens] = vlm.model_params.image_token_id

        # Test all models return same format
        models_and_inputs = [
            (transformer, {"input_ids": input_ids, "attention_mask": attention_mask}),
            (transformer_hf, {"input_ids": input_ids, "attention_mask": attention_mask}),
            (vlm, {"input_ids": input_ids_vlm, "image": image, "attention_mask": attention_mask}),
            (vlm_hf, {"input_ids": input_ids, "image": image, "attention_mask": attention_mask}),
        ]

        for model, inputs in models_and_inputs:
            # Test without hidden states
            result = model(**inputs, output_hidden_states=False)
            assert len(result) == 3
            logits, past_key_values, hidden_states = result
            assert logits is not None
            assert hidden_states is None

            # Test with hidden states
            result = model(**inputs, output_hidden_states=True)
            assert len(result) == 3
            logits, past_key_values, hidden_states = result
            assert logits is not None
            assert hidden_states is not None
            assert isinstance(hidden_states, list)
