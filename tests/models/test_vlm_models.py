from unittest.mock import Mock, patch

import pytest
import torch

from lbm2.models import create_model
from lbm2.models.utils import compute_num_image_tokens
from lbm2.models.vlm import ModalityProjector
from lbm2.params.model_params import ViTParams
from lbm2.params.train_experiment_params import load_experiment_params_from_yaml, load_params_from_yaml


class TestModalityProjector:
    @pytest.fixture
    def vit_configs(self):
        return load_params_from_yaml(ViTParams, "tests/params/dummy_configs/dummy_vit_config.yaml")

    @pytest.fixture
    def projector(self, vit_configs):
        output_dim = vit_configs.hidden_dim * (vit_configs.projector_pixel_shuffle_factor**2)
        return ModalityProjector(vit_configs, output_dim)

    def test_modality_projector_forward(self, projector):
        """Test modality projector forward pass"""
        batch_size, seq_len = 2, 16  # 4x4 sequence
        input_tensor = torch.randn(batch_size, seq_len, 100)  # hidden_dim = 100

        output = projector(input_tensor)

        # After pixel shuffle: 100 * 2^2 = 400, then projected to output_dim
        expected_seq_len = seq_len // 4  # 16 // 4 = 4
        assert output.shape == (batch_size, expected_seq_len, 400)

    def test_modality_projector_pixel_shuffle_perfect_square(self, projector):
        """Test pixel shuffle with perfect square sequence length"""
        batch_size, seq_len = 2, 16  # 4x4 sequence
        input_tensor = torch.randn(batch_size, seq_len, 100)

        # Should not raise error
        output = projector(input_tensor)
        assert output.shape[1] == 4  # 16 // 4

    def test_modality_projector_pixel_shuffle_error_not_perfect_square(self, vit_configs):
        """Test pixel shuffle error with non-perfect square sequence length"""
        output_dim = vit_configs.hidden_dim * (vit_configs.projector_pixel_shuffle_factor**2)
        projector = ModalityProjector(vit_configs, output_dim)
        batch_size, seq_len = 2, 15  # Not a perfect square

        input_tensor = torch.randn(batch_size, seq_len, 100)

        with pytest.raises(AssertionError):
            projector(input_tensor)

    def test_modality_projector_pixel_shuffle_error_not_divisible(self):
        """Test pixel shuffle error when sequence root not divisible by scale factor"""
        # Create a config with scale factor 3 (not divisible by 4)
        vit_cfg = load_params_from_yaml(ViTParams, "tests/params/dummy_configs/dummy_vit_config.yaml")
        object.__setattr__(vit_cfg, "projector_pixel_shuffle_factor", 3)
        output_dim = vit_cfg.hidden_dim * (vit_cfg.projector_pixel_shuffle_factor**2)
        projector = ModalityProjector(vit_cfg, output_dim)
        batch_size, seq_len = 2, 16  # 4x4 sequence

        input_tensor = torch.randn(batch_size, seq_len, 100)

        with pytest.raises(AssertionError):
            projector(input_tensor)


class TestVLM:
    @pytest.fixture
    def vlm_config(self):
        return load_experiment_params_from_yaml("tests/params/dummy_configs/dummy_vlm_config.yaml").model

    @pytest.fixture
    def vlm(self, vlm_config):
        vlm = create_model(vlm_config)
        # Use a safe in-vocab token id for image tokens to avoid embedding OOB
        import builtins

        builtins.object.__setattr__(vlm.model_params, "image_token_id", 0)
        return vlm

    def test_vlm_forward_basic(self, vlm):
        """Test basic forward pass without hidden states"""
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        batch_size, seq_len = 2, num_image_tokens + 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # Compute expected number of image tokens based on ViT config
        input_ids[0, 0:num_image_tokens] = vlm.model_params.image_token_id
        input_ids[1, 0:num_image_tokens] = vlm.model_params.image_token_id

        image = torch.randn(batch_size, 3, vit_cfg.img_size, vit_cfg.img_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        logits, past_key_values, hidden_states = vlm(
            input_ids=input_ids, image=image, attention_mask=attention_mask, output_hidden_states=False
        )

        assert logits.shape == (batch_size, seq_len, 1000)  # vocab_size = 1000
        assert past_key_values is None
        assert hidden_states is None

    def test_vlm_forward_with_hidden_states(self, vlm):
        """Test forward pass with hidden states returned"""
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        batch_size, seq_len = 2, num_image_tokens + 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # Set some tokens as image tokens
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        input_ids[0, 0:num_image_tokens] = vlm.model_params.image_token_id
        input_ids[1, 0:num_image_tokens] = vlm.model_params.image_token_id

        image = torch.randn(batch_size, 3, vit_cfg.img_size, vit_cfg.img_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        vlm.eval()
        logits, past_key_values, hidden_states = vlm(
            input_ids=input_ids, image=image, attention_mask=attention_mask, output_hidden_states=True, use_cache=True
        )

        assert logits.shape == (batch_size, seq_len, 1000)
        assert past_key_values is not None
        assert isinstance(hidden_states, list)
        assert len(hidden_states) == 2  # n_layers = 2

    def test_vlm_forward_image_token_mismatch(self, vlm):
        """Test error when image token count doesn't match image embedding count"""
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        batch_size, seq_len = 2, num_image_tokens + 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # Set wrong number of image tokens
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        wrong = max(1, num_image_tokens // 2)
        input_ids[0, 0:wrong] = vlm.model_params.image_token_id  # mismatch count

        image = torch.randn(batch_size, 3, vit_cfg.img_size, vit_cfg.img_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        with pytest.raises(AssertionError):
            vlm(input_ids=input_ids, image=image, attention_mask=attention_mask)

    def test_vlm_properties(self, vlm):
        """Test VLM properties"""
        assert vlm.hidden_dim == 128  # transformer.hidden_dim = 128
        assert vlm.num_hidden_layers == 2  # transformer.n_layers = 2

    def test_vlm_generate(self, vlm):
        """Test VLM generation"""
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        batch_size, seq_len = 2, num_image_tokens + 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        input_ids[:, 0:num_image_tokens] = vlm.model_params.image_token_id
        image = torch.randn(batch_size, 3, vit_cfg.img_size, vit_cfg.img_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        generated = vlm.generate(input_ids, image, attention_mask, max_new_tokens=5)

        assert generated.shape == (batch_size, seq_len + 5)


class TestVLMHF:
    @pytest.fixture
    def vlm_hf_config(self):
        return load_experiment_params_from_yaml("tests/params/dummy_configs/dummy_vlm_hf_config.yaml").model

    @patch("lbm2.models.vlm_hf.AutoModelForVision2Seq.from_pretrained")
    def test_vlm_hf_forward_basic(self, mock_from_pretrained, vlm_hf_config):
        """Test basic forward pass without hidden states"""
        # Mock the HF model
        mock_model = Mock()
        mock_model.logits = torch.randn(2, 10, 1000)
        mock_model.past_key_values = None

        # Mock the forward method to return the expected values
        mock_output = Mock()
        mock_output.logits = mock_model.logits
        mock_output.past_key_values = mock_model.past_key_values
        mock_model.return_value = mock_output
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        image = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        logits, past_key_values, hidden_states = vlm(
            input_ids=input_ids, image=image, attention_mask=attention_mask, output_hidden_states=False
        )

        assert logits.shape == (batch_size, seq_len, 1000)
        assert past_key_values is None
        assert hidden_states is None

    @patch("lbm2.models.vlm_hf.AutoModelForVision2Seq.from_pretrained")
    def test_vlm_hf_forward_with_hidden_states(self, mock_from_pretrained, vlm_hf_config):
        """Test forward pass with hidden states returned"""
        # Mock the HF model
        mock_model = Mock()
        mock_model.logits = torch.randn(2, 10, 1000)
        mock_model.past_key_values = None
        mock_model.hidden_states = [torch.randn(2, 10, 128) for _ in range(2)]

        # Mock the forward method to return the expected values
        mock_output = Mock()
        mock_output.logits = mock_model.logits
        mock_output.past_key_values = mock_model.past_key_values
        mock_output.hidden_states = mock_model.hidden_states
        mock_model.return_value = mock_output
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        image = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        logits, past_key_values, hidden_states = vlm(
            input_ids=input_ids, image=image, attention_mask=attention_mask, output_hidden_states=True
        )

        assert logits.shape == (batch_size, seq_len, 1000)
        assert past_key_values is None
        assert isinstance(hidden_states, list)
        assert len(hidden_states) == 2

    @patch("lbm2.models.vlm_hf.AutoModelForVision2Seq.from_pretrained")
    def test_vlm_hf_forward_hidden_states_fallback(self, mock_from_pretrained, vlm_hf_config):
        """Test forward pass with hidden states fallback to last_hidden_state"""
        # Mock the HF model
        mock_model = Mock()
        mock_model.logits = torch.randn(2, 10, 1000)
        mock_model.past_key_values = None
        mock_model.hidden_states = [torch.randn(2, 10, 128) for _ in range(2)]

        # Mock the forward method to return the expected values
        mock_output = Mock()
        mock_output.logits = mock_model.logits
        mock_output.past_key_values = mock_model.past_key_values
        mock_output.hidden_states = mock_model.hidden_states
        mock_model.return_value = mock_output
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        image = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        logits, past_key_values, hidden_states = vlm(
            input_ids=input_ids, image=image, attention_mask=attention_mask, output_hidden_states=True
        )

        assert logits.shape == (batch_size, seq_len, 1000)
        assert past_key_values is None
        assert isinstance(hidden_states, list)
        # Should create hidden states based on num_hidden_layers

    @patch("lbm2.models.vlm_hf.AutoModelForVision2Seq.from_pretrained")
    def test_vlm_hf_properties(self, mock_from_pretrained, vlm_hf_config):
        """Test VLM HF properties"""
        # Mock the HF model with config
        mock_model = Mock()
        mock_config_obj = Mock()
        mock_config_obj.hidden_size = 128
        mock_config_obj.num_hidden_layers = 2
        mock_model.config = mock_config_obj
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        # Test that properties can be accessed
        assert hasattr(vlm, "hidden_dim")
        assert hasattr(vlm, "num_hidden_layers")
        assert isinstance(vlm.hidden_dim, int)
        assert isinstance(vlm.num_hidden_layers, int)

    @patch("lbm2.models.vlm_hf.AutoModelForVision2Seq.from_pretrained")
    def test_vlm_hf_set_num_action_layers(self, mock_from_pretrained, vlm_hf_config):
        """Test setting number of action layers"""
        # Mock the HF model
        mock_model = Mock()
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        # Initially should be None
        assert vlm._limit_hidden_states_to_last_n is None

        # Set to 3 layers
        vlm.set_num_action_layers(3)
        assert vlm._limit_hidden_states_to_last_n == 3

        # Set to 0 layers
        vlm.set_num_action_layers(0)
        assert vlm._limit_hidden_states_to_last_n == 0

    @patch("lbm2.models.vlm_hf.AutoModelForVision2Seq.from_pretrained")
    def test_vlm_hf_grad_checkpointing(self, mock_from_pretrained, vlm_hf_config):
        """Test gradient checkpointing methods"""
        # Mock the HF model with gradient checkpointing methods
        mock_model = Mock()
        mock_model.gradient_checkpointing_enable = Mock()
        mock_model.gradient_checkpointing_disable = Mock()
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        # Test that gradient checkpointing methods exist and can be called
        vlm.set_grad_checkpointing(True)
        vlm.set_grad_checkpointing(False)

    @patch("lbm2.models.vlm_hf.AutoModelForVision2Seq.from_pretrained")
    def test_vlm_hf_grad_checkpointing_no_methods(self, mock_from_pretrained, vlm_hf_config):
        """Test gradient checkpointing when methods don't exist"""
        # Mock the HF model without gradient checkpointing methods
        mock_model = Mock()
        # Don't add gradient_checkpointing methods
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        # Should not raise error even if methods don't exist
        vlm.set_grad_checkpointing(True)
        vlm.set_grad_checkpointing(False)

    @patch("lbm2.models.vlm_hf.AutoModelForVision2Seq.from_pretrained")
    def test_vlm_hf_generate(self, mock_from_pretrained, vlm_hf_config):
        """Test VLM HF generation"""
        # Mock the HF model
        mock_model = Mock()
        mock_model.logits = torch.randn(2, 15, 1000)  # 10 + 5 new tokens
        mock_model.past_key_values = None

        # Mock the forward method to return the expected values
        mock_output = Mock()
        mock_output.logits = mock_model.logits
        mock_output.past_key_values = mock_model.past_key_values
        mock_model.return_value = mock_output
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        image = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        generated = vlm.generate(input_ids, image, attention_mask, max_new_tokens=5)

        assert generated.shape == (batch_size, seq_len + 5)
