from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from lbm2.models.batch_handlers import (
    StableDiffusionBatchHandler,
    TransformerBatchHandler,
    VLMBatchHandler,
    create_batch_handler,
)


class TestBatchHandlerBase:
    """Test the abstract BatchHandler base class methods."""

    def test_slice_inputs_for_accumulation_with_tensors(self):
        """Test slicing tensor inputs for gradient accumulation."""
        handler = TransformerBatchHandler()  # Use concrete implementation

        model_inputs = {
            "input_ids": torch.randint(0, 1000, (8, 10)),
            "attention_mask": torch.ones(8, 10, dtype=torch.bool),
            "output_hidden_states": False,  # Non-tensor value
        }

        sliced_inputs = handler.slice_inputs_for_accumulation(model_inputs, 2, 6)

        assert sliced_inputs["input_ids"].shape == (4, 10)  # [2:6] = 4 samples
        assert sliced_inputs["attention_mask"].shape == (4, 10)
        assert sliced_inputs["output_hidden_states"] is False  # Non-tensor unchanged

    def test_slice_inputs_for_accumulation_with_scalars(self):
        """Test slicing inputs with scalar tensors."""
        handler = TransformerBatchHandler()

        model_inputs = {
            "input_ids": torch.randint(0, 1000, (8, 10)),
            "scalar_value": torch.tensor(5.0),  # 0-dim tensor
            "non_tensor": "some_string",
        }

        sliced_inputs = handler.slice_inputs_for_accumulation(model_inputs, 1, 3)

        assert sliced_inputs["input_ids"].shape == (2, 10)
        assert sliced_inputs["scalar_value"] == torch.tensor(5.0)  # Scalar unchanged
        assert sliced_inputs["non_tensor"] == "some_string"  # Non-tensor unchanged


class TestTransformerBatchHandler:
    """Test the TransformerBatchHandler class."""

    @pytest.fixture
    def handler(self):
        return TransformerBatchHandler()

    @pytest.fixture
    def mock_cfg(self):
        cfg = Mock()
        cfg.data.seq_len = 8
        cfg.data.pad_token_id = 0
        return cfg

    @pytest.fixture
    def sample_batch(self):
        return {
            "input_ids": torch.randint(1, 1000, (2, 12)),
            "attention_mask": torch.ones(2, 12, dtype=torch.bool),
        }

    @pytest.fixture
    def sample_batch_no_mask(self):
        return {
            "input_ids": torch.randint(1, 1000, (2, 12)),
            "attention_mask": None,
        }

    def test_prepare_inputs_with_attention_mask(self, handler, sample_batch):
        """Test prepare_inputs with attention mask."""
        device = torch.device("cpu")
        model_dtype = torch.float32
        cfg = Mock()

        inputs = handler.prepare_inputs(sample_batch, device, model_dtype, cfg)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "output_hidden_states" in inputs
        assert inputs["input_ids"].dtype == torch.long
        assert inputs["attention_mask"].dtype == torch.bool
        assert inputs["output_hidden_states"] is False
        assert inputs["input_ids"].shape == (2, 12)
        assert inputs["attention_mask"].shape == (2, 12)

    def test_prepare_inputs_without_attention_mask(self, handler, sample_batch_no_mask):
        """Test prepare_inputs without attention mask."""
        device = torch.device("cpu")
        model_dtype = torch.float32
        cfg = Mock()

        inputs = handler.prepare_inputs(sample_batch_no_mask, device, model_dtype, cfg)

        assert "input_ids" in inputs
        assert "attention_mask" not in inputs
        assert "output_hidden_states" in inputs
        assert inputs["input_ids"].dtype == torch.long
        assert inputs["output_hidden_states"] is False

    def test_prepare_inputs_and_targets_with_mask(self, handler, sample_batch, mock_cfg):
        """Test prepare_inputs_and_targets with attention mask."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        model_inputs, targets = handler.prepare_inputs_and_targets(sample_batch, device, model_dtype, mock_cfg)

        # Check model inputs
        assert "input_ids" in model_inputs
        assert "attention_mask" in model_inputs
        assert "output_hidden_states" in model_inputs
        assert model_inputs["input_ids"].shape == (2, 8)  # seq_len from config
        assert model_inputs["attention_mask"].shape == (2, 8)
        assert model_inputs["output_hidden_states"] is False

        # Check targets
        assert targets.shape == (2, 8)
        assert targets.dtype == torch.long

    def test_prepare_inputs_and_targets_without_mask(self, handler, sample_batch_no_mask, mock_cfg):
        """Test prepare_inputs_and_targets without attention mask."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        model_inputs, targets = handler.prepare_inputs_and_targets(sample_batch_no_mask, device, model_dtype, mock_cfg)

        # Check model inputs
        assert "input_ids" in model_inputs
        assert "attention_mask" not in model_inputs
        assert "output_hidden_states" in model_inputs
        assert model_inputs["input_ids"].shape == (2, 8)

        # Check targets
        assert targets.shape == (2, 8)

    def test_compute_loss(self, handler, mock_cfg):
        """Test compute_loss method."""
        # Mock model outputs
        outputs = Mock()
        outputs.logits = torch.randn(2, 8, 1000)  # batch_size=2, seq_len=8, vocab_size=1000

        targets = torch.randint(0, 1000, (2, 8))
        loss_fn = F.cross_entropy

        loss = handler.compute_loss(outputs, targets, loss_fn, mock_cfg)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert not torch.isnan(loss)


class TestVLMBatchHandler:
    """Test the VLMBatchHandler class."""

    @pytest.fixture
    def handler(self):
        return VLMBatchHandler()

    @pytest.fixture
    def mock_cfg_vlm(self):
        cfg = Mock()
        cfg.model.type = "vlm"
        cfg.data.seq_len = 8
        cfg.data.pad_token_id = 0
        cfg.data.image_token_id = 32000
        return cfg

    @pytest.fixture
    def mock_cfg_vlm_hf(self):
        cfg = Mock()
        cfg.model.type = "vlm_hf"
        cfg.data.seq_len = 8
        cfg.data.pad_token_id = 0
        cfg.data.image_token_id = 32000
        return cfg

    @pytest.fixture
    def sample_vlm_batch(self):
        return {
            "input_ids": torch.randint(1, 1000, (2, 12)),
            "attention_mask": torch.ones(2, 12, dtype=torch.bool),
            "pixel_values": torch.randn(2, 3, 224, 224),
        }

    @pytest.fixture
    def sample_vlm_batch_no_image(self):
        return {
            "input_ids": torch.randint(1, 1000, (2, 12)),
            "attention_mask": torch.ones(2, 12, dtype=torch.bool),
        }

    def test_prepare_inputs_vlm_with_image(self, handler, sample_vlm_batch, mock_cfg_vlm):
        """Test prepare_inputs for VLM with image."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        inputs = handler.prepare_inputs(sample_vlm_batch, device, model_dtype, mock_cfg_vlm)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "image" in inputs  # Custom VLM uses 'image'
        assert "pixel_values" not in inputs
        assert "output_hidden_states" in inputs
        assert inputs["input_ids"].dtype == torch.long
        assert inputs["attention_mask"].dtype == torch.bool
        assert inputs["image"].dtype == torch.float32
        assert inputs["image"].shape == (2, 3, 224, 224)

    def test_prepare_inputs_vlm_hf_with_image(self, handler, sample_vlm_batch, mock_cfg_vlm_hf):
        """Test prepare_inputs for VLM HF with image."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        inputs = handler.prepare_inputs(sample_vlm_batch, device, model_dtype, mock_cfg_vlm_hf)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "pixel_values" in inputs  # HF VLM uses 'pixel_values'
        assert "image" not in inputs
        assert inputs["pixel_values"].dtype == torch.float32
        assert inputs["pixel_values"].shape == (2, 3, 224, 224)

    def test_prepare_inputs_without_image(self, handler, sample_vlm_batch_no_image, mock_cfg_vlm):
        """Test prepare_inputs without image data."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        inputs = handler.prepare_inputs(sample_vlm_batch_no_image, device, model_dtype, mock_cfg_vlm)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "image" not in inputs
        assert "pixel_values" not in inputs

    def test_prepare_inputs_and_targets_vlm(self, handler, sample_vlm_batch, mock_cfg_vlm):
        """Test prepare_inputs_and_targets for VLM."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        model_inputs, targets = handler.prepare_inputs_and_targets(sample_vlm_batch, device, model_dtype, mock_cfg_vlm)

        # Check model inputs
        assert "input_ids" in model_inputs
        assert "attention_mask" in model_inputs
        assert "image" in model_inputs
        assert "output_hidden_states" in model_inputs
        assert model_inputs["input_ids"].shape == (2, 8)
        assert model_inputs["image"].shape == (2, 3, 224, 224)

        # Check targets
        assert targets.shape == (2, 8)

    def test_prepare_inputs_and_targets_vlm_hf(self, handler, sample_vlm_batch, mock_cfg_vlm_hf):
        """Test prepare_inputs_and_targets for VLM HF."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        model_inputs, targets = handler.prepare_inputs_and_targets(
            sample_vlm_batch, device, model_dtype, mock_cfg_vlm_hf
        )

        # Check model inputs
        assert "input_ids" in model_inputs
        assert "attention_mask" in model_inputs
        assert "pixel_values" in model_inputs
        assert "image" not in model_inputs
        assert model_inputs["pixel_values"].shape == (2, 3, 224, 224)

    def test_compute_loss_with_masking(self, handler, mock_cfg_vlm):
        """Test compute_loss with pad and image token masking."""
        # Mock model outputs
        outputs = Mock()
        outputs.logits = torch.randn(2, 8, 1000)

        # Create targets with pad and image tokens
        targets = torch.randint(1, 1000, (2, 8))
        targets[0, 0] = mock_cfg_vlm.data.pad_token_id  # Add pad token
        targets[0, 1] = mock_cfg_vlm.data.image_token_id  # Add image token

        loss_fn = F.cross_entropy

        loss = handler.compute_loss(outputs, targets, loss_fn, mock_cfg_vlm)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestStableDiffusionBatchHandler:
    """Test the StableDiffusionBatchHandler class."""

    @pytest.fixture
    def handler(self):
        return StableDiffusionBatchHandler()

    @pytest.fixture
    def mock_cfg_diffusion(self):
        cfg = Mock()
        cfg.model.use_flow_matching_scheduler = False
        return cfg

    @pytest.fixture
    def mock_cfg_flow_matching(self):
        cfg = Mock()
        cfg.model.use_flow_matching_scheduler = True
        return cfg

    @pytest.fixture
    def sample_diffusion_batch(self):
        return {
            "input_ids": torch.randint(1, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.bool),
            "pixel_values": torch.randn(2, 3, 64, 64),
        }

    @pytest.fixture
    def sample_diffusion_batch_no_mask(self):
        return {
            "input_ids": torch.randint(1, 1000, (2, 10)),
            "pixel_values": torch.randn(2, 3, 64, 64),
        }

    def test_prepare_inputs_with_mask(self, handler, sample_diffusion_batch, mock_cfg_diffusion):
        """Test prepare_inputs with attention mask."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        inputs = handler.prepare_inputs(sample_diffusion_batch, device, model_dtype, mock_cfg_diffusion)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "image" in inputs
        assert "noise" in inputs
        assert inputs["input_ids"].dtype == torch.long
        assert inputs["attention_mask"].dtype == torch.bool
        assert inputs["image"].dtype == torch.float32
        assert inputs["noise"].dtype == torch.float32
        assert inputs["image"].shape == (2, 3, 64, 64)
        assert inputs["noise"].shape == (2, 3, 64, 64)

    def test_prepare_inputs_without_mask(self, handler, sample_diffusion_batch_no_mask, mock_cfg_diffusion):
        """Test prepare_inputs without attention mask."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        inputs = handler.prepare_inputs(sample_diffusion_batch_no_mask, device, model_dtype, mock_cfg_diffusion)

        assert "input_ids" in inputs
        assert "attention_mask" not in inputs
        assert "image" in inputs
        assert "noise" in inputs

    def test_prepare_inputs_and_targets_standard_diffusion(self, handler, sample_diffusion_batch, mock_cfg_diffusion):
        """Test prepare_inputs_and_targets for standard diffusion."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        # Set seed for reproducible noise
        torch.manual_seed(42)

        model_inputs, targets = handler.prepare_inputs_and_targets(
            sample_diffusion_batch, device, model_dtype, mock_cfg_diffusion
        )

        # Check model inputs
        assert "input_ids" in model_inputs
        assert "attention_mask" in model_inputs
        assert "image" in model_inputs
        assert "noise" in model_inputs
        assert model_inputs["image"].shape == (2, 3, 64, 64)
        assert model_inputs["noise"].shape == (2, 3, 64, 64)

        # For standard diffusion, targets should be the noise
        assert targets.shape == (2, 3, 64, 64)
        assert torch.allclose(targets, model_inputs["noise"])

    def test_prepare_inputs_and_targets_flow_matching(self, handler, sample_diffusion_batch, mock_cfg_flow_matching):
        """Test prepare_inputs_and_targets for flow matching."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        # Set seed for reproducible noise
        torch.manual_seed(42)

        model_inputs, targets = handler.prepare_inputs_and_targets(
            sample_diffusion_batch, device, model_dtype, mock_cfg_flow_matching
        )

        # Check model inputs
        assert "input_ids" in model_inputs
        assert "image" in model_inputs
        assert "noise" in model_inputs

        # For flow matching, targets should be (noise - image)
        assert targets.shape == (2, 3, 64, 64)
        expected_targets = model_inputs["noise"] - model_inputs["image"]
        assert torch.allclose(targets, expected_targets)

    def test_compute_loss(self, handler, mock_cfg_diffusion):
        """Test compute_loss method."""
        # Mock model outputs (predicted noise)
        predicted_noise = torch.randn(2, 3, 64, 64)
        targets = torch.randn(2, 3, 64, 64)
        loss_fn = F.mse_loss

        loss = handler.compute_loss(predicted_noise, targets, loss_fn, mock_cfg_diffusion)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestBatchHandlerFactory:
    """Test the create_batch_handler factory function."""

    def test_create_transformer_handler(self):
        """Test creating transformer batch handler."""
        handler = create_batch_handler("transformer")
        assert isinstance(handler, TransformerBatchHandler)

    def test_create_transformer_hf_handler(self):
        """Test creating transformer_hf batch handler."""
        handler = create_batch_handler("transformer_hf")
        assert isinstance(handler, TransformerBatchHandler)

    def test_create_vlm_handler(self):
        """Test creating vlm batch handler."""
        handler = create_batch_handler("vlm")
        assert isinstance(handler, VLMBatchHandler)

    def test_create_vlm_hf_handler(self):
        """Test creating vlm_hf batch handler."""
        handler = create_batch_handler("vlm_hf")
        assert isinstance(handler, VLMBatchHandler)

    def test_create_stable_diffusion_handler(self):
        """Test creating stable_diffusion batch handler."""
        handler = create_batch_handler("stable_diffusion")
        assert isinstance(handler, StableDiffusionBatchHandler)

    def test_create_handler_unsupported_type(self):
        """Test creating handler for unsupported model type."""
        with pytest.raises(ValueError, match="Batch handler not supported for model type: unsupported_type"):
            create_batch_handler("unsupported_type")
