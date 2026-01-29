from unittest.mock import Mock

import pytest
import torch

from vla_foundry.losses import get_loss_function
from vla_foundry.models.batch_handlers import (
    DiffusionPolicyBatchHandler,
    StableDiffusionBatchHandler,
    TransformerBatchHandler,
    VLMBatchHandler,
)
from vla_foundry.models.registry import create_batch_handler


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
        cfg.z_loss_coefficient = 1e-4
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

        model_inputs, targets, mask = handler.prepare_inputs_and_targets(sample_batch, device, model_dtype, mock_cfg)

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

        # Check mask - TransformerBatchHandler should return None
        assert mask is None

    def test_prepare_inputs_and_targets_without_mask(self, handler, sample_batch_no_mask, mock_cfg):
        """Test prepare_inputs_and_targets without attention mask."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        model_inputs, targets, mask = handler.prepare_inputs_and_targets(
            sample_batch_no_mask, device, model_dtype, mock_cfg
        )

        # Check model inputs
        assert "input_ids" in model_inputs
        assert "attention_mask" not in model_inputs
        assert "output_hidden_states" in model_inputs
        assert model_inputs["input_ids"].shape == (2, 8)

        # Check targets
        assert targets.shape == (2, 8)

        # Check mask - TransformerBatchHandler should return None
        assert mask is None

    def test_compute_loss(self, handler, mock_cfg):
        """Test compute_loss method."""
        # Mock model outputs
        outputs = Mock()
        outputs.logits = torch.randn(2, 8, 1000)  # batch_size=2, seq_len=8, vocab_size=1000

        targets = torch.randint(0, 1000, (2, 8))
        loss_fn = get_loss_function("cross_entropy", mock_cfg)

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
        cfg.z_loss_coefficient = 1e-4
        return cfg

    @pytest.fixture
    def mock_cfg_vlm_hf(self):
        cfg = Mock()
        cfg.model.type = "vlm_hf"
        cfg.data.seq_len = 8
        cfg.data.pad_token_id = 0
        cfg.data.image_token_id = 32000
        cfg.z_loss_coefficient = 1e-4
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
        assert "pixel_values" in inputs
        assert "output_hidden_states" in inputs
        assert inputs["input_ids"].dtype == torch.long
        assert inputs["attention_mask"].dtype == torch.bool
        assert inputs["pixel_values"].dtype == torch.float32
        assert inputs["pixel_values"].shape == (2, 3, 224, 224)

    def test_prepare_inputs_vlm_hf_with_image(self, handler, sample_vlm_batch, mock_cfg_vlm_hf):
        """Test prepare_inputs for VLM HF with image."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        inputs = handler.prepare_inputs(sample_vlm_batch, device, model_dtype, mock_cfg_vlm_hf)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "pixel_values" in inputs
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

        model_inputs, targets, mask = handler.prepare_inputs_and_targets(
            sample_vlm_batch, device, model_dtype, mock_cfg_vlm
        )

        # Check model inputs
        assert "input_ids" in model_inputs
        assert "attention_mask" in model_inputs
        assert "pixel_values" in model_inputs
        assert "output_hidden_states" in model_inputs
        assert model_inputs["input_ids"].shape == (2, 8)
        assert model_inputs["pixel_values"].shape == (2, 3, 224, 224)

        # Check targets
        assert targets.shape == (2, 8)

        # Check mask - VLMBatchHandler should return a mask based on pad and image tokens
        assert mask is not None
        assert mask.shape == (2, 8)
        assert mask.dtype == torch.bool

    def test_prepare_inputs_and_targets_vlm_hf(self, handler, sample_vlm_batch, mock_cfg_vlm_hf):
        """Test prepare_inputs_and_targets for VLM HF."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        model_inputs, targets, mask = handler.prepare_inputs_and_targets(
            sample_vlm_batch, device, model_dtype, mock_cfg_vlm_hf
        )

        # Check model inputs
        assert "input_ids" in model_inputs
        assert "attention_mask" in model_inputs
        assert "pixel_values" in model_inputs
        assert "image" not in model_inputs
        assert model_inputs["pixel_values"].shape == (2, 3, 224, 224)

        # Check mask - VLMBatchHandler should return a mask based on pad and image tokens
        assert mask is not None
        assert mask.shape == (2, 8)
        assert mask.dtype == torch.bool

    def test_prepare_inputs_and_targets_vlm_mask_creation(self, handler, mock_cfg_vlm):
        """Test VLM mask creation with specific pad and image tokens."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        # Create a batch with known pad and image tokens
        batch_with_special_tokens = {
            "input_ids": torch.tensor(
                [
                    [1, 2, mock_cfg_vlm.data.pad_token_id, mock_cfg_vlm.data.image_token_id, 5, 6, 7, 8, 9, 10, 11, 12],
                    [
                        13,
                        mock_cfg_vlm.data.image_token_id,
                        15,
                        16,
                        mock_cfg_vlm.data.pad_token_id,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                    ],
                ]
            ),
            "attention_mask": torch.ones(2, 12, dtype=torch.bool),
            "pixel_values": torch.randn(2, 3, 224, 224),
        }

        model_inputs, targets, mask = handler.prepare_inputs_and_targets(
            batch_with_special_tokens, device, model_dtype, mock_cfg_vlm
        )

        # Check that mask correctly identifies pad and image tokens
        assert mask is not None
        assert mask.shape == (2, 8)  # seq_len from config
        assert mask.dtype == torch.bool

        # Check specific positions where we expect mask to be True
        # Note: targets are chunked to seq_len=8, so positions may shift
        # We'll check that at least some positions are masked
        assert mask.any(), "Mask should have some True values for pad/image tokens"

        # Verify that mask corresponds to pad_token_id and image_token_id in targets
        pad_positions = targets == mock_cfg_vlm.data.pad_token_id
        image_positions = targets == mock_cfg_vlm.data.image_token_id
        expected_mask = pad_positions | image_positions

        assert torch.equal(mask, expected_mask), "Mask should match pad and image token positions"

    def test_compute_loss_with_masking(self, handler, mock_cfg_vlm):
        """Test compute_loss with pad and image token masking."""
        # Mock model outputs
        outputs = Mock()
        outputs.logits = torch.randn(2, 8, 50000)  # Large vocab to accommodate image_token_id=32000

        # Create targets with pad and image tokens
        targets = torch.randint(1, 1000, (2, 8)).long()
        targets[0, 0] = mock_cfg_vlm.data.pad_token_id  # Add pad token
        targets[0, 1] = mock_cfg_vlm.data.image_token_id  # Add image token

        loss_fn = get_loss_function("cross_entropy", mock_cfg_vlm)

        loss = handler.compute_loss(outputs, targets, loss_fn, mock_cfg_vlm)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_compute_loss_with_mask_parameter(self, handler, mock_cfg_vlm):
        """Test compute_loss with explicit mask parameter."""
        # Mock model outputs
        outputs = Mock()
        outputs.logits = torch.randn(2, 8, 50000)

        # Create targets and mask
        targets = torch.randint(1, 1000, (2, 8)).long()
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[0, 0] = True  # Mask first position
        mask[1, 3] = True  # Mask another position

        loss_fn = get_loss_function("cross_entropy", mock_cfg_vlm)

        # Test loss computation with mask
        loss = handler.compute_loss(outputs, targets, loss_fn, mock_cfg_vlm, mask=mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

        # Verify that masked positions are set to -100 (ignore index for cross entropy)
        # This is done internally by the VLMBatchHandler.compute_loss method
        # We can't directly inspect the modified targets, but we can verify the loss is computed


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

        model_inputs, targets, mask = handler.prepare_inputs_and_targets(
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

        # Check mask - StableDiffusionBatchHandler should return None
        assert mask is None

    def test_prepare_inputs_and_targets_flow_matching(self, handler, sample_diffusion_batch, mock_cfg_flow_matching):
        """Test prepare_inputs_and_targets for flow matching."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        # Set seed for reproducible noise
        torch.manual_seed(42)

        model_inputs, targets, mask = handler.prepare_inputs_and_targets(
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

        # Check mask - StableDiffusionBatchHandler should return None
        assert mask is None

    def test_compute_loss(self, handler, mock_cfg_diffusion):
        """Test compute_loss method."""
        # Mock model outputs (predicted noise)
        predicted_noise = torch.randn(2, 3, 64, 64)
        targets = torch.randn(2, 3, 64, 64)
        loss_fn = get_loss_function("mse", mock_cfg_diffusion)

        loss = handler.compute_loss(predicted_noise, targets, loss_fn, mock_cfg_diffusion)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestDiffusionPolicyBatchHandler:
    """Test the DiffusionPolicyBatchHandler class."""

    @pytest.fixture
    def handler(self):
        return DiffusionPolicyBatchHandler()

    @pytest.fixture
    def mock_cfg_diffusion_policy(self):
        cfg = Mock()
        cfg.model.type = "diffusion_policy"
        return cfg

    @pytest.fixture
    def sample_diffusion_policy_batch(self):
        """Sample batch data for diffusion policy."""
        return {
            "input_ids": torch.randint(1, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.bool),
            "pixel_values": torch.randn(2, 3, 224, 224),
            "actions": torch.randn(2, 16, 7),  # batch_size=2, seq_len=16, action_dim=7
            "past_mask": torch.ones(2, 16, dtype=torch.bool),
            "future_mask": torch.zeros(2, 16, dtype=torch.bool),
        }

    @pytest.fixture
    def sample_diffusion_policy_batch_no_attention_mask(self):
        """Sample batch data for diffusion policy without attention mask."""
        return {
            "input_ids": torch.randint(1, 1000, (2, 10)),
            "pixel_values": torch.randn(2, 3, 224, 224),
            "actions": torch.randn(2, 16, 7),
            "past_mask": torch.ones(2, 16, dtype=torch.bool),
            "future_mask": torch.zeros(2, 16, dtype=torch.bool),
        }

    def test_prepare_inputs_with_attention_mask(
        self, handler, sample_diffusion_policy_batch, mock_cfg_diffusion_policy
    ):
        """Test prepare_inputs with attention mask."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        inputs = handler.prepare_inputs(sample_diffusion_policy_batch, device, model_dtype, mock_cfg_diffusion_policy)

        # Check all required fields are present
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "pixel_values" in inputs
        assert "actions" in inputs
        assert "noise" in inputs
        assert "past_mask" in inputs
        assert "future_mask" in inputs

        # Check data types
        assert inputs["input_ids"].dtype == torch.long
        assert inputs["attention_mask"].dtype == torch.bool
        assert inputs["pixel_values"].dtype == torch.float32
        assert inputs["actions"].dtype == torch.float32
        assert inputs["noise"].dtype == torch.float32
        assert inputs["past_mask"].dtype == torch.bool
        assert inputs["future_mask"].dtype == torch.bool

        # Check shapes
        assert inputs["input_ids"].shape == (2, 10)
        assert inputs["attention_mask"].shape == (2, 10)
        assert inputs["pixel_values"].shape == (2, 3, 224, 224)
        assert inputs["actions"].shape == (2, 16, 7)
        assert inputs["noise"].shape == (2, 16, 7)  # Same shape as actions
        assert inputs["past_mask"].shape == (2, 16)
        assert inputs["future_mask"].shape == (2, 16)

        # Check device
        assert inputs["input_ids"].device == device
        assert inputs["pixel_values"].device == device
        assert inputs["actions"].device == device
        assert inputs["noise"].device == device

    def test_prepare_inputs_without_attention_mask(
        self, handler, sample_diffusion_policy_batch_no_attention_mask, mock_cfg_diffusion_policy
    ):
        """Test prepare_inputs without attention mask."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        inputs = handler.prepare_inputs(
            sample_diffusion_policy_batch_no_attention_mask, device, model_dtype, mock_cfg_diffusion_policy
        )

        # Check that attention_mask is None when not provided
        assert inputs["attention_mask"] is None

        # Check other required fields are still present
        assert "input_ids" in inputs
        assert "pixel_values" in inputs
        assert "actions" in inputs
        assert "noise" in inputs
        assert "past_mask" in inputs
        assert "future_mask" in inputs

    def test_prepare_inputs_and_targets(self, handler, sample_diffusion_policy_batch, mock_cfg_diffusion_policy):
        """Test prepare_inputs_and_targets method."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        # Set seed for reproducible noise
        torch.manual_seed(42)

        model_inputs, targets, mask = handler.prepare_inputs_and_targets(
            sample_diffusion_policy_batch, device, model_dtype, mock_cfg_diffusion_policy
        )

        # Check model inputs structure
        assert "input_ids" in model_inputs
        assert "pixel_values" in model_inputs
        assert "actions" in model_inputs
        assert "noise" in model_inputs
        assert "past_mask" in model_inputs
        assert "future_mask" in model_inputs

        # Check shapes match original batch
        assert model_inputs["input_ids"].shape == (2, 10)
        assert model_inputs["pixel_values"].shape == (2, 3, 224, 224)
        assert model_inputs["actions"].shape == (2, 16, 7)
        assert model_inputs["noise"].shape == (2, 16, 7)

        # Check targets are computed correctly (noise - actions)
        assert targets.shape == (2, 16, 7)
        expected_targets = model_inputs["noise"] - model_inputs["actions"]
        assert torch.allclose(targets, expected_targets)

        # Check mask - DiffusionPolicyBatchHandler should return None
        assert mask is None

    def test_compute_loss(self, handler, mock_cfg_diffusion_policy):
        """Test compute_loss method."""
        # Mock model outputs (predicted direction)
        predicted_direction = torch.randn(2, 16, 7)
        targets = torch.randn(2, 16, 7)
        loss_fn = get_loss_function("mse", mock_cfg_diffusion_policy)

        loss = handler.compute_loss(predicted_direction, targets, loss_fn, mock_cfg_diffusion_policy)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert not torch.isnan(loss)

    def test_compute_loss_with_mask(self, handler, mock_cfg_diffusion_policy):
        """Test compute_loss method with mask parameter."""
        # Mock model outputs and targets
        predicted_direction = torch.randn(2, 16, 7)
        targets = torch.randn(2, 16, 7)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[0, :8] = False  # Mask out first half of first batch
        loss_fn = get_loss_function("mse", mock_cfg_diffusion_policy)

        loss = handler.compute_loss(predicted_direction, targets, loss_fn, mock_cfg_diffusion_policy, mask=mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_noise_generation_randomness(self, handler, sample_diffusion_policy_batch, mock_cfg_diffusion_policy):
        """Test that noise generation produces different values across calls."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        # Generate inputs twice without setting seed
        inputs1 = handler.prepare_inputs(sample_diffusion_policy_batch, device, model_dtype, mock_cfg_diffusion_policy)
        inputs2 = handler.prepare_inputs(sample_diffusion_policy_batch, device, model_dtype, mock_cfg_diffusion_policy)

        # Noise should be different across calls
        assert not torch.allclose(inputs1["noise"], inputs2["noise"])

        # But actions should be the same (they come from the batch)
        assert torch.allclose(inputs1["actions"], inputs2["actions"])

    def test_slice_inputs_for_accumulation(self, handler, sample_diffusion_policy_batch, mock_cfg_diffusion_policy):
        """Test slicing inputs for gradient accumulation."""
        device = torch.device("cpu")
        model_dtype = torch.float32

        inputs = handler.prepare_inputs(sample_diffusion_policy_batch, device, model_dtype, mock_cfg_diffusion_policy)

        # Slice to get first batch element only
        sliced_inputs = handler.slice_inputs_for_accumulation(inputs, 0, 1)

        # Check that tensor dimensions are correctly sliced
        assert sliced_inputs["input_ids"].shape == (1, 10)  # batch_size reduced from 2 to 1
        assert sliced_inputs["pixel_values"].shape == (1, 3, 224, 224)
        assert sliced_inputs["actions"].shape == (1, 16, 7)
        assert sliced_inputs["noise"].shape == (1, 16, 7)
        assert sliced_inputs["past_mask"].shape == (1, 16)
        assert sliced_inputs["future_mask"].shape == (1, 16)

        # Check that None values pass through unchanged
        inputs_no_mask = inputs.copy()
        inputs_no_mask["attention_mask"] = None
        sliced_inputs_no_mask = handler.slice_inputs_for_accumulation(inputs_no_mask, 0, 1)
        assert sliced_inputs_no_mask["attention_mask"] is None


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

    def test_create_diffusion_policy_handler(self):
        """Test creating diffusion_policy batch handler."""
        handler = create_batch_handler("diffusion_policy")
        assert isinstance(handler, DiffusionPolicyBatchHandler)

    def test_create_handler_unsupported_type(self):
        """Test creating handler for unsupported model type."""
        with pytest.raises(ValueError, match="Batch handler for model type 'unsupported_type' is not registered"):
            create_batch_handler("unsupported_type")
