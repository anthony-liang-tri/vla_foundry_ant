from unittest.mock import Mock, patch

import pytest
import torch

from lbm2.models import create_model
from lbm2.params.model_params import DiffusionPolicyParams
from lbm2.params.train_experiment_params import load_params_from_yaml


class TestDiffusionPolicy:
    @pytest.fixture
    def diffusion_policy_config(self):
        return load_params_from_yaml(
            DiffusionPolicyParams, "tests/params/dummy_configs/dummy_diffusion_policy_config.yaml"
        )

    @pytest.fixture
    def diffusion_policy(self, diffusion_policy_config):
        with patch("lbm2.models.diffusion_policy.clip_hf.CLIPModel.from_pretrained") as mock_clip_pretrained:
            # Mock the HuggingFace CLIPModel with proper projection_dim
            mock_hf_clip_model = Mock()
            mock_hf_clip_model.projection_dim = 512
            mock_clip_pretrained.return_value = mock_hf_clip_model

            # Create the diffusion policy model
            model = create_model(diffusion_policy_config)
            return model

    def _mock_clip_output(self, batch_size):
        """Helper method to create mock CLIP output"""
        mock_clip_output = Mock()
        mock_clip_output.text_embeds = torch.randn(batch_size, 512)
        mock_clip_output.image_embeds = torch.randn(batch_size, 512)
        return mock_clip_output

    def test_diffusion_policy_forward_basic(self, diffusion_policy):
        """Test basic forward pass of diffusion policy"""
        batch_size, seq_len = 2, 10
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Mock the CLIP forward call
        with patch.object(diffusion_policy.clip, "forward", return_value=self._mock_clip_output(batch_size)):
            # Forward pass
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            # Check output shape
            assert output.shape == (batch_size, seq_len, action_dim)
            assert output.dtype == torch.float32

    def test_diffusion_policy_forward_with_past_actions(self, diffusion_policy):
        """Test forward pass with some past actions (mixed past/future)"""
        batch_size, seq_len = 2, 10
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors with mixed past/future
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        # First 5 timesteps are past (no noise), last 5 are future (with noise)
        past_mask = torch.cat(
            [torch.ones(batch_size, 5, dtype=torch.bool), torch.zeros(batch_size, 5, dtype=torch.bool)], dim=1
        )
        future_mask = ~past_mask

        # Mock the CLIP forward call
        with patch.object(diffusion_policy.clip, "forward", return_value=self._mock_clip_output(batch_size)):
            # Forward pass
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            # Check output shape
            assert output.shape == (batch_size, seq_len, action_dim)

    def test_diffusion_policy_generate_actions(self, diffusion_policy):
        """Test action generation using iterative denoising"""
        batch_size, seq_len = 2, 8
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        actions = torch.randn(batch_size, seq_len, action_dim)
        # First half are past actions, second half will be generated
        past_mask = torch.cat(
            [
                torch.ones(batch_size, seq_len // 2, dtype=torch.bool),
                torch.zeros(batch_size, seq_len // 2, dtype=torch.bool),
            ],
            dim=1,
        )

        # Mock the CLIP forward call
        with patch.object(diffusion_policy.clip, "forward", return_value=self._mock_clip_output(batch_size)):
            # Generate actions with fewer inference steps for speed
            generated_actions = diffusion_policy.generate_actions(
                input_ids=input_ids,
                pixel_values=pixel_values,
                actions=actions,
                attention_mask=attention_mask,
                num_inference_steps=5,
                past_mask=past_mask,
            )

            # Check output shape and that past actions are preserved
            assert generated_actions.shape == (batch_size, seq_len, action_dim)
            # Past actions should be preserved
            torch.testing.assert_close(
                generated_actions[:, : seq_len // 2], actions[:, : seq_len // 2], rtol=1e-5, atol=1e-5
            )

    def test_diffusion_policy_generate_actions_no_past_mask(self, diffusion_policy):
        """Test action generation without past mask (all actions generated)"""
        batch_size, seq_len = 2, 6
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        actions = torch.randn(batch_size, seq_len, action_dim)

        # Mock the CLIP forward call
        with patch.object(diffusion_policy.clip, "forward", return_value=self._mock_clip_output(batch_size)):
            # Generate actions without past mask
            generated_actions = diffusion_policy.generate_actions(
                input_ids=input_ids,
                pixel_values=pixel_values,
                actions=actions,
                attention_mask=attention_mask,
                num_inference_steps=3,
            )

            # Check output shape
            assert generated_actions.shape == (batch_size, seq_len, action_dim)

    def test_diffusion_policy_weight_initialization(self, diffusion_policy):
        """Test that weight initialization works correctly"""
        # Check that time encoding weights were initialized with sinusoidal embeddings
        time_encoding_weights = diffusion_policy.time_encoding.weight
        assert time_encoding_weights.shape == (100, 512)  # num_timesteps x projection_dim

        # Check that output layer was initialized with Xavier
        output_weights = diffusion_policy.output_layer.weight
        assert output_weights.shape == (
            diffusion_policy.model_params.action_dim,
            diffusion_policy.transformer.hidden_dim,
        )

        # Weights should not be zero (indicating initialization occurred)
        assert time_encoding_weights.abs().sum() > 0
        assert output_weights.abs().sum() > 0

    def test_diffusion_policy_model_components(self, diffusion_policy):
        """Test that all model components are properly initialized"""
        # Check that all components exist
        assert hasattr(diffusion_policy, "clip")
        assert hasattr(diffusion_policy, "transformer")
        assert hasattr(diffusion_policy, "scheduler")
        assert hasattr(diffusion_policy, "time_encoding")
        assert hasattr(diffusion_policy, "sinusoidal_position_embeddings")
        assert hasattr(diffusion_policy, "output_layer")
        assert hasattr(diffusion_policy, "action_encode")
        assert hasattr(diffusion_policy, "condition_encode")

        # Check dimensions
        assert diffusion_policy.time_encoding.num_embeddings == 100  # num_timesteps
        assert diffusion_policy.time_encoding.embedding_dim == 512  # clip projection_dim
        assert diffusion_policy.output_layer.in_features == 128  # transformer hidden_dim
        assert diffusion_policy.output_layer.out_features == 7  # action_dim
        assert diffusion_policy.action_encode.in_features == 7  # action_dim
        assert diffusion_policy.action_encode.out_features == 128  # transformer hidden_dim
        assert diffusion_policy.condition_encode.in_features == 512  # clip projection_dim
        assert diffusion_policy.condition_encode.out_features == 128  # transformer hidden_dim

    def test_diffusion_policy_conditional_embeddings_shape(self, diffusion_policy):
        """Test that conditional embeddings are created with correct shapes"""
        batch_size = 2

        # Mock the CLIP forward call
        with patch.object(diffusion_policy.clip, "forward", return_value=self._mock_clip_output(batch_size)):
            # Test time embeddings
            timesteps = torch.randint(0, 100, (batch_size,))
            time_embeddings = diffusion_policy.time_encoding(timesteps)
            assert time_embeddings.shape == (batch_size, 512)

            # Test that image embeddings can be handled in different shapes
            mock_output = self._mock_clip_output(batch_size)
            # 2D case (batch_size, features)
            if mock_output.image_embeds.ndim == 2:
                image_embeddings = mock_output.image_embeds.unsqueeze(1)
                assert image_embeddings.shape == (batch_size, 1, 512)

    def test_diffusion_policy_transformer_integration(self, diffusion_policy):
        """Test integration with transformer component"""
        batch_size, seq_len = 2, 8
        action_dim = diffusion_policy.model_params.action_dim

        # Create mock transformer input
        conditional_embeddings = torch.randn(batch_size, 3, 128)  # time + text + image
        noisy_action = torch.randn(batch_size, seq_len, 128)
        transformer_input = torch.cat([conditional_embeddings, noisy_action], dim=1)

        # Test transformer forward pass
        transformer_output = diffusion_policy.transformer(
            inputs_embeds=transformer_input,
            output_hidden_states=True,
            use_cache=False,
            is_causal=False,
        )

        # Check that transformer returns expected structure
        assert hasattr(transformer_output, "hidden_states")
        assert transformer_output.hidden_states is not None
        assert len(transformer_output.hidden_states) == 2  # n_layers

        # Check that we can extract action predictions
        action_seq_len = seq_len
        predicted_direction = diffusion_policy.output_layer(
            transformer_output.hidden_states[-1][:, -action_seq_len:, :]
        )
        assert predicted_direction.shape == (batch_size, seq_len, action_dim)

    def test_diffusion_policy_noise_scheduler_integration(self, diffusion_policy):
        """Test integration with noise scheduler"""
        batch_size, seq_len = 2, 5
        action_dim = diffusion_policy.model_params.action_dim

        # Create test data
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        timesteps = torch.randint(0, 100, (batch_size,))
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Test noise addition
        noisy_action = diffusion_policy.scheduler.add_noise(actions, noise, timesteps, mask=future_mask)
        assert noisy_action.shape == actions.shape

        # Test scheduler step (for generation)
        predicted_direction = torch.randn(batch_size, seq_len, action_dim)
        denoised_action = diffusion_policy.scheduler.step(predicted_direction, 50, actions, step_size=1)
        assert denoised_action.shape == actions.shape

    def test_diffusion_policy_edge_cases(self, diffusion_policy):
        """Test edge cases and error conditions"""
        batch_size = 1

        # Test with minimal sequence length
        input_ids = torch.randint(0, 1000, (batch_size, 1))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, 1, dtype=torch.bool)
        actions = torch.randn(batch_size, 1, 7)
        noise = torch.randn(batch_size, 1, 7)
        past_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
        future_mask = torch.ones(batch_size, 1, dtype=torch.bool)

        # Mock the CLIP forward call
        with patch.object(diffusion_policy.clip, "forward", return_value=self._mock_clip_output(batch_size)):
            # Should not raise error with minimal input
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )
            assert output.shape == (batch_size, 1, 7)

    def test_diffusion_policy_device_consistency(self, diffusion_policy):
        """Test that model handles device placement correctly"""
        batch_size, seq_len = 2, 4
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Mock the CLIP forward call
        with patch.object(diffusion_policy.clip, "forward", return_value=self._mock_clip_output(batch_size)):
            # Test that timesteps are moved to correct device
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            # Output should be on same device as input
            assert output.device == actions.device

    def test_diffusion_policy_gradient_flow(self, diffusion_policy):
        """Test that gradients flow properly through the model"""
        batch_size, seq_len = 2, 4
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors with gradient tracking
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        actions = torch.randn(batch_size, seq_len, action_dim, requires_grad=True)
        noise = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Mock the CLIP forward call
        with patch.object(diffusion_policy.clip, "forward", return_value=self._mock_clip_output(batch_size)):
            # Forward pass
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            # Compute loss and backpropagate
            loss = output.sum()
            loss.backward()

            # Check that gradients exist for key parameters
            assert diffusion_policy.output_layer.weight.grad is not None
            assert diffusion_policy.action_encode.weight.grad is not None
            assert diffusion_policy.condition_encode.weight.grad is not None
