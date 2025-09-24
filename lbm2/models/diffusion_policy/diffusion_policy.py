from typing import Union

import torch

from lbm2.models.base_model import BaseModel
from lbm2.models.diffusion.noise_scheduler import NoiseScheduler
from lbm2.models.diffusion.unet import SinusoidalPositionEmbeddings
from lbm2.models.diffusion_policy.clip_hf import CLIPHF
from lbm2.models.transformer import Transformer
from lbm2.models.transformer_hf import TransformerHF
from lbm2.params.model_params import DiffusionPolicyParams


class DiffusionPolicy(BaseModel):
    def __init__(
        self,
        model_params: DiffusionPolicyParams,
        clip: CLIPHF,
        transformer: Union[Transformer, TransformerHF],
        noise_scheduler: NoiseScheduler,
    ):
        super().__init__(model_params)
        self.clip = clip
        self.transformer = transformer
        self.scheduler = noise_scheduler
        self.time_encoding = torch.nn.Embedding(noise_scheduler.num_timesteps, clip.model.projection_dim)
        self.sinusoidal_position_embeddings = SinusoidalPositionEmbeddings(clip.model.projection_dim)
        self.output_layer = torch.nn.Linear(transformer.hidden_dim, model_params.action_dim)
        self.action_encode = torch.nn.Linear(model_params.action_dim, transformer.hidden_dim)
        self.condition_encode = torch.nn.Linear(clip.model.projection_dim, transformer.hidden_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize time encoding weights with sinusoidal position embeddings
        timesteps = torch.arange(self.time_encoding.weight.shape[0])
        with torch.no_grad():
            self.time_encoding.weight.copy_(self.sinusoidal_position_embeddings.forward(timesteps))

        # Initialize output layer weights with Xavier initialization
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, input_ids, pixel_values, attention_mask, actions, noise, past_mask, future_mask):
        # Sample random timesteps
        timesteps = torch.randint(0, self.scheduler.num_timesteps, (input_ids.shape[0],)).to(actions.device)  # [bsz]

        # Sample action to denoise
        noisy_action = self.scheduler.add_noise(actions, noise, timesteps, mask=future_mask)
        noisy_action = torch.where(future_mask.unsqueeze(-1), noisy_action, actions)
        noisy_action = self.action_encode(noisy_action)

        # Create condition embeddings
        ## Image and text embeddings
        out_clip = self.clip(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        text_embeddings = out_clip.text_embeds
        image_embeddings = out_clip.image_embeds
        ## Time embeddings
        time_embeddings = self.time_encoding(timesteps)

        # Create conditional embeddings sequence
        if image_embeddings.ndim == 2:
            image_embeddings = image_embeddings.unsqueeze(1)
        time_embeddings = time_embeddings.unsqueeze(1)
        text_embeddings = text_embeddings.unsqueeze(1)
        conditional_embeddings = torch.cat([time_embeddings, text_embeddings, image_embeddings], dim=1)
        conditional_embeddings = self.condition_encode(conditional_embeddings)

        # Create transformer input
        transformer_input = torch.cat([conditional_embeddings, noisy_action], dim=1)

        # Pass through transformer
        transformer_output = self.transformer(
            inputs_embeds=transformer_input,
            output_hidden_states=True,
            use_cache=False,
        )

        # Extract predicted direction to denoise the action
        action_seq_len = noise.shape[1]
        predicted_direction = self.output_layer(transformer_output.hidden_states[-1][:, -action_seq_len:, :])

        return predicted_direction

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids,
        pixel_values,
        actions,
        attention_mask=None,
        num_inference_steps=None,
        past_mask=None,
    ):
        """
        Generate actions using iterative denoising through the diffusion process.

        Args:
            input_ids: Text input token IDs
            pixel_values: Input images/pixel values
            actions: Input actions (past timesteps are given in the same sequence, others can be noise)
            attention_mask: Optional attention mask for text
            num_inference_steps: Number of denoising steps (defaults to scheduler.num_timesteps)
            past_mask: Optional mask indicating which actions are from past (1) vs future (0)

        Returns:
            Generated actions
        """
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_timesteps

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Create condition embeddings (same as in forward)
        out_clip = self.clip(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        text_embeddings = out_clip.text_embeds
        image_embeddings = out_clip.image_embeds

        # Initialize actions with noise (preserve past actions if past_mask provided)
        if past_mask is not None:
            original_past_actions = actions.clone() * past_mask[:, :, None].float()
            # Keep past actions, add noise to future actions
            actions = actions * past_mask[:, :, None].float() + torch.randn_like(actions) * (
                1 - past_mask[:, :, None].float()
            )
        else:
            # All actions are noise
            actions = torch.randn_like(actions)

        # Iterative denoising - similar to flow VLM approach
        step_size = max(1, self.scheduler.num_timesteps // num_inference_steps)
        text_embeddings = text_embeddings.unsqueeze(1)
        for step in range(self.scheduler.num_timesteps - 1, 0, -step_size):
            # Create time embeddings for current timestep
            timesteps = torch.tensor([step] * batch_size, device=device)
            time_embeddings = self.time_encoding(timesteps)

            # Encode current actions
            action_encoding = self.action_encode(actions)

            # Create conditional embeddings sequence
            if image_embeddings.ndim == 2:
                image_embeddings = image_embeddings.unsqueeze(1)
            time_embeddings = time_embeddings.unsqueeze(1)
            conditional_embeddings = torch.cat([time_embeddings, text_embeddings, image_embeddings], dim=1)
            conditional_embeddings = self.condition_encode(conditional_embeddings)

            # Create transformer input
            transformer_input = torch.cat([conditional_embeddings, action_encoding], dim=1)

            # Pass through transformer
            transformer_output = self.transformer(
                inputs_embeds=transformer_input,
                output_hidden_states=True,
                use_cache=False,
            )

            # Extract predicted direction to denoise the action
            action_seq_len = actions.shape[1]
            predicted_direction = self.output_layer(transformer_output.hidden_states[-1][:, -action_seq_len:, :])

            # Denoise actions using scheduler step
            predicted_actions = self.scheduler.step(predicted_direction, step, actions, step_size=step_size)

            # Preserve past actions if mask provided
            if past_mask is not None:
                # Keep original past actions, update only future actions
                past_mask_expanded = past_mask[:, :, None].to(actions.dtype)
                actions = original_past_actions + predicted_actions * (1 - past_mask_expanded)
            else:
                actions = predicted_actions

        return actions
