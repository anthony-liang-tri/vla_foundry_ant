from typing import Union

import torch

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.diffusion.noise_scheduler import NoiseScheduler
from vla_foundry.models.diffusion.unet import SinusoidalPositionEmbeddings
from vla_foundry.models.diffusion_policy.clip_hf import CLIPHF
from vla_foundry.models.transformer import Transformer
from vla_foundry.models.transformer_hf import TransformerHF
from vla_foundry.params.model_params import DiffusionPolicyParams


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
        self.proprioception_dim = model_params.proprioception_dim
        self.time_encoding = torch.nn.Embedding(noise_scheduler.num_timesteps, clip.get_projection_dim())
        self.sinusoidal_position_embeddings = SinusoidalPositionEmbeddings(clip.get_projection_dim())
        self.output_layer = torch.nn.Linear(transformer.hidden_dim, model_params.action_dim)
        self.action_encode = torch.nn.Linear(model_params.action_dim, transformer.hidden_dim)
        self.condition_encode = torch.nn.Linear(clip.get_projection_dim(), transformer.hidden_dim)
        self.proprioception_encode = (
            torch.nn.Linear(self.proprioception_dim, transformer.hidden_dim) if self.proprioception_dim > 0 else None
        )
        self.input_noise_std = model_params.input_noise_std
        self.disable_text = model_params.disable_text
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize time encoding weights with sinusoidal position embeddings
        timesteps = torch.arange(self.time_encoding.weight.shape[0])
        with torch.no_grad():
            self.time_encoding.weight.copy_(self.sinusoidal_position_embeddings.forward(timesteps))

        # Initialize output layer weights with Xavier initialization
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        if self.proprioception_encode is not None:
            torch.nn.init.xavier_uniform_(self.proprioception_encode.weight)

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        attention_mask_images,
        actions,
        noise,
        past_mask,
        future_mask,
        proprioception=None,
    ):
        # Sample random timesteps
        timesteps = torch.randint(0, self.scheduler.num_timesteps, (actions.shape[0],)).to(actions.device)  # [bsz]

        # Sample action to denoise
        noisy_action = self.scheduler.add_noise(actions, noise, timesteps, mask=future_mask)
        noisy_action = torch.where(future_mask.unsqueeze(-1), noisy_action, actions)
        if self.input_noise_std > 0:
            noisy_action = noisy_action + torch.randn_like(noisy_action) * self.input_noise_std
        noisy_action = self.action_encode(noisy_action)

        # Create condition embeddings
        ## Image and text embeddings
        out_clip = self.clip(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
        )
        text_embeddings = out_clip.text_embeds
        image_embeddings = out_clip.image_embeds
        ## Time embeddings (B, 1, D)
        time_embeddings = self.time_encoding(timesteps).unsqueeze(1)
        conditional_embeddings = [time_embeddings]

        # Create conditional embeddings sequence
        if not self.disable_text and text_embeddings is not None:
            # (B, D) -> (B, 1, D)
            text_embeddings = text_embeddings.unsqueeze(1)
            conditional_embeddings.append(text_embeddings)
        if image_embeddings is not None:
            # if multiple images per sample, (B, N, D) else (B, D) -> (B, 1, D)
            if image_embeddings.ndim == 2:
                image_embeddings = image_embeddings.unsqueeze(1)
            conditional_embeddings.append(image_embeddings)
        # (B, 1 + 1 + N, D)
        conditional_embeddings = torch.cat(conditional_embeddings, dim=1)
        # [B, 1 + 1 + N, C] -> [B, 1 + 1 + N, transformer.hidden_dim]
        conditional_embeddings = self.condition_encode(conditional_embeddings)

        # Create transformer input (B, 1 + 1 + N + T, D)
        transformer_input_parts = [conditional_embeddings]
        if self.proprioception_encode is not None and proprioception is not None:
            proprio_embeddings = self.proprioception_encode(proprioception)
            if self.input_noise_std > 0:
                proprio_embeddings = proprio_embeddings + torch.randn_like(proprio_embeddings) * self.input_noise_std
            transformer_input_parts.append(proprio_embeddings)
        transformer_input_parts.append(noisy_action)
        transformer_input = torch.cat(transformer_input_parts, dim=1)

        # Pass through transformer
        transformer_output = self.transformer(
            inputs_embeds=transformer_input,
            output_hidden_states=True,
            use_cache=False,
        )

        # Extract predicted direction to denoise the action (B, 1 + 1 + N + T, D) -> (B, T, D)
        action_seq_len = noise.shape[1]
        # [B, 1 + 1 + N + action_seq_len, transformer.hidden_dim] -> [B, action_seq_len, transformer.hidden_dim]
        predicted_direction = self.output_layer(transformer_output.hidden_states[-1][:, -action_seq_len:, :])

        return predicted_direction

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids,
        pixel_values,
        actions,
        attention_mask=None,
        attention_mask_images=None,
        num_inference_steps=None,
        past_mask=None,
        proprioception=None,
        **kwargs,  # Ignore extra params like point_cloud (used by other models)
    ):
        """
        Generate actions using iterative denoising through the diffusion process.

        Args:
            input_ids: Text input token IDs
            pixel_values: Input images/pixel values
            actions: Input actions (past timesteps are given in the same sequence, others can be noise)
            attention_mask: Optional attention mask for text
            attention_mask_images: Optional attention mask for camera images
            num_inference_steps: Number of denoising steps (defaults to scheduler.num_timesteps)
            past_mask: Optional mask indicating which actions are from past (1) vs future (0)

        Returns:
            Generated actions
        """
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_timesteps

        batch_size = actions.shape[0]
        device = actions.device

        # Create condition embeddings (same as in forward)
        out_clip = self.clip(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
        )
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
        if not self.disable_text and text_embeddings is not None:
            # (B, D) -> (B, 1, D)
            text_embeddings = text_embeddings.unsqueeze(1)
        if image_embeddings is not None and image_embeddings.ndim == 2:
            # if multiple images per sample, (B, N, D) else (B, D) -> (B, 1, D)
            image_embeddings = image_embeddings.unsqueeze(1)
        proprio_embeddings = None
        if self.proprioception_encode is not None and proprioception is not None:
            proprio_embeddings = self.proprioception_encode(proprioception)
        for step in range(self.scheduler.num_timesteps - 1, 0, -step_size):
            # Create time embeddings for current timestep (B, 1, D)
            timesteps = torch.tensor([step] * batch_size, device=device)
            time_embeddings = self.time_encoding(timesteps).unsqueeze(1)
            conditional_embeddings = [time_embeddings]

            if not self.disable_text and text_embeddings is not None:
                conditional_embeddings.append(text_embeddings)
            if image_embeddings is not None:
                conditional_embeddings.append(image_embeddings)

            # Create conditional embeddings sequence (B, 1 + 1 + N, D)
            conditional_embeddings = torch.cat(conditional_embeddings, dim=1)
            conditional_embeddings = self.condition_encode(conditional_embeddings)

            # Encode current actions
            action_encoding = self.action_encode(actions)

            # Create transformer input (B, 1 + 1 + N + T, D)
            transformer_input_parts = [conditional_embeddings]
            if proprio_embeddings is not None:
                transformer_input_parts.append(proprio_embeddings)
            transformer_input_parts.append(action_encoding)
            transformer_input = torch.cat(transformer_input_parts, dim=1)

            # Pass through transformer
            transformer_output = self.transformer(
                inputs_embeds=transformer_input,
                output_hidden_states=True,
                use_cache=False,
            )

            # Extract predicted direction to denoise the action (B, 1 + 1 + N + T, D) -> (B, T, D)
            action_seq_len = actions.shape[1]
            predicted_direction = self.output_layer(transformer_output.hidden_states[-1][:, -action_seq_len:, :])

            # Denoise actions using scheduler step
            predicted_actions = self.scheduler.step(predicted_direction, step, actions, step_size=step_size)

            # Preserve past actions if mask provided
            if past_mask is not None:
                # Keep original past actions, update only future actions
                # (B, T) -> (B, T, 1)
                past_mask_expanded = past_mask[:, :, None].to(actions.dtype)
                actions = original_past_actions + predicted_actions * (1 - past_mask_expanded)
            else:
                actions = predicted_actions

        return actions
