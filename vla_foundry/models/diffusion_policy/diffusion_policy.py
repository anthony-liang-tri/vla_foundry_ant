import torch

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.diffusion.noise_scheduler import NoiseScheduler
from vla_foundry.models.diffusion.unet import SinusoidalPositionEmbeddings
from vla_foundry.models.transformer import Transformer
from vla_foundry.models.transformer_hf import TransformerHF
from vla_foundry.models.vision_language_backbones import BaseBackboneWrapper
from vla_foundry.params.model_params import DiffusionPolicyParams


class DiffusionPolicy(BaseModel):
    def __init__(
        self,
        model_params: DiffusionPolicyParams,
        vision_language_backbone: BaseBackboneWrapper,
        transformer: Transformer | TransformerHF,
        noise_scheduler: NoiseScheduler,
    ):
        super().__init__(model_params)
        self.vision_language_backbone = vision_language_backbone
        self.transformer = transformer
        self.scheduler = noise_scheduler
        self.proprioception_dim = model_params.proprioception_dim

        backbone_dim = vision_language_backbone.get_conditioning_embeddings_dim()
        self.time_encoding = torch.nn.Embedding(noise_scheduler.num_timesteps, backbone_dim)
        self.sinusoidal_position_embeddings = SinusoidalPositionEmbeddings(backbone_dim)
        self.output_layer = torch.nn.Linear(transformer.hidden_dim, model_params.action_dim)
        self.action_encode = torch.nn.Linear(model_params.action_dim, transformer.hidden_dim)
        self.condition_encode = torch.nn.Linear(backbone_dim, transformer.hidden_dim)
        self.proprioception_encode = (
            torch.nn.Linear(self.proprioception_dim, transformer.hidden_dim) if self.proprioception_dim > 0 else None
        )

        self.diffusion_step_conditioning = model_params.diffusion_step_conditioning
        self.input_noise_std = model_params.input_noise_std
        self.num_action_head_repeats = model_params.num_action_head_repeats
        self.use_flow_matching_scheduler = (
            model_params.use_flow_matching_scheduler and not model_params.use_diffusers_scheduler
        )
        self.action_denoising_mask = model_params.action_denoising_mask
        self.flow_matching_target = model_params.flow_matching_target
        self.flow_matching_timestep_sampling = model_params.flow_matching_timestep_sampling
        self.flow_matching_time_embedding = model_params.flow_matching_time_embedding
        self.flow_matching_sigma_min = model_params.flow_matching_sigma_min
        self.flow_matching_beta_s = model_params.flow_matching_beta_s
        self.flow_matching_beta_alpha = model_params.flow_matching_beta_alpha
        self.flow_matching_beta_beta = model_params.flow_matching_beta_beta
        if self.flow_matching_time_embedding == "continuous":
            self.continuous_time_mlp = torch.nn.Sequential(
                torch.nn.Linear(backbone_dim, 2 * backbone_dim),
                torch.nn.GELU(),
                torch.nn.Linear(2 * backbone_dim, backbone_dim),
                torch.nn.GELU(),
            )
        else:
            self.continuous_time_mlp = None
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

    def _sample_timesteps(self, batch_size, device):
        if not self.use_flow_matching_scheduler or self.flow_matching_timestep_sampling == "uniform_discrete":
            timesteps = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=device)
            tau = timesteps.to(dtype=torch.float32) / self.scheduler.num_timesteps
            return timesteps, tau

        if self.flow_matching_timestep_sampling == "uniform":
            tau = torch.rand(batch_size, device=device)
        elif self.flow_matching_timestep_sampling == "beta":
            beta_dist = torch.distributions.Beta(
                torch.tensor(self.flow_matching_beta_alpha, device=device),
                torch.tensor(self.flow_matching_beta_beta, device=device),
            )
            tau = self.flow_matching_beta_s * (1.0 - beta_dist.sample((batch_size,)))
            tau = tau.clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unknown flow_matching_timestep_sampling: {self.flow_matching_timestep_sampling}")

        if self.flow_matching_time_embedding == "continuous":
            return tau, tau

        discrete_timesteps = (tau * (self.scheduler.num_timesteps - 1)).round().long()
        return discrete_timesteps, tau

    def _time_embeddings(self, timesteps):
        if torch.is_floating_point(timesteps):
            time_embeddings = self.sinusoidal_position_embeddings(timesteps)
            if self.continuous_time_mlp is not None:
                time_embeddings = self.continuous_time_mlp(time_embeddings)
            return time_embeddings.unsqueeze(1)
        return self.time_encoding(timesteps).unsqueeze(1)

    def _effective_denoise_mask(self, actions, past_mask=None, future_mask=None):
        if self.action_denoising_mask == "future":
            return future_mask
        if self.action_denoising_mask == "valid":
            if past_mask is None and future_mask is None:
                return None
            if past_mask is None:
                return future_mask
            if future_mask is None:
                return past_mask
            return past_mask | future_mask
        if self.action_denoising_mask == "all":
            return torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        raise ValueError(f"Unknown action_denoising_mask: {self.action_denoising_mask}")

    def _add_noise(self, actions, noise, time_condition, tau, denoise_mask):
        if self.use_flow_matching_scheduler and self.flow_matching_target == "data_minus_noise":
            tau = tau.to(device=actions.device, dtype=actions.dtype)
            tau_expanded = tau.view([-1] + [1] * (actions.ndim - 1))
            noisy_action = tau_expanded * actions + (
                1 - (1 - self.flow_matching_sigma_min) * tau_expanded
            ) * noise
            if denoise_mask is not None:
                mask = denoise_mask
                while mask.ndim < actions.ndim:
                    mask = mask.unsqueeze(-1)
                noisy_action = torch.where(mask, noisy_action, actions)
            return noisy_action

        scheduler_timesteps = time_condition
        if torch.is_floating_point(scheduler_timesteps):
            scheduler_timesteps = tau.to(device=actions.device, dtype=actions.dtype) * self.scheduler.num_timesteps
        noisy_action = self.scheduler.add_noise(actions, noise, scheduler_timesteps, mask=denoise_mask)
        if denoise_mask is not None:
            noisy_action = torch.where(denoise_mask.unsqueeze(-1), noisy_action, actions)
        return noisy_action

    def _build_transformer_input(self, backbone_embeddings, time_embeddings, noisy_action, proprio_embeddings=None):
        """Build transformer input by combining conditioning, time, and action embeddings.

        Supports two time conditioning strategies:
        - CONCAT: Prepend time as a separate token [time, backbone] → [B, 1+N, D]
        - ADD: Add time to backbone embeddings element-wise → [B, N, D]

        Args:
            backbone_embeddings: [B, N, backbone_dim] from vision-language backbone
            time_embeddings: [B, 1, backbone_dim] from time encoding
            noisy_action: [B, T, transformer_dim] encoded noisy actions
            proprio_embeddings: Optional [B, P, transformer_dim] encoded proprioception

        Returns:
            transformer_input: [B, C+P+T, transformer_dim]
        """
        if self.diffusion_step_conditioning == "add":
            conditional_embeddings = backbone_embeddings + time_embeddings
        elif self.diffusion_step_conditioning == "concat":
            conditional_embeddings = torch.cat([time_embeddings, backbone_embeddings], dim=1)
        else:
            raise ValueError(f"Unknown diffusion_step_conditioning: {self.diffusion_step_conditioning}")

        conditional_embeddings = self.condition_encode(conditional_embeddings)

        parts = [conditional_embeddings]
        if proprio_embeddings is not None:
            parts.append(proprio_embeddings)
        parts.append(noisy_action)

        return torch.cat(parts, dim=1)

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
        **kwargs,
    ):
        # Sample random timesteps
        timesteps, tau = self._sample_timesteps(actions.shape[0], actions.device)

        # Sample action to denoise
        denoise_mask = self._effective_denoise_mask(actions, past_mask=past_mask, future_mask=future_mask)
        noisy_action = self._add_noise(actions, noise, timesteps, tau, denoise_mask)
        if self.input_noise_std > 0:
            noisy_action = noisy_action + torch.randn_like(noisy_action) * self.input_noise_std
        noisy_action = self.action_encode(noisy_action)

        # Get backbone embeddings (handles text+image concatenation)
        backbone_output = self.vision_language_backbone.get_action_conditioning(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
            **kwargs,
        )

        backbone_embeddings = backbone_output.embeddings
        num_repeats = self.num_action_head_repeats
        if num_repeats is not None and num_repeats > 1:
            # Verify action-side inputs were tiled to [B*N] by the batch handler
            vlm_batch_size = input_ids.shape[0]
            assert actions.shape[0] == vlm_batch_size * num_repeats, (
                f"Expected actions batch size {vlm_batch_size * num_repeats} (vlm_batch={vlm_batch_size} * "
                f"num_repeats={num_repeats}), got {actions.shape[0]}"
            )
            assert noise.shape[0] == vlm_batch_size * num_repeats, (
                f"Expected noise batch size {vlm_batch_size * num_repeats}, got {noise.shape[0]}"
            )
            assert future_mask.shape[0] == vlm_batch_size * num_repeats, (
                f"Expected future_mask batch size {vlm_batch_size * num_repeats}, got {future_mask.shape[0]}"
            )
            if proprioception is not None:
                assert proprioception.shape[0] == vlm_batch_size * num_repeats, (
                    f"Expected proprioception batch size {vlm_batch_size * num_repeats}, got {proprioception.shape[0]}"
                )
            # Tile backbone embeddings to match the action batch size [B*N]
            backbone_embeddings = backbone_embeddings.repeat_interleave(num_repeats, dim=0)

        # Time embeddings (batch, 1, backbone_dim) — batch is [B*N] when repeating, else [B]
        time_embeddings = self._time_embeddings(timesteps)

        # Proprioception embeddings (already tiled to [B*N] by the batch handler when num_repeats > 1)
        proprio_embeddings = None
        if self.proprioception_encode is not None and proprioception is not None:
            proprio_embeddings = self.proprioception_encode(proprioception)
            if self.input_noise_std > 0:
                proprio_embeddings = proprio_embeddings + torch.randn_like(proprio_embeddings) * self.input_noise_std

        # Build transformer input using time conditioning strategy
        transformer_input = self._build_transformer_input(
            backbone_embeddings=backbone_embeddings,
            time_embeddings=time_embeddings,
            noisy_action=noisy_action,
            proprio_embeddings=proprio_embeddings,
        )

        # Pass through transformer
        transformer_output = self.transformer(
            inputs_embeds=transformer_input,
            output_hidden_states=True,
            use_cache=False,
        )

        # Extract predicted direction to denoise the action (B, 1+N+P+T, D) -> (B, T, D)
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
        attention_mask_images=None,
        num_inference_steps=None,
        past_mask=None,
        future_mask=None,
        proprioception=None,
        guidance_target: torch.Tensor | None = None,
        guidance_scale: float = 0.0,
        guidance_mask: torch.Tensor | None = None,
        sigma_d_obs: float = 0.2,
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
            future_mask: Optional mask indicating which action slots should be denoised
            proprioception: Optional proprioception input
            guidance_target: Target actions for Pi-GDM guidance (previous chunk predictions)
            guidance_scale: Multiplier for β = guidance_scale * n (1.0 = recommended default)
            guidance_mask: Per-timestep mask weighting the guidance residual
            sigma_d_obs: Observation noise std for Pi-GDM (default 0.2)
            **kwargs: Model-specific args

        Returns:
            Generated actions
        """
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_timesteps
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be > 0, got {num_inference_steps}")

        use_guidance = guidance_scale > 0 and guidance_target is not None
        # β = guidance_scale * n (num_inference_steps) as in the RTC paper.
        # This gives critically-damped guidance: β·dt = guidance_scale ≈ 1.
        beta = guidance_scale * num_inference_steps

        batch_size = actions.shape[0]
        device = actions.device

        # Precompute backbone embeddings (reused across all denoising steps)
        backbone_output = self.vision_language_backbone.get_action_conditioning(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
            **kwargs,
        )

        # Initialize only denoised slots with noise. Training uses the same
        # effective mask, so generation should preserve the same slots.
        denoise_mask = self._effective_denoise_mask(actions, past_mask=past_mask, future_mask=future_mask)
        preserve_mask = None if denoise_mask is None else ~denoise_mask.to(device=device, dtype=torch.bool)

        if preserve_mask is not None:
            preserve_mask_expanded = preserve_mask[:, :, None].to(actions.dtype)
            original_preserved_actions = actions.clone() * preserve_mask_expanded
            actions = actions * preserve_mask_expanded + torch.randn_like(actions) * (1 - preserve_mask_expanded)
        else:
            # All actions are noise
            actions = torch.randn_like(actions)
            original_preserved_actions = None

        # Precompute proprioception embedding
        proprio_embeddings = None
        if self.proprioception_encode is not None and proprioception is not None:
            proprio_embeddings = self.proprioception_encode(proprioception)

        if self.use_flow_matching_scheduler and self.flow_matching_target == "data_minus_noise":
            if use_guidance:
                raise ValueError("Pi-GDM guidance is not implemented for data_minus_noise flow matching.")
            dt = 1.0 / num_inference_steps
            for step_idx in range(num_inference_steps):
                tau = step_idx / num_inference_steps
                if self.flow_matching_time_embedding == "continuous":
                    timesteps = torch.full((batch_size,), tau, device=device, dtype=actions.dtype)
                else:
                    discrete_t = round(tau * (self.scheduler.num_timesteps - 1))
                    timesteps = torch.full((batch_size,), discrete_t, device=device, dtype=torch.long)
                time_embeddings = self._time_embeddings(timesteps)

                action_encoding = self.action_encode(actions)
                transformer_input = self._build_transformer_input(
                    backbone_embeddings=backbone_output.embeddings,
                    time_embeddings=time_embeddings,
                    noisy_action=action_encoding,
                    proprio_embeddings=proprio_embeddings,
                )
                transformer_output = self.transformer(
                    inputs_embeds=transformer_input,
                    output_hidden_states=True,
                    use_cache=False,
                )
                action_seq_len = actions.shape[1]
                predicted_direction = self.output_layer(
                    transformer_output.hidden_states[-1][:, -action_seq_len:, :]
                )
                predicted_actions = actions + predicted_direction * dt
                clamp_range = getattr(self.scheduler, "clamp_range", None)
                if clamp_range is not None:
                    predicted_actions = predicted_actions.clamp(
                        clamp_range[0],
                        clamp_range[1],
                    )
                if preserve_mask is not None:
                    actions = original_preserved_actions + predicted_actions * (1 - preserve_mask_expanded)
                else:
                    actions = predicted_actions
            return actions

        # Iterative denoising loop. Diffusers schedulers own their inference
        # timestep spacing; the local DDPM scheduler uses the historical stride.
        if hasattr(self.scheduler, "get_inference_timesteps"):
            inference_timesteps = self.scheduler.get_inference_timesteps(num_inference_steps, device=device)
            step_size = 1
        else:
            step_size = max(1, self.scheduler.num_timesteps // num_inference_steps)
            inference_timesteps = list(range(self.scheduler.num_timesteps - 1, -1, -step_size))
            if inference_timesteps[-1] != 0:
                inference_timesteps.append(0)

        for step in inference_timesteps:
            step_int = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
            # Create timesteps for current step
            timesteps = torch.full((batch_size,), step_int, device=device, dtype=torch.long)
            time_embeddings = self._time_embeddings(timesteps)

            # Encode current actions
            action_encoding = self.action_encode(actions)

            # Build transformer input using time conditioning strategy
            transformer_input = self._build_transformer_input(
                backbone_embeddings=backbone_output.embeddings,
                time_embeddings=time_embeddings,
                noisy_action=action_encoding,
                proprio_embeddings=proprio_embeddings,
            )

            # Pass through transformer
            transformer_output = self.transformer(
                inputs_embeds=transformer_input,
                output_hidden_states=True,
                use_cache=False,
            )

            # Extract predicted direction to denoise the action
            action_seq_len = actions.shape[1]
            predicted_direction = self.output_layer(transformer_output.hidden_states[-1][:, -action_seq_len:, :])

            # Pi-GDM guidance (replace approximation, J ≈ I).
            # Weight = min(β, raw_weight) with β = guidance_scale (fixed).
            # Fixed β gives step-count invariance (see generate_actions docstring).
            if use_guidance:
                tau = step_int / self.scheduler.num_timesteps
                x0_hat = actions - tau * predicted_direction
                residual = guidance_target - x0_hat
                if guidance_mask is not None:
                    if guidance_mask.dim() == 1:
                        guidance_mask = guidance_mask.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
                    elif guidance_mask.dim() == 2:
                        guidance_mask = guidance_mask.unsqueeze(0)  # [1, T, D]
                    residual = residual * guidance_mask
                sigma_sq = sigma_d_obs**2
                denom = (1 - tau) * tau * sigma_sq + 1e-8
                raw_weight = (tau**2 + sigma_sq * (1 - tau) ** 2) / denom
                adaptive_scale = min(beta, raw_weight)
                predicted_direction = predicted_direction - adaptive_scale * residual

            # Denoise actions using scheduler step
            predicted_actions = self.scheduler.step(predicted_direction, step, actions, step_size=step_size)

            # Preserve non-denoised slots if a mask provided them.
            if preserve_mask is not None:
                actions = original_preserved_actions + predicted_actions * (1 - preserve_mask_expanded)
            else:
                actions = predicted_actions

        return actions
