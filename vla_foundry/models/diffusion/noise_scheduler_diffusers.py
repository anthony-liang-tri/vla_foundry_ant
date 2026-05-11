import torch
from diffusers import DDPMScheduler

from vla_foundry.models.diffusion.noise_scheduler import NoiseScheduler
from vla_foundry.params.model_params import NoiseSchedulerParams


class NoiseSchedulerDDPMDiffusers(NoiseScheduler):
    def __init__(self, params: NoiseSchedulerParams):
        super().__init__(params)
        self.num_timesteps = params.num_timesteps
        if params.clamp_range is not None:
            assert -params.clamp_range[0] == params.clamp_range[1], (
                "clamp_range must be symmetric for DDPM Diffusers scheduler"
            )
            clip_params = {
                "clip_sample": True,
                "clip_sample_range": params.clamp_range[1],
            }
        else:
            clip_params = {
                "clip_sample": False,
                "clip_sample_range": None,
            }
        self.clamp_range = params.clamp_range
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.num_timesteps,
            **clip_params,
        )

    def add_noise(self, x_start, noise, timesteps, mask=None):
        normal_result = self.scheduler.add_noise(x_start, noise, timesteps)
        if mask is not None:
            mask_expanded = mask
            while mask_expanded.ndim < x_start.ndim:
                mask_expanded = mask_expanded.unsqueeze(-1)
            mask_expanded = mask_expanded.to(dtype=x_start.dtype)
            output = mask_expanded * normal_result + (1 - mask_expanded) * x_start
        else:
            output = normal_result
        if self.clamp_range is not None:
            output = output.clamp(self.clamp_range[0], self.clamp_range[1])
        return output

    def step(self, model_output, timestep, sample, step_size=1):
        if int(step_size) > 1:
            prev_t = int(timestep) - int(step_size)
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep].to(device=sample.device, dtype=sample.dtype)
            if prev_t >= 0:
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t].to(
                    device=sample.device,
                    dtype=sample.dtype,
                )
            else:
                alpha_prod_t_prev = torch.ones((), device=sample.device, dtype=sample.dtype)

            pred_original_sample = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
            output = (
                torch.sqrt(alpha_prod_t_prev) * pred_original_sample
                + torch.sqrt(1 - alpha_prod_t_prev) * model_output
            )
            if self.clamp_range is not None:
                output = output.clamp(self.clamp_range[0], self.clamp_range[1])
            return output

        output = self.scheduler.step(model_output, timestep, sample).prev_sample
        if self.clamp_range is not None:
            output = output.clamp(self.clamp_range[0], self.clamp_range[1])
        return output


class FlowMatchingScheduler(NoiseScheduler):
    def __init__(self, params: NoiseSchedulerParams):
        super().__init__(params)
        self.num_timesteps = params.num_timesteps
        self.clamp_range = params.clamp_range

    def add_noise(self, x_start, noise, timesteps, mask=None):
        """
        Add noise to the input data. If mask is provided, only add noise to the unmasked parts.
        Args:
            x_start: The input data.
            noise: The noise to add.
            timesteps: The timesteps used to scale the noise.
            mask: The mask 1: should add noise, 0: should not add noise.
        Returns:
            The noisy data.
        """
        # Determine how many dimensions to add by comparing timesteps and x_start shapes
        # timesteps is typically (batch_size,) and x_start is (batch_size, ..., feature)
        # We need to add dimensions to timesteps to match x_start's broadcasting requirements
        target_ndim = x_start.ndim
        current_ndim = timesteps.ndim
        dims_to_add = target_ndim - current_ndim

        # Create the view shape: keep first dimension (-1), add 1s for the remaining dimensions
        view_shape = [-1] + [1] * dims_to_add
        # Compute a scale in the same dtype as inputs to avoid dtype promotion (e.g., bf16 -> fp32)
        scale = timesteps.view(*view_shape).to(dtype=x_start.dtype) / self.num_timesteps
        if mask is not None:
            # Ensure mask can broadcast with x_start by expanding missing dimensions
            # mask should have first dimensions matching x_start, and we'll expand the rest
            mask_expanded = mask
            while mask_expanded.ndim < x_start.ndim:
                mask_expanded = mask_expanded.unsqueeze(-1)
            mask_expanded = mask_expanded.to(dtype=x_start.dtype)

            output = x_start + scale * (noise - x_start) * mask_expanded
        else:
            output = x_start + scale * (noise - x_start)
        if self.clamp_range is not None:
            output = output.clamp(self.clamp_range[0], self.clamp_range[1])
        return output

    def step(self, model_output, timestep, sample, step_size=1):
        output = sample - model_output * (step_size / self.num_timesteps)
        if self.clamp_range is not None:
            output = output.clamp(self.clamp_range[0], self.clamp_range[1])
        return output
