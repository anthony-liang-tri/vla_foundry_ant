from typing import Union

from diffusers import DDPMScheduler

from vla_foundry.models.diffusion.noise_scheduler import NoiseScheduler
from vla_foundry.params.model_params import NoiseSchedulerParams


class NoiseSchedulerDDPMDiffusers(NoiseScheduler):
    def __init__(self, params: NoiseSchedulerParams):
        super(NoiseSchedulerDDPMDiffusers, self).__init__(params)
        self.num_timesteps = params.num_timesteps
        self.scheduler = DDPMScheduler(num_train_timesteps=self.num_timesteps)

    def add_noise(self, x_start, noise, timesteps):
        return self.scheduler.add_noise(x_start, noise, timesteps)

    def step(self, model_output, timestep, sample):
        return self.scheduler.step(model_output, timestep, sample).prev_sample


class FlowMatchingScheduler:
    def __init__(self, params: Union[int, NoiseSchedulerParams]):
        if isinstance(params, int):
            # This class only uses one parameter so here we also allow an int to be passed in
            self.num_timesteps = params
        else:
            self.num_timesteps = params.num_timesteps

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

            output = x_start + scale * (noise - x_start) * mask_expanded
            return output
        else:
            return x_start + scale * (noise - x_start)

    def step(self, model_output, timestep, sample, step_size=1):
        return sample - model_output * (step_size / self.num_timesteps)
