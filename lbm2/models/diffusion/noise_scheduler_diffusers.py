from diffusers import DDPMScheduler

from lbm2.params.model_params import NoiseSchedulerParams


class NoiseSchedulerDDPMDiffusers:
    def __init__(self, params: NoiseSchedulerParams):
        self.num_timesteps = params.num_timesteps
        self.scheduler = DDPMScheduler(num_train_timesteps=self.num_timesteps)

    def add_noise(self, x_start, noise, timesteps):
        return self.scheduler.add_noise(x_start, noise, timesteps)

    def step(self, model_output, timestep, sample):
        return self.scheduler.step(model_output, timestep, sample).prev_sample


class FlowMatchingScheduler:
    def __init__(self, params: NoiseSchedulerParams):
        self.num_timesteps = params.num_timesteps

    def add_noise(self, x_start, noise, timesteps):
        return x_start + timesteps.view(-1, 1, 1, 1) / self.num_timesteps * (noise - x_start)

    def step(self, model_output, timestep, sample):
        return sample - model_output / self.num_timesteps
