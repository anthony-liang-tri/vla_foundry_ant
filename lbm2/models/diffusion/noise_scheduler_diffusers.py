from diffusers import DDPMScheduler


class NoiseSchedulerDDPMDiffusers:
    def __init__(self, model_configs):
        self.model_configs = model_configs
        self.num_timesteps = model_configs.noise_scheduler.num_timesteps
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_timesteps)

    def add_noise(self, x_start, noise, timesteps):
        return self.noise_scheduler.add_noise(x_start, noise, timesteps)

    def step(self, model_output, timestep, sample):
        return self.noise_scheduler.step(model_output, timestep, sample).prev_sample


class FlowMatchingScheduler:
    def __init__(self, model_configs):
        self.num_timesteps = model_configs.noise_scheduler.num_timesteps

    def add_noise(self, x_start, noise, timesteps):
        return x_start + timesteps.view(-1, 1, 1, 1) / self.num_timesteps * (noise - x_start)

    def step(self, model_output, timestep, sample):
        return sample - model_output / self.num_timesteps
