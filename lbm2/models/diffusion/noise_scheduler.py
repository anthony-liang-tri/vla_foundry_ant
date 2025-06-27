import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseSchedulerDDPM(nn.Module):
    def __init__(self, model_configs):
        super().__init__()
        self.model_configs = model_configs
        self.num_timesteps = model_configs.diffusion_noise_scheduler_num_timesteps
        self.beta_start = model_configs.diffusion_noise_scheduler_beta_start
        self.beta_end = model_configs.diffusion_noise_scheduler_beta_end
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer("posterior_variance", betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

    def add_noise(self, x_start, noise, timesteps):
        # x_start, noise shape [bsz, channels, h, w]
        # self.sqrt_alphas_cumprod[timesteps] shape [bsz]
        return (
            self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1) * x_start +
            self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1) * noise
        )   # [bsz, channels, h, w]

    def step(self, model_output, timestep, sample):
        """Reverse process single step"""
        t = timestep
       
        # Get coefficients
        alpha_t = self.alphas[t]  # scalar
        alpha_cumprod_t = self.alphas_cumprod[t]  # scalar
        alpha_cumprod_t_prev = self.alphas_cumprod_prev[t]  # scalar
        beta_t = self.betas[t]  # scalar
       
        # Compute predicted original sample
        pred_original_sample = (sample - torch.sqrt(1 - alpha_cumprod_t) * model_output) / torch.sqrt(alpha_cumprod_t)  # [batch_size, channels, height, width]
       
        # Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = torch.sqrt(alpha_cumprod_t_prev) * beta_t / (1 - alpha_cumprod_t)  # scalar
        current_sample_coeff = torch.sqrt(alpha_t) * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)  # scalar
       
        # Compute predicted previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample  # [batch_size, channels, height, width]
       
        # Add noise if not the last timestep
        if t > 0:
            noise = torch.randn_like(sample)  # [batch_size, channels, height, width]
            variance = torch.sqrt(self.posterior_variance[t]) * noise  # [batch_size, channels, height, width]
            pred_prev_sample = pred_prev_sample + variance  # [batch_size, channels, height, width]
       
        return pred_prev_sample  # [batch_size, channels, height, width]
