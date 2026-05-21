"""LeRobot-compatible Diffusion Policy.

This is a local VLA Foundry integration of the policy used by the
``lerobot/diffusion_pusht`` artifact. The architecture, normalization, DDPM
objective, and sampling path follow the LeRobot implementation pinned by that
artifact at commit ``3c0a209f9fac4d2a57617e686a7f2a2309144ba2``. The optional
flow-matching objective follows LeRobot's current MultiTask DiT objective while
keeping this PushT policy architecture fixed.
"""

import json
import math
import os
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.fsdp_block import FSDPBlock
from vla_foundry.models.registry import register_model
from vla_foundry.params.model_params import LeRobotDiffusionPolicyParams, ModelParams

_PUSHT_FALLBACK_STATS = {
    "action": {
        "min": [12.0, 25.0],
        "max": [511.0, 511.0],
    },
    "observation.image": {
        "mean": [[[0.9719857573509216]], [[0.9805848598480225]], [[0.9775370359420776]]],
        "std": [[[0.09939701110124588]], [[0.07211199402809143]], [[0.07754173874855042]]],
    },
    "observation.state": {
        "min": [13.45642375946045, 32.93829345703125],
        "max": [496.14617919921875, 510.9578857421875],
    },
}


def _load_lerobot_stats(stats_path: str | None) -> dict:
    if stats_path and os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    return _PUSHT_FALLBACK_STATS


def _as_float_tensor(stats: dict, key: str, stat_name: str) -> torch.Tensor:
    return torch.as_tensor(stats[key][stat_name], dtype=torch.float32)


def _make_noise_scheduler(name: str, **kwargs) -> DDPMScheduler | DDIMScheduler:
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    if name == "DDIM":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"Unsupported noise scheduler type {name}")


def _get_device_from_parameters(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def _get_output_shape(module: nn.Module, input_shape: tuple[int, ...]) -> tuple[int, ...]:
    was_training = module.training
    module.eval()
    with torch.inference_mode():
        output = module(torch.zeros(input_shape))
    module.train(was_training)
    return tuple(output.shape)


def _replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, key in replace_list:
        parent_module = root_module
        if parents:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(key)]
            parent_module[int(key)] = func(src_module)
        else:
            src_module = getattr(parent_module, key)
            setattr(parent_module, key, func(src_module))

    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class SpatialSoftmax(nn.Module):
    """Robomimic-style spatial soft-argmax used by LeRobot DiffusionPolicy."""

    def __init__(self, input_shape: tuple[int, int, int], num_kp: int | None = None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)

        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        return expected_xy.view(-1, self._out_c, 2)


class DiffusionRgbEncoder(nn.Module):
    """LeRobot RGB encoder: crop, ResNet18, spatial softmax, Linear+ReLU."""

    def __init__(self, config: LeRobotDiffusionPolicyParams):
        super().__init__()
        crop_shape = tuple(config.crop_shape) if config.crop_shape is not None else None
        if crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(crop_shape) if config.crop_is_random else self.center_crop
            )
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("GroupNorm replacement is incompatible with pretrained BatchNorm weights.")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        image_shape = tuple(config.image_shape)
        dummy_shape_h_w = crop_shape if crop_shape is not None else image_shape[1:]
        dummy_shape = (1, image_shape[0], *dummy_shape_h_w)
        feature_map_shape = _get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            x = self.maybe_random_crop(x) if self.training else self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        return self.relu(self.out(x))


class DiffusionSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DiffusionConv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DiffusionConditionalResidualBlock1d(FSDPBlock):
    """1D residual block with FiLM conditioning, matching LeRobot DiffusionPolicy."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        return out + self.residual_conv(x)


class DiffusionConditionalUnet1d(nn.Module):
    """LeRobot 1D Conditional U-Net with global FiLM conditioning."""

    def __init__(self, config: LeRobotDiffusionPolicyParams, global_cond_dim: int):
        super().__init__()
        self.config = config

        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        cond_dim = config.diffusion_step_embed_dim + global_cond_dim
        down_dims = list(config.down_dims)
        in_out = [(config.action_dim, down_dims[0])] + list(zip(down_dims[:-1], down_dims[1:], strict=True))

        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(down_dims[-1], down_dims[-1], **common_res_block_kwargs),
                DiffusionConditionalResidualBlock1d(down_dims[-1], down_dims[-1], **common_res_block_kwargs),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(down_dims[0], down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(down_dims[0], config.action_dim, 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond: Tensor | None = None) -> Tensor:
        x = einops.rearrange(x, "b t d -> b d t")
        timesteps_embed = self.diffusion_step_encoder(timestep)
        global_feature = (
            torch.cat([timesteps_embed, global_cond], axis=-1) if global_cond is not None else timesteps_embed
        )

        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return einops.rearrange(x, "b d t -> b t d")


class LeRobotDiffusionModel(nn.Module):
    def __init__(self, config: LeRobotDiffusionPolicyParams):
        super().__init__()
        self.config = config

        global_cond_dim = config.state_dim
        if config.num_cameras > 0:
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(config.num_cameras)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * config.num_cameras
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * config.num_cameras

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = (
            self.noise_scheduler.config.num_train_timesteps
            if config.num_inference_steps is None
            else config.num_inference_steps
        )

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        global_cond_feats = [batch["observation.state"]]

        if self.config.num_cameras > 0:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [encoder(images) for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> Tensor:
        if self.config.objective == "flow_matching":
            return self._flow_matching_sample(
                batch_size=batch_size,
                global_cond=global_cond,
                generator=generator,
                num_inference_steps=num_inference_steps,
            )

        device = _get_device_from_parameters(self)
        dtype = _get_dtype_from_parameters(self)
        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(num_inference_steps or self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            timestep = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], timestep, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _flow_matching_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> Tensor:
        device = _get_device_from_parameters(self)
        dtype = _get_dtype_from_parameters(self)
        x = torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        num_steps = num_inference_steps or self.config.num_integration_steps
        time_grid = torch.linspace(0, 1, num_steps + 1, device=device)

        if self.config.integration_method == "euler":
            return self._euler_integrate(x, time_grid, global_cond)
        if self.config.integration_method == "rk4":
            return self._rk4_integrate(x, time_grid, global_cond)
        raise ValueError(f"Unsupported integration method {self.config.integration_method}")

    def _flow_velocity(self, x: Tensor, t_scalar: float, global_cond: Tensor | None) -> Tensor:
        t_batch = torch.full((x.shape[0],), t_scalar, dtype=x.dtype, device=x.device)
        return self.unet(x, t_batch, global_cond=global_cond)

    def _euler_integrate(self, x_init: Tensor, time_grid: Tensor, global_cond: Tensor | None) -> Tensor:
        x = x_init
        for i in range(len(time_grid) - 1):
            t_scalar = time_grid[i].item()
            dt = (time_grid[i + 1] - time_grid[i]).item()
            velocity = self._flow_velocity(x, t_scalar, global_cond)
            x = x + dt * velocity
        return x

    def _rk4_integrate(self, x_init: Tensor, time_grid: Tensor, global_cond: Tensor | None) -> Tensor:
        x = x_init
        for i in range(len(time_grid) - 1):
            t = time_grid[i].item()
            dt = (time_grid[i + 1] - time_grid[i]).item()

            k1 = self._flow_velocity(x, t, global_cond)
            k2 = self._flow_velocity(x + dt * k1 / 2, t + dt / 2, global_cond)
            k3 = self._flow_velocity(x + dt * k2 / 2, t + dt / 2, global_cond)
            k4 = self._flow_velocity(x + dt * k3, t + dt, global_cond)

            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def generate_action_horizon(self, batch: dict[str, Tensor], num_inference_steps: int | None = None) -> Tensor:
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps
        global_cond = self._prepare_global_conditioning(batch)
        return self.conditional_sample(batch_size, global_cond=global_cond, num_inference_steps=num_inference_steps)

    def generate_actions(self, batch: dict[str, Tensor], num_inference_steps: int | None = None) -> Tensor:
        actions = self.generate_action_horizon(batch, num_inference_steps=num_inference_steps)
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        assert set(batch).issuperset({"observation.state", "observation.images", "action", "action_is_pad"})
        assert batch["observation.state"].shape[1] == self.config.n_obs_steps
        assert batch["action"].shape[1] == self.config.horizon

        if self.config.objective == "flow_matching":
            return self._compute_flow_matching_loss(batch)
        if self.config.objective != "diffusion":
            raise ValueError(f"Unsupported objective {self.config.objective}")

        global_cond = self._prepare_global_conditioning(batch)
        trajectory = batch["action"]
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        if self.config.do_mask_loss_for_padding:
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)
        return loss.mean()

    def _sample_flow_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        if self.config.timestep_sampling_strategy == "uniform":
            return torch.rand(batch_size, device=device)
        if self.config.timestep_sampling_strategy == "beta":
            beta_dist = torch.distributions.Beta(
                self.config.timestep_sampling_alpha,
                self.config.timestep_sampling_beta,
            )
            u = beta_dist.sample((batch_size,)).to(device)
            return self.config.timestep_sampling_s * (1.0 - u)
        raise ValueError(f"Unsupported timestep sampling strategy {self.config.timestep_sampling_strategy}")

    def _compute_flow_matching_loss(self, batch: dict[str, Tensor]) -> Tensor:
        global_cond = self._prepare_global_conditioning(batch)
        data = batch["action"]
        noise = torch.randn_like(data)
        t = self._sample_flow_timesteps(data.shape[0], data.device)
        t_expanded = t.view(-1, 1, 1)
        x_t = t_expanded * data + (1 - (1 - self.config.sigma_min) * t_expanded) * noise

        target_velocity = data - (1 - self.config.sigma_min) * noise
        predicted_velocity = self.unet(x_t, t, global_cond=global_cond)
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")

        if self.config.do_mask_loss_for_padding:
            mask = ~batch["action_is_pad"].unsqueeze(-1)
            num_valid = mask.sum() * loss.shape[-1]
            return (loss * mask).sum() / num_valid.clamp_min(1)

        return loss.mean()


class LeRobotDiffusionPolicy(BaseModel):
    def __init__(self, model_params: LeRobotDiffusionPolicyParams):
        super().__init__(model_params)
        if model_params.state_dim <= 0:
            raise ValueError("LeRobot Diffusion Policy requires proprioception/state conditioning.")
        if model_params.horizon % (2 ** len(model_params.down_dims)) != 0:
            raise ValueError("horizon must be divisible by the U-Net downsampling factor.")

        stats = _load_lerobot_stats(model_params.stats_path)
        self.register_buffer("image_mean", _as_float_tensor(stats, "observation.image", "mean"))
        self.register_buffer("image_std", _as_float_tensor(stats, "observation.image", "std"))
        self.register_buffer("state_min", _as_float_tensor(stats, "observation.state", "min"))
        self.register_buffer("state_max", _as_float_tensor(stats, "observation.state", "max"))
        self.register_buffer("action_min", _as_float_tensor(stats, "action", "min"))
        self.register_buffer("action_max", _as_float_tensor(stats, "action", "max"))

        self.diffusion = LeRobotDiffusionModel(model_params)

    def _normalize_visual(self, x: Tensor) -> Tensor:
        return (x - self.image_mean) / (self.image_std + 1e-8)

    def _normalize_state(self, x: Tensor) -> Tensor:
        x = (x - self.state_min) / (self.state_max - self.state_min + 1e-8)
        return x * 2 - 1

    def _normalize_action(self, x: Tensor) -> Tensor:
        x = (x - self.action_min) / (self.action_max - self.action_min + 1e-8)
        return x * 2 - 1

    def _unnormalize_action(self, x: Tensor) -> Tensor:
        x = (x + 1) / 2
        return x * (self.action_max - self.action_min) + self.action_min

    def _reshape_images(self, pixel_values: Tensor) -> Tensor:
        if pixel_values.ndim == 6:
            return pixel_values
        if pixel_values.ndim != 5:
            raise ValueError(f"Expected pixel_values as [B, N, C, H, W], got {tuple(pixel_values.shape)}")
        batch_size, num_images, channels, height, width = pixel_values.shape
        expected_images = self.model_params.n_obs_steps * self.model_params.num_cameras
        if num_images != expected_images:
            raise ValueError(f"Expected {expected_images} images per sample, got {num_images}")
        return pixel_values.reshape(
            batch_size,
            self.model_params.n_obs_steps,
            self.model_params.num_cameras,
            channels,
            height,
            width,
        )

    def _make_batch(
        self,
        pixel_values: Tensor,
        proprioception: Tensor,
        actions: Tensor | None = None,
        action_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        images = self._normalize_visual(self._reshape_images(pixel_values))
        state = self._normalize_state(proprioception)
        batch = {
            "observation.images": images,
            "observation.state": state,
        }
        if actions is not None:
            batch["action"] = self._normalize_action(actions)
            if action_is_pad is None:
                action_is_pad = torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)
            batch["action_is_pad"] = action_is_pad
        return batch

    def forward(
        self,
        input_ids: Tensor | None = None,
        pixel_values: Tensor | None = None,
        actions: Tensor | None = None,
        proprioception: Tensor | None = None,
        action_is_pad: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        del input_ids, kwargs
        if pixel_values is None or actions is None or proprioception is None:
            raise ValueError("pixel_values, actions, and proprioception are required for training.")
        batch = self._make_batch(pixel_values, proprioception, actions=actions, action_is_pad=action_is_pad)
        return {"loss": self.diffusion.compute_loss(batch)}

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: Tensor | None = None,
        pixel_values: Tensor | None = None,
        actions: Tensor | None = None,
        attention_mask: Tensor | None = None,
        attention_mask_images: Tensor | None = None,
        num_inference_steps: int | None = None,
        past_mask: Tensor | None = None,
        future_mask: Tensor | None = None,
        proprioception: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        del input_ids, actions, attention_mask, attention_mask_images, past_mask, future_mask, kwargs
        if pixel_values is None or proprioception is None:
            raise ValueError("pixel_values and proprioception are required for action generation.")
        batch = self._make_batch(pixel_values, proprioception)
        action_horizon = self.diffusion.generate_action_horizon(batch, num_inference_steps=num_inference_steps)
        return self._unnormalize_action(action_horizon)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        return None


@register_model("lerobot_diffusion_policy")
def create_lerobot_diffusion_policy(model_params: ModelParams, load_pretrained: bool = True):
    del load_pretrained
    return LeRobotDiffusionPolicy(model_params)
