from dataclasses import dataclass, fields
from typing import List


def add_diffusion_params(parser):
    parser.add_argument(
        "--diffusion-use-diffusers-unet",
        action="store_true",
    )
    parser.add_argument(
        "--diffusion-use-diffusers-scheduler",
        action="store_true",
    )
    parser.add_argument(
        "--diffusion-use-flow-matching-scheduler",
        action="store_true",
    )
    parser.add_argument(
        "--diffusion-noise-scheduler-num-timesteps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--diffusion-noise-scheduler-beta-start",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--diffusion-noise-scheduler-beta-end",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--diffusion-unet-in-channels",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--diffusion-unet-out-channels",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--diffusion-unet-time-emb-dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--diffusion-unet-text-emb-dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--diffusion-unet-channels",
        type=str,
        default="128,256,512,1024",
    )

@dataclass(frozen=True)
class DiffusionParams:
    diffusion_use_diffusers_unet: bool
    diffusion_use_diffusers_scheduler: bool
    diffusion_use_flow_matching_scheduler: bool
    diffusion_noise_scheduler_num_timesteps: int
    diffusion_noise_scheduler_beta_start: int
    diffusion_noise_scheduler_beta_end: int
    diffusion_unet_in_channels: int
    diffusion_unet_out_channels: int
    diffusion_unet_time_emb_dim: int
    diffusion_unet_text_emb_dim: int
    diffusion_unet_channels: List[int]

    @classmethod
    def from_args(cls, args):
        init_kwargs = {
            f.name: getattr(args, f.name)
            for f in fields(cls)
            if hasattr(args, f.name)
        }
        return cls(**init_kwargs)