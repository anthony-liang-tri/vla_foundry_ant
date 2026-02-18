"""Diffusion Policy models and CLIP encoders."""

from vla_foundry.models.diffusion_policy.clip_hf import CLIPHF
from vla_foundry.models.diffusion_policy.clip_openclip import CLIP_OpenCLIP
from vla_foundry.models.diffusion_policy.diffusion_policy import DiffusionPolicy
from vla_foundry.models.registry import register_model
from vla_foundry.params.model_params import ModelParams


@register_model("clip_openclip")
def create_clip_openclip(model_params: ModelParams, load_pretrained: bool = True):
    return CLIP_OpenCLIP(model_params)


@register_model("clip_hf")
@register_model("clip_backbone")
def create_clip_hf(model_params: ModelParams, load_pretrained: bool = True):
    return CLIPHF(model_params, load_pretrained=load_pretrained)


@register_model("diffusion_policy")
def create_diffusion_policy(model_params: ModelParams, load_pretrained: bool = True):
    from vla_foundry.models.diffusion import create_noise_scheduler
    from vla_foundry.models.registry import create_model
    from vla_foundry.models.vision_language_backbones import get_vision_language_backbone

    vision_language_backbone = get_vision_language_backbone(model_params.vision_language_backbone, load_pretrained)
    transformer = create_model(model_params.transformer, load_pretrained)
    noise_scheduler = create_noise_scheduler(model_params)
    return DiffusionPolicy(model_params, vision_language_backbone, transformer, noise_scheduler)


__all__ = [
    "CLIPHF",
    "CLIP_OpenCLIP",
    "DiffusionPolicy",
]
