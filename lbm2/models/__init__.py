import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

from lbm2.models.batch_handlers import create_batch_handler
from lbm2.models.diffusion.noise_scheduler import NoiseSchedulerDDPM
from lbm2.models.diffusion.noise_scheduler_diffusers import FlowMatchingScheduler, NoiseSchedulerDDPMDiffusers
from lbm2.models.diffusion.stable_diffusion import StableDiffusion
from lbm2.models.diffusion.unet import CrossAttentionBlock, ResnetBlock, SelfAttentionBlock, UNet
from lbm2.models.diffusion.unet_diffusers import UNetDiffusers
from lbm2.models.diffusion_policy.clip_hf import CLIPHF
from lbm2.models.diffusion_policy.clip_openclip import CLIP_OpenCLIP
from lbm2.models.diffusion_policy.diffusion_policy import DiffusionPolicy
from lbm2.models.transformer import Transformer, TransformerBlock
from lbm2.models.transformer_hf import TransformerHF
from lbm2.models.vit import ViT
from lbm2.models.vit_hf import ViTHF
from lbm2.models.vlm import VLM
from lbm2.models.vlm_hf import VLMHF
from lbm2.params.model_params import ModelParams


def create_noise_scheduler(model_params: ModelParams):
    if model_params.use_diffusers_scheduler:
        noise_scheduler = NoiseSchedulerDDPMDiffusers(model_params.noise_scheduler)
    elif model_params.use_flow_matching_scheduler:
        noise_scheduler = FlowMatchingScheduler(model_params.noise_scheduler)
    else:
        noise_scheduler = NoiseSchedulerDDPM(model_params.noise_scheduler)
    return noise_scheduler


def create_model(model_params: ModelParams):
    if model_params.type == "transformer":
        model = Transformer(model_params)
    elif model_params.type == "transformer_hf":
        model = TransformerHF(model_params)
    elif model_params.type == "vlm":
        transformer = (
            Transformer(model_params.transformer)
            if model_params.transformer.type == "transformer"
            else TransformerHF(model_params.transformer)
        )
        vit = ViT(model_params.vit) if model_params.vit.type == "vit" else ViTHF(model_params.vit)
        model = VLM(model_params, transformer, vit)
    elif model_params.type == "vlm_hf":
        model = VLMHF(model_params)
    elif model_params.type == "stable_diffusion":
        clip = create_model(model_params.clip) if model_params.clip.hf_pretrained is not None else None
        unet = UNetDiffusers(model_params.unet) if model_params.use_diffusers_unet else UNet(model_params.unet)
        noise_scheduler = create_noise_scheduler(model_params)
        model = StableDiffusion(model_params, clip, unet, noise_scheduler)
    elif model_params.type == "clip_openclip":
        model = CLIP_OpenCLIP(model_params)
    elif model_params.type == "clip_hf":
        model = CLIPHF(model_params)
    elif model_params.type == "diffusion_policy":
        clip = create_model(model_params.clip)
        transformer = create_model(model_params.transformer)
        noise_scheduler = create_noise_scheduler(model_params)
        model = DiffusionPolicy(model_params, clip, transformer, noise_scheduler)
    else:
        raise ValueError(f"{model_params.type} not supported!")
    return model


def get_model_block(model_type: str, model_params: ModelParams):
    if model_type == "transformer":
        return (TransformerBlock,)
    elif model_type == "transformer_hf":
        config = AutoConfig.from_pretrained(model_params.hf_pretrained)
        model = AutoModelForCausalLM.from_config(config)
        for _name, module in model.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                return (type(module[0]),)
        raise ValueError("Could not find model block class.")
    elif model_type == "vlm":
        return (TransformerBlock,)
    elif model_type == "vlm_hf":
        config = AutoConfig.from_pretrained(model_params.hf_pretrained)
        model = AutoModelForVision2Seq.from_config(config)

        block_types = set()

        # Find text/language model blocks
        for attr in ["language_model", "text_model"]:
            if hasattr(model.model, attr):
                for _name, module in getattr(model.model, attr).named_modules():
                    if isinstance(module, nn.ModuleList) and len(module) > 0:
                        block_types.add(type(module[0]))

        # Find vision model blocks
        if hasattr(model.model, "vision_model") and hasattr(model.model.vision_model, "encoder"):
            for _name, module in model.model.vision_model.encoder.named_modules():
                if isinstance(module, nn.ModuleList) and len(module) > 0:
                    block_types.add(type(module[0]))

        if not block_types:
            raise ValueError("Could not find any model block classes.")

        return tuple(block_types)
    elif model_type == "stable_diffusion":
        if model_params.use_diffusers_unet:
            from diffusers.models.unets.unet_2d_blocks import (
                AttnDownBlock2D,
                AttnUpBlock2D,
                DownBlock2D,
                UNetMidBlock2D,
                UpBlock2D,
            )

            return (
                DownBlock2D,
                UpBlock2D,
                UNetMidBlock2D,
                AttnUpBlock2D,
                AttnDownBlock2D,
            )
        else:
            return (ResnetBlock, SelfAttentionBlock, CrossAttentionBlock)
    elif model_type == "clip_hf":
        from transformers.models.clip.modeling_clip import CLIPEncoderLayer

        return (CLIPEncoderLayer,)
    elif model_type == "clip_openclip":
        import open_clip

        return (open_clip.transformer.ResidualAttentionBlock,)
    elif model_type == "diffusion_policy":
        transformer_block = get_model_block(model_params.transformer.type, model_params.transformer)
        clip_block = get_model_block(model_params.clip.type, model_params.clip)
        return (*transformer_block, *clip_block)
    else:
        raise ValueError(f"get_model_block (used for FSDP) not supported for {model_type}")
