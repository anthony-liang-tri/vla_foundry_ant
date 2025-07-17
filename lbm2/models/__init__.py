from pathlib import Path

import torch.nn as nn
import yaml

from lbm2.models.diffusion.noise_scheduler import NoiseSchedulerDDPM
from lbm2.models.diffusion.noise_scheduler_diffusers import FlowMatchingScheduler, NoiseSchedulerDDPMDiffusers
from lbm2.models.diffusion.stable_diffusion import StableDiffusion
from lbm2.models.diffusion.unet import CrossAttentionBlock, ResnetBlock, SelfAttentionBlock, UNet
from lbm2.models.diffusion.unet_diffusers import UNetDiffusers
from lbm2.models.transformer import Transformer, TransformerBlock
from lbm2.models.transformer_hf import TransformerHF
from lbm2.models.vit import ViT
from lbm2.models.vit_hf import ViTHF
from lbm2.models.vlm import VLM
from lbm2.models.vlm_hf import VLMHF


def create_model(model_configs):
    if model_configs.type == "transformer":
        model = Transformer(model_configs)
    elif model_configs.type == "transformer_hf":
        model = TransformerHF(model_configs)
    elif model_configs.type == "vlm":
        transformer = Transformer(model_configs.transformer)
        vit = ViT(model_configs.vit) if model_configs.vit.type == "vit" else ViTHF(model_configs.vit)
        if model_configs.vit_freeze:
            for param in vit.parameters():
                param.requires_grad = False
        model = VLM(model_configs, transformer, vit)
    elif model_configs.type == "vlm_hf":
        model = VLMHF(model_configs)
    elif model_configs.type == "stable_diffusion":
        unet = UNetDiffusers(model_configs) if model_configs.use_diffusers_unet else UNet(model_configs)
        if model_configs.use_diffusers_scheduler:
            noise_scheduler = NoiseSchedulerDDPMDiffusers(model_configs)
        elif model_configs.use_flow_matching_scheduler:
            noise_scheduler = FlowMatchingScheduler(model_configs)
        else:
            noise_scheduler = NoiseSchedulerDDPM(model_configs)
        model = StableDiffusion(model_configs, noise_scheduler, unet)
    else:
        raise ValueError(f"{model_configs.type} not supported!")
    return model


def get_model_block(model_type, model_configs):
    if model_type == "transformer":
        return {TransformerBlock}
    elif model_type == "transformer_hf":
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(model_configs.hf_pretrained)
        model = AutoModelForCausalLM.from_config(config)
        for _name, module in model.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                return {type(module[0])}
        raise ValueError("Could not find model block class.")
    elif model_type == "vlm":
        return {TransformerBlock}
    elif model_type == "vlm_hf":
        from transformers import AutoConfig, AutoModelForVision2Seq

        config = AutoConfig.from_pretrained(model_configs.hf_pretrained)
        model = AutoModelForVision2Seq.from_config(config)
        for attr in ["language_model", "text_model"]:
            if hasattr(model.model, attr):
                for _name, module in getattr(model.model, attr).named_modules():
                    if isinstance(module, nn.ModuleList) and len(module) > 0:
                        return {type(module[0])}
        raise ValueError("Could not find model block class.")
    elif model_type == "stable_diffusion":
        if model_configs.use_diffusers_unet:
            from diffusers.models.unets.unet_2d_blocks import (
                AttnDownBlock2D,
                AttnUpBlock2D,
                DownBlock2D,
                UNetMidBlock2D,
                UpBlock2D,
            )

            return {
                DownBlock2D,
                UpBlock2D,
                UNetMidBlock2D,
                AttnUpBlock2D,
                AttnDownBlock2D,
            }
        else:
            return {ResnetBlock, SelfAttentionBlock, CrossAttentionBlock}
    else:
        raise ValueError(f"get_model_block (used for FSDP) not supported for {model_type}")
