import yaml
import torch.nn as nn
from pathlib import Path
from models.transformer import Transformer, TransformerBlock
from models.transformer_hf import TransformerHF
from models.vit import ViT
from models.vit_hf import ViTHF
from models.vlm import VLM
from models.vlm_hf import VLMHF


def create_model(model_configs):
    if "_hf" not in model_configs.model_type:
        current_dir = Path(__file__).parent
        with open(f'{current_dir}/../config_presets/models/{model_configs.model}.yaml', 'r') as f:
            model_configs_from_file = yaml.safe_load(f)
            for key, value in model_configs_from_file.items():
                if hasattr(model_configs, key):
                    object.__setattr__(model_configs, key, value)       # Bypass Frozen=True
                else:
                    raise AttributeError(f"Attribute '{key}' not found in {type(model_configs).__name__}")
    else:
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_configs.model)
        vocab_size = getattr(hf_config, 'vocab_size', getattr(getattr(hf_config, 'text_config', None), 'vocab_size', None))
        if vocab_size is not None:
            object.__setattr__(model_configs, 'vocab_size', vocab_size)
        else:
            raise AttributeError("Could not find vocab_size in hf_config")

    if model_configs.model_type == "transformer":
        model = Transformer(model_configs)
    elif model_configs.model_type == "transformer_hf":
        model = TransformerHF(model_configs)
    elif model_configs.model_type == "vlm":
        transformer = Transformer(model_configs)
        if model_configs.vit_pretrained is not None:
            vit = ViTHF(model_configs)
        else:
            vit = ViT(model_configs)
        model = VLM(model_configs, transformer, vit)
    elif model_configs.model_type == "vlm_hf":
        model = VLMHF(model_configs)
    else:
        raise ValueError(f"{model_configs.model_type} not supported!")
    return model


def get_model_block(model_type, model_name):
    if model_type == "transformer":
        return {TransformerBlock} 
    elif model_type == "transformer_hf":
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)
        for name, module in model.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                return {type(module[0])}
        raise ValueError("Could not find model block class.")
    elif model_type == "vlm":
        return {TransformerBlock}
    elif model_type == "vlm_hf":
        from transformers import AutoConfig, AutoModelForVision2Seq
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_config(config)
        for attr in ["language_model", "text_model"]:
            if hasattr(model.model, attr):
                for name, module in getattr(model.model, attr).named_modules():
                    if isinstance(module, nn.ModuleList) and len(module) > 0:
                        return {type(module[0])}
        raise ValueError("Could not find model block class.")
    else:
        raise ValueError(f"get_model_block (used for FSDP) not supported for {model_type}")