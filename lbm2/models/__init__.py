import yaml
import torch.nn as nn
from pathlib import Path
from models.transformer import Transformer, TransformerBlock
from models.transformer_hf import TransformerHF
from models.vit import ViT
from models.vlm import VLM


def create_model(model_configs):
    if model_configs.model_type != "transformer_hf":
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
        object.__setattr__(model_configs, 'vocab_size', hf_config.vocab_size)

    if model_configs.model_type == "transformer":
        model = Transformer(model_configs)
    elif model_configs.model_type == "transformer_hf":
        model = TransformerHF(model_configs)
    elif model_configs.model_type == "vlm":
        transformer = Transformer(model_configs)
        vit = ViT(model_configs)
        model = VLM(model_configs, transformer, vit)
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
    else:
        raise ValueError(f"get_model_block (used for FSDP) not supported for {model_type}")