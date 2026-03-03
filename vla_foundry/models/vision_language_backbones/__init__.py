from vla_foundry.models.vision_language_backbones.base import BaseBackboneWrapper
from vla_foundry.models.vision_language_backbones.clip_hf_backbone import CLIPBackboneWrapper
from vla_foundry.models.vision_language_backbones.vlm_hf_backbone import VLMHFBackboneWrapper
from vla_foundry.params.model_params import CLIPBackboneParams, VLMBackboneParams


def get_vision_language_backbone(backbone_params, load_pretrained: bool = True):
    if isinstance(backbone_params, CLIPBackboneParams):
        return CLIPBackboneWrapper(backbone_params, load_pretrained)
    elif isinstance(backbone_params, VLMBackboneParams):
        return VLMHFBackboneWrapper(backbone_params, load_pretrained)
    else:
        raise ValueError(f"Unsupported vision language backbone type: {backbone_params.type}")
