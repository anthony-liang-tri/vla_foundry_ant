"""ViT backbone wrapper for action policy conditioning.

Wraps a standalone ViT (no text encoder) and provides patch embeddings
as conditioning tokens for the diffusion transformer.
"""

import einops
import torch

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.model_outputs.backbone_output import VisionLanguageBackboneOutput
from vla_foundry.models.vision_language_backbones.base import BaseBackboneWrapper
from vla_foundry.models.vit import ViT
from vla_foundry.params.model_params import ViTBackboneParams


class ViTBackboneWrapper(BaseBackboneWrapper):
    """Wraps a ViT and provides patch embeddings for action policy conditioning."""

    def __init__(self, backbone_params: ViTBackboneParams, load_pretrained: bool = True):
        # Skip BaseBackboneWrapper.__init__ which calls create_model with the wrong type.
        # Instead, initialize BaseModel directly and create the ViT ourselves.
        BaseModel.__init__(self, backbone_params)
        self._model = ViT(backbone_params)

    def get_conditioning_embeddings_dim(self) -> int:
        return self._model.model_params.hidden_dim

    def get_action_conditioning(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ) -> VisionLanguageBackboneOutput:
        # pixel_values: [B, N, C, H, W] where N = number of camera images
        # ViT.forward handles 5D input via einops rearrange
        embeddings = self._model(pixel_values)  # [B, N, num_patches, hidden_dim]
        # Flatten camera and patch dims into a single sequence
        if embeddings.ndim == 4:
            embeddings = einops.rearrange(embeddings, "b n t d -> b (n t) d")
        return VisionLanguageBackboneOutput(embeddings=embeddings)

    def _extract_action_conditioning(self, outputs) -> VisionLanguageBackboneOutput:
        # Not used — we override get_action_conditioning directly
        raise NotImplementedError
