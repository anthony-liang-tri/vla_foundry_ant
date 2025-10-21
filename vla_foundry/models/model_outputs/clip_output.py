from typing import Optional

import torch

from vla_foundry.models.model_outputs.base_output import BaseOutput


class CLIPOutput(BaseOutput):
    """
    Custom output class for CLIP models.
    """

    def __init__(
        self,
        image_embeds: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
    ):
        self.image_embeds = image_embeds
        self.text_embeds = text_embeds
