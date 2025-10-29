from typing import Optional

import torch

from vla_foundry.models.model_outputs.base_output import BaseOutput


class CLIPOutput(BaseOutput):
    def __init__(
        self,
        text_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        text_model_output: Optional[torch.Tensor] = None,
        vision_model_output: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.text_embeds = text_embeds
        self.image_embeds = image_embeds
        self.text_model_output = text_model_output
        self.vision_model_output = vision_model_output
