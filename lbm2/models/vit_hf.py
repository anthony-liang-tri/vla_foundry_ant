import timm
import torch

from lbm2.models.base_model import BaseModel
from lbm2.params.model_params import ViTHFParams


class ViTHF(BaseModel):
    def __init__(self, model_params: ViTHFParams):
        super().__init__(model_params)
        self.model_name = model_params.hf_pretrained
        self.model = timm.create_model(self.model_name, num_classes=0, pretrained=True)

    def forward(self, image):
        image_embeddings = self.model.forward_intermediates(image)[0]
        return image_embeddings

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError
