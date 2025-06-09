import torch
import torch.nn as nn
import timm

class ViTHF(nn.Module):
    def __init__(self, model_configs):
        super().__init__()
        self.model_configs = model_configs
        self.model_name = model_configs.vit_pretrained
        self.model = timm.create_model(self.model_name, num_classes=0, pretrained=True)

    def forward(self, image):
        image_embeddings = self.model.forward_intermediates(image)[0]
        return image_embeddings

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError

