import torch
from transformers import CLIPModel

from lbm2.models.base_model import BaseModel
from lbm2.params.model_params import CLIPHFParams


class CLIPHF(BaseModel):
    def __init__(self, model_params: CLIPHFParams):
        super().__init__(model_params)
        self.model_name = model_params.hf_pretrained
        self.model = CLIPModel.from_pretrained(self.model_name)

    def _post_init(self):
        super()._post_init()
        if self.model_params.freeze_text_encoder:
            self.freeze_text_encoder()
        if self.model_params.freeze_image_encoder:
            self.freeze_image_encoder()

    def freeze_text_encoder(self):
        self.model.text_model.requires_grad_(False)

    def freeze_image_encoder(self):
        self.model.vision_model.requires_grad_(False)

    def get_projection_dim(self):
        return self.model.projection_dim

    def forward(self, input_ids, pixel_values, attention_mask):
        if pixel_values.ndim == 5:
            # Handle multiple images per sample
            # [B, N, C, H, W] -> [B*N, C, H, W]
            num_images = pixel_values.shape[1]
            pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
            out = self.model(input_ids, pixel_values, attention_mask)
            out.image_embeds = out.image_embeds.view(input_ids.shape[0], num_images, -1, *out.image_embeds.shape[2:])
            out.text_embeds = out.text_embeds.view(input_ids.shape[0], -1, *out.text_embeds.shape[2:])
            return out
        else:
            return self.model(input_ids, pixel_values, attention_mask)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError
