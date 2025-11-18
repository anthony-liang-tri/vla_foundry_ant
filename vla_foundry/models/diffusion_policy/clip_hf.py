import torch
import torch.nn.functional as F
from transformers import CLIPModel

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.model_outputs.clip_output import CLIPOutput
from vla_foundry.params.model_params import CLIPHFParams


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

    def forward(self, input_ids, pixel_values, attention_mask, attention_mask_images):
        if input_ids is not None:
            text_output = self.model.text_model(input_ids).pooler_output
            text_embeds = F.normalize(text_output, dim=-1)
            text_embeds = self.model.text_projection(text_embeds)
        else:
            text_output = None
            text_embeds = None
        if pixel_values is None:
            vision_output = None
            image_embeds = None
        elif pixel_values.ndim == 5:
            # Handle multiple images per sample
            # [B, N, C, H, W] -> [B*N, C, H, W]
            num_images = pixel_values.shape[1]
            pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
            vision_output = self.model.vision_model(pixel_values).pooler_output
            image_embeds = self.model.visual_projection(vision_output)
            # [B*N, C, ...] -> [B, N, C, ...]
            vision_output = vision_output.view(input_ids.shape[0], num_images, -1, *vision_output.shape[2:])
            image_embeds = image_embeds.view(input_ids.shape[0], num_images, -1, *image_embeds.shape[2:])
            image_embeds = F.normalize(image_embeds, dim=-1)
        else:
            assert pixel_values.ndim == 4, "Pixel values must be of dimension 4 or 5 but got {pixel_values.ndim}"
            vision_output = self.model.vision_model(pixel_values).pooler_output
            image_embeds = self.model.visual_projection(vision_output)
            image_embeds = F.normalize(image_embeds, dim=-1)

        # Zero out embeddings where mask is False
        if image_embeds is not None and attention_mask_images is not None:
            # [B, N, D] * [B, N, 1] -> [B, N, D]
            image_embeds = image_embeds * attention_mask_images.unsqueeze(-1)

        return CLIPOutput(
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_output,
            vision_model_output=vision_output,
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError
