import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq

class VLMHF(nn.Module):
    def __init__(self, model_configs):
        super().__init__()
        self.model_configs = model_configs
        self.model_name = model_configs.model
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)

    def forward(self, input_ids, image, attention_mask=None):
        out = self.model(input_ids=input_ids, pixel_values=image, attention_mask=attention_mask)
        return out.logits, out.past_key_values

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError