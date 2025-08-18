import torch
from transformers import AutoModelForCausalLM

from lbm2.models.base_model import BaseModel
from lbm2.params.model_params import TransformerHFParams


class TransformerHF(BaseModel):
    def __init__(self, model_params: TransformerHFParams):
        super().__init__(model_params)
        self.model_name = model_params.hf_pretrained
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, use_cache=False, attention_mask=None):
        assert input_ids is not None and inputs_embeds is None
        out = self.model(input_ids, past_key_values=past_key_values, use_cache=use_cache, attention_mask=attention_mask)
        return out.logits, out.past_key_values

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError
