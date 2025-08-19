import torch
from transformers import AutoModelForCausalLM

from lbm2.models.base_model import BaseModel
from lbm2.models.utils import get_hidden_dim_hf, get_num_hidden_layers_hf
from lbm2.params.model_params import TransformerHFParams


class TransformerHF(BaseModel):
    def __init__(self, model_params: TransformerHFParams):
        super().__init__(model_params)
        self.model_name = model_params.hf_pretrained
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def forward(
        self,
        *args,
        output_hidden_states=False,
        **kwargs,
    ):
        out = self.model(
            *args,
            **kwargs,
        )
        return out.logits, out.past_key_values, (out.hidden_states if output_hidden_states else None)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError

    @property
    def hidden_dim(self) -> int:
        return get_hidden_dim_hf(self.model.config)

    @property
    def num_hidden_layers(self) -> int:
        return get_num_hidden_layers_hf(self.model.config)
