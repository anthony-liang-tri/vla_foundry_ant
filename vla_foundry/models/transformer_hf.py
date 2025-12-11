import torch
from transformers import AutoConfig, AutoModelForCausalLM

from vla_foundry.models.transformer_base import TransformerBase
from vla_foundry.models.utils import get_hidden_dim_hf, get_num_hidden_layers_hf
from vla_foundry.params.model_params import TransformerHFParams


class TransformerHF(TransformerBase):
    def __init__(self, model_params: TransformerHFParams, load_pretrained: bool = True):
        super().__init__(model_params)
        self.model_name = model_params.hf_pretrained
        if load_pretrained:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        else:
            config = AutoConfig.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_config(config)

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        return out

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError

    @property
    def hidden_dim(self) -> int:
        return get_hidden_dim_hf(self.model.config)

    @property
    def num_hidden_layers(self) -> int:
        return get_num_hidden_layers_hf(self.model.config)

    def resize_token_embeddings(self, token_id: int = None) -> int:
        """Extend the embedding vocabulary of the underlying LM."""
        embed = self.model.resize_token_embeddings(token_id)
        if token_id is None:
            return embed.num_embeddings
        else:
            return token_id
