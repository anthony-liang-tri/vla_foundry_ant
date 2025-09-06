from typing import Optional, Tuple, Union

import torch

from lbm2.models.model_outputs.base_output import BaseOutput


class TransformerOutput(BaseOutput):
    """
    Custom output class that mimics transformers.modeling_outputs.CausalLMOutputWithPast
    but doesn't depend on HuggingFace transformers library.

    This allows us to provide HF-compatible API without requiring HF dependencies
    for non-HF models.
    """

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        hidden_states: Optional[Union[Tuple[torch.Tensor], list]] = None,
        attentions: Optional[Tuple[torch.Tensor]] = None,
        loss: Optional[torch.Tensor] = None,
    ):
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.loss = loss
