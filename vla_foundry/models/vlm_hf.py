import torch
from transformers import AutoConfig, AutoModelForVision2Seq

from vla_foundry.models.transformer_base import TransformerBase
from vla_foundry.models.utils import get_hidden_dim_hf, get_num_hidden_layers_hf
from vla_foundry.params.model_params import VLMHFParams


class VLMHF(TransformerBase):
    def __init__(self, model_params: VLMHFParams, load_pretrained: bool = True):
        super().__init__(model_params)
        self.model_name = model_params.hf_pretrained
        if load_pretrained:
            self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)
        else:
            config = AutoConfig.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_config(config)
        self._limit_hidden_states_to_last_n = None

    def forward(self, input_ids, pixel_values, attention_mask=None, output_hidden_states=False, **kwargs):
        out = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values.to(dtype=torch.bfloat16),
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        if self._limit_hidden_states_to_last_n is not None and output_hidden_states:
            out.hidden_states = out.hidden_states[-self._limit_hidden_states_to_last_n :]

        return out

    def resize_token_embeddings(self, token_id: int = None) -> int:
        """Ensure the token embedding matrix can index the provided token.

        If token_id is None, attempt to read it from the registry for this model name.
        This should be called during model setup, not inside the forward loop.
        """
        if token_id is None:
            token_id = int(self.model.get_input_embeddings().num_embeddings) + 1

        if token_id > self.model.get_input_embeddings().num_embeddings:
            print(f"Resizing token embeddings from {self.model.get_input_embeddings().num_embeddings} to {token_id}")
            self.model.resize_token_embeddings(token_id, mean_resizing=False)
        return token_id

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            if enable:
                self.model.gradient_checkpointing_enable()
            else:
                self.model.gradient_checkpointing_disable()

    @property
    def hidden_dim(self) -> int:
        return get_hidden_dim_hf(self.model.config)

    @property
    def num_hidden_layers(self) -> int:
        return get_num_hidden_layers_hf(self.model.config)

    def generate(self, input_ids, pixel_values, attention_mask, max_new_tokens=20):
        """Generate text tokens using the VLM HF model"""
        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        generated = input_ids.clone()
        attn_mask = attention_mask.clone()

        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids=generated, pixel_values=pixel_values, attention_mask=attn_mask)
            last_output = outputs.logits[:, -1, :]
            next_token = torch.argmax(last_output, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            # Update attention mask: 1 for non-padding tokens
            next_token_mask = torch.ones_like(next_token, dtype=attn_mask.dtype)
            attn_mask = torch.cat([attn_mask, next_token_mask], dim=-1)

        return generated
