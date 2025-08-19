import torch
from transformers import AutoModelForVision2Seq

from lbm2.models.base_model import BaseModel
from lbm2.models.utils import get_hidden_dim_hf, get_hidden_states_hf, get_num_hidden_layers_hf
from lbm2.params.model_params import VLMHFParams


class VLMHF(BaseModel):
    def __init__(self, model_params: VLMHFParams):
        super().__init__(model_params)
        self.model_name = model_params.hf_pretrained
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)
        self._limit_hidden_states_to_last_n = None

    def forward(self, input_ids, image, attention_mask=None, output_hidden_states=False):
        out = self.model(
            input_ids=input_ids,
            pixel_values=image.to(dtype=torch.bfloat16),
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if output_hidden_states:
            # Try to pull text hidden states from common attributes
            hidden_states = get_hidden_states_hf(out)

            # Normalize to list of tensors [layers][-1 dims]
            if hidden_states is None and hasattr(out, "last_hidden_state"):
                last = out.last_hidden_state
                hidden_states = [last for _ in range(self.num_hidden_layers)]
            elif isinstance(hidden_states, tuple):
                hidden_states = list(hidden_states)

            # Optionally keep only the last N layers to reduce memory
            if isinstance(hidden_states, list) and self._limit_hidden_states_to_last_n is not None:
                n = self._limit_hidden_states_to_last_n
                hidden_states = hidden_states[-n:] if n > 0 else []

            # Ensure hidden_states is a list before iterating
            if hidden_states is None:
                hidden_states = []
            elif not isinstance(hidden_states, list):
                hidden_states = [hidden_states]

            # Convert to image dtype if we have hidden states
            if hidden_states:
                hidden_states = [h.to(dtype=image.dtype) for h in hidden_states]

        return out.logits.to(dtype=image.dtype), out.past_key_values, (hidden_states if output_hidden_states else None)

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

    def set_num_action_layers(self, num_layers: int):
        self._limit_hidden_states_to_last_n = num_layers

    def generate(self, input_ids, image, attention_mask, max_new_tokens=20):
        """Generate text tokens using the VLM HF model"""
        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        generated = input_ids.clone()
        attn_mask = attention_mask.clone()

        for _ in range(max_new_tokens):
            outputs, _, _ = self.forward(input_ids=generated, image=image, attention_mask=attn_mask)
            last_output = outputs[:, -1, :]
            next_token = torch.argmax(last_output, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            # Update attention mask: 1 for non-padding tokens
            next_token_mask = torch.ones_like(next_token, dtype=attn_mask.dtype)
            attn_mask = torch.cat([attn_mask, next_token_mask], dim=-1)

        return generated
