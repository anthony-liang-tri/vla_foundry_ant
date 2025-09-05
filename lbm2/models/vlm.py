import torch
import torch.nn as nn
from einops import rearrange

from lbm2.models.transformer_base import TransformerBase
from lbm2.params.model_params import ViTParams, VLMParams


class ModalityProjector(nn.Module):
    def __init__(self, vit_params: ViTParams, output_dim: int):
        super().__init__()
        self.input_dim = vit_params.hidden_dim * (vit_params.projector_pixel_shuffle_factor**2)
        self.output_dim = output_dim
        self.scale_factor = vit_params.projector_pixel_shuffle_factor

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # equivalent to:
    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x):
        if x.ndim == 4:
            bsz, cams, seq, embed_dim = x.size()
        else:
            cams = 0
            bsz, seq, embed_dim = x.size()  # x shape [bsz, 16*16, embed_dim]
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq, (
            f"seq_root**2 = {seq_root**2}, seq = {seq}"
        )  # Sequence length must be a perfect square for pixel shuffle
        assert seq_root % self.scale_factor == 0, (
            f"seq_root % self.scale_factor = {seq_root % self.scale_factor}, self.scale_factor = {self.scale_factor}"
        )  # Sequence root must be divisible by scale factor

        # Verified equivalent to original_pixel_shuffle implementation (see tests/models/test_pixelshuffle.py)
        if cams == 0:
            x = rearrange(
                x,
                "n (w w_scale h h_scale) c -> n (w h) (w_scale h_scale c)",
                w_scale=self.scale_factor,
                h_scale=self.scale_factor,
                w=seq_root // self.scale_factor,
                h=seq_root // self.scale_factor,
            )
        else:
            x = rearrange(
                x,
                "n cams (w w_scale h h_scale) c -> n (cams w h) (w_scale h_scale c)",
                w_scale=self.scale_factor,
                h_scale=self.scale_factor,
                w=seq_root // self.scale_factor,
                h=seq_root // self.scale_factor,
            )
        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)
        return x


class VLM(TransformerBase):
    def __init__(self, model_params: VLMParams, transformer, vit):
        super().__init__(model_params)
        self.vit = vit
        self.transformer = transformer
        self.projection = ModalityProjector(model_params.vit, model_params.transformer.hidden_dim)

    def forward(self, input_ids, image, attention_mask=None, output_hidden_states=False, use_cache=False, **kwargs):
        # image shape [bsz, 3, image_size, image_size]
        # input_ids and attention_mask should already allot tokens for the image
        image_embd = self.vit(image)
        image_embd = self.projection(image_embd)  # [bsz, 16*16, lm_hidden_dim]
        token_embd = self.transformer.embeddings(input_ids).to(image_embd.dtype)
        special_image_mask = (input_ids == self.model_params.image_token_id).unsqueeze(-1)
        assert special_image_mask.sum().item() == image_embd.shape[0] * image_embd.shape[1]
        special_image_mask = special_image_mask.expand_as(token_embd).to(token_embd.device)
        inputs_embeds = token_embd.masked_scatter(special_image_mask, image_embd)

        # Call transformer's forward method directly to get logits and past_key_values and hidden_states
        logits, past_key_values, hidden_states = self.transformer(
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            **kwargs,
        )
        return logits, past_key_values, hidden_states

    def set_grad_checkpointing(self, enable: bool = True):
        """Optional: enable gradient checkpointing on the underlying LM if supported."""
        self.transformer.set_grad_checkpointing(enable)
        self.vit.set_grad_checkpointing(enable)

    def resize_token_embeddings(self, token_id: int = None) -> int:
        """Extend the embedding vocabulary of the underlying LM."""
        return self.transformer.resize_token_embeddings(token_id)

    @property
    def hidden_dim(self) -> int:
        return self.model_params.transformer.hidden_dim

    @property
    def num_hidden_layers(self) -> int:
        return self.model_params.transformer.n_layers

    def generate(
        self,
        input_ids: torch.Tensor,
        image: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 20,
    ) -> torch.Tensor:
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
            # Note: You could enable the generation to break earlier than max_new_tokens when it detects a eos token,
            # but this does not work in batched generation (output tensors need to have the same size)

        return generated
