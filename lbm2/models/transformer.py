import math

import torch
from torch import nn

from lbm2.activations import get_feed_forward
from lbm2.attention import get_attn_func
from lbm2.models.base_model import BaseModel
from lbm2.norms import get_norm_class
from lbm2.params.model_params import TransformerParams
from lbm2.positional_embedding import get_pos_embed


class CustomAttn(nn.Module):
    def __init__(self, layer_id: int, model_params: TransformerParams):
        super().__init__()
        self.n_heads = model_params.n_heads
        self.hidden_dim = model_params.hidden_dim
        self.head_dim = model_params.hidden_dim // model_params.n_heads
        self.in_proj = nn.Linear(self.hidden_dim, 3 * self.n_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_dim, bias=False)
        self.pos_embed = get_pos_embed(model_params)
        self.attn_fn = get_attn_func(model_params.attn_name)
        self.apply_qk_norm = model_params.qk_norm

        # initialize norm layers for queries and keys if needed
        self.norm_type = get_norm_class(model_params.norm_type)
        self.q_norm = (
            self.norm_type(
                self.n_heads * self.head_dim,
                eps=model_params.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            self.norm_type(
                self.n_heads * self.head_dim,
                eps=model_params.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )

        self.layer_id = layer_id
        self.reset_parameters()

    def reset_parameters(self):
        # initialize weights by trunc_normal(1/sqrt(fan_in))
        std = 1.0 / math.sqrt(self.hidden_dim)
        torch.nn.init.trunc_normal_(self.in_proj.weight, std=std, a=-3 * std, b=3 * std)
        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = std / math.sqrt(2 * (self.layer_id + 1))
        torch.nn.init.trunc_normal_(self.out_proj.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor, is_causal=True, past_key_value=None, use_cache=False, attention_mask=None):
        batchsize, seq_len, hidden_dim = x.shape
        queries, keys, vals = self.in_proj(x).chunk(3, dim=-1)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = queries.view(batchsize, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(batchsize, seq_len, self.n_heads, self.head_dim)
        vals = vals.view(batchsize, seq_len, self.n_heads, self.head_dim)

        past_length = 0 if past_key_value is None else past_key_value[0].shape[1]
        queries, keys, vals = self.pos_embed(queries, keys, vals, offset=past_length)

        if past_key_value is not None and use_cache:
            keys = torch.cat([past_key_value[0], keys], dim=1)
            vals = torch.cat([past_key_value[1], vals], dim=1)

        if use_cache:
            past_key_value = [keys, vals]

        output = self.attn_fn(
            queries,
            keys,
            vals,
            is_causal=is_causal,
            attention_mask=attention_mask,
        )
        output = output.view(batchsize, seq_len, -1)
        return self.out_proj(output), past_key_value


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_params: TransformerParams):
        super().__init__()
        self.n_heads = model_params.n_heads
        self.hidden_dim = model_params.hidden_dim

        self.head_dim = model_params.hidden_dim // model_params.n_heads
        self.attention = CustomAttn(layer_id, model_params)
        self.ffn_type = model_params.ffn_type
        self.feed_forward, self.ffn_hidden_dim = get_feed_forward(self.ffn_type, self.hidden_dim)

        self.layer_id = layer_id
        self.norm_type = get_norm_class(model_params.norm_type)
        self.attention_norm = self.norm_type(
            model_params.hidden_dim,
            eps=model_params.norm_eps,
        )
        self.ffn_norm = self.norm_type(
            model_params.hidden_dim,
            eps=model_params.norm_eps,
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.ffn_type == "swiglu":
            # initialize weights trunc_normal(1/sqrt(fan_in))
            std = 1.0 / math.sqrt(self.hidden_dim)
            torch.nn.init.trunc_normal_(self.feed_forward.w12.weight, std=std, a=-3 * std, b=3 * std)
            # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
            std = 1.0 / math.sqrt(self.hidden_dim)
            std = std / math.sqrt(2 * (self.layer_id + 1))
            torch.nn.init.trunc_normal_(self.feed_forward.w3.weight, std=std, a=-3 * std, b=3 * std)
        elif self.ffn_type == "gelu":
            std = 1.0 / math.sqrt(self.hidden_dim)
            torch.nn.init.trunc_normal_(self.feed_forward[0].weight, std=std, a=-3 * std, b=3 * std)

            std = 1.0 / math.sqrt(self.hidden_dim)
            std = std / math.sqrt(2 * (self.layer_id + 1))
            torch.nn.init.trunc_normal_(self.feed_forward[2].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x, past_key_value=None, use_cache=False, attention_mask=None):
        h, past_key_value = self.attention(
            self.attention_norm(x),
            is_causal=True,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        h = x + h
        ffn_out = self.feed_forward(self.ffn_norm(h))
        out = h + ffn_out
        return out, past_key_value


class Transformer(BaseModel):
    def __init__(self, model_params: TransformerParams):
        super().__init__(model_params)
        # for convenience we often share param names with llama
        self.hidden_dim = model_params.hidden_dim
        self.vocab_size = model_params.vocab_size
        self.n_layers = model_params.n_layers
        self.max_seq_len = model_params.max_seq_len
        self.norm_type = get_norm_class(model_params.norm_type)
        self.post_embed_norm = (
            self.norm_type(
                model_params.hidden_dim,
                eps=model_params.norm_eps,
            )
            if model_params.post_embed_norm
            else nn.Identity()
        )
        self.weight_tying = model_params.weight_tying
        self.embeddings = nn.Embedding(model_params.vocab_size, model_params.hidden_dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(model_params.n_layers):
            self.layers.append(TransformerBlock(layer_id, model_params))

        # get class for normalization layers
        self.norm = self.norm_type(
            model_params.hidden_dim,
            eps=model_params.norm_eps,
        )
        self.output = nn.Linear(model_params.hidden_dim, model_params.vocab_size, bias=False)
        if self.weight_tying:
            self.embeddings.weight = self.output.weight
        self.grad_checkpointing = False
        self.reset_parameters()

    def reset_parameters(self):
        # initialize weight 1/sqrt(dim)
        # this is 1/fan_in for output, as is default, and Maciej Kilian tried another option
        # for the embed layer (from RWKV paper) but this was better.
        std = 1.0 / math.sqrt(self.hidden_dim)
        torch.nn.init.trunc_normal_(self.output.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.embeddings.weight, std=std, a=-3 * std, b=3 * std)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(
        self,
        input_ids=None,
        input_embeds=None,
        past_key_values=None,
        use_cache=False,
        attention_mask=None,
        output_hidden_states=False,
    ):
        """
        Args:
            input
            past_key_values
            use_cache (bool)
            attention_mask (torch.Tensor): Shape (batch_size, sequence_len), indicates tokens that should not be
                attended to. attention_mask[s, i] = False indicates that token i should not be attended to by any other
                token for sequence s.
            output_hidden_states (bool): Whether to return the hidden states of the transformer.
        """
        if input_ids is not None:
            x = self.embeddings(input_ids)
        elif input_embeds is not None:
            x = input_embeds
        else:
            raise ValueError("Either input_ids or input_embeds must be provided.")

        x = self.post_embed_norm(x)

        if past_key_values is None:
            past_key_values = [None] * self.n_layers
        elif isinstance(past_key_values, tuple):
            past_key_values = list(past_key_values)
        hidden_states = []
        for i, layer in enumerate(self.layers):
            if self.grad_checkpointing:
                x, past_key_values[i] = torch.utils.checkpoint.checkpoint(
                    layer, x, past_key_values[i], use_cache, attention_mask
                )
            else:
                x, past_key_values[i] = layer(x, past_key_values[i], use_cache=use_cache, attention_mask=attention_mask)
            if output_hidden_states:
                hidden_states.append(x)
        if past_key_values[0] is None:
            past_key_values = None
        x = self.norm(x)
        output = self.output(x)
        # follow llama in casting this to float.
        return output.float(), past_key_values, (hidden_states if output_hidden_states else None)

    def generate(self, input_ids, attention_mask, max_new_tokens=20):
        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        generated = input_ids.clone()
        attn_mask = attention_mask.clone()

        for _ in range(max_new_tokens):
            outputs, _, _ = self.forward(input_ids=generated, attention_mask=attn_mask)
            last_output = outputs[:, -1, :]
            next_token = torch.argmax(last_output, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            # Update attention mask: 1 for non-padding tokens
            next_token_mask = torch.ones_like(next_token, dtype=attn_mask.dtype)
            attn_mask = torch.cat([attn_mask, next_token_mask], dim=-1)
            # Note: You could enable the generation to break earlier than max_new_tokens when it detects a eos token,
            # but this does not work in batched generation (output tensors need to have the same size)

        return generated
