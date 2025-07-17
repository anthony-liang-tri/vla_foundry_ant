import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    def __init__(self, vit_configs, output_dim):
        super().__init__()
        self.vit_configs = vit_configs
        self.input_dim = vit_configs.vit_hidden_dim * (vit_configs.projector_pixel_shuffle_factor**2)
        self.output_dim = output_dim
        self.scale_factor = vit_configs.projector_pixel_shuffle_factor

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()  # x shape [bsz, 16*16, embed_dim]
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq  # Sequence length must be a perfect square for pixel shuffle
        assert seq_root % self.scale_factor == 0  # Sequence root must be divisible by scale factor

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)  # [bsz, 16, 16, embed_dim]
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor

        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)

        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)
        return x


class VLM(nn.Module):
    def __init__(self, model_configs, transformer, vit):
        super().__init__()
        self.model_configs = model_configs
        self.vit = vit
        self.transformer = transformer
        self.projection = ModalityProjector(model_configs.vit, model_configs.transformer.hidden_dim)

    def forward(self, input_ids, image, attention_mask=None):
        # image shape [bsz, 3, image_size, image_size]
        # input_ids and attention_mask should already allot tokens for the image
        image_embd = self.vit(image)
        image_embd = self.projection(image_embd)  # [bsz, 16*16, lm_hidden_dim]
        token_embd = self.transformer.embeddings(input_ids).to(image_embd.dtype)
        special_image_mask = (input_ids == self.model_configs.image_token_id).unsqueeze(-1)
        assert special_image_mask.sum().item() == image_embd.shape[0] * image_embd.shape[1]
        special_image_mask = special_image_mask.expand_as(token_embd).to(token_embd.device)
        inputs_embeds = token_embd.masked_scatter(special_image_mask, image_embd)

        logits, _ = self.transformer(input_embeds=inputs_embeds, attention_mask=attention_mask)
        return logits, _

    def generate(self, input_ids, image, attention_mask, max_new_tokens=20):
        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        generated = input_ids.clone()
        attn_mask = attention_mask.clone()

        for _ in range(max_new_tokens):
            outputs, _ = self.forward(input_ids=generated, image=image, attention_mask=attn_mask)
            last_output = outputs[:, -1, :]
            next_token = torch.argmax(last_output, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            # Update attention mask: 1 for non-padding tokens
            next_token_mask = torch.ones_like(next_token, dtype=attn_mask.dtype)
            attn_mask = torch.cat([attn_mask, next_token_mask], dim=-1)
            # Note: You could enable the generation to break earlier than max_new_tokens when it detects a eos token,
            # but this does not work in batched generation (output tensors need to have the same size)

        return generated
