import torch
import torch.nn as nn

class ModalityProjector(nn.Module):
    def __init__(self, model_configs):
        super().__init__()
        self.model_configs = model_configs
        self.input_dim = model_configs.vit_hidden_dim * (model_configs.projector_pixel_shuffle_factor**2)
        self.output_dim = model_configs.hidden_dim
        self.scale_factor = model_configs.projector_pixel_shuffle_factor

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq # Sequence length must be a perfect square for pixel shuffle
        assert seq_root % self.scale_factor == 0 # Sequence root must be divisible by scale factor

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
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
        self.projection = ModalityProjector(model_configs)

    def forward(self, input_ids, image, attention_mask=None, targets=None):
        # image shape [bsz, 3, image_size, image_size]
        image_embd = self.vit(image)        
        image_embd = self.projection(image_embd)
        token_embd = self.transformer.embeddings(input_ids)

        # Adjust attention mask to account for image tokens
        if attention_mask is not None:
            # Create mask of 1s for image tokens (all image tokens should be attended to)
            batch_size = image_embd.size(0)
            img_seq_len = image_embd.size(1)
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
            
            # Combine image and token attention masks
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

        logits, _ = self.transformer(input_embeds=token_embd, attention_mask=attention_mask) # Not logits yet, but easier to return like this
        return logits, _

