from positional_embedding.head_rotary import HeadRotaryWithCast
from positional_embedding.rotary import RotaryWithCast
from positional_embedding.llama_rotary import LLaMARotaryWithCast
from positional_embedding.none import identity_with_cast


def get_pos_embed(model_configs):
    head_dim = model_configs.hidden_dim // model_configs.n_heads
    if model_configs.positional_embedding_type == "rotary":
        return RotaryWithCast(head_dim, model_configs.seq_len)
    elif model_configs.positional_embedding_type == "llama_rotary":
        return LLaMARotaryWithCast(head_dim, model_configs.n_heads, model_configs.seq_len)
    elif model_configs.positional_embedding_type == "head_rotary":
        return HeadRotaryWithCast(head_dim, model_configs.seq_len)
    elif model_configs.positional_embedding_type == "none":
        return identity_with_cast
    else:
        raise RuntimeError(f"Unknown positional embedding type {model_configs.positional_embedding_type}")
