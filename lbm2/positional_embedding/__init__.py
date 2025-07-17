from lbm2.positional_embedding.none import identity_with_cast
from lbm2.positional_embedding.rotary import RotaryWithCast


def get_pos_embed(model_configs):
    head_dim = model_configs.hidden_dim // model_configs.n_heads
    if model_configs.positional_embedding_type == "rotary":
        return RotaryWithCast(head_dim, model_configs.max_seq_len)
    elif model_configs.positional_embedding_type == "none":
        return identity_with_cast
    else:
        raise RuntimeError(f"Unknown positional embedding type {model_configs.positional_embedding_type}")
