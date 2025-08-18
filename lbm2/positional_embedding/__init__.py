from lbm2.model_utils import Float32Module
from lbm2.params.model_params import TransformerParams
from lbm2.positional_embedding.none import identity_with_cast
from lbm2.positional_embedding.rotary import RotaryWithCast


def get_pos_embed(model_params: TransformerParams):
    head_dim = model_params.hidden_dim // model_params.n_heads
    if model_params.positional_embedding_type == "rotary":
        return RotaryWithCast(head_dim, model_params.max_seq_len)
    elif model_params.positional_embedding_type == "rotary_float32":
        return Float32Module(RotaryWithCast(head_dim, model_params.max_seq_len), cast_outputs_back=True)
    elif model_params.positional_embedding_type == "none":
        return identity_with_cast
    else:
        raise RuntimeError(f"Unknown positional embedding type {model_params.positional_embedding_type}")
