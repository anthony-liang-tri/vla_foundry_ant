from dataclasses import dataclass, fields
    

def add_model_params(parser):
    parser.add_argument(
        "--model-type",
        type=str,
        default="transformer",
        help="model config to load"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model_config to use. Read from yaml"
    )
    parser.add_argument(
        "--norm-type",
        type=str,
        default="default_layer_norm",
        choices=[
            "default_layer_norm",
            "lp_layer_norm",
            "gain_only_lp_layer_norm",
            "gain_only_layer_norm",
            "no_wb_layer_norm",
            "rms_norm",
        ],
        help="Type of normalization to employ in the model.",
    )
    parser.add_argument(
        "--ffn-type",
        type=str,
        choices=["swiglu", "gelu"],
        default="swiglu",
        help="Type of feedforward layer to use.",
    )
    parser.add_argument(
        "--qk-norm",
        action="store_true",
        default=False,
        help="apply --model-norm to qk as in: https://arxiv.org/abs/2302.05442.",
    )
    parser.add_argument(
        "--positional-embedding-type",
        type=str,
        choices=["rotary", "head_rotary", "llama_rotary", "none"],
        default="rotary",
        help="Type of positional embedding to use.",
    )
    parser.add_argument(
        "--attn-name",
        type=str,
        default="auto",
        choices=["auto", "torch_attn", "custom_attn"],
        help="type of attention to use",
    )
    parser.add_argument(
        "--hidden-dim", 
        type=int,
        default=96,
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50432,
    )
    parser.add_argument(
        "--post-embed-norm",
        action="store_true",
        default=False,
        help="Whether or nor to layernorm after embedding layer"
    )
    parser.add_argument(
        "--norm-eps",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--weight-tying",
        action="store_true",
        default=False,
    )


@dataclass(frozen=True)
class ModelParams:
    model_type: str
    model: str
    
    norm_type: str
    ffn_type: str
    qk_norm: bool
    positional_embedding_type: str
    attn_name: str
    hidden_dim: int
    n_layers: int
    n_heads: int
    seq_len: int
    vocab_size: int
    post_embed_norm: bool
    norm_eps: float
    weight_tying: bool

    # These are defined in the add_params of other files but we use them here
    vit_interpolation_mode: str
    vit_hidden_dim: int
    vit_inter_dim: int
    vit_patch_size: int
    vit_img_size: int
    vit_n_heads: int
    vit_dropout: float
    vit_n_layers: int
    vit_ln_eps: float
    vit_cls_flag: bool
    projector_pixel_shuffle_factor: int


    @classmethod
    def from_args(cls, args):
        init_kwargs = {
            f.name: getattr(args, f.name)
            for f in fields(cls)
            if hasattr(args, f.name)
        }
        return cls(**init_kwargs)