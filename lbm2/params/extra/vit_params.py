from dataclasses import dataclass, fields
    

def add_vit_params(parser):
    parser.add_argument(
        "--vit-pretrained",
        type=str,
    )
    parser.add_argument(
        "--vit-freeze",
        action="store_true",
    )
    parser.add_argument(
        "--vit-interpolation-mode",
        type=str,
        default="bicubic",
    )
    parser.add_argument(
        "--vit-hidden-dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--vit-inter-dim",
        type=int,
        default=3072,
    )
    parser.add_argument(
        "--vit-patch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--vit-img-size",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--vit-img-num-tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--vit-n-heads",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--vit-dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--vit-n-layers",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--vit-ln-eps",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--vit-cls-flag",
        action="store_true",
    )
    parser.add_argument(
        "--projector-pixel-shuffle-factor",
        type=int,
        default=2,
    )


@dataclass(frozen=True)
class ViTParams:
    vit_pretrained: str
    vit_freeze: bool
    vit_interpolation_mode: str
    vit_hidden_dim: int
    vit_inter_dim: int
    vit_patch_size: int
    vit_img_size: int
    vit_img_num_tokens: int
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