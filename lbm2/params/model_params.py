from dataclasses import dataclass, field
from typing import List, Union

import draccus

from lbm2.params.base_params import BaseParams


def register_model_params(key: str):
    """
    Registers a ModelParams subclass and sets its type attribute.
    Use decorator wrapper because draccus's model selection with --model.type doesn't
    automatically populate the attribute cfg.model.type
    """

    def decorator(cls):
        registered_cls = ModelParams.register_subclass(key)(cls)
        registered_cls._type = key
        return registered_cls

    return decorator


@dataclass(frozen=True)
class ModelParams(draccus.ChoiceRegistry, BaseParams):
    type: str = field(default=None)
    resume_from_checkpoint: str = field(default=None)
    resume_weights_only: bool = field(default=False)

    def __init__(self):
        raise NotImplementedError("ModelParams should not be instantiated directly. Use a subclass with model.type=...")

    def __post_init__(self):
        super().__post_init__()
        if self.type is None:
            object.__setattr__(self, "type", getattr(self, "_type", None))


@register_model_params("transformer")
@dataclass(frozen=True)
class TransformerParams(ModelParams):
    norm_type: str = field(default="default_layer_norm")
    ffn_type: str = field(default="swiglu")
    qk_norm: bool = field(default=False)
    positional_embedding_type: str = field(default="rotary")
    attn_name: str = field(default="torch_attn")
    hidden_dim: int = field(default=96)
    n_layers: int = field(default=8)
    n_heads: int = field(default=4)
    vocab_size: int = field(default=50432)
    post_embed_norm: bool = field(default=False)
    norm_eps: float = field(default=1e-5)
    weight_tying: bool = field(default=False)
    max_seq_len: int = field(default=2048)


@register_model_params("transformer_hf")
@dataclass(frozen=True)
class TransformerHFParams(ModelParams):
    hf_pretrained: str = field(default=None)


@register_model_params("vit")
@dataclass(frozen=True)
class ViTParams(ModelParams):
    vit_pretrained: str = field(default=None)
    vit_freeze: bool = field(default=False)
    vit_interpolation_mode: str = field(default="bicubic")
    vit_hidden_dim: int = field(default=768)
    vit_inter_dim: int = field(default=3072)
    vit_patch_size: int = field(default=16)
    vit_img_size: int = field(default=384)
    vit_n_heads: int = field(default=12)
    vit_dropout: float = field(default=0.0)
    vit_n_layers: int = field(default=12)
    vit_ln_eps: float = field(default=1e-6)
    vit_cls_flag: bool = field(default=False)
    projector_pixel_shuffle_factor: int = field(default=1)


@register_model_params("vit_hf")
@dataclass(frozen=True)
class ViTHFParams(ModelParams):
    hf_pretrained: str = field(default=None)
    vit_hidden_dim: int = field(default=768)
    projector_pixel_shuffle_factor: int = field(default=1)


@register_model_params("vlm")
@dataclass(frozen=True)
class VLMParams(ModelParams):
    vit: Union[ViTParams, ViTHFParams] = field(default_factory=ViTParams)
    transformer: Union[TransformerParams, TransformerHFParams] = field(default_factory=TransformerParams)
    vit_freeze: bool = field(default=False)
    image_token_id: int = field(default=None)

    def init_shared_attributes(self, cfg):
        object.__setattr__(self, "image_token_id", cfg.data.image_token_id)


@register_model_params("vlm_hf")
@dataclass(frozen=True)
class VLMHFParams(ModelParams):
    hf_pretrained: str = field(default=None)


@register_model_params("unet")
@dataclass(frozen=True)
class UNetParams(ModelParams):
    in_channels: int = field(default=3)
    out_channels: int = field(default=3)
    time_emb_dim: int = field(default=256)
    text_emb_dim: int = field(default=512)
    channels: List[int] = field(default_factory=list)
    image_size: int = field(default=128)
    time_mlp_float32: bool = field(default=False)


@register_model_params("noise_scheduler")
@dataclass(frozen=True)
class NoiseSchedulerParams(ModelParams):
    num_timesteps: int = field(default=1000)
    beta_start: int = field(default=0.0001)
    beta_end: int = field(default=0.02)


@register_model_params("stable_diffusion")
@dataclass(frozen=True)
class DiffusionParams(ModelParams):
    unet: UNetParams = field(default_factory=UNetParams)
    noise_scheduler: NoiseSchedulerParams = field(default_factory=NoiseSchedulerParams)

    use_diffusers_unet: bool = field(default=False)
    use_diffusers_scheduler: bool = field(default=False)
    use_flow_matching_scheduler: bool = field(default=False)

    @property
    def image_size(self):
        return self.unet.image_size
