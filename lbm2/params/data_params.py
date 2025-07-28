from dataclasses import dataclass, field
from typing import List

import draccus

from lbm2.data.processor import get_processor
from lbm2.params.base_params import BaseParams


def register_data_params(key: str):
    """
    Registers a DataParams subclass and sets its type attribute.
    Use decorator wrapper because draccus's model selection with --data.type doesn't
    automatically populate the attribute cfg.data.type
    """

    def decorator(cls):
        registered_cls = DataParams.register_subclass(key)(cls)
        registered_cls._type = key
        return registered_cls

    return decorator


@dataclass(frozen=True)
class DataParams(draccus.ChoiceRegistry, BaseParams):
    type: str = field(default=None)
    dataset_manifest: List[str] = field(default_factory=list)
    dataset_weighting: List[float] = field(default_factory=list)
    dataset_modality: List[str] = field(default_factory=list)
    allow_multiple_epochs: bool = False
    num_workers: int = field(default=1)
    seq_len: int = field(default=2048)
    shuffle: bool = field(default=True)
    shuffle_buffer_size: int = field(default=2000)
    shuffle_initial: int = field(default=500)

    # Shared attributes. Overwritten in init_shared_attributes.
    seed: int = field(default=42)

    def __init__(self):
        raise NotImplementedError("DataParams should not be instantiated directly. Use a subclass with data.type=...")

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "dataset_weighting", [float(i) for i in self.dataset_weighting])
        if not self.shuffle:
            object.__setattr__(self, "shuffle_buffer_size", 0)
            object.__setattr__(self, "shuffle_initial", 0)
        if self.type is None:
            object.__setattr__(self, "type", getattr(self.__class__, "_type", None))

    def init_shared_attributes(self, cfg):
        object.__setattr__(self, "seed", cfg.hparams.seed)


@register_data_params("text")
@dataclass(frozen=True)
class TextDataParams(DataParams):
    pass


@register_data_params("text_untokenized")
@dataclass(frozen=True)
class TextUntokenizedDataParams(DataParams):
    tokenizer: str = field(default="EleutherAI/gpt-neox-20b")


@register_data_params("image_caption")
@dataclass(frozen=True)
class ImageCaptionDataParams(DataParams):
    processor: str = field(default="google/paligemma-3b-pt-224")
    processor_loaded = None
    img_num_tokens: int = field(default=256)
    image_size: int = field(default=224)

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)
        if hasattr(cfg.model, "image_size") and cfg.model.image_size is not None:
            object.__setattr__(self, "image_size", cfg.model.image_size)

    @property
    def image_token_id(self):
        if self.processor_loaded is None:
            object.__setattr__(self, "processor_loaded", get_processor(self))
        return self.processor_loaded.image_token_id

    @property
    def pad_token_id(self):
        if self.processor_loaded is None:
            object.__setattr__(self, "processor_loaded", get_processor(self))
        return self.processor_loaded.tokenizer.pad_token_id
