from dataclasses import dataclass, field

from lbm2.data.processor import get_processor
from lbm2.params.base_data_params import DataParams
from lbm2.params.robotics.normalization_params import FieldNormalizationParams, NormalizationParams


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


@register_data_params("robotics")
@dataclass(frozen=True)
class LBMDataParams(DataParams):
    """
    Configuration for robotics dataset field definitions and normalization.

    This dataclass defines which fields correspond to proprioception and actions,
    and how they should be normalized, replacing hardcoded field names in training scripts.
    """

    dataset_statistics: list[str] = field(default_factory=list)
    processor: str = field(default="google/paligemma-3b-pt-224")
    img_num_tokens: int = field(default=256)
    image_size: int = field(default=224)
    num_images: int = field(default=1)
    add_action_token: bool = field(default=False)

    proprioception_fields: list[str] = field(default_factory=list)
    action_fields: list[str] = field(default_factory=list)
    exclude_fields: list[str] = field(default_factory=list)
    normalization: NormalizationParams = field(default_factory=NormalizationParams)

    def __post_init__(self):
        super().__post_init__()
        # Global default normalization parameters
        enabled = self.normalization.enabled
        method = self.normalization.method
        scope = self.normalization.scope
        epsilon = self.normalization.epsilon

        # Field-specific normalization parameters
        normalization_fields = self.normalization.field_configs

        # For all used fields (proprioception and action), add default normalization parameters if not specified
        for field_name in self.proprioception_fields + self.action_fields:
            if field_name not in normalization_fields:
                normalization_fields[field_name] = FieldNormalizationParams(method=method, scope=scope, epsilon=epsilon)

        # Update normalization parameters with field-specific parameters
        object.__setattr__(
            self,
            "normalization",
            NormalizationParams(
                field_configs=normalization_fields,
                enabled=enabled,
                method=method,
                scope=scope,
                epsilon=epsilon,
            ),
        )
