from dataclasses import dataclass, field

from lbm2.data.processor import get_processor
from lbm2.params.base_data_params import DataParams
from lbm2.params.robotics.augmentation_params import DataAugmentationParams
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
    augmentation: DataAugmentationParams = field(default_factory=DataAugmentationParams)

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
class RoboticsDataParams(DataParams):
    """
    Configuration for robotics dataset field definitions and normalization.

    This dataclass defines which fields correspond to proprioception and actions,
    and how they should be normalized, replacing hardcoded field names in training scripts.
    """

    dataset_statistics: list[str] = field(default_factory=list)
    processor: str = field(default=None)
    img_num_tokens: int = field(default=256)
    image_size: int = field(default=224)
    num_images: int = field(default=None)
    # Language instruction types to use: "original", "randomized", "verbose", "alternative"
    language_instruction_types: list[str] = field(default_factory=lambda: ["original"])

    proprioception_fields: list[str] = field(default_factory=list)
    action_fields: list[str] = field(default_factory=list)
    intrinsics_fields: list[str] = field(default_factory=list)
    extrinsics_fields: list[str] = field(default_factory=list)
    normalization: NormalizationParams = field(default_factory=NormalizationParams)
    augmentation: DataAugmentationParams = field(default_factory=DataAugmentationParams)

    action_dim: int = field(default=None)

    def __post_init__(self):
        super().__post_init__()

        # Validate language instruction types
        valid_types = {"original", "randomized", "verbose", "alternative"}
        invalid_types = set(self.language_instruction_types) - valid_types
        if invalid_types:
            raise ValueError(f"Invalid language instruction types: {invalid_types}. Valid types are: {valid_types}")

        # For all used fields (proprioception and action), add default normalization parameters if not specified
        normalization_fields = self.normalization.field_configs
        for field_name in self.proprioception_fields + self.action_fields:
            if field_name not in normalization_fields:
                normalization_fields[field_name] = FieldNormalizationParams(
                    method=self.normalization.method, scope=self.normalization.scope, epsilon=self.normalization.epsilon
                )

        # Update normalization parameters with field-specific parameters
        object.__setattr__(self.normalization, "field_configs", normalization_fields)

        # Compute action dimension by summing the dimension of all action fields (known from normalization parameters)
        # Need to import here to avoid circular import
        from lbm2.data.robotics.normalization import RoboticsNormalizer

        normalizer = RoboticsNormalizer(
            normalization_params=self.normalization, statistics_path=self.dataset_statistics
        )
        action_dim = 0
        for field_name in self.action_fields:
            action_dim += len(normalizer.stats[field_name]["mean"])
        if self.action_dim is None:
            object.__setattr__(self, "action_dim", action_dim)
        else:
            assert self.action_dim == action_dim, (
                f"Action dimension mismatch, \
            the user-provided action dimension {self.action_dim} does not match \
            the computed action dimension {action_dim}. Please provide the correct action dimension or \
            set action_dim to None to automatically compute it from the action fields. \
            This could also be a discrepancy between the action fields and the normalization parameters."
            )

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)
        if cfg.data.processor:
            if hasattr(cfg.model, "hf_pretrained"):
                object.__setattr__(self, "processor", cfg.model.hf_pretrained)
            elif hasattr(cfg.model, "vlm_params") and hasattr(cfg.model.vlm_params, "hf_pretrained"):
                object.__setattr__(self, "processor", cfg.model.vlm_params.hf_pretrained)
        self.normalization.init_shared_attributes(cfg)
