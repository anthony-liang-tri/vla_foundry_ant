import logging
from dataclasses import dataclass, field
from typing import ClassVar

from vla_foundry.data.processor import get_processor
from vla_foundry.params.base_data_params import DataParams
from vla_foundry.params.robotics.augmentation_params import DataAugmentationParams
from vla_foundry.params.robotics.normalization_params import FieldNormalizationParams, NormalizationParams


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
    tokenizer_loaded = None

    @property
    def pad_token_id(self):
        if self.tokenizer_loaded is None:
            from vla_foundry.data.tokenizer import get_tokenizer

            tokenizer = get_tokenizer(self.tokenizer)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            object.__setattr__(self, "tokenizer_loaded", tokenizer)
        return self.tokenizer_loaded.pad_token_id


@register_data_params("image_caption")
@dataclass(frozen=True)
class ImageCaptionDataParams(DataParams):
    processor: str = field(default="google/paligemma-3b-pt-224")
    processor_kwargs: dict = field(default_factory=dict)
    processor_loaded = None
    img_num_tokens: int = field(default=256)
    image_size: int = field(default=224)
    tokenizer: str = field(default=None)
    augmentation: DataAugmentationParams = field(default_factory=DataAugmentationParams)

    def __post_init__(self):
        super().__post_init__()
        if self.tokenizer is None:
            object.__setattr__(self, "tokenizer", self.processor)

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
    val_dataset_statistics: list[str] = field(default_factory=list)
    processor: str = field(default=None)
    processor_kwargs: dict = field(default_factory=dict)

    img_num_tokens: int = field(default=256)
    image_size: int = field(default=224)
    max_text_seq_len: int | None = field(default=None)

    # Language instruction types to use: "original", "randomized", "verbose", "alternative"
    language_instruction_types: list[str] = field(default_factory=lambda: ["original"])

    camera_names: list[str] = field(default_factory=list)
    image_indices: list[int] = field(default_factory=list)
    image_names: list[str] = field(default_factory=list)
    pad_missing_images: bool = field(default=False)
    mask_padded_images: bool = field(default=False)
    proprioception_fields: list[str] = field(default_factory=list)
    tactile_fields: list[str] = field(default_factory=list)
    action_fields: list[str] = field(default_factory=list)
    pose_groups: list[dict[str, str]] = field(default_factory=list)
    intrinsics_fields: list[str] = field(default_factory=list)
    extrinsics_fields: list[str] = field(default_factory=list)
    use_point_cloud: bool = field(default=False)
    # Total number of points for FPS sampling. If left as None, resolve_robotics_data_fields
    # fills it from the dataset-side preprocessing_config.yaml (when use_point_cloud=True).
    point_cloud_num_points: int | None = field(default=None)
    normalization: NormalizationParams = field(default_factory=NormalizationParams)
    augmentation: DataAugmentationParams = field(default_factory=DataAugmentationParams)

    lowdim_past_timesteps: int | None = field(default=None)
    lowdim_future_timesteps: int | None = field(default=None)
    action_dim: int = field(default=None)
    proprioception_dim: int | None = field(default=None)

    # Set to True by vla_foundry.params.resolve.resolve_derived_fields. This is
    # runtime state, not user config, so it is intentionally not a dataclass field.
    _resolved: ClassVar[bool] = False

    def __post_init__(self):
        try:
            self._post_init_impl()
        except (TypeError, ValueError, KeyError) as e:
            raise RuntimeError(f"RoboticsDataParams initialization failed: {type(e).__name__}: {e}") from e

    def _post_init_impl(self):
        super().__post_init__()

        if self.mask_padded_images and not self.pad_missing_images:
            raise ValueError("mask_padded_images requires pad_missing_images to be True")

        # Validate language instruction types
        valid_types = {"original", "randomized", "verbose", "alternative"}
        invalid_types = set(self.language_instruction_types) - valid_types
        if invalid_types:
            raise ValueError(f"Invalid language instruction types: {invalid_types}. Valid types are: {valid_types}")

        # If no pose groups are provided, they must be explicitly configured
        # Pose groups are now required to be explicitly specified in configuration
        if not self.pose_groups:
            logging.warning(
                "No pose groups specified. Relative coordinate transformations will not be available. "
                "Please add pose_groups to your data configuration if you need relative coordinates."
            )

        # Compute image_names from camera_names and image_indices
        if (self.image_names is None or len(self.image_names) == 0) and self.camera_names and self.image_indices:
            image_names = [f"{cname}_t{idx}" for idx in self.image_indices for cname in self.camera_names]
            object.__setattr__(self, "image_names", image_names)

        # For all used fields (proprioception and action), add default normalization parameters if not specified
        normalization_fields = self.normalization.field_configs
        for field_name in self.proprioception_fields + self.action_fields:
            if field_name not in normalization_fields:
                normalization_fields[field_name] = FieldNormalizationParams(
                    method=self.normalization.method, scope=self.normalization.scope, epsilon=self.normalization.epsilon
                )

        # Update normalization parameters with field-specific parameters
        object.__setattr__(self.normalization, "field_configs", normalization_fields)

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)
        if cfg.data.processor:
            if hasattr(cfg.model, "hf_pretrained"):
                object.__setattr__(self, "processor", cfg.model.hf_pretrained)
            elif hasattr(cfg.model, "vlm_params") and hasattr(cfg.model.vlm_params, "hf_pretrained"):
                object.__setattr__(self, "processor", cfg.model.vlm_params.hf_pretrained)
        if self.lowdim_past_timesteps is None:
            object.__setattr__(self, "lowdim_past_timesteps", self.normalization.lowdim_past_timesteps)
        if self.lowdim_future_timesteps is None:
            object.__setattr__(self, "lowdim_future_timesteps", self.normalization.lowdim_future_timesteps)
