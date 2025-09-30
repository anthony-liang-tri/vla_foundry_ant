from dataclasses import dataclass, field
from typing import Tuple

from lbm2.params.base_params import BaseParams


@dataclass(frozen=True)
class BaseAugmentationParams(BaseParams):
    """
    Base class for augmentation parameters
    """

    enabled: bool = field(default=False)


@dataclass(frozen=True)
class ColorJitterParams(BaseAugmentationParams):
    """
    Configuration for color jitter parameters
    """

    brightness: float = field(default=0.2)
    contrast: float = field(default=0.4)
    saturation: float = field(default=0.2)
    hue: Tuple[float, float] = field(default_factory=lambda: (-0.05, 0.05))

    def __post_init__(self):
        if self.brightness < 0 or self.contrast < 0 or self.saturation < 0:
            raise ValueError("brightness, contrast, and saturation must be >= 0.")

        # Ensure hue is a tuple of size two
        if not isinstance(self.hue, tuple) or len(self.hue) != 2:
            raise ValueError("hue must be a tuple of size two (lo, hi).")

        lo, hi = self.hue
        if not (-0.5 <= lo <= 0.5 and -0.5 <= hi <= 0.5 and lo <= hi):
            raise ValueError("hue must be a (lo, hi) tuple within [-0.5, 0.5] and lo <= hi.")


@dataclass(frozen=True)
class RandomCropParams(BaseAugmentationParams):
    """
    Configuration for random crop parameters
    """

    shape: Tuple[int, int] = field(default=(224, 224))

    def __post_init__(self):
        h, w = self.shape
        if not (isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0):
            raise ValueError("random_crop_shape must be a tuple of positive integers (H, W).")


@dataclass(frozen=True)
class ImageAugmentationParams(BaseParams):
    """
    Configuration for image augmentation
    """

    color_jitter: ColorJitterParams = field(default_factory=ColorJitterParams)
    random_crop: RandomCropParams = field(default_factory=RandomCropParams)

    @property
    def augmentations(self):
        # Get all fields that are enabled
        return [k for k, v in self.__dataclass_fields__.items() if getattr(self, k).enabled]


@dataclass(frozen=True)
class DataAugmentationParams(BaseParams):
    """
    Configuration for data augmentation at dataloading time
    """

    enabled: bool = field(default=True)

    image: ImageAugmentationParams = field(default_factory=ImageAugmentationParams)
    # ... eventually add more types of augmentation for other modalities as necessary
