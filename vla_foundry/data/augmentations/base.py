from torchvision import transforms

from vla_foundry.data.augmentations.random_ratio_crop import RandomRatioCrop
from vla_foundry.params.robotics.augmentation_params import DataAugmentationParams


class Augmentations:
    def __init__(self, augmentation_params: DataAugmentationParams):
        self.augmentation_params = augmentation_params
        self.construct_transforms()

    def construct_transforms(self):
        if self.augmentation_params is None or not self.augmentation_params.enabled:
            self.transforms = transforms.Compose([])
            return

        self.transforms = []

        # Add crop augmentation
        if (crop := self.augmentation_params.image.get("crop", None)) and crop.enabled:
            crop_h, crop_w = crop.shape
            if crop.mode == "center":
                if crop_h <= 1.0 and crop_w <= 1.0:
                    raise ValueError("Center crop with ratio-based shape is not supported. Use absolute pixel values.")
                self.transforms.append(transforms.CenterCrop((int(crop_h), int(crop_w))))
            else:  # random mode
                if crop_h <= 1.0 and crop_w <= 1.0:
                    self.transforms.append(RandomRatioCrop((crop_h, crop_w)))
                elif crop_h > 1.0 and crop_w > 1.0:
                    self.transforms.append(transforms.RandomCrop((int(crop_h), int(crop_w))))
                else:
                    raise ValueError(f"Invalid crop shape: {crop.shape}")

        # Add color jitter augmentation
        if (color_jitter := self.augmentation_params.image.get("color_jitter", None)) and color_jitter.enabled:
            self.transforms.append(
                transforms.ColorJitter(
                    brightness=color_jitter.brightness,
                    contrast=color_jitter.contrast,
                    saturation=color_jitter.saturation,
                    hue=color_jitter.hue,
                )
            )

        self.transforms = transforms.Compose(self.transforms)

    def apply_transforms(self, sample):
        for k, v in sample.items():
            if k.endswith(("jpg", "png", "jpeg", "webp")):
                sample[k] = self.transforms(v)
        return sample
