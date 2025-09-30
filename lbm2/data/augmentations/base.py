from torchvision import transforms

from lbm2.params.robotics.augmentation_params import DataAugmentationParams


class Augmentations:
    def __init__(self, augmentation_params: DataAugmentationParams):
        self.augmentation_params = augmentation_params
        self.construct_transforms()

    def construct_transforms(self):
        if self.augmentation_params is None or not self.augmentation_params.enabled:
            self.transforms = transforms.Compose([])
            return

        self.transforms = []

        # Add random crop augmentation
        if (random_crop := self.augmentation_params.image.get("random_crop", None)) and random_crop.enabled:
            crop_h, crop_w = random_crop.shape
            self.transforms.append(transforms.RandomCrop((crop_h, crop_w)))

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
