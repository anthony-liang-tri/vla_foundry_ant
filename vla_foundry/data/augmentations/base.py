from torchvision import transforms

from vla_foundry.data.augmentations.point_cloud_color_jitter import PointCloudColorJitter
from vla_foundry.data.augmentations.random_ratio_crop import RandomRatioCrop
from vla_foundry.params.robotics.augmentation_params import DataAugmentationParams


class Augmentations:
    def __init__(self, augmentation_params: DataAugmentationParams):
        self.augmentation_params = augmentation_params
        self.construct_transforms()

    def construct_transforms(self):
        self.image_transforms = []
        self.point_cloud_transforms = []

        if self.augmentation_params is None or not self.augmentation_params.enabled:
            return

        # Add crop augmentation
        if (crop := self.augmentation_params.image.get("crop", None)) and crop.enabled:
            crop_h, crop_w = crop.shape
            if crop.mode == "center":
                if crop_h <= 1.0 and crop_w <= 1.0:
                    raise ValueError("Center crop with ratio-based shape is not supported. Use absolute pixel values.")
                self.image_transforms.append(transforms.CenterCrop((int(crop_h), int(crop_w))))
            else:  # random mode
                if crop_h <= 1.0 and crop_w <= 1.0:
                    self.image_transforms.append(RandomRatioCrop((crop_h, crop_w)))
                elif crop_h > 1.0 and crop_w > 1.0:
                    self.image_transforms.append(transforms.RandomCrop((int(crop_h), int(crop_w))))
                else:
                    raise ValueError(f"Invalid crop shape: {crop.shape}")

        # Add color jitter augmentation for images
        if (color_jitter := self.augmentation_params.image.get("color_jitter", None)) and color_jitter.enabled:
            self.image_transforms.append(
                transforms.ColorJitter(
                    brightness=color_jitter.brightness,
                    contrast=color_jitter.contrast,
                    saturation=color_jitter.saturation,
                    hue=color_jitter.hue,
                )
            )

        self.image_transforms = transforms.Compose(self.image_transforms)

        # Add color jitter augmentation for point clouds
        if (
            pc_color_jitter := self.augmentation_params.point_cloud.get("color_jitter", None)
        ) and pc_color_jitter.enabled:
            self.point_cloud_transforms.append(
                PointCloudColorJitter(
                    brightness=pc_color_jitter.brightness,
                    contrast=pc_color_jitter.contrast,
                    saturation=pc_color_jitter.saturation,
                    hue=pc_color_jitter.hue,
                )
            )

        self.point_cloud_transforms = transforms.Compose(self.point_cloud_transforms)

    def apply_transforms(self, sample):
        """Apply augmentations to images and point clouds in the sample.

        Args:
            sample: Dictionary with image keys (ending in image extensions) and
                   optional 'point_cloud' key with (T, N, 6) numpy array

        Returns:
            Modified sample with augmented images and point clouds
        """
        # Apply image transforms
        for k, v in sample.items():
            if k.endswith(("jpg", "png", "jpeg", "webp")):
                sample[k] = self.image_transforms(v)

        # Apply point cloud transforms
        if "point_cloud.npz" in sample and sample["point_cloud.npz"] is not None:
            # Extract data from npz format: {'data': array(...)}
            point_cloud_data = sample["point_cloud.npz"]["data"]
            point_cloud_data = self.point_cloud_transforms(point_cloud_data)
            # Put transformed data back in npz format
            sample["point_cloud.npz"]["data"] = point_cloud_data

        return sample
