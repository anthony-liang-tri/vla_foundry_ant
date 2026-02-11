"""Point cloud color jitter augmentation."""

import numpy as np
import torch
from torchvision import transforms


class PointCloudColorJitter:
    """Apply color jitter augmentation to point cloud RGB channels.

    This augmentation treats the point cloud RGB values as a 2D image and applies
    torchvision ColorJitter. The point cloud is expected to have shape (T, N, C)
    where C=6 for XYZRGB format.
    """

    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        """Initialize point cloud color jitter.

        Args:
            brightness: How much to jitter brightness. brightness_factor is chosen
                uniformly from [max(0, 1 - brightness), 1 + brightness]
            contrast: How much to jitter contrast. contrast_factor is chosen
                uniformly from [max(0, 1 - contrast), 1 + contrast]
            saturation: How much to jitter saturation. saturation_factor is chosen
                uniformly from [max(0, 1 - saturation), 1 + saturation]
            hue: How much to jitter hue. hue_factor is chosen uniformly from
                [-hue, hue] or a tuple (min, max). Should have 0 <= hue <= 0.5

        Note:
            RGB values are clipped to [0, 1] after augmentation.
        """
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, point_cloud: np.ndarray) -> np.ndarray:
        """Apply color jitter to point cloud RGB channels.

        Args:
            point_cloud: (T, N, C) numpy array where C=6 for XYZRGB

        Returns:
            (T, N, C) numpy array with jittered RGB channels
        """
        # Only apply if point cloud has RGB channels (6 channels for XYZRGB)
        if point_cloud.shape[-1] != 6:
            return point_cloud

        # Extract RGB channels (last 3 channels, in [0, 1] range)
        xyz = point_cloud[..., :3]  # (T, N, 3)
        rgb = point_cloud[..., 3:6]  # (T, N, 3)

        # Convert to tensor and reshape: (T, N, 3) -> (3, T, N)
        # Treat the point cloud as a "3-channel image" with height=T, width=N
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (3, T, N)

        # Apply color jitter
        rgb_tensor = self.color_jitter(rgb_tensor)

        # Convert back: (3, T, N) -> (T, N, 3)
        rgb_augmented = rgb_tensor.permute(1, 2, 0).numpy()

        # Clip to [0, 1] range (color jitter can produce values outside this range)
        rgb_augmented = np.clip(rgb_augmented, 0.0, 1.0)

        # Concatenate XYZ (unchanged) with augmented RGB
        return np.concatenate([xyz, rgb_augmented], axis=-1)
