#!/usr/bin/env python3
"""
Pytest tests for augmentation_params.py.
"""

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms

from lbm2.data.augmentations.base import Augmentations
from lbm2.params.robotics.augmentation_params import (
    ColorJitterParams,
    DataAugmentationParams,
    ImageAugmentationParams,
    RandomCropParams,
)


def create_dummy_image(size=(256, 256)):
    """
    Create a dummy image with random colors for testing purposes.

    Args:
        size (tuple): The size of the image (width, height).

    Returns:
        PIL.Image.Image: A dummy image.
    """
    array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array)


@pytest.mark.parametrize(
    "brightness,contrast,saturation,hue,should_raise",
    [
        (-0.1, 0.2, 0.3, (-0.1, 0.1), True),  # Invalid brightness
        (0.1, -0.2, 0.3, (-0.1, 0.1), True),  # Invalid contrast
        (0.1, 0.2, -0.3, (-0.1, 0.1), True),  # Invalid saturation
        (0.1, 0.2, 0.3, (-0.6, 0.1), True),  # Invalid hue (low out of range)
        (0.1, 0.2, 0.3, (-0.1, 0.6), True),  # Invalid hue (high out of range)
        (0.1, 0.2, 0.3, (0.1, -0.1), True),  # Invalid hue (lo > hi)
        (0.1, 0.2, 0.3, (0.1), True),  # Hue not a tuple of size two
        (0.1, 0.2, 0.3, (0.1, -0.1), True),  # Invalid hue range
        (0.1, 0.2, 0.3, (-0.1, 0.1), False),  # Valid parameters
    ],
)
def test_color_jitter_params_validation(brightness, contrast, saturation, hue, should_raise):
    """Test validation logic in ColorJitterParams."""
    if should_raise:
        with pytest.raises(ValueError):
            ColorJitterParams(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    else:
        ColorJitterParams(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)  # No error


@pytest.mark.parametrize(
    "random_crop_factory,should_raise",
    [
        (lambda: RandomCropParams(shape=(128, 128), enabled=True), False),  # Valid random crop
        (lambda: RandomCropParams(shape=(-128, 128), enabled=True), True),  # Invalid shape (negative height)
        (lambda: RandomCropParams(shape=(128, -128), enabled=True), True),  # Invalid shape (negative width)
        (lambda: RandomCropParams(shape=(0, 128), enabled=True), True),  # Invalid shape (zero height)
        (lambda: RandomCropParams(shape=(128, 0), enabled=True), True),  # Invalid shape (zero width)
        (lambda: RandomCropParams(shape=(128, 128), enabled=False), False),  # Valid when disabled
    ],
)
def test_image_augmentation_params_random_crop_validation(random_crop_factory, should_raise):
    """Test validation logic for random crop in ImageAugmentationParams."""
    if should_raise:
        with pytest.raises(ValueError):
            ImageAugmentationParams(
                random_crop=random_crop_factory(),
            )
    else:
        ImageAugmentationParams(
            random_crop=random_crop_factory(),
        )  # Should not raise


@pytest.mark.parametrize(
    "color_jitter_factory,should_raise",
    [
        (
            lambda: ColorJitterParams(brightness=0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=True),
            False,
        ),  # Valid color jitter
        (
            lambda: ColorJitterParams(brightness=-0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=True),
            True,
        ),  # Invalid brightness
        (
            lambda: ColorJitterParams(brightness=0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=False),
            False,
        ),  # Valid when disabled
    ],
)
def test_image_augmentation_params_color_jitter_validation(color_jitter_factory, should_raise):
    """Test validation logic for color jitter in ImageAugmentationParams."""
    if should_raise:
        with pytest.raises(ValueError):
            ImageAugmentationParams(
                color_jitter=color_jitter_factory(),
            )
    else:
        ImageAugmentationParams(
            color_jitter=color_jitter_factory(),
        )  # No error


@pytest.mark.parametrize(
    "random_crop_factory,color_jitter_factory,should_raise",
    [
        (
            lambda: RandomCropParams(shape=(128, 128), enabled=True),
            lambda: ColorJitterParams(brightness=0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=True),
            False,
        ),  # Valid color jitter and random crop
        (
            lambda: RandomCropParams(shape=(-128, 128), enabled=True),
            lambda: ColorJitterParams(brightness=0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=True),
            True,
        ),  # Invalid random_crop_shape
        (
            lambda: RandomCropParams(shape=(128, 128), enabled=True),
            lambda: ColorJitterParams(brightness=-0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=True),
            True,
        ),  # Invalid color_jitter parameters
        (
            lambda: RandomCropParams(shape=(128, 128), enabled=False),
            lambda: ColorJitterParams(brightness=0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=False),
            False,
        ),  # Valid when both disabled
    ],
)
def test_image_augmentation_params_combined_validation(random_crop_factory, color_jitter_factory, should_raise):
    """Test combined validation logic in ImageAugmentationParams."""
    if should_raise:
        with pytest.raises(ValueError):
            ImageAugmentationParams(
                random_crop=random_crop_factory(),
                color_jitter=color_jitter_factory(),
            )
    else:
        ImageAugmentationParams(
            random_crop=random_crop_factory(),
            color_jitter=color_jitter_factory(),
        )  # Should not raise


@pytest.mark.parametrize(
    "augmentation_params,has_transforms",
    [
        (None, False),  # No augmentation parameters provided
        (
            DataAugmentationParams(enabled=False),  # Disabled augmentations
            False,  # No transforms expected
        ),
        (
            DataAugmentationParams(),  # Default (empty augmentations)
            False,  # No transforms expected
        ),
        (
            DataAugmentationParams(
                image=ImageAugmentationParams(
                    random_crop=RandomCropParams(shape=(128, 128), enabled=True),
                )
            ),
            True,  # Should have transforms
        ),
        (
            DataAugmentationParams(
                image=ImageAugmentationParams(
                    color_jitter=ColorJitterParams(
                        brightness=0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=True
                    ),
                )
            ),
            True,  # Should have transforms
        ),
        (
            DataAugmentationParams(
                image=ImageAugmentationParams(
                    random_crop=RandomCropParams(shape=(128, 128), enabled=True),
                    color_jitter=ColorJitterParams(
                        brightness=0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=True
                    ),
                )
            ),
            True,  # Should have transforms
        ),
    ],
)
def test_augmentations_class_creation(augmentation_params, has_transforms):
    """Test the creation of the Augmentations class."""
    augmentations = Augmentations(augmentation_params)

    # Check that transforms are created appropriately
    if has_transforms:
        assert isinstance(augmentations.transforms, transforms.Compose)
        assert len(augmentations.transforms.transforms) > 0
    else:
        # Should either be empty Compose or have no transforms
        if isinstance(augmentations.transforms, transforms.Compose):
            assert len(augmentations.transforms.transforms) == 0


def test_augmentations_class_invalid_params():
    """Test that invalid augmentation parameters raise appropriate errors."""
    # Invalid random_crop_shape
    with pytest.raises(ValueError):
        Augmentations(
            DataAugmentationParams(
                image=ImageAugmentationParams(
                    random_crop=RandomCropParams(shape=(-128, 128), enabled=True),  # Invalid shape
                )
            )
        )

    # Invalid color_jitter parameters
    with pytest.raises(ValueError):
        Augmentations(
            DataAugmentationParams(
                image=ImageAugmentationParams(
                    color_jitter=ColorJitterParams(
                        brightness=-0.2,  # Invalid brightness
                        contrast=0.3,
                        saturation=0.4,
                        hue=(-0.1, 0.1),
                        enabled=True,
                    ),
                )
            )
        )


def test_random_crop_augmentation():
    """
    Test that the random crop augmentation works as expected.
    """
    augmentation_params = DataAugmentationParams(
        image=ImageAugmentationParams(
            random_crop=RandomCropParams(shape=(128, 128), enabled=True),
        )
    )
    augmentations = Augmentations(augmentation_params)

    # Check that transforms were created
    assert isinstance(augmentations.transforms, transforms.Compose)
    assert len(augmentations.transforms.transforms) > 0

    # Test the random crop augmentation
    img = create_dummy_image(size=(256, 256))
    cropped_img = augmentations.transforms(img)

    # Verify that the cropped image has the correct size
    assert cropped_img.size == (128, 128)


def test_color_jitter_augmentation():
    """
    Test that the color jitter augmentation works as expected.
    """
    augmentation_params = DataAugmentationParams(
        image=ImageAugmentationParams(
            color_jitter=ColorJitterParams(brightness=0.5, contrast=0.5, saturation=0.5, hue=(-0.1, 0.1), enabled=True),
        )
    )
    augmentations = Augmentations(augmentation_params)

    # Check that transforms were created
    assert isinstance(augmentations.transforms, transforms.Compose)
    assert len(augmentations.transforms.transforms) > 0

    # Test the color jitter augmentation
    img = create_dummy_image()
    jittered_img = augmentations.transforms(img)

    # Verify that the jittered image is different from the original
    original_array = np.array(img)
    jittered_array = np.array(jittered_img)
    assert not np.array_equal(original_array, jittered_array), "Color jitter should modify the image."


def test_combined_augmentations():
    """
    Test that the pipeline works correctly with both random crop and color jitter.
    """
    augmentation_params = DataAugmentationParams(
        image=ImageAugmentationParams(
            random_crop=RandomCropParams(shape=(128, 128), enabled=True),
            color_jitter=ColorJitterParams(brightness=0.5, contrast=0.5, saturation=0.5, hue=(-0.1, 0.1), enabled=True),
        )
    )
    augmentations = Augmentations(augmentation_params)

    # Check that transforms were created
    assert isinstance(augmentations.transforms, transforms.Compose)
    assert len(augmentations.transforms.transforms) == 2  # Should have both transforms

    # Test the combined augmentation pipeline
    img = create_dummy_image(size=(256, 256))
    transformed_img = augmentations.transforms(img)

    # Verify that the transformed image has the correct size (cropped)
    assert transformed_img.size == (128, 128)

    # Verify that the image was modified (color jitter should make it different)
    # We can't easily test the exact transformation, but we can check the size
    # and that the transforms were applied in sequence


def test_apply_transforms_method():
    """
    Test the apply_transforms method that processes samples with image files.
    """
    augmentation_params = DataAugmentationParams(
        image=ImageAugmentationParams(
            random_crop=RandomCropParams(shape=(128, 128), enabled=True),
        )
    )
    augmentations = Augmentations(augmentation_params)

    # Create a sample with image files
    img = create_dummy_image(size=(256, 256))
    sample = {
        "image1.jpg": img,
        "image2.png": img,
        "non_image_data": "some text",
        "metadata": {"key": "value"},
    }

    # Apply transforms
    transformed_sample = augmentations.apply_transforms(sample)

    # Verify that image files were transformed
    assert transformed_sample["image1.jpg"].size == (128, 128)
    assert transformed_sample["image2.png"].size == (128, 128)

    # Verify that non-image data was not modified
    assert transformed_sample["non_image_data"] == "some text"
    assert transformed_sample["metadata"] == {"key": "value"}


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])
