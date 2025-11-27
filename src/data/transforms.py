"""
Data Augmentation Transforms using Albumentations
Optimized for plant disease detection with leaf images.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=224):
    """
    Get training augmentation pipeline.

    Augmentations chosen specifically for plant leaf images:
    - Geometric: Flips, rotations (leaves can be oriented any direction)
    - Color: Brightness/contrast variations (different lighting conditions)
    - Noise: Adds robustness to image quality variations

    Args:
        image_size (int): Target image size (default: 224 for ResNet/MobileNet)

    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    return A.Compose([
        # Resize to target size
        A.Resize(image_size, image_size),

        # Geometric transformations
        A.HorizontalFlip(p=0.5),  # Leaves can be mirrored
        A.VerticalFlip(p=0.3),    # Less common but possible
        A.Rotate(limit=30, p=0.5, border_mode=0),  # Natural rotation variations

        # Combined geometric transformation
        A.ShiftScaleRotate(
            shift_limit=0.1,    # Small shifts
            scale_limit=0.2,    # Zoom in/out
            rotate_limit=30,    # Additional rotation
            border_mode=0,
            p=0.5
        ),

        # Color transformations (lighting conditions)
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        # Color jitter for additional color variations
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.3
        ),

        # Add noise for robustness
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

        # Normalize using ImageNet statistics
        # (transfer learning models are pretrained on ImageNet)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
            max_pixel_value=255.0
        ),

        # Convert to PyTorch tensor
        ToTensorV2()
    ])


def get_val_transforms(image_size=224):
    """
    Get validation/test transforms (no augmentation).

    Only resize and normalize - no augmentation for validation/testing.

    Args:
        image_size (int): Target image size

    Returns:
        albumentations.Compose: Transform pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def get_no_augmentation_transforms(image_size=224):
    """
    Get transforms without augmentation (for ablation study).

    Used to compare performance with vs without augmentation.

    Args:
        image_size (int): Target image size

    Returns:
        albumentations.Compose: Transform pipeline
    """
    return get_val_transforms(image_size)


def get_test_time_augmentation_transforms(image_size=224):
    """
    Get transforms for test-time augmentation (TTA).

    TTA applies multiple augmentations at test time and averages predictions
    to improve robustness. Use this for final evaluation if needed.

    Args:
        image_size (int): Target image size

    Returns:
        list: List of transform pipelines for TTA
    """
    # Base transform (no augmentation)
    base = get_val_transforms(image_size)

    # Horizontal flip
    hflip = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Vertical flip
    vflip = A.Compose([
        A.Resize(image_size, image_size),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Slight rotation
    rotate = A.Compose([
        A.Resize(image_size, image_size),
        A.Rotate(limit=15, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return [base, hflip, vflip, rotate]


# Custom transform for specific use cases
def get_efficientnet_transforms(image_size=300):
    """
    Get transforms optimized for EfficientNet.
    EfficientNet typically uses 300x300 images.

    Args:
        image_size (int): Target size (default: 300 for EfficientNet-B3)

    Returns:
        albumentations.Compose: Transform pipeline
    """
    return get_train_transforms(image_size=image_size)
