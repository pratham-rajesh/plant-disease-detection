"""
PyTorch Dataset for Plant Disease Detection
Memory-efficient implementation with lazy image loading.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class PlantDiseaseDataset(Dataset):
    """
    Plant Disease Dataset - Memory Efficient Implementation

    Loads images on-the-fly instead of storing in memory.
    Supports albumentations transforms for augmentation.

    Args:
        csv_file (str): Path to CSV file with columns ['image_path', 'label']
        transform (albumentations.Compose, optional): Albumentations transforms

    Returns:
        tuple: (image_tensor, label) where image is [C, H, W] tensor
    """

    def __init__(self, csv_file, transform=None):
        """
        Initialize dataset from CSV file.

        CSV format:
            image_path,label,class_name
            /path/to/image1.jpg,0,Apple___Apple_scab
            /path/to/image2.jpg,1,Apple___Black_rot
        """
        self.data_frame = pd.read_csv(csv_file)
        self.image_paths = self.data_frame['image_path'].values
        self.labels = self.data_frame['label'].values
        self.transform = transform

        # Validate dataset
        self._validate_dataset()

    def _validate_dataset(self):
        """Validate that all image files exist."""
        missing_files = []
        for idx, path in enumerate(self.image_paths[:100]):  # Check first 100
            if not os.path.exists(path):
                missing_files.append(path)

        if missing_files:
            print(f"Warning: {len(missing_files)} missing files detected in first 100 samples")
            print(f"Example: {missing_files[0]}")

    def __len__(self):
        """Return total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and return a single sample.

        Args:
            idx (int): Index of sample to load

        Returns:
            tuple: (image_tensor, label)
                - image_tensor: [C, H, W] float tensor
                - label: int class index
        """
        # Load image
        img_path = self.image_paths[idx]

        try:
            # Open image and convert to RGB
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)  # Convert to numpy array for albumentations

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Get label
        label = int(self.labels[idx])

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']  # Already a tensor if using ToTensorV2
        else:
            # Convert to tensor manually if no transform
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label

    def get_class_distribution(self):
        """
        Get distribution of classes in dataset.

        Returns:
            dict: {class_idx: count}
        """
        from collections import Counter
        return dict(Counter(self.labels))

    def get_sample_weights(self):
        """
        Calculate sample weights for weighted sampling.
        Useful for handling class imbalance.

        Returns:
            torch.Tensor: Weight for each sample
        """
        class_counts = self.get_class_distribution()
        num_samples = len(self.labels)

        # Calculate weight for each class (inverse frequency)
        class_weights = {
            cls: num_samples / count
            for cls, count in class_counts.items()
        }

        # Assign weight to each sample
        sample_weights = [class_weights[label] for label in self.labels]

        return torch.DoubleTensor(sample_weights)


class PlantDiseaseDatasetFromPaths(Dataset):
    """
    Alternative dataset that accepts image paths and labels directly.
    Useful when you don't have a CSV file.
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize dataset from lists.

        Args:
            image_paths (list): List of image file paths
            labels (list): List of corresponding labels
            transform (albumentations.Compose, optional): Transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        assert len(image_paths) == len(labels), \
            f"Mismatch: {len(image_paths)} images but {len(labels)} labels"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        # Get label
        label = int(self.labels[idx])

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label
