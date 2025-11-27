"""
Baseline CNN Model - Built from Scratch
Demonstrates understanding of CNN fundamentals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Baseline Convolutional Neural Network for Plant Disease Classification.

    Architecture:
        Input (224x224x3)
        ├─ Conv Block 1: 3→32 channels
        ├─ Conv Block 2: 32→64 channels
        ├─ Conv Block 3: 64→128 channels
        ├─ Conv Block 4: 128→256 channels
        ├─ Global Average Pooling
        ├─ FC1: 256→512
        ├─ FC2: 512→256
        └─ FC3: 256→num_classes

    Each Conv Block contains:
        - Conv2d (3x3 kernel)
        - BatchNorm2d
        - ReLU activation
        - MaxPool2d (2x2)

    Design Choices (for documentation):
        - BatchNorm: Reduces internal covariate shift, enables higher learning rates
        - ReLU: Prevents vanishing gradients, computationally efficient
        - MaxPool: Spatial dimension reduction, translation invariance
        - Global Avg Pool: Reduces parameters dramatically, prevents overfitting
        - Dropout: Regularization technique, prevents co-adaptation of neurons
    """

    def __init__(self, num_classes=39, dropout_rate=0.5):
        """
        Initialize Baseline CNN.

        Args:
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout probability
        """
        super(BaselineCNN, self).__init__()

        # Conv Block 1: 3 → 32 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224 → 112
        )

        # Conv Block 2: 32 → 64 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 → 56
        )

        # Conv Block 3: 64 → 128 channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 → 28
        )

        # Conv Block 4: 128 → 256 channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28 → 14
        )

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6)  # Slightly less dropout
        )

        self.fc3 = nn.Linear(256, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, 3, 224, 224]

        Returns:
            torch.Tensor: Logits [B, num_classes]
        """
        # Convolutional layers
        x = self.conv1(x)  # [B, 32, 112, 112]
        x = self.conv2(x)  # [B, 64, 56, 56]
        x = self.conv3(x)  # [B, 128, 28, 28]
        x = self.conv4(x)  # [B, 256, 14, 14]

        # Global average pooling
        x = self.global_avg_pool(x)  # [B, 256, 1, 1]
        x = torch.flatten(x, 1)  # [B, 256]

        # Fully connected layers
        x = self.fc1(x)  # [B, 512]
        x = self.fc2(x)  # [B, 256]
        x = self.fc3(x)  # [B, num_classes]

        return x

    def get_num_parameters(self):
        """
        Calculate total number of trainable parameters.

        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_baseline_cnn(num_classes=39, pretrained=False):
    """
    Factory function to create Baseline CNN.

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Not applicable for baseline (built from scratch)

    Returns:
        BaselineCNN: Model instance
    """
    model = BaselineCNN(num_classes=num_classes)
    print(f"Baseline CNN created with {model.get_num_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test model
    model = get_baseline_cnn(num_classes=39)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.get_num_parameters():,}")
