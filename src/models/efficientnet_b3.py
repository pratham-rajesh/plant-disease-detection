"""
EfficientNet-B3 Model with Transfer Learning
State-of-the-art efficiency through compound scaling.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetB3(nn.Module):
    """
    EfficientNet-B3 for Plant Disease Classification.

    EfficientNet uses compound scaling to balance depth, width, and resolution.
    Achieves state-of-the-art accuracy with fewer parameters than ResNet.

    Input Size: 300x300 (optimal for EfficientNet-B3)
    Parameters: ~10M (vs 23M for ResNet50)
    """

    def __init__(self, num_classes=39, pretrained=True, dropout_rate=0.3):
        """
        Initialize EfficientNet-B3 model.

        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Use ImageNet pretrained weights
            dropout_rate (float): Dropout probability
        """
        super(EfficientNetB3, self).__init__()

        # Load pretrained EfficientNet-B3
        if pretrained:
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
            self.efficientnet = models.efficientnet_b3(weights=weights)
        else:
            self.efficientnet = models.efficientnet_b3(weights=None)

        # Get number of features
        num_features = self.efficientnet.classifier[1].in_features  # 1536 for B3

        # Replace classifier
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, 3, 300, 300]

        Returns:
            torch.Tensor: Logits [B, num_classes]
        """
        return self.efficientnet(x)

    def freeze_backbone(self):
        """Freeze feature extractor for Phase 1 training."""
        for param in self.efficientnet.features.parameters():
            param.requires_grad = False

        for param in self.efficientnet.classifier.parameters():
            param.requires_grad = True

        print("✓ EfficientNet-B3 backbone frozen (Phase 1: Feature Extraction)")

    def unfreeze_backbone(self):
        """Unfreeze all layers for Phase 2 fine-tuning."""
        for param in self.efficientnet.parameters():
            param.requires_grad = True

        print("✓ EfficientNet-B3 backbone unfrozen (Phase 2: Fine-Tuning)")

    def get_num_parameters(self, trainable_only=True):
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


def get_efficientnet_b3(num_classes=39, pretrained=True):
    """
    Factory function to create EfficientNet-B3 model.

    Note: EfficientNet-B3 uses 300x300 input size (not 224x224).

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Use ImageNet pretrained weights

    Returns:
        EfficientNetB3: Model instance
    """
    model = EfficientNetB3(num_classes=num_classes, pretrained=pretrained)
    total_params = model.get_num_parameters(trainable_only=False)
    trainable_params = model.get_num_parameters(trainable_only=True)

    print(f"EfficientNet-B3 created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Note: Use 300x300 input size for optimal performance")

    return model


if __name__ == "__main__":
    # Test model
    model = get_efficientnet_b3(num_classes=39, pretrained=True)
    x = torch.randn(2, 3, 300, 300)  # Note: 300x300 for EfficientNet-B3
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
