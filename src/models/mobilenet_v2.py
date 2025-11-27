"""
MobileNetV2 Model with Transfer Learning
Lightweight model optimized for mobile/edge deployment.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV2(nn.Module):
    """
    MobileNetV2 for Plant Disease Classification.

    Lightweight architecture using depthwise separable convolutions.
    Ideal for deployment on resource-constrained devices.

    Key Features:
        - Inverted residual blocks
        - Linear bottlenecks
        - Only ~3.5M parameters (vs 23M for ResNet50)
    """

    def __init__(self, num_classes=39, pretrained=True, dropout_rate=0.2):
        """
        Initialize MobileNetV2 model.

        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Use ImageNet pretrained weights
            dropout_rate (float): Dropout probability (MobileNet uses less dropout)
        """
        super(MobileNetV2, self).__init__()

        # Load pretrained MobileNetV2
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
            self.mobilenet = models.mobilenet_v2(weights=weights)
        else:
            self.mobilenet = models.mobilenet_v2(weights=None)

        # Get number of features from last layer
        num_features = self.mobilenet.classifier[1].in_features  # 1280 for MobileNetV2

        # Replace classifier
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, 3, 224, 224]

        Returns:
            torch.Tensor: Logits [B, num_classes]
        """
        return self.mobilenet(x)

    def freeze_backbone(self):
        """
        Freeze feature extractor, train only classifier.
        Phase 1 of transfer learning.
        """
        # Freeze features
        for param in self.mobilenet.features.parameters():
            param.requires_grad = False

        # Unfreeze classifier
        for param in self.mobilenet.classifier.parameters():
            param.requires_grad = True

        print("✓ MobileNetV2 backbone frozen (Phase 1: Feature Extraction)")

    def unfreeze_backbone(self):
        """
        Unfreeze all layers for fine-tuning.
        Phase 2 of transfer learning.
        """
        for param in self.mobilenet.parameters():
            param.requires_grad = True

        print("✓ MobileNetV2 backbone unfrozen (Phase 2: Fine-Tuning)")

    def get_num_parameters(self, trainable_only=True):
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


def get_mobilenet_v2(num_classes=39, pretrained=True):
    """
    Factory function to create MobileNetV2 model.

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Use ImageNet pretrained weights

    Returns:
        MobileNetV2: Model instance
    """
    model = MobileNetV2(num_classes=num_classes, pretrained=pretrained)
    total_params = model.get_num_parameters(trainable_only=False)
    trainable_params = model.get_num_parameters(trainable_only=True)

    print(f"MobileNetV2 created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Pretrained: {pretrained}")

    return model


if __name__ == "__main__":
    # Test model
    model = get_mobilenet_v2(num_classes=39, pretrained=True)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
