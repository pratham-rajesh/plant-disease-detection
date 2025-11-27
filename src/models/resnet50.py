"""
ResNet50 Model with Transfer Learning
Pretrained on ImageNet, fine-tuned for plant disease detection.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    """
    ResNet50 for Plant Disease Classification.

    Uses transfer learning from ImageNet pretrained weights.
    Replaces final FC layer with custom classifier for 39 classes.

    Architecture:
        - ResNet50 backbone (pretrained on ImageNet)
        - Custom classifier: 2048 → 512 → num_classes
        - Dropout for regularization
    """

    def __init__(self, num_classes=39, pretrained=True, dropout_rate=0.5):
        """
        Initialize ResNet50 model.

        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Use ImageNet pretrained weights
            dropout_rate (float): Dropout probability
        """
        super(ResNet50, self).__init__()

        # Load pretrained ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.resnet = models.resnet50(weights=weights)
        else:
            self.resnet = models.resnet50(weights=None)

        # Get number of features from last layer
        num_features = self.resnet.fc.in_features  # 2048 for ResNet50

        # Replace final FC layer with custom classifier
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, 3, 224, 224]

        Returns:
            torch.Tensor: Logits [B, num_classes]
        """
        return self.resnet(x)

    def freeze_backbone(self):
        """
        Freeze all layers except the final classifier.
        Used for Phase 1 of transfer learning (feature extraction).
        """
        # Freeze all parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze only the final FC layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        print("✓ ResNet50 backbone frozen (Phase 1: Feature Extraction)")

    def unfreeze_backbone(self):
        """
        Unfreeze all layers.
        Used for Phase 2 of transfer learning (fine-tuning).
        """
        for param in self.resnet.parameters():
            param.requires_grad = True

        print("✓ ResNet50 backbone unfrozen (Phase 2: Fine-Tuning)")

    def get_num_parameters(self, trainable_only=True):
        """
        Get number of parameters.

        Args:
            trainable_only (bool): Count only trainable parameters

        Returns:
            int: Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


def get_resnet50(num_classes=39, pretrained=True):
    """
    Factory function to create ResNet50 model.

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Use ImageNet pretrained weights

    Returns:
        ResNet50: Model instance
    """
    model = ResNet50(num_classes=num_classes, pretrained=pretrained)
    total_params = model.get_num_parameters(trainable_only=False)
    trainable_params = model.get_num_parameters(trainable_only=True)

    print(f"ResNet50 created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Pretrained: {pretrained}")

    return model


if __name__ == "__main__":
    # Test model
    model = get_resnet50(num_classes=39, pretrained=True)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test freezing
    print("\n" + "="*50)
    model.freeze_backbone()
    print(f"Trainable params after freeze: {model.get_num_parameters():,}")

    model.unfreeze_backbone()
    print(f"Trainable params after unfreeze: {model.get_num_parameters():,}")
