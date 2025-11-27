"""
Loss functions for plant disease detection.
Handles class imbalance through weighted loss.
"""

import torch
import torch.nn as nn


def calculate_class_weights(class_counts, num_classes, device='cpu'):
    """
    Calculate class weights for imbalanced dataset.

    Formula: weight_i = total_samples / (num_classes * count_i)
    This gives higher weight to minority classes.

    Args:
        class_counts (dict): {class_idx: count}
        num_classes (int): Total number of classes
        device (str): Device to place tensor on

    Returns:
        torch.Tensor: Class weights [num_classes]
    """
    total_samples = sum(class_counts.values())
    weights = []

    for i in range(num_classes):
        count = class_counts.get(i, 1)  # Default to 1 to avoid division by zero
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    return torch.FloatTensor(weights).to(device)


def get_weighted_loss(class_counts, num_classes=39, device='cpu'):
    """
    Get weighted CrossEntropyLoss for imbalanced classes.

    Args:
        class_counts (dict): Class distribution
        num_classes (int): Number of classes
        device (str): Device

    Returns:
        nn.CrossEntropyLoss: Weighted loss function
    """
    class_weights = calculate_class_weights(class_counts, num_classes, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"✓ Weighted CrossEntropyLoss created")
    print(f"  Min weight: {class_weights.min().item():.4f}")
    print(f"  Max weight: {class_weights.max().item():.4f}")
    print(f"  Mean weight: {class_weights.mean().item():.4f}")

    return criterion


def get_standard_loss():
    """
    Get standard CrossEntropyLoss (no class weighting).

    Returns:
        nn.CrossEntropyLoss: Standard loss function
    """
    return nn.CrossEntropyLoss()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    An alternative to weighted CrossEntropy.
    Focuses training on hard examples.

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
