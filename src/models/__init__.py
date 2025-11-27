"""Model factory and utilities."""

from .baseline_cnn import get_baseline_cnn
from .resnet50 import get_resnet50
from .mobilenet_v2 import get_mobilenet_v2
from .efficientnet_b3 import get_efficientnet_b3


def get_model(model_name, num_classes=39, pretrained=True):
    """
    Factory function to create models.

    Args:
        model_name (str): Model name ('baseline', 'resnet50', 'mobilenet', 'efficientnet')
        num_classes (int): Number of classes
        pretrained (bool): Use pretrained weights (ignored for baseline)

    Returns:
        nn.Module: Model instance
    """
    model_name = model_name.lower()

    if model_name == 'baseline':
        return get_baseline_cnn(num_classes)
    elif model_name == 'resnet50':
        return get_resnet50(num_classes, pretrained)
    elif model_name == 'mobilenet':
        return get_mobilenet_v2(num_classes, pretrained)
    elif model_name == 'efficientnet':
        return get_efficientnet_b3(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
