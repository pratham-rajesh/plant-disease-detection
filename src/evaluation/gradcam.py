"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
For visualizing what the CNN is looking at.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


class GradCAM:
    """
    Grad-CAM implementation for CNN visualization.

    Shows which parts of the image the model focuses on for predictions.
    Useful for interpretability and debugging.
    """

    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.

        Args:
            model (nn.Module): Model to visualize
            target_layer (nn.Module): Layer to generate CAM from (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        """
        Generate Class Activation Map.

        Args:
            input_image (torch.Tensor): Input image [1, C, H, W]
            target_class (int): Target class index (if None, uses predicted class)

        Returns:
            numpy.ndarray: CAM heatmap [H, W]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_image)

        # If target class not specified, use predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Calculate weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # [H, W]

        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def visualize(
        self,
        input_image,
        cam,
        original_image=None,
        alpha=0.4,
        colormap=cv2.COLORMAP_JET
    ):
        """
        Create visualization of CAM overlaid on original image.

        Args:
            input_image (torch.Tensor): Input tensor [1, C, H, W]
            cam (numpy.ndarray): CAM heatmap [H, W]
            original_image (PIL.Image): Original image (before transforms)
            alpha (float): Transparency of overlay
            colormap: OpenCV colormap

        Returns:
            tuple: (heatmap_image, overlay_image)
        """
        # Get image dimensions
        if original_image is not None:
            img_size = original_image.size  # (W, H)
        else:
            # Use input tensor size
            img_size = (input_image.shape[3], input_image.shape[2])  # (W, H)

        # Resize CAM to match original image size
        cam_resized = cv2.resize(cam, img_size)

        # Convert CAM to heatmap
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Get original image as numpy array
        if original_image is not None:
            img_array = np.array(original_image)
        else:
            # Denormalize input tensor
            img_tensor = input_image[0].cpu()
            img_array = img_tensor.permute(1, 2, 0).numpy()
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
            img_array = (img_array * 255).astype(np.uint8)

        # Create overlay
        overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)

        return heatmap, overlay


def generate_gradcam_visualization(
    model,
    image_path,
    transform,
    target_layer,
    device,
    class_names=None,
    target_class=None,
    save_path=None
):
    """
    Generate and save Grad-CAM visualization for an image.

    Args:
        model (nn.Module): Trained model
        image_path (str): Path to image file
        transform: Image transformation pipeline
        target_layer (nn.Module): Target layer for CAM
        device: torch.device
        class_names (list): List of class names
        target_class (int): Target class (if None, uses prediction)
        save_path (str): Path to save visualization

    Returns:
        dict: Results including prediction and CAM
    """
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    image_array = np.array(original_image)

    # Apply transforms
    transformed = transform(image=image_array)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor, target_class)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_class = output.argmax(dim=1).item()
        pred_prob = probs[pred_class].item()

    # Create visualization
    heatmap, overlay = gradcam.visualize(input_tensor, cam, original_image)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Heatmap
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(overlay)
    pred_label = class_names[pred_class] if class_names else f"Class {pred_class}"
    axes[2].set_title(
        f'Prediction: {pred_label}\nConfidence: {pred_prob*100:.1f}%',
        fontsize=12,
        fontweight='bold'
    )
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Grad-CAM visualization saved to {save_path}")

    plt.close()

    return {
        'prediction': pred_class,
        'confidence': pred_prob,
        'cam': cam,
        'heatmap': heatmap,
        'overlay': overlay
    }


def get_target_layer(model, model_type='resnet50'):
    """
    Get the appropriate target layer for Grad-CAM based on model type.

    Args:
        model (nn.Module): Model
        model_type (str): Type of model

    Returns:
        nn.Module: Target layer
    """
    if model_type.lower() == 'resnet50':
        # For ResNet, use last conv layer
        return model.resnet.layer4[-1]
    elif model_type.lower() == 'mobilenet':
        # For MobileNet, use last conv layer
        return model.mobilenet.features[-1]
    elif model_type.lower() == 'efficientnet':
        # For EfficientNet, use last conv layer
        return model.efficientnet.features[-1]
    elif model_type.lower() == 'baseline':
        # For baseline CNN, use last conv block
        return model.conv4
    else:
        raise ValueError(f"Unknown model type: {model_type}")
