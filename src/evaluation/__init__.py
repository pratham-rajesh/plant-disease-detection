"""Evaluation modules."""

from .metrics import evaluate_model, print_metrics, calculate_inference_time
from .gradcam import GradCAM, generate_gradcam_visualization, get_target_layer
from .visualizations import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_roc_curves,
    plot_model_comparison
)
