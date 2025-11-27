"""
Visualization utilities for evaluation results.
Creates all required plots for the 20% visualization requirement.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history (dict): Training history with keys: train_loss, val_loss, train_acc, val_acc
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")

    plt.show()


def plot_confusion_matrix(
    cm,
    class_names,
    normalize=True,
    save_path=None,
    figsize=(16, 14),
    cmap='Blues'
):
    """
    Plot confusion matrix heatmap.

    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list): List of class names
        normalize (bool): Normalize by row (true labels)
        save_path (str): Path to save figure
        figsize (tuple): Figure size
        cmap (str): Colormap
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix (Counts)'

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations (only for small matrices or show top values)
    if len(class_names) <= 10:  # Only annotate if not too many classes
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8
            )

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")

    plt.show()


def plot_per_class_metrics(
    metrics,
    class_names,
    metric_name='f1',
    save_path=None,
    top_n=None
):
    """
    Plot per-class performance metrics as horizontal bar chart.

    Args:
        metrics (numpy.ndarray): Metric values for each class
        class_names (list): List of class names
        metric_name (str): Name of metric (for title)
        save_path (str): Path to save figure
        top_n (int): Show only top N classes (if None, show all)
    """
    # Sort by metric value
    sorted_indices = np.argsort(metrics)

    if top_n:
        sorted_indices = sorted_indices[-top_n:]

    sorted_metrics = metrics[sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, len(sorted_indices) * 0.4))

    # Color code by performance
    colors = []
    for val in sorted_metrics:
        if val >= 0.9:
            colors.append('green')
        elif val >= 0.7:
            colors.append('orange')
        else:
            colors.append('red')

    y_pos = np.arange(len(sorted_names))
    bars = ax.barh(y_pos, sorted_metrics, color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel(metric_name.upper(), fontsize=12, fontweight='bold')
    ax.set_title(
        f'Per-Class {metric_name.upper()} Scores',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlim([0, 1.0])
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_metrics)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class {metric_name} plot saved to {save_path}")

    plt.show()


def plot_roc_curves(metrics, class_names, save_path=None, top_n=10):
    """
    Plot ROC curves for top N classes by AUC.

    Args:
        metrics (dict): Metrics dictionary with per_class_auc and probabilities
        class_names (list): List of class names
        save_path (str): Path to save figure
        top_n (int): Number of top classes to plot
    """
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

    if metrics.get('per_class_auc') is None:
        print("Warning: AUC scores not available")
        return

    # Get top N classes by AUC
    auc_scores = metrics['per_class_auc']
    valid_indices = ~np.isnan(auc_scores)
    top_indices = np.argsort(auc_scores[valid_indices])[-top_n:]

    # Convert labels to one-hot
    num_classes = len(class_names)
    labels_onehot = label_binarize(
        metrics['labels'],
        classes=range(num_classes)
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.rainbow(np.linspace(0, 1, top_n))

    for i, (idx, color) in enumerate(zip(top_indices, colors)):
        fpr, tpr, _ = roc_curve(labels_onehot[:, idx], metrics['probabilities'][:, idx])
        auc = auc_scores[idx]

        ax.plot(
            fpr, tpr,
            color=color,
            lw=2,
            label=f'{class_names[idx][:30]} (AUC = {auc:.3f})'
        )

    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')

    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'ROC Curves - Top {top_n} Classes by AUC', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curves saved to {save_path}")

    plt.show()


def plot_model_comparison(results_dict, save_path=None):
    """
    Compare multiple models across different metrics.

    Args:
        results_dict (dict): {model_name: metrics_dict}
        save_path (str): Path to save figure
    """
    models = list(results_dict.keys())
    metrics_to_plot = ['accuracy', 'weighted_f1', 'mean_auc']
    metric_names = ['Accuracy', 'Weighted F1', 'Mean AUC']

    # Extract values
    values = []
    for metric in metrics_to_plot:
        metric_values = [results_dict[model].get(metric, 0) for model in models]
        values.append(metric_values)

    # Plot
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (vals, name, color) in enumerate(zip(values, metric_names, colors)):
        offset = width * (i - 1)
        ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.8)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison Across Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison plot saved to {save_path}")

    plt.show()
