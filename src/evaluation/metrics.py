"""
Evaluation metrics for plant disease detection.
Calculates accuracy, precision, recall, F1-score, confusion matrix, etc.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from tqdm.auto import tqdm


def evaluate_model(model, dataloader, device, num_classes=39):
    """
    Evaluate model on a dataset.

    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Data loader
        device: torch.device
        num_classes (int): Number of classes

    Returns:
        dict: Dictionary containing all metrics
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # Get predictions
            _, preds = outputs.max(1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    metrics = {}

    # Overall accuracy
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    metrics['per_class_precision'] = precision
    metrics['per_class_recall'] = recall
    metrics['per_class_f1'] = f1
    metrics['per_class_support'] = support

    # Weighted metrics (accounts for class imbalance)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    metrics['weighted_precision'] = weighted_precision
    metrics['weighted_recall'] = weighted_recall
    metrics['weighted_f1'] = weighted_f1

    # Macro metrics (treats all classes equally)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    metrics['macro_precision'] = macro_precision
    metrics['macro_recall'] = macro_recall
    metrics['macro_f1'] = macro_f1

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds)

    # Classification report (string)
    metrics['classification_report'] = classification_report(
        all_labels, all_preds, zero_division=0
    )

    # ROC-AUC (one-vs-rest)
    try:
        # Convert labels to one-hot for AUC calculation
        from sklearn.preprocessing import label_binarize
        labels_onehot = label_binarize(all_labels, classes=range(num_classes))

        # Calculate AUC for each class
        auc_scores = []
        for i in range(num_classes):
            if len(np.unique(labels_onehot[:, i])) > 1:  # Need both classes present
                auc = roc_auc_score(labels_onehot[:, i], all_probs[:, i])
                auc_scores.append(auc)
            else:
                auc_scores.append(np.nan)

        metrics['per_class_auc'] = np.array(auc_scores)
        metrics['mean_auc'] = np.nanmean(auc_scores)
    except Exception as e:
        print(f"Warning: Could not calculate AUC scores: {e}")
        metrics['per_class_auc'] = None
        metrics['mean_auc'] = None

    # Store raw predictions and probabilities
    metrics['predictions'] = all_preds
    metrics['labels'] = all_labels
    metrics['probabilities'] = all_probs

    return metrics


def print_metrics(metrics, class_names=None):
    """
    Print evaluation metrics in a readable format.

    Args:
        metrics (dict): Metrics dictionary from evaluate_model
        class_names (list): List of class names (optional)
    """
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    # Overall metrics
    print(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")

    print(f"\nMacro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")

    if metrics.get('mean_auc'):
        print(f"\nMean ROC-AUC: {metrics['mean_auc']:.4f}")

    # Per-class metrics (top 5 and bottom 5)
    print("\n" + "-" * 60)
    print("Per-Class Metrics (Top 5 by F1-Score):")
    print("-" * 60)

    f1_scores = metrics['per_class_f1']
    top_5_indices = np.argsort(f1_scores)[-5:][::-1]

    for idx in top_5_indices:
        class_name = class_names[idx] if class_names else f"Class {idx}"
        print(f"{class_name:40s} | P: {metrics['per_class_precision'][idx]:.3f} "
              f"| R: {metrics['per_class_recall'][idx]:.3f} "
              f"| F1: {metrics['per_class_f1'][idx]:.3f} "
              f"| Support: {int(metrics['per_class_support'][idx])}")

    print("\n" + "-" * 60)
    print("Per-Class Metrics (Bottom 5 by F1-Score):")
    print("-" * 60)

    bottom_5_indices = np.argsort(f1_scores)[:5]

    for idx in bottom_5_indices:
        class_name = class_names[idx] if class_names else f"Class {idx}"
        print(f"{class_name:40s} | P: {metrics['per_class_precision'][idx]:.3f} "
              f"| R: {metrics['per_class_recall'][idx]:.3f} "
              f"| F1: {metrics['per_class_f1'][idx]:.3f} "
              f"| Support: {int(metrics['per_class_support'][idx])}")

    print("\n" + "=" * 60)


def calculate_inference_time(model, dataloader, device, num_samples=100):
    """
    Calculate average inference time.

    Args:
        model (nn.Module): Model
        dataloader (DataLoader): Data loader
        device: torch.device
        num_samples (int): Number of samples to test

    Returns:
        float: Average inference time in milliseconds
    """
    import time

    model.eval()
    times = []

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_samples:
                break

            images = images.to(device)

            # Measure time
            start = time.time()
            _ = model(images)
            end = time.time()

            # Time per image
            batch_time = (end - start) * 1000  # Convert to ms
            per_image_time = batch_time / images.size(0)
            times.append(per_image_time)

    avg_time = np.mean(times)
    return avg_time
