"""
Early Stopping implementation to prevent overfitting.
"""

import numpy as np


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.

    Args:
        patience (int): How many epochs to wait after last improvement
        min_delta (float): Minimum change to qualify as improvement
        mode (str): 'min' for loss, 'max' for accuracy
    """

    def __init__(self, patience=7, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf if mode == 'min' else -np.Inf

    def __call__(self, val_metric):
        """
        Check if should stop training.

        Args:
            val_metric (float): Validation metric (loss or accuracy)

        Returns:
            bool: True if should stop training
        """
        if self.mode == 'min':
            score = -val_metric
        else:
            score = val_metric

        if self.best_score is None:
            self.best_score = score
            return False

        # Check if improved
        if score < self.best_score + self.min_delta:
            # No improvement
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            # Improved
            self.best_score = score
            self.counter = 0
            return False

        return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
