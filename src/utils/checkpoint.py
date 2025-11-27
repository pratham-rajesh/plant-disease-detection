"""
Checkpoint management utilities.
"""

import os
import torch
import glob


def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth'):
    """
    Save checkpoint.

    Args:
        state (dict): State dictionary
        checkpoint_dir (str): Directory to save checkpoint
        filename (str): Checkpoint filename
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load checkpoint.

    Args:
        checkpoint_path (str): Path to checkpoint
        model (nn.Module): Model to load weights into
        optimizer: Optimizer to load state (optional)
        scheduler: Scheduler to load state (optional)
        device (str): Device to map tensors to

    Returns:
        dict: Checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def get_latest_checkpoint(checkpoint_dir):
    """
    Get path to latest checkpoint.

    Args:
        checkpoint_dir (str): Directory containing checkpoints

    Returns:
        str: Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))

    if not checkpoints:
        return None

    # Sort by modification time
    latest = max(checkpoints, key=os.path.getmtime)
    return latest


def delete_old_checkpoints(checkpoint_dir, keep_n=3):
    """
    Delete old checkpoints, keeping only the n most recent.

    Args:
        checkpoint_dir (str): Directory containing checkpoints
        keep_n (int): Number of checkpoints to keep
    """
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))

    if len(checkpoints) <= keep_n:
        return

    # Sort by modification time
    checkpoints.sort(key=os.path.getmtime, reverse=True)

    # Delete old checkpoints
    for checkpoint in checkpoints[keep_n:]:
        os.remove(checkpoint)
        print(f"Deleted old checkpoint: {checkpoint}")
