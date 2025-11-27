"""
DataLoader utilities for Plant Disease Detection
Optimized for Colab Pro memory constraints.
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def get_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
    use_weighted_sampler=False
):
    """
    Create a DataLoader with optimized settings for Colab.

    Args:
        dataset (Dataset): PyTorch Dataset instance
        batch_size (int): Batch size (default: 32 for Colab)
        shuffle (bool): Shuffle data (ignored if using weighted sampler)
        num_workers (int): Number of worker processes (default: 2 for Colab)
        pin_memory (bool): Pin memory for faster GPU transfer
        drop_last (bool): Drop last incomplete batch
        use_weighted_sampler (bool): Use weighted sampling for class imbalance

    Returns:
        DataLoader: Configured DataLoader instance
    """

    # Use weighted sampler if requested
    sampler = None
    if use_weighted_sampler:
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # Sampler and shuffle are mutually exclusive

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=True if num_workers > 0 else False
    )

    return loader


def get_train_dataloader(dataset, batch_size=32, use_weighted_sampler=False):
    """
    Get training DataLoader with appropriate settings.

    Args:
        dataset: Training dataset
        batch_size (int): Batch size
        use_weighted_sampler (bool): Use weighted sampling

    Returns:
        DataLoader: Training data loader
    """
    return get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,  # Drop last batch to avoid small batches
        use_weighted_sampler=use_weighted_sampler
    )


def get_val_dataloader(dataset, batch_size=32):
    """
    Get validation DataLoader with appropriate settings.

    Args:
        dataset: Validation dataset
        batch_size (int): Batch size

    Returns:
        DataLoader: Validation data loader
    """
    return get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        use_weighted_sampler=False
    )


def get_test_dataloader(dataset, batch_size=32):
    """
    Get test DataLoader with appropriate settings.

    Args:
        dataset: Test dataset
        batch_size (int): Batch size

    Returns:
        DataLoader: Test data loader
    """
    return get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        use_weighted_sampler=False
    )


def calculate_steps_per_epoch(dataset_size, batch_size, drop_last=True):
    """
    Calculate number of steps per epoch.

    Args:
        dataset_size (int): Number of samples in dataset
        batch_size (int): Batch size
        drop_last (bool): Whether last batch is dropped

    Returns:
        int: Steps per epoch
    """
    if drop_last:
        return dataset_size // batch_size
    else:
        return (dataset_size + batch_size - 1) // batch_size


# Memory optimization utilities
def clear_cuda_cache():
    """Clear CUDA cache to free up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_optimal_batch_size(model, device, initial_batch_size=64):
    """
    Find optimal batch size that fits in GPU memory.

    This is a simple binary search approach. Start with initial_batch_size
    and reduce until it fits in memory.

    Args:
        model: PyTorch model
        device: torch.device
        initial_batch_size (int): Starting batch size to try

    Returns:
        int: Optimal batch size
    """
    batch_size = initial_batch_size

    while batch_size > 1:
        try:
            # Try to run a dummy forward pass
            dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
            with torch.no_grad():
                _ = model(dummy_input)

            # If successful, return this batch size
            clear_cuda_cache()
            print(f"Optimal batch size found: {batch_size}")
            return batch_size

        except RuntimeError as e:
            if "out of memory" in str(e):
                # Reduce batch size and try again
                clear_cuda_cache()
                batch_size = batch_size // 2
                print(f"Reducing batch size to {batch_size}")
            else:
                raise e

    return batch_size
