"""
Trainer class for model training with TensorBoard logging.
Handles the complete training loop with checkpointing and early stopping.
"""

import os
import time
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Trainer for Plant Disease Detection Models.

    Features:
        - Training and validation loops
        - TensorBoard logging
        - Model checkpointing (save best model)
        - Early stopping
        - Learning rate scheduling
        - GPU memory management
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        checkpoint_dir,
        tensorboard_dir,
        early_stopping=None,
        max_epochs=30
    ):
        """
        Initialize Trainer.

        Args:
            model (nn.Module): Model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: torch.device
            checkpoint_dir (str): Directory to save checkpoints
            tensorboard_dir (str): Directory for TensorBoard logs
            early_stopping: EarlyStopping instance (optional)
            max_epochs (int): Maximum number of epochs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping = early_stopping
        self.max_epochs = max_epochs

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

    def train_one_epoch(self, epoch):
        """
        Train for one epoch.

        Args:
            epoch (int): Current epoch number

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.max_epochs} [Train]')

        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """
        Validate the model.

        Args:
            epoch (int): Current epoch number

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.max_epochs} [Val]')

            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """
        Save model checkpoint.

        Args:
            epoch (int): Current epoch
            val_loss (float): Validation loss
            val_acc (float): Validation accuracy
            is_best (bool): Is this the best model so far?
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'training_history': self.training_history
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'✓ Best model saved (val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%)')

    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """
        Log metrics to TensorBoard.

        Args:
            epoch (int): Current epoch
            train_loss (float): Training loss
            train_acc (float): Training accuracy
            val_loss (float): Validation loss
            val_acc (float): Validation accuracy
            lr (float): Current learning rate
        """
        # Log scalars
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
        self.writer.add_scalar('Learning_Rate', lr, epoch)

        # Log to history
        self.training_history['train_loss'].append(train_loss)
        self.training_history['train_acc'].append(train_acc)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['val_acc'].append(val_acc)
        self.training_history['lr'].append(lr)

    def fit(self):
        """
        Train the model for multiple epochs.

        Returns:
            dict: Training history
        """
        print("=" * 60)
        print("TRAINING START")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate(epoch)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics
            self.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.max_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            # Update best metrics
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                is_best = True

            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, val_acc, is_best)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping
            if self.early_stopping:
                if self.early_stopping(val_loss):
                    print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
                    break

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("-" * 60)

        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Best val acc: {self.best_val_acc:.2f}%")
        print("=" * 60)

        self.writer.close()

        return self.training_history

    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.training_history = checkpoint.get('training_history', self.training_history)

        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Best val acc: {self.best_val_acc:.2f}%")
