"""
Training Callbacks

Callbacks for monitoring and controlling training process.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import Dict, Optional, Any
import torch
import logging
from pathlib import Path


class Callback:
    """
    Base callback class.

    All callbacks should inherit from this and implement hook methods.

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback.

    Stops training when validation loss stops improving.

    Example:
        early_stop = EarlyStopping(patience=5, min_delta=0.001)
        trainer = Trainer(model, callbacks=[early_stop])

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor ('val_loss', 'train_loss', etc.)
            mode: 'min' or 'max' (minimize or maximize metric)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Check if should stop training."""
        if logs is None or self.monitor not in logs:
            return

        current_value = logs[self.monitor]

        if self.mode == "min":
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta

        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best {self.monitor}: {self.best_value:.6f}")


class ModelCheckpoint(Callback):
    """
    Model checkpoint callback.

    Saves model at specified intervals or when metric improves.

    Example:
        checkpoint = ModelCheckpoint(
            filepath='models/best_model.pt',
            save_best_only=True,
            monitor='val_loss'
        )

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        filepath: str,
        save_best_only: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        save_freq: int = 1,
    ):
        """
        Initialize model checkpoint.

        Args:
            filepath: Path to save model
            save_best_only: Only save when metric improves
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_freq: Save frequency in epochs

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.save_freq = save_freq

        self.best_value = float("inf") if mode == "min" else float("-inf")

        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Save model if criteria are met."""
        if (epoch + 1) % self.save_freq != 0:
            return

        should_save = True

        if self.save_best_only and logs and self.monitor in logs:
            current_value = logs[self.monitor]

            if self.mode == "min":
                improved = current_value < self.best_value
            else:
                improved = current_value > self.best_value

            if improved:
                self.best_value = current_value
                print(f"Metric improved to {current_value:.6f}, saving model...")
            else:
                should_save = False

        if should_save:
            # Model saving will be handled by trainer
            print(f"Saved model to {self.filepath}")


class LearningRateScheduler(Callback):
    """
    Learning rate scheduling callback.

    Adjusts learning rate during training.

    Example:
        lr_scheduler = LearningRateScheduler(
            schedule=lambda epoch: 0.001 * (0.95 ** epoch)
        )

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, schedule: callable, verbose: bool = True):
        """
        Initialize LR scheduler.

        Args:
            schedule: Function that takes epoch and returns learning rate
            verbose: Whether to print LR changes

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Update learning rate."""
        new_lr = self.schedule(epoch)

        if self.verbose:
            print(f"Epoch {epoch}: Learning rate = {new_lr:.6f}")

        # Store in logs for trainer to use
        if logs is not None:
            logs["learning_rate"] = new_lr


class TensorBoardLogger(Callback):
    """
    TensorBoard logging callback.

    Logs metrics to TensorBoard for visualization.

    Example:
        tb_logger = TensorBoardLogger(log_dir='runs/experiment1')

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, log_dir: str = "runs"):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.log_dir = log_dir
        self.writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log metrics to TensorBoard."""
        if self.writer is None or logs is None:
            return

        for metric_name, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(metric_name, value, epoch)

    def on_train_end(self, logs: Optional[Dict] = None):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
