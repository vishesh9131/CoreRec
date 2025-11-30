"""
Training utilities for CoreRec models.

Provides early stopping, checkpointing, and learning rate scheduling
to improve training efficiency and model quality.

Author: CoreRec Team
"""

import os
import logging
import pickle
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import numpy as np


logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.

    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in monitored metric to qualify as improvement
        mode: 'min' for metrics that should be minimized (loss), 'max' for metrics
              that should be maximized (accuracy, AUC)
        verbose: Whether to print early stopping messages
        restore_best_weights: Whether to restore model weights from epoch with best metric

    Example:
        early_stop = EarlyStopping(patience=5, min_delta=0.001, mode='min')

        for epoch in range(epochs):
            train_loss = train_epoch()
            val_loss = validate()

            if early_stop(val_loss, model):
                print(f"Early stopping triggered at epoch {epoch}")
                break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        if mode == "min":
            self.min_delta *= -1

    def __call__(self, metric: float, model: Optional[Any] = None) -> bool:
        """
        Check if training should stop.

        Args:
            metric: Current validation metric value
            model: Model object (optional, for saving best weights)

        Returns:
            True if training should stop, False otherwise
        """
        score = -metric if self.mode == "min" else metric

        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best_weights:
                self.best_weights = self._save_weights(model)
        elif self.monitor_op(score - self.min_delta, self.best_score):
            # Improvement detected
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = self._save_weights(model)
            if self.verbose:
                logger.info(f"Validation metric improved to {metric:.6f}")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"No improvement for {self.counter}/{self.patience} epochs. "
                    f"Best: {self.best_score:.6f}"
                )

            if self.counter >= self.patience:
                self.early_stop = True
                if (
                    self.restore_best_weights
                    and model is not None
                    and self.best_weights is not None
                ):
                    self._restore_weights(model, self.best_weights)
                    if self.verbose:
                        logger.info("Restored model weights from best epoch")
                return True

        return False

    def _save_weights(self, model: Any) -> Dict:
        """Save model weights to dictionary."""
        try:
            # Try PyTorch models
            import torch

            if hasattr(model, "state_dict"):
                return {k: v.clone().cpu()
                        for k, v in model.state_dict().items()}
        except ImportError:
            pass

        # Fallback: try to pickle the model
        try:
            return pickle.dumps(model)
        except Exception as e:
            logger.warning(f"Could not save model weights: {e}")
            return None

    def _restore_weights(self, model: Any, weights: Dict):
        """Restore model weights from dictionary."""
        try:
            # Try PyTorch models
            import torch

            if hasattr(model, "load_state_dict") and isinstance(weights, dict):
                # Move weights to model's device
                device = next(
                    model.parameters()).device if hasattr(
                    model, "parameters") else "cpu"
                weights_on_device = {k: v.to(device)
                                     for k, v in weights.items()}
                model.load_state_dict(weights_on_device)
                return
        except Exception:
            pass

        # Fallback: try to unpickle
        try:
            if isinstance(weights, bytes):
                restored_model = pickle.loads(weights)
                model.__dict__.update(restored_model.__dict__)
        except Exception as e:
            logger.warning(f"Could not restore model weights: {e}")


class ModelCheckpoint:
    """
    Save model checkpoints during training.

    Args:
        filepath: Path to save checkpoint (can include format strings like
                  '{epoch}' and '{metric:.4f}')
        monitor: Metric name to monitor
        save_best_only: If True, only save when monitored metric improves
        mode: 'min' or 'max'
        save_weights_only: If True, only save model weights not full model
        verbose: Whether to print checkpoint save messages

    Example:
        checkpoint = ModelCheckpoint(
            filepath='checkpoints/model_epoch{epoch}_loss{val_loss:.4f}.pkl',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )

        for epoch in range(epochs):
            train_loss = train_epoch()
            val_loss = validate()
            checkpoint(epoch, {'val_loss': val_loss}, model)
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = False,
        mode: str = "min",
        save_weights_only: bool = False,
        verbose: bool = True,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        self.best = None
        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def __call__(self, epoch: int, metrics: Dict[str, float], model: Any):
        """
        Save checkpoint if conditions are met.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric values
            model: Model to save
        """
        metric_value = metrics.get(self.monitor)

        if metric_value is None:
            logger.warning(
                f"Metric '{
                    self.monitor}' not found in metrics dict")
            return

        # Create directory if it doesn't exist
        filepath = self.filepath.format(epoch=epoch, **metrics)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Check if we should save
        should_save = False
        if not self.save_best_only:
            should_save = True
        elif self.best is None:
            should_save = True
            self.best = metric_value
        elif self.monitor_op(metric_value, self.best):
            should_save = True
            self.best = metric_value

        if should_save:
            try:
                if self.save_weights_only:
                    # Save only weights
                    if hasattr(model, "state_dict"):
                        import torch

                        torch.save(model.state_dict(), filepath)
                    else:
                        with open(filepath, "wb") as f:
                            pickle.dump(model, f)
                else:
                    # Save full model
                    if hasattr(model, "save"):
                        model.save(filepath)
                    else:
                        with open(filepath, "wb") as f:
                            pickle.dump(model, f)

                if self.verbose:
                    logger.info(f"Saved checkpoint to {filepath}")
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")


class LearningRateScheduler:
    """
    Adjust learning rate during training.

    Args:
        optimizer: PyTorch optimizer
        mode: 'step', 'exponential', or 'reduce_on_plateau'
        step_size: For 'step' mode, number of epochs between LR reductions
        gamma: Multiplicative factor for LR reduction
        patience: For 'reduce_on_plateau', number of epochs to wait
        min_lr: Minimum learning rate
        verbose: Whether to print LR changes

    Example:
        scheduler = LearningRateScheduler(optimizer, mode='reduce_on_plateau', patience=3)

        for epoch in range(epochs):
            train()
            val_loss = validate()
            scheduler.step(val_loss)
    """

    def __init__(
        self,
        optimizer: Any,
        mode: str = "step",
        step_size: int = 10,
        gamma: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-7,
        verbose: bool = True,
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.step_size = step_size
        self.gamma = gamma
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose

        self.current_epoch = 0
        self.best_metric = None
        self.epochs_since_improvement = 0

    def step(self, metric: Optional[float] = None):
        """
        Update learning rate based on epoch or metric.

        Args:
            metric: Current metric value (required for 'reduce_on_plateau')
        """
        self.current_epoch += 1

        if self.mode == "step":
            if self.current_epoch % self.step_size == 0:
                self._reduce_lr()
        elif self.mode == "exponential":
            self._reduce_lr()
        elif self.mode == "reduce_on_plateau":
            if metric is None:
                raise ValueError(
                    "metric required for 'reduce_on_plateau' mode")

            if self.best_metric is None or metric < self.best_metric:
                self.best_metric = metric
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1

                if self.epochs_since_improvement >= self.patience:
                    self._reduce_lr()
                    self.epochs_since_improvement = 0

    def _reduce_lr(self):
        """Reduce learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group["lr"]
            new_lr = max(old_lr * self.gamma, self.min_lr)
            param_group["lr"] = new_lr

            if self.verbose and new_lr != old_lr:
                logger.info(
                    f"Reduced learning rate: {old_lr:.2e} -> {new_lr:.2e}")


def get_optimizer(model_parameters, optimizer_name: str = "adam", **kwargs):
    """
    Get optimizer instance by name.

    Args:
        model_parameters: Model parameters to optimize
        optimizer_name: Name of optimizer ('adam', 'sgd', 'rmsprop', etc.)
        **kwargs: Optimizer-specific arguments (lr, weight_decay, etc.)

    Returns:
        Optimizer instance

    Example:
        optimizer = get_optimizer(model.parameters(), 'adam', lr=0.001, weight_decay=1e-5)
    """
    try:
        import torch.optim as optim
    except ImportError:
        raise ImportError("PyTorch is required for optimizers")

    optimizer_name = optimizer_name.lower()

    optimizers = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
        "adagrad": optim.Adagrad,
        "adadelta": optim.Adadelta,
        "adamw": optim.AdamW,
    }

    if optimizer_name not in optimizers:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. " f"Available: {
                list(
                    optimizers.keys())}")

    return optimizers[optimizer_name](model_parameters, **kwargs)
