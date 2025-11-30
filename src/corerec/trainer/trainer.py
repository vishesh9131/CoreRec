"""
Trainer module for CoreRec framework.

This module provides a trainer for training recommendation models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
from typing import Dict, List, Any, Callable, Optional, Union
from datetime import datetime

from corerec.core.base_model import BaseModel
from corerec.trainer.callbacks import Callback, EarlyStopping


class Trainer:
    """
    Trainer for recommendation models.

    This class provides utilities for training and evaluating recommendation models.

    Attributes:
        model (BaseModel): Model to train
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler
        device (torch.device): Device to train on
        callbacks (List[Callback]): List of callbacks
        checkpoint_dir (str): Directory to save checkpoints
        log_dir (str): Directory to save logs
    """

    def __init__(
        self,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callback]] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        """Initialize the trainer.

        Args:
            model (BaseModel): Model to train
            optimizer (torch.optim.Optimizer): Optimizer
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler
            device (Optional[torch.device]): Device to train on
            callbacks (Optional[List[Callback]]): List of callbacks
            checkpoint_dir (str): Directory to save checkpoints
            log_dir (str): Directory to save logs
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.callbacks = callbacks or []
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # Move model to device
        self.model.to(self.device)

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Set up logger
        self.logger = logging.getLogger("Trainer")
        file_handler = logging.FileHandler(os.path.join(log_dir, "trainer.log"))
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

        # Initialize history
        self.history = {"train_loss": [], "val_loss": [], "val_metrics": {}}

        self.logger.info(f"Trainer initialized with device: {self.device}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        validation_freq: int = 1,
        save_freq: int = 1,
        metrics: Optional[Dict[str, Callable]] = None,
    ):
        """Train the model.

        Args:
            train_loader (DataLoader): DataLoader for training data
            val_loader (Optional[DataLoader]): DataLoader for validation data
            epochs (int): Number of epochs to train
            validation_freq (int): Frequency of validation in epochs
            save_freq (int): Frequency of saving checkpoints in epochs
            metrics (Optional[Dict[str, Callable]]): Dictionary of metric functions
        """
        # Initialize metrics
        metrics = metrics or {}

        # Call on_train_begin for all callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)

        # Train loop
        for epoch in range(epochs):
            # Call on_epoch_begin for all callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)

            # Train epoch
            train_metrics = self._train_epoch(train_loader, epoch)

            # Validate if needed
            val_metrics = {}
            if val_loader is not None and (epoch + 1) % validation_freq == 0:
                val_metrics = self._validate_epoch(val_loader, epoch, metrics)

            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            if val_metrics:
                self.history["val_loss"].append(val_metrics["val_loss"])
                for metric_name, metric_value in val_metrics.items():
                    if metric_name not in self.history["val_metrics"]:
                        self.history["val_metrics"][metric_name] = []
                    self.history["val_metrics"][metric_name].append(metric_value)

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # For ReduceLROnPlateau, we need to pass a metric
                    self.scheduler.step(val_metrics.get("val_loss", train_metrics["loss"]))
                else:
                    # For other schedulers, we just step
                    self.scheduler.step()

            # Save checkpoint if needed
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(epoch)

            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Call on_epoch_end for all callbacks
            stop_training = False
            for callback in self.callbacks:
                if callback.on_epoch_end(self, epoch, train_metrics, val_metrics):
                    stop_training = True

            if stop_training:
                self.logger.info(f"Training stopped at epoch {epoch+1}")
                break

        # Call on_train_end for all callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data
            epoch (int): Current epoch

        Returns:
            Dict[str, float]: Dictionary with training metrics
        """
        self.model.train()

        # Initialize metrics
        metrics = {"loss": 0.0, "steps": 0}

        # Train loop
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Call on_batch_begin for all callbacks
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch_idx)

            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            # Train step
            result = self.model.train_step(batch, self.optimizer)

            # Update metrics
            metrics["loss"] += result["loss"]
            metrics["steps"] += 1

            # Log progress
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx+1}: loss={result['loss']:.4f}, elapsed={elapsed:.2f}s"
                )
                start_time = time.time()

            # Call on_batch_end for all callbacks
            for callback in self.callbacks:
                callback.on_batch_end(self, batch_idx, result)

        # Compute average metrics
        metrics["loss"] /= max(1, metrics["steps"])

        return metrics

    def _validate_epoch(
        self, val_loader: DataLoader, epoch: int, metrics: Dict[str, Callable]
    ) -> Dict[str, float]:
        """Validate the model for one epoch.

        Args:
            val_loader (DataLoader): DataLoader for validation data
            epoch (int): Current epoch
            metrics (Dict[str, Callable]): Dictionary of metric functions

        Returns:
            Dict[str, float]: Dictionary with validation metrics
        """
        self.model.eval()

        # Initialize metrics
        val_metrics = {"val_loss": 0.0, "steps": 0}

        # Initialize metric accumulators
        metric_values = {name: 0.0 for name in metrics}

        # Validation loop
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Validate step
                result = self.model.validate_step(batch)

                # Update metrics
                val_metrics["val_loss"] += result["val_loss"]
                val_metrics["steps"] += 1

                # Compute additional metrics
                for name, metric_fn in metrics.items():
                    if "labels" in batch and "predictions" in result:
                        metric_values[name] += metric_fn(batch["labels"], result["predictions"])

        # Compute average metrics
        val_metrics["val_loss"] /= max(1, val_metrics["steps"])

        # Add additional metrics
        for name in metrics:
            val_metrics[f"val_{name}"] = metric_values[name] / max(1, val_metrics["steps"])

        return val_metrics

    def _log_metrics(
        self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]
    ):
        """Log metrics for an epoch.

        Args:
            epoch (int): Current epoch
            train_metrics (Dict[str, float]): Dictionary with training metrics
            val_metrics (Dict[str, float]): Dictionary with validation metrics
        """
        # Create message
        message = f"Epoch {epoch+1}: train_loss={train_metrics['loss']:.4f}"

        # Add validation metrics if available
        if val_metrics:
            message += f", val_loss={val_metrics.get('val_loss', 0):.4f}"
            for name, value in val_metrics.items():
                if name not in ["val_loss", "steps"]:
                    message += f", {name}={value:.4f}"

        # Log message
        self.logger.info(message)

    def save_checkpoint(self, epoch: int, metrics: Optional[Dict[str, float]] = None):
        """Save a checkpoint of the model.

        Args:
            epoch (int): Current epoch
            metrics (Optional[Dict[str, float]]): Dictionary with metrics
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Create checkpoint path
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1}_{timestamp}.pt")

        # Create checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "timestamp": timestamp,
        }

        # Add scheduler state if available
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Add metrics if available
        if metrics is not None:
            checkpoint["metrics"] = metrics

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint of the model.

        Args:
            checkpoint_path (str): Path to checkpoint
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if available
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load history if available
        if "history" in checkpoint:
            self.history = checkpoint["history"]

        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")

        return checkpoint["epoch"]

    def predict(self, data_loader: DataLoader) -> List[Dict[str, Any]]:
        """Make predictions with the model.

        Args:
            data_loader (DataLoader): DataLoader for prediction data

        Returns:
            List[Dict[str, Any]]: List of predictions
        """
        self.model.eval()

        # Initialize predictions
        predictions = []

        # Prediction loop
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Get predictions
                outputs = self.model(batch)

                # Process predictions
                for i in range(len(outputs)):
                    prediction = {"output": outputs[i].cpu().numpy()}

                    # Add batch data if available
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor) and batch[key].size(
                            0
                        ) == outputs.size(0):
                            prediction[key] = batch[key][i].cpu().numpy()

                    predictions.append(prediction)

        return predictions
