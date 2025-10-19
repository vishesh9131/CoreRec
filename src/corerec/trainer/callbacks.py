"""
Callbacks for CoreRec trainer.

This module provides callback classes for the trainer,
enabling customization of the training process.
"""

import os
import torch
import numpy as np
import time
import logging
import mlflow
import wandb
from typing import Dict, Any, Optional, List, Union, Callable
from abc import ABC, abstractmethod


class Callback(ABC):
    """
    Base class for all callbacks.
    
    Callbacks provide a way to customize the training process by
    executing code at various points during training.
    """
    
    def on_train_begin(self, trainer):
        """Called at the beginning of training.
        
        Args:
            trainer: Trainer instance
        """
        pass
    
    def on_train_end(self, trainer):
        """Called at the end of training.
        
        Args:
            trainer: Trainer instance
        """
        pass
    
    def on_epoch_begin(self, trainer, epoch):
        """Called at the beginning of an epoch.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch
        """
        pass
    
    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        """Called at the end of an epoch.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch
            train_metrics: Dictionary with training metrics
            val_metrics: Dictionary with validation metrics
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        return False
    
    def on_batch_begin(self, trainer, batch_idx):
        """Called at the beginning of a batch.
        
        Args:
            trainer: Trainer instance
            batch_idx: Current batch index
        """
        pass
    
    def on_batch_end(self, trainer, batch_idx, logs):
        """Called at the end of a batch.
        
        Args:
            trainer: Trainer instance
            batch_idx: Current batch index
            logs: Dictionary with batch metrics
        """
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback.
    
    This callback stops training when a monitored metric has stopped improving.
    
    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement
        monitor (str): Metric to monitor
        mode (str): 'min' or 'max' depending on whether the monitored metric should be minimized or maximized
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, monitor: str = 'val_loss', mode: str = 'min'):
        """Initialize the early stopping callback.
        
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement
            monitor (str): Metric to monitor
            mode (str): 'min' or 'max' depending on whether the monitored metric should be minimized or maximized
        """
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        # Initialize variables
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.best_epoch = 0
        
        # Set comparison function
        self.is_better = self._is_less if mode == 'min' else self._is_greater
    
    def _is_less(self, current: float, best: float) -> bool:
        """Check if current value is less than best value by at least min_delta.
        
        Args:
            current (float): Current value
            best (float): Best value
            
        Returns:
            bool: True if current < best - min_delta, False otherwise
        """
        return current < best - self.min_delta
    
    def _is_greater(self, current: float, best: float) -> bool:
        """Check if current value is greater than best value by at least min_delta.
        
        Args:
            current (float): Current value
            best (float): Best value
            
        Returns:
            bool: True if current > best + min_delta, False otherwise
        """
        return current > best + self.min_delta
    
    def on_train_begin(self, trainer):
        """Reset variables at the beginning of training.
        
        Args:
            trainer: Trainer instance
        """
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float('inf') if self.mode == 'min' else -float('inf')
        self.best_epoch = 0
    
    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        """Check if training should stop at the end of an epoch.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch
            train_metrics: Dictionary with training metrics
            val_metrics: Dictionary with validation metrics
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        # Get monitored metric
        if self.monitor.startswith('val_') and not val_metrics:
            # No validation metrics available
            return False
        
        if self.monitor.startswith('val_'):
            current = val_metrics.get(self.monitor, None)
        else:
            current = train_metrics.get(self.monitor, None)
        
        if current is None:
            # Monitored metric not available
            return False
        
        # Check if current value is better than best value
        if self.is_better(current, self.best_value):
            # Reset wait counter
            self.wait = 0
            self.best_value = current
            self.best_epoch = epoch
        else:
            # Increment wait counter
            self.wait += 1
            if self.wait >= self.patience:
                # Stop training
                self.stopped_epoch = epoch
                trainer.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                trainer.logger.info(f"Best epoch: {self.best_epoch + 1}, best {self.monitor}: {self.best_value:.4f}")
                return True
        
        return False


class ModelCheckpoint(Callback):
    """
    Model checkpoint callback.
    
    This callback saves the model after every epoch.
    
    Attributes:
        filepath (str): Path to save the model
        monitor (str): Metric to monitor
        mode (str): 'min' or 'max' depending on whether the monitored metric should be minimized or maximized
        save_best_only (bool): Whether to save only the best model
        save_weights_only (bool): Whether to save only the model weights
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = False,
        save_weights_only: bool = False,
        save_freq: int = 1,
        verbose: int = 0
    ):
        """Initialize the model checkpoint callback.
        
        Args:
            filepath (str): Path to save the model
            monitor (str): Metric to monitor
            mode (str): 'min' or 'max' depending on whether the monitored metric should be minimized or maximized
            save_best_only (bool): Whether to save only the best model
            save_weights_only (bool): Whether to save only the model weights
            save_freq (int): Frequency of saving checkpoints in epochs
            verbose (int): Verbosity level
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.verbose = verbose
        
        # Set comparison function
        self.is_better = self._is_less if mode == 'min' else self._is_greater
        self.best_value = float('inf') if mode == 'min' else -float('inf')
    
    def _is_less(self, current: float, best: float) -> bool:
        """Check if current value is less than best value.
        
        Args:
            current (float): Current value
            best (float): Best value
            
        Returns:
            bool: True if current < best, False otherwise
        """
        return current < best
    
    def _is_greater(self, current: float, best: float) -> bool:
        """Check if current value is greater than best value.
        
        Args:
            current (float): Current value
            best (float): Best value
            
        Returns:
            bool: True if current > best, False otherwise
        """
        return current > best
    
    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        """Save the model at the end of an epoch.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch
            train_metrics: Dictionary with training metrics
            val_metrics: Dictionary with validation metrics
            
        Returns:
            bool: False (never stops training)
        """
        # Check if it's time to save
        if (epoch + 1) % self.save_freq != 0:
            return False
        
        # Get monitored metric
        if self.monitor.startswith('val_') and not val_metrics:
            # No validation metrics available
            return False
        
        if self.monitor.startswith('val_'):
            current = val_metrics.get(self.monitor, None)
        else:
            current = train_metrics.get(self.monitor, None)
        
        if current is None:
            # Monitored metric not available
            return False
        
        # Check if current value is better than best value
        if not self.save_best_only or self.is_better(current, self.best_value):
            # Update best value
            self.best_value = current
            
            # Create filepath
            filepath = self.filepath.format(epoch=epoch + 1, **train_metrics, **val_metrics)
            
            # Save model
            if self.save_weights_only:
                torch.save(trainer.model.state_dict(), filepath)
            else:
                trainer.save_checkpoint(epoch, val_metrics)
            
            if self.verbose > 0:
                trainer.logger.info(f"Model saved to {filepath}")
        
        return False


class MLflowLogger(Callback):
    """
    MLflow logger callback.
    
    This callback logs metrics and parameters to MLflow.
    
    Attributes:
        experiment_name (str): Name of the MLflow experiment
        run_name (str): Name of the MLflow run
        log_model (bool): Whether to log the model
    """
    
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        log_model: bool = False,
        log_artifacts: bool = False,
        artifact_paths: Optional[List[str]] = None,
        tracking_uri: Optional[str] = None,
        registered_model_name: Optional[str] = None
    ):
        """Initialize the MLflow logger callback.
        
        Args:
            experiment_name (Optional[str]): Name of the MLflow experiment
            run_name (Optional[str]): Name of the MLflow run
            log_model (bool): Whether to log the model
            log_artifacts (bool): Whether to log artifacts
            artifact_paths (Optional[List[str]]): List of artifact paths
            tracking_uri (Optional[str]): MLflow tracking URI
            registered_model_name (Optional[str]): Name of the registered model
        """
        super().__init__()
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_model = log_model
        self.log_artifacts = log_artifacts
        self.artifact_paths = artifact_paths or []
        self.tracking_uri = tracking_uri
        self.registered_model_name = registered_model_name
    
    def on_train_begin(self, trainer):
        """Set up MLflow at the beginning of training.
        
        Args:
            trainer: Trainer instance
        """
        # Set up MLflow
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set experiment
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)
        
        # Start run
        mlflow.start_run(run_name=self.run_name)
        
        # Log model parameters
        params = {}
        for name, module in trainer.model.named_modules():
            if name:
                params[f"model.{name}.num_parameters"] = sum(p.numel() for p in module.parameters())
        
        # Log parameters
        mlflow.log_params(params)
    
    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        """Log metrics at the end of an epoch.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch
            train_metrics: Dictionary with training metrics
            val_metrics: Dictionary with validation metrics
            
        Returns:
            bool: False (never stops training)
        """
        # Log metrics
        metrics = {**train_metrics}
        if val_metrics:
            metrics.update(val_metrics)
        
        # Add epoch to metrics
        metrics['epoch'] = epoch
        
        # Log metrics
        mlflow.log_metrics(metrics, step=epoch)
        
        return False
    
    def on_train_end(self, trainer):
        """Clean up MLflow at the end of training.
        
        Args:
            trainer: Trainer instance
        """
        # Log model
        if self.log_model:
            mlflow.pytorch.log_model(
                trainer.model,
                "model",
                registered_model_name=self.registered_model_name
            )
        
        # Log artifacts
        if self.log_artifacts:
            for path in self.artifact_paths:
                mlflow.log_artifact(path)
        
        # End run
        mlflow.end_run()


class WandbLogger(Callback):
    """
    Weights & Biases logger callback.
    
    This callback logs metrics and parameters to Weights & Biases.
    
    Attributes:
        project (str): Name of the W&B project
        name (str): Name of the W&B run
        config (Dict[str, Any]): Configuration to log
        log_model (bool): Whether to log the model
    """
    
    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_model: bool = False,
        log_code: bool = False,
        log_freq: int = 1,
        save_code: bool = False
    ):
        """Initialize the W&B logger callback.
        
        Args:
            project (Optional[str]): Name of the W&B project
            name (Optional[str]): Name of the W&B run
            config (Optional[Dict[str, Any]]): Configuration to log
            log_model (bool): Whether to log the model
            log_code (bool): Whether to log code
            log_freq (int): Frequency of logging in batches
            save_code (bool): Whether to save code
        """
        super().__init__()
        self.project = project
        self.name = name
        self.config = config or {}
        self.log_model = log_model
        self.log_code = log_code
        self.log_freq = log_freq
        self.save_code = save_code
        self.step = 0
    
    def on_train_begin(self, trainer):
        """Set up W&B at the beginning of training.
        
        Args:
            trainer: Trainer instance
        """
        # Initialize W&B
        wandb.init(
            project=self.project,
            name=self.name,
            config=self.config,
            save_code=self.save_code
        )
        
        # Watch model
        if self.log_model:
            wandb.watch(trainer.model)
    
    def on_batch_end(self, trainer, batch_idx, logs):
        """Log metrics at the end of a batch.
        
        Args:
            trainer: Trainer instance
            batch_idx: Current batch index
            logs: Dictionary with batch metrics
        """
        # Increment step
        self.step += 1
        
        # Log metrics
        if self.step % self.log_freq == 0:
            wandb.log(logs, step=self.step)
    
    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        """Log metrics at the end of an epoch.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch
            train_metrics: Dictionary with training metrics
            val_metrics: Dictionary with validation metrics
            
        Returns:
            bool: False (never stops training)
        """
        # Log metrics
        metrics = {**train_metrics}
        if val_metrics:
            metrics.update(val_metrics)
        
        # Add epoch to metrics
        metrics['epoch'] = epoch
        
        # Log metrics
        wandb.log(metrics, step=self.step)
        
        return False
    
    def on_train_end(self, trainer):
        """Clean up W&B at the end of training.
        
        Args:
            trainer: Trainer instance
        """
        # Finish W&B run
        wandb.finish() 