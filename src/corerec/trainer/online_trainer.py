"""
Online trainer for continual learning in recommendation systems.

This module provides an online trainer for continual learning in
recommendation systems, enabling models to be updated with new data
while in production.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Tuple, Any, Union, Optional
import logging
import time
import os
import threading
import queue
import numpy as np
import pandas as pd
from datetime import datetime

from corerec.core.base_model import BaseModel
from corerec.data.streaming_dataloader import StreamingDataLoader
from corerec.trainer.parameter_sync import ParameterSync, GradientAccumulator


class OnlineTrainer:
    """
    Online trainer for continual learning in recommendation systems.
    
    This class provides utilities for online training of recommendation models,
    enabling them to be updated with new data while in production.
    
    Attributes:
        model (BaseModel): Model to train
        optimizer (torch.optim.Optimizer): Optimizer
        data_queue (queue.Queue): Queue for receiving new training data
        device (torch.device): Device to train on
        checkpoint_dir (str): Directory to save checkpoints
        log_dir (str): Directory to save logs
    """
    
    def __init__(
        self,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs'
    ):
        """Initialize the online trainer.
        
        Args:
            model (BaseModel): Model to train
            optimizer (torch.optim.Optimizer): Optimizer
            device (torch.device): Device to train on
            config (Dict[str, Any]): Configuration
            checkpoint_dir (str): Directory to save checkpoints
            log_dir (str): Directory to save logs
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger('OnlineTrainer')
        file_handler = logging.FileHandler(os.path.join(log_dir, 'online_trainer.log'))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        
        # Set up data queue for receiving new data
        self.data_queue = queue.Queue()
        
        # Set up training thread
        self.training_thread = None
        self.stop_training = threading.Event()
        
        # Set up gradient accumulator
        self.gradient_accumulator = GradientAccumulator(
            model=model,
            accumulation_steps=config.get('accumulation_steps', 1)
        )
        
        # Set up parameter sync if distributed
        self.param_sync = None
        if dist.is_initialized():
            self.param_sync = ParameterSync(
                model=model,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                sync_interval=config.get('sync_interval', 10)
            )
        
        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'train_steps': 0,
            'updates': 0,
            'last_update_time': time.time()
        }
        
        self.logger.info("Online trainer initialized")
    
    def add_training_data(self, data: Dict[str, torch.Tensor]):
        """Add new training data to the queue.
        
        Args:
            data (Dict[str, torch.Tensor]): Batch of training data
        """
        self.data_queue.put(data)
    
    def add_training_data_from_dataframe(self, df: pd.DataFrame):
        """Add new training data from a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with training data
        """
        # Convert DataFrame to tensors
        batch = {}
        for col in df.columns:
            if col in ['user_id', 'item_id']:
                # Convert IDs to long tensors
                batch[col] = torch.tensor(df[col].values, dtype=torch.long)
            elif col == 'rating' or col == 'label':
                # Convert ratings/labels to float tensors
                batch[col] = torch.tensor(df[col].values, dtype=torch.float32)
            else:
                # Other columns as float tensors
                try:
                    batch[col] = torch.tensor(df[col].values, dtype=torch.float32)
                except:
                    # Skip columns that can't be converted to tensors
                    pass
        
        # Add to queue
        self.add_training_data(batch)
    
    def start_training(self):
        """Start the online training process in a separate thread."""
        if self.training_thread is not None and self.training_thread.is_alive():
            self.logger.warning("Training thread is already running")
            return
        
        # Reset stop flag
        self.stop_training.clear()
        
        # Create training thread
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        self.logger.info("Online training started")
    
    def stop_training_thread(self):
        """Stop the online training process."""
        if self.training_thread is None or not self.training_thread.is_alive():
            self.logger.warning("Training thread is not running")
            return
        
        # Set stop flag
        self.stop_training.set()
        
        # Wait for thread to finish
        self.training_thread.join()
        
        self.logger.info("Online training stopped")
    
    def _training_loop(self):
        """Training loop for online training."""
        self.model.train()
        
        # Training parameters
        max_queue_size = self.config.get('max_queue_size', 1000)
        min_batch_size = self.config.get('min_batch_size', 32)
        max_wait_time = self.config.get('max_wait_time', 60)  # seconds
        checkpoint_interval = self.config.get('checkpoint_interval', 1000)  # updates
        
        while not self.stop_training.is_set():
            # Wait for enough data
            start_wait_time = time.time()
            
            while (self.data_queue.qsize() < min_batch_size and 
                  time.time() - start_wait_time < max_wait_time and
                  not self.stop_training.is_set()):
                time.sleep(0.1)
            
            # Check if we should stop
            if self.stop_training.is_set():
                break
            
            # Process available data
            num_samples = min(self.data_queue.qsize(), max_queue_size)
            
            if num_samples == 0:
                # No data available
                continue
            
            # Collect samples
            samples = []
            for _ in range(num_samples):
                try:
                    samples.append(self.data_queue.get_nowait())
                except queue.Empty:
                    break
            
            # Process samples
            for batch in samples:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Synchronize parameters (if distributed)
                if self.param_sync is not None:
                    self.param_sync.sync()
                
                # Train step
                result = self.model.train_step(batch, self.optimizer)
                
                # Accumulate gradients
                self.gradient_accumulator.accumulate_gradients()
                
                # Apply accumulated gradients
                self.gradient_accumulator.apply_accumulated_gradients()
                
                # Update metrics
                self.metrics['train_loss'].append(result['loss'])
                self.metrics['train_steps'] += 1
                
                # Keep only recent losses
                if len(self.metrics['train_loss']) > 100:
                    self.metrics['train_loss'] = self.metrics['train_loss'][-100:]
            
            # Update counter
            self.metrics['updates'] += 1
            self.metrics['last_update_time'] = time.time()
            
            # Log progress
            if self.metrics['updates'] % 10 == 0:
                avg_loss = np.mean(self.metrics['train_loss'])
                self.logger.info(f"Update {self.metrics['updates']}: avg_loss={avg_loss:.4f}, queue_size={self.data_queue.qsize()}")
            
            # Save checkpoint
            if self.metrics['updates'] % checkpoint_interval == 0:
                self.save_checkpoint()
    
    def save_checkpoint(self):
        """Save a checkpoint of the model."""
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create checkpoint path
        checkpoint_path = os.path.join(self.checkpoint_dir, f"online_model_{timestamp}.pt")
        
        # Create checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'timestamp': timestamp
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics.
        
        Returns:
            Dict[str, Any]: Dictionary with metrics
        """
        return {
            'avg_loss': np.mean(self.metrics['train_loss']) if self.metrics['train_loss'] else float('nan'),
            'updates': self.metrics['updates'],
            'train_steps': self.metrics['train_steps'],
            'last_update_time': self.metrics['last_update_time'],
            'queue_size': self.data_queue.qsize()
        } 