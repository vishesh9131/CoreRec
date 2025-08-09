#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for CoreRec MonolithModel.

This script handles distributed training of the MonolithModel for large-scale
recommendation systems.
"""

import os
import sys
import argparse
import logging
import json
import yaml
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer

from corerec.engines.monolith.monolith_model import MonolithModel
from corerec.data.streaming_dataloader import StreamingDataLoader
from corerec.trainer.parameter_sync import ParameterSync
from corerec.utils.config import load_config
from corerec.utils.logging import setup_logging
from corerec.utils.seed import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a CoreRec MonolithModel')
    
    # Configuration
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=None, help='World size for distributed training')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12355', help='Master port for distributed training')
    
    return parser.parse_args()


def setup_distributed(args):
    """Set up distributed training environment."""
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='env://',
        world_size=args.world_size,
        rank=args.local_rank
    )
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        
    logging.info(f"Initialized distributed process: rank {args.local_rank}/{args.world_size}")


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_data_loader(config: Dict[str, Any], args, is_training: bool = True) -> StreamingDataLoader:
    """Create a data loader.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        args: Command line arguments
        is_training (bool): Whether this is a training data loader
        
    Returns:
        StreamingDataLoader: Data loader
    """
    data_config = config['data']
    
    # Get file paths
    if is_training:
        file_paths = data_config['train_files']
    else:
        file_paths = data_config.get('val_files', [])
        
    # Create data loader
    batch_size = args.batch_size or data_config.get('batch_size', 1024)
    
    loader = StreamingDataLoader(
        file_paths=file_paths,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=data_config.get('num_workers', 4),
        user_id_col=data_config.get('user_id_col', 'user_id'),
        item_id_col=data_config.get('item_id_col', 'item_id'),
        rating_col=data_config.get('rating_col', 'rating'),
        file_format=data_config.get('file_format', 'csv')
    )
    
    return loader


def create_model(config: Dict[str, Any], args) -> MonolithModel:
    """Create a model.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        args: Command line arguments
        
    Returns:
        MonolithModel: Model
    """
    model_config = config['model']
    
    # Create model
    model = MonolithModel(
        name=model_config.get('name', 'monolith_model'),
        config=model_config
    )
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Wrap in DDP if distributed
    if args.distributed:
        # Find modules using sparse gradients
        sparse_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding) and module.sparse:
                sparse_modules.append(name)
                
        # Configure find_unused_parameters to avoid deadlocks with sparse gradients
        model = DDP(
            model,
            device_ids=[args.local_rank] if torch.cuda.is_available() else None,
            output_device=args.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=len(sparse_modules) > 0,
            broadcast_buffers=False  # Important for models with large buffers
        )
    
    return model


def create_optimizer(model: nn.Module, config: Dict[str, Any], args) -> torch.optim.Optimizer:
    """Create an optimizer.
    
    Args:
        model (nn.Module): Model
        config (Dict[str, Any]): Configuration dictionary
        args: Command line arguments
        
    Returns:
        torch.optim.Optimizer: Optimizer
    """
    train_config = config['training']
    
    # Get learning rate
    lr = args.learning_rate or train_config.get('learning_rate', 0.001)
    
    # Create parameter groups with different learning rates
    # - Embedding tables with sparse gradients: lower learning rate
    # - Other parameters: normal learning rate
    embedding_params = []
    other_params = []
    
    if args.distributed:
        # For DDP, the model is wrapped
        module = model.module
    else:
        module = model
        
    for name, param in module.named_parameters():
        if 'embedding_tables' in name and getattr(param, 'sparse', False):
            embedding_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': embedding_params, 'lr': lr * 0.1, 'sparse': True},
        {'params': other_params, 'lr': lr}
    ]
    
    # Create optimizer
    optimizer_name = train_config.get('optimizer', 'adam').lower()
    
    if args.distributed and train_config.get('use_zero', False):
        # Use ZeroRedundancyOptimizer for distributed training
        if optimizer_name == 'adagrad':
            optimizer = ZeroRedundancyOptimizer(
                param_groups,
                optimizer_class=torch.optim.Adagrad,
                lr=lr
            )
        else:  # Default to Adam
            optimizer = ZeroRedundancyOptimizer(
                param_groups,
                optimizer_class=torch.optim.Adam,
                lr=lr
            )
    else:
        # Use standard optimizer
        if optimizer_name == 'adagrad':
            optimizer = torch.optim.Adagrad(param_groups, lr=lr)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=train_config.get('momentum', 0.9)
            )
        else:  # Default to Adam
            optimizer = torch.optim.Adam(param_groups, lr=lr)
    
    return optimizer


def train_epoch(
    model: nn.Module,
    data_loader: StreamingDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    param_sync: Optional[ParameterSync] = None
) -> Dict[str, float]:
    """Train the model for one epoch.
    
    Args:
        model (nn.Module): Model
        data_loader (StreamingDataLoader): Data loader
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        epoch (int): Epoch number
        args: Command line arguments
        param_sync (Optional[ParameterSync]): Parameter synchronization helper
        
    Returns:
        Dict[str, float]: Dictionary with training metrics
    """
    model.train()
    
    # Initialize metrics
    metrics = {
        'loss': 0.0,
        'steps': 0
    }
    
    # Train model
    start_time = time.time()
    
    for batch_idx, batch in enumerate(data_loader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Synchronize parameters (if distributed)
        if param_sync is not None:
            param_sync.sync()
        
        # Train step
        result = model.train_step(batch, optimizer)
        
        # Update metrics
        metrics['loss'] += result['loss']
        metrics['steps'] += 1
        
        # Log progress
        if (batch_idx + 1) % 100 == 0 and (args.local_rank == 0 or not args.distributed):
            elapsed = time.time() - start_time
            logging.info(f"Epoch {epoch}, Batch {batch_idx+1}: loss={result['loss']:.4f}, elapsed={elapsed:.2f}s")
            start_time = time.time()
    
    # Compute average metrics
    metrics['loss'] /= max(1, metrics['steps'])
    
    return metrics


def validate(
    model: nn.Module,
    data_loader: StreamingDataLoader,
    device: torch.device,
    args
) -> Dict[str, float]:
    """Validate the model.
    
    Args:
        model (nn.Module): Model
        data_loader (StreamingDataLoader): Data loader
        device (torch.device): Device to validate on
        args: Command line arguments
        
    Returns:
        Dict[str, float]: Dictionary with validation metrics
    """
    model.eval()
    
    # Initialize metrics
    metrics = {
        'val_loss': 0.0,
        'val_accuracy': 0.0,
        'steps': 0
    }
    
    # Validate model
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Validate step
            result = model.validate_step(batch)
            
            # Update metrics
            metrics['val_loss'] += result['val_loss']
            metrics['val_accuracy'] += result['val_accuracy']
            metrics['steps'] += 1
    
    # Compute average metrics
    metrics['val_loss'] /= max(1, metrics['steps'])
    metrics['val_accuracy'] /= max(1, metrics['steps'])
    
    return metrics


def save_model(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, float], args):
    """Save model checkpoint.
    
    Args:
        model (nn.Module): Model
        optimizer (torch.optim.Optimizer): Optimizer
        epoch (int): Epoch number
        metrics (Dict[str, float]): Dictionary with metrics
        args: Command line arguments
    """
    # Only save on master process
    if args.distributed and args.local_rank != 0:
        return
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create checkpoint
    if args.distributed:
        # Use model's special state dict method for distributed training
        model_state_dict = model.module.get_state_dict_for_save()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(args.model_dir, f"monolith_model_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    logging.info(f"Model saved to {checkpoint_path}")


def train(config: Dict[str, Any], args):
    """Train the model.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        args: Command line arguments
    """
    # Set up logging
    if args.distributed:
        log_file = os.path.join(args.log_dir, f"train_rank_{args.local_rank}.log")
    else:
        log_file = os.path.join(args.log_dir, "train.log")
    setup_logging(log_file)
    
    # Set seed for reproducibility
    set_seed(args.seed + args.local_rank if args.distributed else args.seed)
    
    # Create data loaders
    train_loader = create_data_loader(config, args, is_training=True)
    val_loader = create_data_loader(config, args, is_training=False)
    
    # Create model
    model = create_model(config, args)
    
    # Create optimizer
    optimizer = create_optimizer(model, config, args)
    
    # Create parameter synchronization helper (if distributed)
    param_sync = None
    if args.distributed:
        param_sync = ParameterSync(model, args.local_rank, args.world_size)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train model
    num_epochs = args.epochs or config['training'].get('epochs', 10)
    
    for epoch in range(num_epochs):
        # Train epoch
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, args, param_sync)
        
        # Validate
        val_metrics = validate(model, val_loader, device, args)
        
        # Log metrics
        metrics = {**train_metrics, **val_metrics}
        if args.local_rank == 0 or not args.distributed:
            logging.info(f"Epoch {epoch} metrics: {metrics}")
        
        # Save model
        if (epoch + 1) % config['training'].get('save_interval', 1) == 0:
            save_model(model, optimizer, epoch, metrics, args)
    
    # Final save
    save_model(model, optimizer, num_epochs, metrics, args)


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up distributed training
    if args.distributed:
        if args.world_size is None:
            args.world_size = int(os.environ.get('WORLD_SIZE', 1))
        setup_distributed(args)
    
    try:
        # Train model
        train(config, args)
    finally:
        # Clean up
        if args.distributed:
            cleanup_distributed()


if __name__ == '__main__':
    main() 