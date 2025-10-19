"""
Parameter synchronization utilities for distributed training.

This module provides utilities for synchronizing parameters across
distributed processes during training.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any, List, Optional


class ParameterSync:
    """
    Helper class for synchronizing parameters across distributed processes.
    
    This class provides utilities for synchronizing parameters efficiently
    during distributed training, especially for models with sparse gradients
    like embedding tables.
    
    Attributes:
        model (nn.Module): Model to synchronize parameters for
        rank (int): Rank of the current process
        world_size (int): Total number of processes
        sync_interval (int): Interval for parameter synchronization
    """
    
    def __init__(self, model: nn.Module, rank: int, world_size: int, sync_interval: int = 10):
        """Initialize the parameter synchronization helper.
        
        Args:
            model (nn.Module): Model to synchronize parameters for
            rank (int): Rank of the current process
            world_size (int): Total number of processes
            sync_interval (int): Interval for parameter synchronization
        """
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.sync_interval = sync_interval
        self.step_counter = 0
        
        # Identify embedding tables with sparse gradients
        self.sparse_param_names = []
        self.sparse_params = []
        
        if dist.is_initialized():
            # For DDP, the model might be wrapped
            if hasattr(model, 'module'):
                module = model.module
            else:
                module = model
                
            for name, param in module.named_parameters():
                if param.requires_grad and getattr(param, 'sparse', False):
                    self.sparse_param_names.append(name)
                    self.sparse_params.append(param)
    
    def sync(self):
        """Synchronize parameters across processes.
        
        This method synchronizes parameters with sparse gradients
        at specified intervals to avoid performance degradation.
        """
        self.step_counter += 1
        
        # Skip if not time to sync or no sparse parameters
        if (self.step_counter % self.sync_interval != 0) or (not self.sparse_params):
            return
        
        # Synchronize sparse parameters
        for name, param in zip(self.sparse_param_names, self.sparse_params):
            # Generate a unique tag for this parameter
            tag = hash(name) % 1000000
            
            # Sync using all-reduce
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data.div_(self.world_size)


class GradientAccumulator:
    """
    Helper class for accumulating gradients across multiple forward passes.
    
    This class provides utilities for gradient accumulation, which is
    useful for training with larger effective batch sizes than what
    can fit in memory.
    
    Attributes:
        model (nn.Module): Model to accumulate gradients for
        accumulation_steps (int): Number of steps to accumulate gradients
    """
    
    def __init__(self, model: nn.Module, accumulation_steps: int = 1):
        """Initialize the gradient accumulator.
        
        Args:
            model (nn.Module): Model to accumulate gradients for
            accumulation_steps (int): Number of steps to accumulate gradients
        """
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
        # Initialize gradient accumulators
        if accumulation_steps > 1:
            self.grad_accumulators = {}
            
            # For DDP, the model might be wrapped
            if hasattr(model, 'module'):
                module = model.module
            else:
                module = model
                
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.grad_accumulators[name] = torch.zeros_like(param.data)
    
    def accumulate_gradients(self):
        """Accumulate gradients.
        
        This method accumulates gradients from the current backward pass.
        """
        self.current_step += 1
        
        # Skip if no accumulation needed
        if self.accumulation_steps <= 1:
            return
            
        # Get model parameters
        if hasattr(self.model, 'module'):
            module = self.model.module
        else:
            module = self.model
            
        # Accumulate gradients
        with torch.no_grad():
            for name, param in module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.grad_accumulators[name].add_(param.grad.data)
                    param.grad.data.zero_()
    
    def apply_accumulated_gradients(self):
        """Apply accumulated gradients.
        
        This method applies accumulated gradients to the model parameters
        after the specified number of accumulation steps.
        """
        # Skip if no accumulation needed or not at the right step
        if self.accumulation_steps <= 1 or self.current_step % self.accumulation_steps != 0:
            return
            
        # Get model parameters
        if hasattr(self.model, 'module'):
            module = self.model.module
        else:
            module = self.model
            
        # Apply accumulated gradients
        with torch.no_grad():
            for name, param in module.named_parameters():
                if param.requires_grad:
                    # Restore accumulated gradients
                    if param.grad is None:
                        param.grad = torch.empty_like(param.data)
                    param.grad.data.copy_(self.grad_accumulators[name].div_(self.accumulation_steps))
                    
                    # Reset accumulator
                    self.grad_accumulators[name].zero_()
        
        # Reset step counter
        self.current_step = 0 