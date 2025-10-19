"""
Model Interface for PyTorch-based Models

Provides interface for models that are pure PyTorch nn.Module instances.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Any


class ModelInterface(nn.Module, ABC):
    """
    Interface for PyTorch-based recommendation models.
    
    Models like DeepFM, AFM, AutoInt that are pure nn.Module should implement this.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass - must be implemented by PyTorch models."""
        pass
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension if applicable."""
        return getattr(self, 'embedding_dim', None)
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

