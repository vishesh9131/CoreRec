"""
TorchRecommender - Bridge between BaseRecommender API and PyTorch Models

This class combines the high-level recommendation API (BaseRecommender)
with PyTorch's nn.Module functionality (BaseModel) to create a unified
interface for deep learning recommender systems.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import pandas as pd
import pickle
import json

from .base_recommender import BaseRecommender
from ..core.base_model import BaseModel


class TorchRecommender(BaseRecommender):
    """
    Unified base class for PyTorch-based recommendation models.
    
    This class bridges the gap between:
    - BaseRecommender: High-level recommendation API (fit, predict, recommend)
    - BaseModel: PyTorch nn.Module with utilities (forward, save, load)
    
    Architecture Pattern:
    
    ┌──────────────────────────────────────────┐
    │      TorchRecommender                     │
    │  (Inherits from BaseRecommender)          │
    │                                           │
    │  ┌─────────────────────────────────────┐ │
    │  │  self.model (BaseModel/nn.Module)   │ │
    │  │  - forward()                         │ │
    │  │  - train_step()                      │ │
    │  │  - validate_step()                   │ │
    │  └─────────────────────────────────────┘ │
    │                                           │
    │  API Methods:                             │
    │  - fit()                                  │
    │  - predict()                              │
    │  - recommend()                            │
    │  - save() / load()                        │
    └──────────────────────────────────────────┘
    
    Usage Example:
    
        class MyDeepModel(BaseModel):
            def __init__(self, config):
                super().__init__("MyModel", config)
                self.layers = nn.Sequential(...)
            
            def forward(self, x):
                return self.layers(x)
        
        class MyRecommender(TorchRecommender):
            def __init__(self, embedding_dim=64):
                config = {'embedding_dim': embedding_dim}
                model = MyDeepModel(config)
                super().__init__(
                    name="MyRecommender",
                    model=model,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
            
            def fit(self, data, **kwargs):
                # Training logic using self.model
                pass
            
            def predict(self, user_id, item_id, **kwargs):
                # Prediction using self.model.forward()
                pass
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self,
        name: str = "TorchRecommender",
        model: Optional[Union[BaseModel, nn.Module]] = None,
        device: str = 'cpu',
        verbose: bool = False
    ):
        """
        Initialize PyTorch-based recommender.
        
        Args:
            name: Model name for identification
            model: PyTorch model instance (BaseModel or nn.Module)
            device: Device to run model on ('cpu', 'cuda', etc.)
            verbose: Whether to print training logs
        """
        super().__init__(name=name, verbose=verbose)
        
        self.model = model
        self.device = torch.device(device)
        
        if self.model is not None:
            self.model.to(self.device)
        
        # Training state
        self.optimizer = None
        self.criterion = None
        self._training_history = []
    
    def fit(self, data: Union[pd.DataFrame, Dict, Any], **kwargs) -> 'TorchRecommender':
        """
        Train the PyTorch recommendation model.
        
        Must be implemented by subclasses with specific training logic.
        
        Args:
            data: Training data
            **kwargs: Additional training parameters (epochs, lr, etc.)
            
        Returns:
            self: For method chaining
        """
        raise NotImplementedError(
            "Subclasses must implement fit() with specific training logic"
        )
    
    def predict(self, user_id: Any, item_id: Any, **kwargs) -> float:
        """
        Predict score for a single user-item pair.
        
        Must be implemented by subclasses.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted score
        """
        raise NotImplementedError(
            "Subclasses must implement predict() for scoring user-item pairs"
        )
    
    def recommend(
        self,
        user_id: Any,
        top_k: int = 10,
        exclude_items: Optional[List[Any]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Generate top-K recommendations for a user.
        
        Must be implemented by subclasses.
        
        Args:
            user_id: User identifier
            top_k: Number of recommendations
            exclude_items: Items to exclude
            **kwargs: Additional parameters
            
        Returns:
            List of recommended item IDs
        """
        raise NotImplementedError(
            "Subclasses must implement recommend() for generating recommendations"
        )
    
    def save(self, path: Union[str, Path], format: str = 'torch') -> None:
        """
        Save model to disk.
        
        Args:
            path: File path to save model
            format: Save format ('torch' for PyTorch, 'pickle' for pickle)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'torch':
            # Save PyTorch model
            state = {
                'model_state_dict': self.model.state_dict() if self.model else None,
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'name': self.name,
                'is_fitted': self.is_fitted,
                'device': str(self.device),
                'training_history': self._training_history,
                'version': self._version
            }
            torch.save(state, path)
            
            if self.verbose:
                print(f"Model saved to {path}")
        
        elif format == 'pickle':
            # Save entire object via pickle
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            
            if self.verbose:
                print(f"Model saved to {path}")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'torch' or 'pickle'")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TorchRecommender':
        """
        Load model from disk.
        
        Args:
            path: File path to load model from
            
        Returns:
            Loaded model instance
        """
        path = Path(path)
        
        # Try to detect format
        try:
            # Try loading as torch checkpoint
            checkpoint = torch.load(path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # This is a torch checkpoint - subclass must override
                raise NotImplementedError(
                    f"{cls.__name__} must implement custom load() to restore "
                    "model architecture from checkpoint"
                )
            else:
                # This might be a pickled object
                with open(path, 'rb') as f:
                    return pickle.load(f)
        
        except (pickle.UnpicklingError, RuntimeError):
            # Try pickle
            with open(path, 'rb') as f:
                return pickle.load(f)
    
    def to(self, device: Union[str, torch.device]) -> 'TorchRecommender':
        """
        Move model to specified device.
        
        Args:
            device: Device to move to
            
        Returns:
            self
        """
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
        return self
    
    def train(self) -> 'TorchRecommender':
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
        return self
    
    def eval(self) -> 'TorchRecommender':
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information.
        
        Returns:
            Dictionary with model metadata
        """
        info = super().get_model_info()
        
        if self.model is not None:
            info.update({
                'device': str(self.device),
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'num_trainable_params': sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
                'model_class': self.model.__class__.__name__
            })
        
        return info
    
    def __repr__(self) -> str:
        """String representation."""
        device_str = str(self.device)
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', device='{device_str}', {fitted_str})"

