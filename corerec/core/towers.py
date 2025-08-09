"""
Towers module for CoreRec framework.

This module contains tower classes for user and item representation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional, List, Tuple
from abc import ABC, abstractmethod

class Tower(nn.Module, ABC):
    """
    Abstract tower class for CoreRec framework.
    
    A tower is a neural network that projects user or item features
    into a common embedding space.
    """
    
    def __init__(self, name: str, input_dim: int, output_dim: int, config: Dict[str, Any]):
        """Initialize the tower.
        
        Args:
            name (str): Name of the tower
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            config (Dict[str, Any]): Tower configuration
        """
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        
        # Build the tower network
        self._build_network()
    
    @abstractmethod
    def _build_network(self):
        """Build the tower network."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the tower.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output embedding
        """
        pass


class MLPTower(Tower):
    """
    Multi-layer perceptron tower.
    
    A simple MLP tower with configurable hidden layers.
    """
    
    def __init__(self, name: str, input_dim: int, output_dim: int, config: Dict[str, Any]):
        """Initialize the MLP tower.
        
        Args:
            name (str): Name of the tower
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            config (Dict[str, Any]): Tower configuration including:
                - hidden_dims (List[int]): List of hidden dimensions
                - dropout (float): Dropout rate
                - activation (str): Activation function ('relu', 'tanh', 'leaky_relu')
                - norm (str): Normalization type ('batch', 'layer', None)
        """
        super().__init__(name, input_dim, output_dim, config)
    
    def _build_network(self):
        """Build the MLP tower network."""
        hidden_dims = self.config.get('hidden_dims', [128, 64])
        dropout_rate = self.config.get('dropout', 0.0)
        activation = self.config.get('activation', 'relu')
        norm = self.config.get('norm', None)
        
        # Create activation function
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        elif activation == 'leaky_relu':
            activation_fn = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Create layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Add normalization if specified
            if norm == 'batch':
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm == 'layer':
                layers.append(nn.LayerNorm(hidden_dim))
                
            layers.append(activation_fn)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        # Create sequential model
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP tower.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output embedding
        """
        return self.network(x)


class UserTower(MLPTower):
    """
    User tower for encoding user features.
    
    This tower is specifically for encoding user features.
    """
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        """Initialize the user tower.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            config (Dict[str, Any]): Tower configuration
        """
        super().__init__('user_tower', input_dim, output_dim, config)


class ItemTower(MLPTower):
    """
    Item tower for encoding item features.
    
    This tower is specifically for encoding item features.
    """
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        """Initialize the item tower.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            config (Dict[str, Any]): Tower configuration
        """
        super().__init__('item_tower', input_dim, output_dim, config)


class TowerFactory:
    """
    Factory class for creating towers.
    
    This class provides methods for creating different types of towers.
    """
    
    @staticmethod
    def create_tower(tower_type: str, input_dim: int, output_dim: int, config: Dict[str, Any]) -> Tower:
        """Create a tower of the specified type.
        
        Args:
            tower_type (str): Type of tower ('mlp', 'user', 'item')
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            config (Dict[str, Any]): Tower configuration
            
        Returns:
            Tower: Instantiated tower
        """
        if tower_type == 'mlp':
            return MLPTower('mlp_tower', input_dim, output_dim, config)
        elif tower_type == 'user':
            return UserTower(input_dim, output_dim, config)
        elif tower_type == 'item':
            return ItemTower(input_dim, output_dim, config)
        else:
            raise ValueError(f"Unknown tower type: {tower_type}") 