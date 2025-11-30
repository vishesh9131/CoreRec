"""
Base tower module for CoreRec framework.

This module contains the AbstractTower base class that all towers must inherit from.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Union, Optional
from abc import ABC, abstractmethod


class AbstractTower(nn.Module, ABC):
    """
    Abstract base class for all towers in CoreRec.

    A tower is a neural network that encodes input features into a fixed-dimension
    representation. Towers can be used for encoding users, items, or any other
    entity in a recommendation system.

    All tower implementations must inherit from this class and implement
    the forward() and encode() methods.

    Architecture:
        Input Features
              ↓
        [AbstractTower]
              ↓
        Encoded Representation
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the abstract tower.

        Args:
            name (str): Name of the tower
            config (Dict[str, Any]): Tower configuration
        """
        super().__init__()
        self.name = name
        self.config = config

    @abstractmethod
    def forward(self, inputs: Any) -> torch.Tensor:
        """Forward pass of the tower.

        This method should transform the input into a fixed-dimension
        representation.

        Args:
            inputs (Any): Input data to encode

        Returns:
            torch.Tensor: Encoded representation
        """
        pass

    @abstractmethod
    def encode(self, inputs: Any) -> torch.Tensor:
        """Encode input data into a tensor representation.

        This is a convenience method that might include pre/post-processing
        around the forward method.

        Args:
            inputs (Any): Input data to encode

        Returns:
            torch.Tensor: Encoded representation
        """
        pass

    @property
    def device(self) -> torch.device:
        """Get the device of the tower.

        Returns:
            torch.device: Device of the tower
        """
        return next(self.parameters()).device

    def __repr__(self) -> str:
        """String representation of the tower.

        Returns:
            str: String representation
        """
        return f"{self.__class__.__name__}(name={self.name})"
