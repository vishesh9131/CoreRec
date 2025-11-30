"""
Base model class for all CoreRec models.

This module provides the BaseModel class which handles forward pass,
model saving and loading, and other common functionality.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional, List, Tuple
import logging
import json
import time


class BaseModel(nn.Module):
    """
    Base class for all CoreRec models.

    This class provides common functionality for model forward pass,
    saving, loading, and other utilities.

    Attributes:
        name (str): Name of the model
        config (Dict[str, Any]): Model configuration
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the base model.

        Args:
            name (str): Name of the model
            config (Dict[str, Any]): Model configuration
        """
        super().__init__()
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def forward(self, *args, **kwargs):
        """Forward pass of the model.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save model weights and configuration.

        Args:
            path (str): Directory path to save the model
            metadata (Optional[Dict[str, Any]]): Additional metadata to save

        Returns:
            str: Path to the saved model file
        """
        os.makedirs(path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(path, f"{self.name}_{timestamp}.pt")

        # Save model state
        state_dict = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "model_name": self.name,
            "timestamp": timestamp,
        }

        if metadata is not None:
            state_dict["metadata"] = metadata

        torch.save(state_dict, save_path)

        # Save config as JSON for easy inspection
        config_path = os.path.join(path, f"{self.name}_{timestamp}_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        self.logger.info(f"Model saved to {save_path}")
        return save_path

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "BaseModel":
        """Load model from saved checkpoint.

        Args:
            path (str): Path to the saved model file
            device (Optional[torch.device]): Device to load the model to

        Returns:
            BaseModel: Loaded model instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        model_name = checkpoint["model_name"]

        # Create model instance
        model = cls(name=model_name, config=config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return model

    def get_model_size(self) -> int:
        """Get model size in terms of parameter count.

        Returns:
            int: Number of parameters in the model
        """
        return sum(p.numel() for p in self.parameters())

    def freeze(self):
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def train_step(
        self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            optimizer (torch.optim.Optimizer): Optimizer instance

        Returns:
            Dict[str, float]: Dictionary with loss values
        """
        raise NotImplementedError("Subclasses must implement train_step method")

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single validation step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data

        Returns:
            Dict[str, float]: Dictionary with validation metrics
        """
        raise NotImplementedError("Subclasses must implement validate_step method")
