"""
MLP tower for recommendation systems.

This module provides a simple MLP tower for encoding user/item features.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Union, Optional
import logging
import numpy as np

from corerec.towers.base_tower import AbstractTower


class MLPTower(AbstractTower):
    """
    Multi-Layer Perceptron (MLP) tower for recommendation systems.

    This tower uses a simple MLP to encode user or item features into
    a fixed-dimension representation.

    Architecture:
        Input Features
              ↓
         [Embedding]  (if categorical)
              ↓
          [Dense Layer]
              ↓
           [ReLU/Tanh]
              ↓
             ...
              ↓
          [Dense Layer]
              ↓
           [Norm/Dropout]
              ↓
        Encoded Representation
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the MLP tower.

        Args:
            name (str): Name of the tower
            config (Dict[str, Any]): Tower configuration including:
                - input_dim (int): Input dimension
                - hidden_dims (List[int]): Dimensions of hidden layers
                - output_dim (int): Output dimension
                - dropout (float): Dropout rate
                - activation (str): Activation function ('relu', 'tanh', 'sigmoid', etc.)
                - use_batch_norm (bool): Whether to use batch normalization
                - use_layer_norm (bool): Whether to use layer normalization
                - categorical_features (Dict[str, int]): Dict mapping feature names to vocabulary sizes
                - embedding_dim (int): Dimension for embedding categorical features
        """
        super().__init__(name, config)

        # Get configuration
        self.input_dim = config.get("input_dim", 64)
        self.hidden_dims = config.get("hidden_dims", [128, 64])
        self.output_dim = config.get("output_dim", 32)
        self.dropout_rate = config.get("dropout", 0.1)
        self.activation_name = config.get("activation", "relu")
        self.use_batch_norm = config.get("use_batch_norm", False)
        self.use_layer_norm = config.get("use_layer_norm", False)

        # Get categorical features config
        self.categorical_features = config.get("categorical_features", {})
        self.embedding_dim = config.get("embedding_dim", 16)

        # Create embedding layers for categorical features
        self.embedding_layers = nn.ModuleDict()
        total_embedding_dim = 0

        for feature_name, vocab_size in self.categorical_features.items():
            self.embedding_layers[feature_name] = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=self.embedding_dim
            )
            total_embedding_dim += self.embedding_dim

        # Calculate the actual input dimension (original + embeddings)
        actual_input_dim = self.input_dim + total_embedding_dim

        # Create MLP layers
        self.mlp_layers = self._create_mlp_layers(actual_input_dim)

    def _create_mlp_layers(self, input_dim: int) -> nn.Sequential:
        """Create MLP layers.

        Args:
            input_dim (int): Input dimension

        Returns:
            nn.Sequential: MLP layers
        """
        # Create activation function
        if self.activation_name == "relu":
            activation = nn.ReLU()
        elif self.activation_name == "leaky_relu":
            activation = nn.LeakyReLU(0.1)
        elif self.activation_name == "tanh":
            activation = nn.Tanh()
        elif self.activation_name == "sigmoid":
            activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")

        # Create layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(self.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch norm
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Layer norm
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            layers.append(activation)

            # Dropout
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))

        return nn.Sequential(*layers)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the MLP tower.

        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary of inputs including:
                - features (torch.Tensor): Dense features
                - categorical features: One or more categorical features

        Returns:
            torch.Tensor: Encoded representations
        """
        # Extract dense features
        if "features" in inputs:
            dense_features = inputs["features"]
        else:
            dense_features = torch.zeros(
                (inputs[list(inputs.keys())[0]].size(0), self.input_dim), device=self.device
            )

        # Process categorical features
        embeddings = []

        for feature_name, embedding_layer in self.embedding_layers.items():
            if feature_name in inputs:
                feature_embedding = embedding_layer(inputs[feature_name])
                embeddings.append(feature_embedding)

        # Combine dense features and embeddings
        combined_features = [dense_features]
        combined_features.extend(embeddings)

        # Concatenate features
        if len(combined_features) > 1:
            combined = torch.cat(combined_features, dim=1)
        else:
            combined = combined_features[0]

        # Pass through MLP
        output = self.mlp_layers(combined)

        return output

    def encode(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Encode inputs.

        Args:
            inputs (Dict[str, Any]): Dictionary of inputs

        Returns:
            torch.Tensor: Encoded representations
        """
        # Convert inputs to tensors if needed
        tensor_inputs = {}

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                tensor_inputs[key] = value
            elif isinstance(value, np.ndarray):
                tensor_inputs[key] = torch.from_numpy(value).to(self.device)
            elif isinstance(value, (list, tuple)) and all(
                isinstance(x, (int, float)) for x in value
            ):
                tensor_inputs[key] = torch.tensor(value, device=self.device)
            elif isinstance(value, (int, float)):
                tensor_inputs[key] = torch.tensor([value], device=self.device)

        # Forward pass
        with torch.no_grad():
            output = self.forward(tensor_inputs)

        return output
