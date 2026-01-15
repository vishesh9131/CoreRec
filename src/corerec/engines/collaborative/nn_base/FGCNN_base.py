import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
import os
import logging
import pickle
from pathlib import Path
from collections import defaultdict

from corerec.base_recommender import BaseCorerec


class FeatureEmbedding(nn.Module):
    """
    Feature embedding module for FGCNN.

    Maps categorical features to dense vectors.

    Architecture:
    ┌─────────────────┐
    │  Sparse Features│
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Embeddings    │
    └─────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, field_dims: List[int], embed_dim: int):
        """
        Initialize feature embedding layer.

        Args:
            field_dims: List of feature field dimensions
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in field_dims])

        # Initialize embeddings with Xavier uniform
        for embed in self.embedding:
            nn.init.xavier_uniform_(embed.weight)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass for feature embedding.

        Args:
            x: Sparse feature tensor of shape (batch_size, num_fields)

        Returns:
            Embedded features of shape (batch_size, num_fields, embed_dim)
        """
        return torch.stack(
            [embedding(x[:, i]) for i, embedding in enumerate(self.embedding)], dim=1
        )


class FeatureGeneration(nn.Module):
    """
    Feature Generation module using Convolutional Neural Networks.

    This module learns local patterns from embeddings and generates new features.

    Architecture:
    ┌─────────────────┐
    │  Embeddings     │
    └────────┬────────┘
             │
             ▼
    ┌────────────────┐
    │    Conv1D      │───► MaxPool ───► RecombinePooling
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │    Conv1D      │───► MaxPool ───► RecombinePooling
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │  New Features  │
    └────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        num_fields: int,
        embed_dim: int,
        channels: List[int] = [64, 32],
        kernel_heights: List[int] = [3, 3],
        pooling_sizes: List[int] = [2, 2],
        recombine_kernels: List[int] = [2, 3],
    ):
        """
        Initialize feature generation module.

        Args:
            num_fields: Number of feature fields
            embed_dim: Embedding dimension
            channels: List of CNN channel sizes for each layer
            kernel_heights: List of kernel heights for each layer
            pooling_sizes: List of pooling sizes for each layer
            recombine_kernels: List of recombine kernel sizes for each layer
        """
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.channels = [1] + channels  # Input channel is 1
        self.kernel_heights = kernel_heights
        self.pooling_sizes = pooling_sizes
        self.recombine_kernels = recombine_kernels

        # Ensure all lists have same length
        assert (
            len(channels) == len(kernel_heights) == len(pooling_sizes) == len(recombine_kernels)
        ), "All hyperparameter lists must have the same length"

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        # Max pooling layers
        self.pool_layers = nn.ModuleList()
        # New feature mapping layers
        self.recombine_layers = nn.ModuleList()

        # Output feature dimensions after each layer
        self.new_feature_dims = []
        h, w = num_fields, embed_dim  # Input dimensions

        # Build layers
        for i in range(len(channels)):
            # CNN layer
            self.conv_layers.append(
                nn.Conv2d(
                    self.channels[i],
                    self.channels[i + 1],
                    kernel_size=(kernel_heights[i], 1),
                    padding=(kernel_heights[i] // 2, 0),
                )
            )

            # Update height after CNN (width unchanged)
            h_after_conv = h

            # Max pooling layer
            self.pool_layers.append(
                nn.MaxPool2d(kernel_size=(pooling_sizes[i], 1), stride=(pooling_sizes[i], 1))
            )

            # Update height after pooling
            h_after_pool = h_after_conv // pooling_sizes[i]

            # Feature recombination layer
            self.recombine_layers.append(
                nn.Conv2d(
                    self.channels[i + 1],
                    self.channels[i + 1] * recombine_kernels[i],
                    kernel_size=(1, 1),
                )
            )

            # Update dimensions for next layer
            h = h_after_pool
            self.new_feature_dims.append((h, self.channels[i + 1] * recombine_kernels[i]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for feature generation.

        Args:
            x: Input tensor of shape (batch_size, num_fields, embed_dim)

        Returns:
            Tuple of (original_features, list of new_features)
            - original_features: Original embedding features
            - new_features: List of generated features from each layer
        """
        batch_size = x.size(0)

        # Reshape input to (batch_size, channels=1, height=num_fields, width=embed_dim)
        x = x.unsqueeze(1)

        # Store output features from each layer
        new_features = []

        # Process through each layer
        for i in range(len(self.channels) - 1):
            # Apply convolution
            conv_out = F.relu(self.conv_layers[i](x))

            # Apply max pooling
            pool_out = self.pool_layers[i](conv_out)

            # Apply feature recombination
            recombine_out = F.relu(self.recombine_layers[i](pool_out))

            # Get actual dimensions from the output tensor
            actual_shape = recombine_out.shape  # [batch_size, channels, height, width]

            # Flatten the last 3 dimensions to create a 2D tensor
            new_feature = recombine_out.view(batch_size, -1)

            # Add to list of new features
            new_features.append(new_feature)

            # Update x for next layer
            x = pool_out

        # Return original features and new features
        return x.squeeze(1), new_features


class FGCNNModel(nn.Module):
    """
    Feature Generation by Convolutional Neural Network model.

    FGCNN automatically generates new features through CNN and recombination,
    then models interactions between features with DNN.

    Architecture:
    ┌─────────────────┐
    │  Input Features │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Embeddings    │
    └─────────┬───────┘
              │
              ▼
    ┌────────────────────┐                 ┌───────────────┐
    │ Feature Generation │────────────────►│   Original    │
    │     with CNN       │                 │   Features    │
    └────────┬───────────┘                 └───────┬───────┘
             │                                     │
             ▼                                     │
    ┌────────────────────┐                         │
    │  Generated Feature │                         │
    │   Combinations     │                         │
    └────────┬───────────┘                         │
             │                                     │
             └────────────────┬────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────┐
    │       Deep Neural Network (DNN)              │
    └─────────────────────┬────────────────────────┘
                          │
                          ▼
    ┌──────────────────────────────────────────────┐
    │                   Output                     │
    └──────────────────────────────────────────────┘

    References:
        - Liu, B., et al. "Feature Generation by Convolutional Neural Network for Click-Through
          Rate Prediction." WWW 2019.

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        field_dims: List[int],
        embed_dim: int = 16,
        channels: List[int] = [64, 32],
        kernel_heights: List[int] = [3, 3],
        pooling_sizes: List[int] = [2, 2],
        recombine_kernels: List[int] = [2, 3],
        dnn_hidden_units: List[int] = [128, 64, 32],
        dropout: float = 0.1,
    ):
        """
        Initialize the FGCNN model.

        Args:
            field_dims: List of feature field dimensions
            embed_dim: Embedding dimension
            channels: List of CNN channel sizes
            kernel_heights: List of kernel heights
            pooling_sizes: List of pooling sizes
            recombine_kernels: List of recombine kernel sizes
            dnn_hidden_units: Hidden unit sizes for DNN
            dropout: Dropout rate
        """
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)

        # Feature embedding layer
        self.embedding = FeatureEmbedding(field_dims, embed_dim)

        # Feature generation layer
        self.feature_generation = FeatureGeneration(
            num_fields=self.num_fields,
            embed_dim=embed_dim,
            channels=channels,
            kernel_heights=kernel_heights,
            pooling_sizes=pooling_sizes,
            recombine_kernels=recombine_kernels,
        )

        # Calculate input dimension for DNN
        self.new_feature_dims = self.feature_generation.new_feature_dims

        # Original features dimension
        original_dim = self.num_fields * embed_dim

        # Generated features dimension
        generated_dim = sum([h * c_r for h, c_r in self.new_feature_dims])

        # Total input dimension for DNN
        dnn_input_dim = original_dim + generated_dim

        # DNN layers
        dnn_layers = []
        input_dim = dnn_input_dim

        for hidden_dim in dnn_hidden_units:
            dnn_layers.append(nn.Linear(input_dim, hidden_dim))
            dnn_layers.append(nn.BatchNorm1d(hidden_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Output layer
        dnn_layers.append(nn.Linear(input_dim, 1))

        # DNN module
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass of the FGCNN model.

        Args:
            x: Input tensor of shape (batch_size, num_fields)

        Returns:
            Prediction tensor of shape (batch_size, 1)
        """
        # Get embeddings
        embedding_output = self.embedding(x)
        batch_size = embedding_output.size(0)

        # Generate new features
        _, new_features = self.feature_generation(embedding_output)

        # Flatten embedding output
        flat_embedding = embedding_output.view(batch_size, -1)

        # Calculate actual input dimension
        input_dim = flat_embedding.size(1)
        for feature in new_features:
            input_dim += feature.size(1)

        # Combine all features - features are already flattened from feature_generation
        combined_features = torch.cat([flat_embedding] + new_features, dim=1)

        # Resize the DNN input layer if needed
        if self.dnn[0].in_features != combined_features.size(1):
            with torch.no_grad():
                old_weight = self.dnn[0].weight
                old_bias = self.dnn[0].bias

                new_linear = nn.Linear(combined_features.size(1), self.dnn[0].out_features).to(
                    combined_features.device
                )
                if combined_features.size(1) > self.dnn[0].in_features:
                    # Expand weights, initialize new weights to zero
                    new_linear.weight[:, : self.dnn[0].in_features] = old_weight
                else:
                    # Take subset of weights
                    new_linear.weight = nn.Parameter(old_weight[:, : combined_features.size(1)])

                new_linear.bias = nn.Parameter(old_bias.clone())
                self.dnn[0] = new_linear

        # Apply DNN
        output = self.dnn(combined_features)

        # Apply sigmoid for binary classification
        return torch.sigmoid(output)


class FGCNN_base(BaseCorerec):
    """
    Feature Generation by Convolutional Neural Network for recommendation.

    FGCNN consists of two components:
    1. Feature Generation - uses CNN to learn local patterns and generate new features
    2. Deep Learning - models interactions between original and generated features

    Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │                     FGCNN_base                            │
    │                                                           │
    │  ┌─────────────┐    ┌───────────────┐    ┌────────────┐  │
    │  │ User & Item │    │Feature Mapping│    │Data Process│  │
    │  └──────┬──────┘    └───────┬───────┘    └──────┬─────┘  │
    │         │                   │                   │         │
    │         └───────────┬───────┴─────────┬─────────┘         │
    │                     │                 │                    │
    │                     ▼                 ▼                    │
    │            ┌────────────────┐  ┌─────────────┐            │
    │            │ FGCNN Model    │  │Training Loop│            │
    │            └────────┬───────┘  └──────┬──────┘            │
    │                     │                 │                    │
    │                     └────────┬────────┘                    │
    │                              │                             │
    │                              ▼                             │
    │                    ┌─────────────────┐                     │
    │                    │Recommendation API│                     │
    │                    └─────────────────┘                     │
    └───────────────────────────────────────────────────────────┘
                           │         │
                           ▼         ▼
                    ┌─────────┐ ┌──────────┐
                    │Prediction│ │Recommend │
                    └─────────┘ └──────────┘

    References:
        - Liu, B., et al. "Feature Generation by Convolutional Neural Network for Click-Through
          Rate Prediction." WWW 2019.

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        name: str = "FGCNN",
        embed_dim: int = 16,
        channels: List[int] = [64, 32],
        kernel_heights: List[int] = [3, 3],
        pooling_sizes: List[int] = [2, 2],
        recombine_kernels: List[int] = [2, 3],
        dnn_hidden_units: List[int] = [128, 64, 32],
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        num_epochs: int = 20,
        patience: int = 5,
        shuffle: bool = True,
        device: str = None,
        seed: int = 42,
        verbose: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the FGCNN model.

        Args:
            name: Model name
            embed_dim: Embedding dimension
            channels: CNN channel sizes for each layer
            kernel_heights: Kernel heights for each CNN layer
            pooling_sizes: Pooling sizes for each layer
            recombine_kernels: Recombine kernel sizes for each layer
            dnn_hidden_units: Hidden unit sizes for DNN
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Number of samples per batch
            num_epochs: Maximum number of training epochs
            patience: Early stopping patience
            shuffle: Whether to shuffle data during training
            device: Device to run model on ('cpu' or 'cuda')
            seed: Random seed for reproducibility
            verbose: Whether to display training progress
            config: Configuration dictionary that overrides the default parameters
        """
        super().__init__(name=name, verbose=verbose)
        self.seed = seed
        self.is_fitted = False

        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Process config if provided
        if config is not None:
            self.embed_dim = config.get("embed_dim", embed_dim)
            self.channels = config.get("channels", channels)
            self.kernel_heights = config.get("kernel_heights", kernel_heights)
            self.pooling_sizes = config.get("pooling_sizes", pooling_sizes)
            self.recombine_kernels = config.get("recombine_kernels", recombine_kernels)
            self.dnn_hidden_units = config.get("dnn_hidden_units", dnn_hidden_units)
            self.dropout = config.get("dropout", dropout)
            self.learning_rate = config.get("learning_rate", learning_rate)
            self.batch_size = config.get("batch_size", batch_size)
            self.num_epochs = config.get("num_epochs", num_epochs)
            self.patience = config.get("patience", patience)
            self.shuffle = config.get("shuffle", shuffle)
        else:
            self.embed_dim = embed_dim
            self.channels = channels
            self.kernel_heights = kernel_heights
            self.pooling_sizes = pooling_sizes
            self.recombine_kernels = recombine_kernels
            self.dnn_hidden_units = dnn_hidden_units
            self.dropout = dropout
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.patience = patience
            self.shuffle = shuffle

        # Validate hyperparameters
        assert (
            len(self.channels)
            == len(self.kernel_heights)
            == len(self.pooling_sizes)
            == len(self.recombine_kernels)
        ), "All layer hyperparameter lists must have the same length"

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Setup logger
        self._setup_logger()

        # Initialize model
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.BCELoss()

        # Initialize data structures
        self.field_names = []
        self.field_dims = []
        self.field_mapping = {}

        if self.verbose:
            self.logger.info(
                f"Initialized {self.name} model with {self.embed_dim} embedding dimensions"
            )

    def _setup_logger(self):
        """Setup logger for the model."""
        self.logger = logging.getLogger(f"{self.name}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False

    def _preprocess_data(self, data: List[Dict[str, Any]]):
        """
        Preprocess data for training.

        Args:
            data: List of dictionaries with features and label
        """
        # Extract field names
        all_fields = set()
        for sample in data:
            for field in sample.keys():
                if field != "label":
                    all_fields.add(field)

        self.field_names = sorted(list(all_fields))
        if self.verbose:
            self.logger.info(f"Identified {len(self.field_names)} fields: {self.field_names}")

        # Create field mappings
        for field in self.field_names:
            self.field_mapping[field] = {}
            values = set()

            # Collect all values for this field
            for sample in data:
                if field in sample:
                    values.add(sample[field])

            # Map values to indices
            for i, value in enumerate(sorted(list(values))):
                self.field_mapping[field][value] = i + 1  # Reserve 0 for unknown/missing

            # Set field dimension
            self.field_dims.append(len(self.field_mapping[field]) + 1)  # +1 for unknown/missing

        if self.verbose:
            self.logger.info(f"Field dimensions: {self.field_dims}")

    def _build_model(self):
        """Build the FGCNN model."""
        self.model = FGCNNModel(
            field_dims=self.field_dims,
            embed_dim=self.embed_dim,
            channels=self.channels,
            kernel_heights=self.kernel_heights,
            pooling_sizes=self.pooling_sizes,
            recombine_kernels=self.recombine_kernels,
            dnn_hidden_units=self.dnn_hidden_units,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.verbose:
            self.logger.info(f"Built FGCNN model with {len(self.field_dims)} fields")
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Total parameters: {total_params}")

    def _prepare_batch(self, batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for training.

        Args:
            batch: List of dictionaries with features and label

        Returns:
            Tuple of (features, labels)
        """
        batch_size = len(batch)

        # Initialize tensors
        features = torch.zeros((batch_size, len(self.field_names)), dtype=torch.long)
        labels = torch.zeros((batch_size, 1), dtype=torch.float)

        # Fill tensors with data
        for i, sample in enumerate(batch):
            # Features
            for j, field in enumerate(self.field_names):
                if field in sample:
                    value = sample[field]
                    field_idx = self.field_mapping[field].get(value, 0)  # Use 0 for unknown values
                    features[i, j] = field_idx

            # Label
            if "label" in sample:
                labels[i, 0] = float(sample["label"])

        return features.to(self.device), labels.to(self.device)

    def fit(self, data: List[Dict[str, Any]]) -> "FGCNN_base":
        """
        Fit the FGCNN model.

        Args:
            data: List of dictionaries with features and label

        Returns:
            Fitted model
        """
        if self.verbose:
            self.logger.info(f"Fitting {self.name} model on {len(data)} samples")

        # Preprocess data
        self._preprocess_data(data)

        # Build model
        self._build_model()

        # Training loop
        num_batches = (len(data) + self.batch_size - 1) // self.batch_size
        best_loss = float("inf")
        patience_counter = 0
        self.loss_history = []

        for epoch in range(self.num_epochs):
            if self.shuffle:
                np.random.shuffle(data)

            epoch_loss = 0

            for i in range(0, len(data), self.batch_size):
                batch = data[i : i + self.batch_size]

                # Prepare data
                features, labels = self._prepare_batch(batch)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(features)

                # Compute loss
                loss = self.loss_fn(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / num_batches
            self.loss_history.append(avg_epoch_loss)

            if self.verbose:
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_epoch_loss:.4f}")

            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        self.is_fitted = True
        return self

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Predict probability for a single sample.

        Args:
            features: Dictionary with feature values

        Returns:
            Predicted probability
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        # Convert features to tensor
        feature_tensor = torch.zeros(1, len(self.field_names), dtype=torch.long)

        for i, field in enumerate(self.field_names):
            if field in features:
                value = features[field]
                field_idx = self.field_mapping.get(field, {}).get(value, 0)
                feature_tensor[0, i] = field_idx

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            feature_tensor = feature_tensor.to(self.device)
            prediction = self.model(feature_tensor).item()

        return prediction

    def recommend(
        self, user_features: Dict[str, Any], item_pool: List[Dict[str, Any]], top_n: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Generate recommendations for a user.

        Args:
            user_features: Dictionary with user features
            item_pool: List of dictionaries with item features
            top_n: Number of recommendations to generate

        Returns:
            List of (item, score) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making recommendations")

        # Score each item in the pool
        scored_items = []
        for item in item_pool:
            # Merge user and item features
            features = {**user_features, **item}

            # Make prediction
            score = self.predict(features)
            scored_items.append((item, score))

        # Sort by score in descending order
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # Return top-n items
        return scored_items[:top_n]

    def save(self, filepath: str):
        """
        Save model to file.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        # Create directory if it doesn't exist
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data to save
        model_data = {
            "model_config": {
                "name": self.name,
                "embed_dim": self.embed_dim,
                "channels": self.channels,
                "kernel_heights": self.kernel_heights,
                "pooling_sizes": self.pooling_sizes,
                "recombine_kernels": self.recombine_kernels,
                "dnn_hidden_units": self.dnn_hidden_units,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "patience": self.patience,
                "shuffle": self.shuffle,
                "seed": self.seed,
                "verbose": self.verbose,
                "dnn_input_dim": self.model.dnn[0].in_features,  # Save actual input dimension
            },
            "data": {
                "field_names": self.field_names,
                "field_dims": self.field_dims,
                "field_mapping": self.field_mapping,
            },
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "is_fitted": self.is_fitted,
        }

        # Save to file
        torch.save(model_data, filepath)

        if self.verbose:
            self.logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "FGCNN_base":
        """
        Load model from file.

        Args:
            filepath: Path to the saved model
            device: Device to load the model on ('cpu' or 'cuda')

        Returns:
            Loaded model
        """
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        # Load data
        model_data = torch.load(filepath, map_location=device)

        # Extract configuration
        config = model_data["model_config"]

        # Create instance
        instance = cls(
            name=config["name"],
            embed_dim=config["embed_dim"],
            channels=config["channels"],
            kernel_heights=config["kernel_heights"],
            pooling_sizes=config["pooling_sizes"],
            recombine_kernels=config["recombine_kernels"],
            dnn_hidden_units=config["dnn_hidden_units"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            num_epochs=config["num_epochs"],
            patience=config.get("patience", 5),
            shuffle=config.get("shuffle", True),
            device=device,
            seed=config["seed"],
            verbose=config["verbose"],
        )

        # Restore data
        instance.field_names = model_data["data"]["field_names"]
        instance.field_dims = model_data["data"]["field_dims"]
        instance.field_mapping = model_data["data"]["field_mapping"]

        # Ensure the model is built
        instance._build_model()

        # Fix DNN input dimension if needed
        dnn_input_dim = config.get("dnn_input_dim")
        if dnn_input_dim and dnn_input_dim != instance.model.dnn[0].in_features:
            # Create new first layer with correct input dimension
            old_layer = instance.model.dnn[0]
            new_layer = nn.Linear(dnn_input_dim, old_layer.out_features).to(device)
            instance.model.dnn[0] = new_layer

        # Load model state
        instance.model.load_state_dict(model_data["model_state"])
        instance.optimizer.load_state_dict(model_data["optimizer_state"])
        instance.is_fitted = model_data["is_fitted"]

        return instance

    def train(self):
        """Required by base class but implemented as fit."""
        pass
