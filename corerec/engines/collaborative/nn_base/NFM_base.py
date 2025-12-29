import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
import os
import yaml
import logging
import pickle
from pathlib import Path
import pandas as pd
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from corerec.api.base_recommender import BaseRecommender


class FactorizationMachineLayer(nn.Module):
    """
    Factorization Machine Layer for feature interactions.

    Computes element-wise product of embedding vectors and their sum.

    Architecture:
    ┌─────────────┐
    │  Embeddings │
    └──────┬──────┘
           │
           ▼
    ┌─────────────────────┐
    │ Sum(embed_i * embed_j) │
    └───────────┬─────────┘
                │
                ▼
    ┌─────────────────────┐
    │   Output Vector     │
    └─────────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, embed_dim: int):
        """
        Initialize the Factorization Machine Layer.

        Args:
            embed_dim: Dimension of embedding vectors.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute second-order feature interactions.

        Args:
            embeddings: Tensor of shape (batch_size, num_fields, embed_dim)
                containing embedding vectors for each field.

        Returns:
            Tensor of shape (batch_size, embed_dim) containing FM interactions.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Square of sum
        sum_embeddings = torch.sum(embeddings, dim=1)  # (batch_size, embed_dim)
        sum_embeddings_squared = sum_embeddings**2  # (batch_size, embed_dim)

        # Sum of squares
        embeddings_squared = embeddings**2  # (batch_size, num_fields, embed_dim)
        sum_squared_embeddings = torch.sum(embeddings_squared, dim=1)  # (batch_size, embed_dim)

        # FM interaction term: 0.5 * (sum^2 - sum(squares))
        fm_interaction = 0.5 * (
            sum_embeddings_squared - sum_squared_embeddings
        )  # (batch_size, embed_dim)

        return fm_interaction


class DNN(nn.Module):
    """
    Deep Neural Network component for NFM model.

    Processes the output of Factorization Machine Layer through fully connected layers.

    Architecture:
    ┌──────────────┐
    │  FM Output   │
    └──────┬───────┘
           │
           ▼
    ┌─────────────────┐
    │  Hidden Layer 1 │───► Batch Norm ───► ReLU ───► Dropout
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Hidden Layer 2 │───► Batch Norm ───► ReLU ───► Dropout
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Output Layer  │
    └─────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1, batch_norm: bool = True
    ):
        """
        Initialize the DNN component.

        Args:
            input_dim: Input dimension (embedding dimension).
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout probability for regularization.
            batch_norm: Whether to use batch normalization.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.dropouts = nn.ModuleList()

        # Input layer
        layer_dims = [input_dim] + hidden_dims

        # Create layers
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_dims[i + 1]))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DNN.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, hidden_dims[-1])

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if self.batch_norms:
                x = self.batch_norms[i](x)

            x = F.relu(x)

            x = self.dropouts[i](x)

        return x


class NFMModel(nn.Module):
    """
    Neural Factorization Machine Model.

    Combines linear terms, factorization machine interactions, and deep neural networks.

    Architecture:
    ┌───────────────┐   ┌───────────────┐
    │ Field Values  │   │   Embeddings  │
    └───────┬───────┘   └───────┬───────┘
            │                   │
            ▼                   ▼
    ┌──────────────┐    ┌─────────────────┐
    │ Linear Terms │    │ FM Interactions │
    └───────┬──────┘    └────────┬────────┘
            │                    │
            │                    ▼
            │           ┌─────────────────┐
            │           │   Deep Layers   │
            │           └────────┬────────┘
            │                    │
            └────────────┬───────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Output Layer  │
                └─────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        field_dims: List[int],
        embed_dim: int = 16,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
        batch_norm: bool = True,
    ):
        """
        Initialize the NFM model.

        Args:
            field_dims: List of feature field dimensions (cardinalities).
            embed_dim: Embedding dimension.
            hidden_dims: List of hidden layer dimensions for DNN.
            dropout: Dropout probability.
            batch_norm: Whether to use batch normalization.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()

        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)

        # Feature embeddings
        self.embeddings = nn.ModuleList(
            [nn.Embedding(field_dim, embed_dim) for field_dim in field_dims]
        )

        # Linear terms (first-order)
        self.linear = nn.ModuleList([nn.Embedding(field_dim, 1) for field_dim in field_dims])

        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))

        # Factorization Machine Layer
        self.fm_layer = FactorizationMachineLayer(embed_dim)

        # Deep Network
        self.dnn = DNN(embed_dim, hidden_dims, dropout, batch_norm)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Xavier initialization.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        for embedding in self.embeddings:
            nn.init.xavier_normal_(embedding.weight)

        for linear in self.linear:
            nn.init.xavier_normal_(linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the NFM model.

        Args:
            x: Input tensor of shape (batch_size, num_fields) containing field indices.

        Returns:
            Predicted scores of shape (batch_size, 1).

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Linear part (first-order)
        linear_part = self.bias
        for i in range(self.num_fields):
            linear_part += self.linear[i](x[:, i]).squeeze(1)

        # Embedding part
        embed_x = [self.embeddings[i](x[:, i]).unsqueeze(1) for i in range(self.num_fields)]
        embed_x = torch.cat(embed_x, dim=1)  # (batch_size, num_fields, embed_dim)

        # Factorization Machine part (second-order)
        fm_interaction = self.fm_layer(embed_x)  # (batch_size, embed_dim)

        # Deep Network part
        dnn_output = self.dnn(fm_interaction)  # (batch_size, hidden_dims[-1])

        # Output layer
        output = self.output_layer(dnn_output).squeeze(1)  # (batch_size)

        # Combine linear part and neural network part
        y = linear_part + output

        # Apply sigmoid for binary classification/recommendation
        y = torch.sigmoid(y)

        return y.view(-1, 1)


class NFM_base(BaseRecommender):
    """
    Neural Factorization Machine (NFM) model for recommendation.

    NFM combines FM for learning feature interactions and neural networks
    for learning high-order feature interactions.

    Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │                     NFM_base                              │
    │                                                           │
    │  ┌─────────────┐    ┌───────────────┐    ┌────────────┐  │
    │  │ User & Item │    │Feature Mapping│    │Data Process│  │
    │  └──────┬──────┘    └───────┬───────┘    └──────┬─────┘  │
    │         │                   │                   │         │
    │         └───────────┬───────┴─────────┬─────────┘         │
    │                     │                 │                    │
    │                     ▼                 ▼                    │
    │            ┌────────────────┐  ┌─────────────┐            │
    │            │ NFM Model      │  │Training Loop│            │
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

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        name: str = "NFM",
        embed_dim: int = 16,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
        batch_norm: bool = True,
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
        Initialize the NFM model.

        Args:
            name: Model name.
            embed_dim: Embedding dimension.
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout probability.
            batch_norm: Whether to use batch normalization.
            learning_rate: Learning rate for optimizer.
            batch_size: Number of samples per batch.
            num_epochs: Maximum number of training epochs.
            patience: Early stopping patience.
            shuffle: Whether to shuffle data during training.
            device: Device to run model on ('cpu' or 'cuda').
            seed: Random seed for reproducibility.
            verbose: Whether to display training progress.
            config: Configuration dictionary that overrides the default parameters.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        self.name = name
        self.seed = seed
        self.verbose = verbose
        self.is_fitted = False

        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Process config if provided
        if config is not None:
            self.embed_dim = config.get("embed_dim", embed_dim)
            self.hidden_dims = config.get("hidden_dims", hidden_dims)
            self.dropout = config.get("dropout", dropout)
            self.batch_norm = config.get("batch_norm", batch_norm)
            self.learning_rate = config.get("learning_rate", learning_rate)
            self.batch_size = config.get("batch_size", batch_size)
            self.num_epochs = config.get("num_epochs", num_epochs)
            self.patience = config.get("patience", patience)
            self.shuffle = config.get("shuffle", shuffle)
        else:
            self.embed_dim = embed_dim
            self.hidden_dims = hidden_dims
            self.dropout = dropout
            self.batch_norm = batch_norm
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.patience = patience
            self.shuffle = shuffle

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

        # Initialize data structures for users, items, and features
        self.user_map = {}
        self.item_map = {}
        self.feature_map = {}
        self.feature_names = []
        self.field_dims = []

        # Initialize hook manager for model introspection
        self.hook_manager = None

        if self.verbose:
            self.logger.info(
                f"Initialized {self.name} model with {self.embed_dim} embedding dimensions"
            )

    def _setup_logger(self):
        """
        Setup logger for the model.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.logger = logging.getLogger(f"{self.name}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False

    def _preprocess_data(self, interactions: List[Tuple]):
        """
        Preprocess interactions data.

        Args:
            interactions: List of (user, item, features) tuples.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Map users and items to internal indices
        for user, item, _ in interactions:
            if user not in self.user_map:
                self.user_map[user] = len(self.user_map)
            if item not in self.item_map:
                self.item_map[item] = len(self.item_map)

        # Extract feature names from the first interaction
        if not self.feature_names and interactions:
            features = interactions[0][2]
            self.feature_names = list(features.keys())

            # Add user and item as fields
            self.feature_names = ["user_id", "item_id"] + self.feature_names

            if self.verbose:
                self.logger.info(f"Extracted features: {self.feature_names}")

        # Map features to internal indices
        for _, _, features in interactions:
            for feature_name, feature_value in features.items():
                if feature_name not in self.feature_map:
                    self.feature_map[feature_name] = {}

                if feature_value not in self.feature_map[feature_name]:
                    self.feature_map[feature_name][feature_value] = len(
                        self.feature_map[feature_name]
                    )

        # Compute field dimensions
        self.field_dims = [len(self.user_map), len(self.item_map)]
        for feature_name in self.feature_names[2:]:  # Skip user_id and item_id
            if feature_name in self.feature_map:
                self.field_dims.append(
                    len(self.feature_map[feature_name]) + 1
                )  # +1 for unknown values
            else:
                self.field_dims.append(1)  # Placeholder for empty features

    def _build_model(self):
        """
        Build NFM model.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.model = NFMModel(
            field_dims=self.field_dims,
            embed_dim=self.embed_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.verbose:
            self.logger.info(f"Built NFM model with {len(self.field_dims)} fields")
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Total parameters: {total_params}")

    def _prepare_batch(self, interactions: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for training.

        Args:
            interactions: List of (user, item, features) tuples.

        Returns:
            Tuple of (features tensor, labels tensor).

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        batch_size = len(interactions)
        feature_tensor = torch.zeros(batch_size, len(self.field_dims), dtype=torch.long)
        labels = torch.ones(batch_size, 1)

        for i, (user, item, features) in enumerate(interactions):
            # User and item indices
            feature_tensor[i, 0] = self.user_map.get(user, 0)
            feature_tensor[i, 1] = self.item_map.get(item, 0)

            # Feature values
            for j, feature_name in enumerate(self.feature_names[2:], 2):
                if feature_name in features:
                    feature_value = features[feature_name]
                    feature_index = self.feature_map.get(feature_name, {}).get(feature_value, 0)
                    feature_tensor[i, j] = feature_index

        return feature_tensor.to(self.device), labels.to(self.device)

    def _generate_negative_samples(self, interactions: List[Tuple]) -> List[Tuple]:
        """
        Generate negative samples for training.

        Args:
            interactions: List of positive (user, item, features) tuples.

        Returns:
            List of negative (user, item, features) tuples.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        negative_samples = []
        items = list(self.item_map.keys())

        for user, item, features in interactions:
            # Sample a random item different from the positive item
            while True:
                neg_item = np.random.choice(items)
                if neg_item != item:
                    break

            negative_samples.append((user, neg_item, features))

        return negative_samples

    def fit(self, interactions: List[Tuple]) -> "NFM_base":
        """
        Fit the NFM model on the given interactions.

        Args:
            interactions: List of (user, item, features) tuples.

        Returns:
            The fitted model.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if self.verbose:
            self.logger.info(f"Fitting {self.name} model on {len(interactions)} interactions")

        # Preprocess data
        self._preprocess_data(interactions)

        # Build model
        self._build_model()

        # Training loop
        num_batches = (len(interactions) + self.batch_size - 1) // self.batch_size
        best_loss = float("inf")
        patience_counter = 0
        self.loss_history = []

        for epoch in range(self.num_epochs):
            if self.shuffle:
                np.random.shuffle(interactions)

            epoch_loss = 0

            for i in range(0, len(interactions), self.batch_size):
                batch_interactions = interactions[i : i + self.batch_size]

                # Generate negative samples
                negative_interactions = self._generate_negative_samples(batch_interactions)

                # Prepare data
                pos_features, pos_labels = self._prepare_batch(batch_interactions)
                neg_features, neg_labels = self._prepare_batch(negative_interactions)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                pos_output = self.model(pos_features)
                neg_output = self.model(neg_features)

                # Set negative sample labels to 0
                neg_labels.fill_(0)

                # Combine positive and negative samples
                features = torch.cat([pos_features, neg_features], dim=0)
                labels = torch.cat([pos_labels, neg_labels], dim=0)
                outputs = torch.cat([pos_output, neg_output], dim=0)

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

    def predict(self, user: Any, item: Any, features: Optional[Dict[str, Any]] = None) -> float:
        """
        Predict the probability of interaction between user and item.

        Args:
            user: User identifier.
            item: Item identifier.
            features: Additional features for prediction.

        Returns:
            Probability of interaction.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        if features is None:
            features = {}

        # Check if user and item exist in mappings
        if user not in self.user_map:
            if self.verbose:
                self.logger.warning(f"Unknown user: {user}")
            return 0.0

        if item not in self.item_map:
            if self.verbose:
                self.logger.warning(f"Unknown item: {item}")
            return 0.0

        # Prepare input
        feature_tensor = torch.zeros(1, len(self.field_dims), dtype=torch.long)
        feature_tensor[0, 0] = self.user_map.get(user, 0)
        feature_tensor[0, 1] = self.item_map.get(item, 0)

        # Feature values
        for j, feature_name in enumerate(self.feature_names[2:], 2):
            if feature_name in features:
                feature_value = features[feature_name]
                feature_index = self.feature_map.get(feature_name, {}).get(feature_value, 0)
                feature_tensor[0, j] = feature_index

        # Move to device
        feature_tensor = feature_tensor.to(self.device)

        # Get prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(feature_tensor).item()

        return prediction

    def recommend(
        self,
        user: Any,
        top_n: int = 10,
        exclude_seen: bool = True,
        features: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Generate recommendations for a user.

        Args:
            user: User identifier.
            top_n: Number of recommendations to generate.
            exclude_seen: Whether to exclude items the user has interacted with.
            features: Additional features for prediction.

        Returns:
            List of (item, score) tuples.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making recommendations")

        if user not in self.user_map:
            if self.verbose:
                self.logger.warning(f"Unknown user: {user}")
            return []

        if features is None:
            features = {}

        # Get predictions for all items
        items = list(self.item_map.keys())
        predictions = []

        for item in items:
            predictions.append((item, self.predict(user, item, features)))

        # Sort by prediction score
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        return predictions[:top_n]

    def register_hook(self, layer_name: str, callback: Optional[callable] = None) -> bool:
        """
        Register a hook for a layer.

        Args:
            layer_name: Name of the layer.
            callback: Callback function.

        Returns:
            Whether the hook was successfully registered.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not hasattr(self, "hook_manager"):
            from corerec.engines.unionizedFilterEngine.nn_base.util.hook_manager import HookManager

            self.hook_manager = HookManager()

        return self.hook_manager.register_hook(self.model, layer_name, callback)

    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        Get activation for a layer.

        Args:
            layer_name: Name of the layer.

        Returns:
            Activation tensor.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not hasattr(self, "hook_manager"):
            return None

        return self.hook_manager.get_activation(layer_name)

    def save(self, filepath: str) -> None:
        """
        Save the model to a file.

        Args:
            filepath: Path to save the model.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        # Create directory if it doesn't exist
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = {
            "name": self.name,
            "embed_dim": self.embed_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "patience": self.patience,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "verbose": self.verbose,
            "user_map": self.user_map,
            "item_map": self.item_map,
            "feature_map": self.feature_map,
            "feature_names": self.feature_names,
            "field_dims": self.field_dims,
            "loss_history": self.loss_history if hasattr(self, "loss_history") else [],
        }

        # Save model state
        model_state = {
            "config": config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(model_state, filepath)

        if self.verbose:
            self.logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "NFM_base":
        """
        Load a model from a file.

        Args:
            filepath: Path to load the model from.
            device: Device to load the model on.

        Returns:
            Loaded model.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Load model state
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        model_state = torch.load(filepath, map_location=device)
        config = model_state["config"]

        # Create new model
        model = cls(
            name=config["name"],
            embed_dim=config["embed_dim"],
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
            batch_norm=config["batch_norm"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            num_epochs=config["num_epochs"],
            patience=config["patience"],
            shuffle=config["shuffle"],
            seed=config["seed"],
            verbose=config["verbose"],
            device=device,
        )

        # Restore mappings
        model.user_map = config["user_map"]
        model.item_map = config["item_map"]
        model.feature_map = config["feature_map"]
        model.feature_names = config["feature_names"]
        model.field_dims = config["field_dims"]

        if "loss_history" in config:
            model.loss_history = config["loss_history"]

        # Build and restore model
        model._build_model()
        model.model.load_state_dict(model_state["model_state_dict"])
        model.optimizer.load_state_dict(model_state["optimizer_state_dict"])

        model.is_fitted = True

        return model

    def export_feature_importance(self) -> Dict[str, float]:
        """
        Export feature importance.

        Returns:
            Dictionary mapping feature names to importance scores.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before exporting feature importance")

        # Get embedding weights
        feature_importance = {}

        # Use embedding norms as importance scores
        with torch.no_grad():
            for i, feature_name in enumerate(self.feature_names):
                if i < len(self.model.embeddings):
                    # Get embedding layer for this feature
                    embedding = self.model.embeddings[i]

                    # Compute average L2 norm of embedding vectors
                    weights = embedding.weight.detach()
                    norms = torch.norm(weights, dim=1).mean().item()

                    feature_importance[feature_name] = norms

        # Normalize importance scores
        total = sum(feature_importance.values())
        if total > 0:
            for feature_name in feature_importance:
                feature_importance[feature_name] /= total

        return feature_importance

    def set_device(self, device: str) -> None:
        """
        Set the device for the model.

        Args:
            device: Device to run model on ('cpu' or 'cuda').

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if self.is_fitted:
            self.device = torch.device(device)
            self.model.to(self.device)
        else:
            self.device = torch.device(device)

    def train(self):
        pass
