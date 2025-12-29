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

from corerec.api.base_recommender import BaseRecommender


class FieldAwareEmbedding(nn.Module):
    """
    Field-aware embedding layer that creates embeddings for each feature-field pair.

    This is the key component of FFM where feature i has a separate embedding vector
    for interacting with field j.

    Architecture:
    ┌─────────────────┐
    │  Features       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │   Field-specific Embeddings     │
    │  (feature i x field j matrix)   │
    └─────────────────────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, field_dims: List[int], num_fields: int, embed_dim: int):
        """
        Initialize field-aware embedding layer.

        Args:
            field_dims: List of feature dimensions for each field
            num_fields: Number of fields
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.ModuleList(
            [
                nn.ModuleList([nn.Embedding(field_dim, embed_dim) for _ in range(num_fields)])
                for field_dim in field_dims
            ]
        )

        # Initialize embeddings with Xavier uniform
        for field_embeddings in self.embedding:
            for embed in field_embeddings:
                nn.init.xavier_uniform_(embed.weight)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Get field-aware embeddings for each feature-field pair.

        Args:
            x: Feature indices tensor of shape (batch_size, num_fields)

        Returns:
            List of field-aware embeddings of shape (batch_size, num_fields, num_fields, embed_dim)
        """
        field_aware_embeddings = []

        # For each field
        for i in range(len(self.embedding)):
            # Get embeddings specific to interactions with each other field
            field_embeddings = []
            for j in range(len(self.embedding)):
                # Skip embeddings for interaction with same field
                if i == j:
                    continue

                # Get embedding for feature i when interacting with field j
                embed = self.embedding[i][j](x[:, i])
                field_embeddings.append(embed)

            # Stack embeddings for this field
            if field_embeddings:
                field_aware_embeddings.append(torch.stack(field_embeddings, dim=1))

        return field_aware_embeddings


class FFMModel(nn.Module):
    """
    Field-aware Factorization Machine model.

    Extends Factorization Machines by modeling separate feature interactions
    for each field combination.

    Architecture:
    ┌─────────────────┐
    │  Features       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │    Linear Terms + Bias          │
    └────────┬──────────────┬─────────┘
             │              │
             ▼              ▼
    ┌─────────────┐  ┌─────────────────────┐
    │  First-order│  │Field-aware Second-  │
    │    Term     │  │order Interactions   │
    └──────┬──────┘  └───────────┬─────────┘
           │                     │
           └─────────┬───────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │           Sum + Sigmoid         │
    └─────────────────────────────────┘

    References:
        - Juan, Y., et al. "Field-aware factorization machines for CTR prediction."
          Proceedings of the 10th ACM Conference on Recommender Systems. 2016.

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 4, dropout: float = 0.0):
        """
        Initialize the FFM model.

        Args:
            field_dims: Cardinality of each field
            embed_dim: Dimension of field-aware embeddings
            dropout: Dropout rate
        """
        super().__init__()
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim

        # Bias and first-order term
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.ModuleList([nn.Embedding(field_dim, 1) for field_dim in field_dims])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Field-aware feature embeddings
        self.field_aware_embedding = FieldAwareEmbedding(field_dims, self.num_fields, embed_dim)

        # Initialize weights
        for embed in self.linear:
            nn.init.xavier_uniform_(embed.weight)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass of FFM model.

        Args:
            x: Input tensor of shape (batch_size, num_fields) containing categorical features

        Returns:
            Predictions of shape (batch_size, 1)
        """
        # Bias term (intercept)
        out = torch.full((x.size(0),), self.bias.item(), device=x.device)

        # First-order term
        first_order = torch.zeros_like(out)
        for i in range(self.num_fields):
            first_order = first_order + self.linear[i](x[:, i]).squeeze(1)

        # Don't apply dropout to categorical indices, but to embeddings later
        # Field-aware embeddings
        field_aware_embeddings = self.field_aware_embedding(x)

        # Second-order interactions
        second_order = torch.zeros_like(out)
        field_pairs = []

        # Compute field-aware interactions
        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                # Get field-aware embeddings
                # Embedding of feature i for field j and feature j for field i
                v_i_j = (
                    field_aware_embeddings[i][:, j - 1]
                    if j > i
                    else field_aware_embeddings[i][:, j]
                )
                v_j_i = (
                    field_aware_embeddings[j][:, i]
                    if i < j
                    else field_aware_embeddings[j][:, i - 1]
                )

                # Apply dropout to embeddings
                if self.training:
                    v_i_j = self.dropout(v_i_j)
                    v_j_i = self.dropout(v_j_i)

                # Compute inner product
                field_pairs.append(torch.sum(v_i_j * v_j_i, dim=1))

        # Sum all field-wise interactions
        if field_pairs:
            second_order = torch.stack(field_pairs, dim=1).sum(dim=1)

        # Add all terms and apply sigmoid
        final_out = out + first_order + second_order
        result = torch.sigmoid(final_out)

        return result.view(-1, 1)


class FFM_base(BaseRecommender):
    """
    Field-aware Factorization Machine (FFM) for recommendation and CTR prediction.

    FFM extends factorization machines by considering field information for feature interactions.
    Each feature has a separate latent vector for each field, which improves modeling of
    complex feature interactions.

    Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │                    FFM_base                               │
    │                                                           │
    │  ┌─────────────┐    ┌───────────────┐    ┌────────────┐  │
    │  │ User & Item │    │Field Detection│    │Data Process│  │
    │  └──────┬──────┘    └───────┬───────┘    └──────┬─────┘  │
    │         │                   │                   │         │
    │         └───────────┬───────┴─────────┬─────────┘         │
    │                     │                 │                    │
    │                     ▼                 ▼                    │
    │            ┌────────────────┐  ┌─────────────┐            │
    │            │ FFM Model      │  │Training Loop│            │
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
        - Juan, Y., et al. "Field-aware factorization machines for CTR prediction."
          Proceedings of the 10th ACM Conference on Recommender Systems. 2016.

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        name: str = "FFM",
        embed_dim: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        num_epochs: int = 20,
        patience: int = 5,
        l2_reg: float = 1e-6,
        shuffle: bool = True,
        device: str = None,
        seed: int = 42,
        verbose: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the FFM model.

        Args:
            name: Model name
            embed_dim: Embedding dimension
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Number of samples per batch
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            l2_reg: L2 regularization strength
            shuffle: Whether to shuffle data during training
            device: Device to run model on ('cpu' or 'cuda')
            seed: Random seed for reproducibility
            verbose: Whether to display training progress
            config: Configuration dictionary
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
            self.dropout = config.get("dropout", dropout)
            self.learning_rate = config.get("learning_rate", learning_rate)
            self.batch_size = config.get("batch_size", batch_size)
            self.num_epochs = config.get("num_epochs", num_epochs)
            self.patience = config.get("patience", patience)
            self.l2_reg = config.get("l2_reg", l2_reg)
            self.shuffle = config.get("shuffle", shuffle)
        else:
            self.embed_dim = embed_dim
            self.dropout = dropout
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.patience = patience
            self.l2_reg = l2_reg
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
        """Build FFM model."""
        self.model = FFMModel(
            field_dims=self.field_dims, embed_dim=self.embed_dim, dropout=self.dropout
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg
        )

        if self.verbose:
            self.logger.info(f"Built FFM model with {len(self.field_dims)} fields")
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

        # Initialize feature and label tensors
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

    def fit(self, data: List[Dict[str, Any]]) -> "FFM_base":
        """
        Fit the FFM model.

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
            Prediction probability
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
                "embed_dim": self.embed_dim,
                "dropout": self.dropout,
                "name": self.name,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "patience": self.patience,
                "l2_reg": self.l2_reg,
                "shuffle": self.shuffle,
                "seed": self.seed,
                "verbose": self.verbose,
            },
            "field_data": {
                "field_names": self.field_names,
                "field_dims": self.field_dims,
                "field_mapping": self.field_mapping,
            },
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss_history": self.loss_history if hasattr(self, "loss_history") else [],
        }

        # Save to file
        torch.save(model_data, filepath)

        if self.verbose:
            self.logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "FFM_base":
        """
        Load model from file.

        Args:
            filepath: Path to the saved model
            device: Device to load the model on

        Returns:
            Loaded model
        """
        # Load model data
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        model_data = torch.load(filepath, map_location=device)

        # Create model instance with saved config
        instance = cls(
            name=model_data["model_config"]["name"],
            embed_dim=model_data["model_config"]["embed_dim"],
            dropout=model_data["model_config"]["dropout"],
            learning_rate=model_data["model_config"]["learning_rate"],
            batch_size=model_data["model_config"]["batch_size"],
            num_epochs=model_data["model_config"]["num_epochs"],
            patience=model_data["model_config"]["patience"],
            l2_reg=model_data["model_config"]["l2_reg"],
            shuffle=model_data["model_config"]["shuffle"],
            seed=model_data["model_config"]["seed"],
            verbose=model_data["model_config"]["verbose"],
            device=device,
        )

        # Restore field data
        instance.field_names = model_data["field_data"]["field_names"]
        instance.field_dims = model_data["field_data"]["field_dims"]
        instance.field_mapping = model_data["field_data"]["field_mapping"]
        instance.loss_history = model_data.get("loss_history", [])

        # Build and load model
        instance._build_model()
        instance.model.load_state_dict(model_data["model_state"])
        instance.optimizer.load_state_dict(model_data["optimizer_state"])

        instance.is_fitted = True
        return instance

    def train(self):
        """Required by base class but implemented as fit."""
        pass
