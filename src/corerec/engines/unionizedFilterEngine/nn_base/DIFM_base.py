from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import yaml
import logging
from pathlib import Path


class InterestFusionLayer(nn.Module):
    """
    Interest Fusion Layer for DIFM.

    Architecture:

    ┌───────────┐   ┌───────────┐
    │  User     │   │   Item    │
    │ Features  │   │ Features  │
    └─────┬─────┘   └─────┬─────┘
          │               │
          └───────┬───────┘
                  │
            ┌─────▼─────┐
            │ Attention │
            │   Layer   │
            └─────┬─────┘
                  │
            ┌─────▼─────┐
            │  Feature  │
            │  Fusion   │
            └─────┬─────┘
                  │
            ┌─────▼─────┐
            │  Output   │
            └───────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim * 2, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
            nn.Sigmoid(),
        )

        self.fusion = nn.Linear(input_dim * 2, input_dim)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        # Concatenate features
        combined = torch.cat([user_features, item_features], dim=-1)

        # Calculate attention weights
        attention_weights = self.attention(combined)

        # Apply fusion
        fused = self.fusion(combined)

        # Apply attention
        output = attention_weights * fused

        return output


class DIFM_base(nn.Module):
    """
    Deep Interest Fusion Machine (DIFM) base implementation.

    Architecture:

    ┌───────────┐
    │  Input    │
    │  Layer    │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │ Embedding │
    │  Layer    │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │ Interest  │
    │  Fusion   │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │   MLP     │
    │  Layer    │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │  Output   │
    │  Layer    │
    └───────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        mlp_dims: List[int] = [128, 64],
        field_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        attention_dim: int = 32,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        seed: int = 42,
    ):
        """
        Initialize DIFM model.

        Args:
            embed_dim: Embedding dimension
            mlp_dims: List of MLP layer dimensions
            field_dims: List of field dimensions (optional)
            dropout: Dropout rate
            attention_dim: Attention layer dimension
            batch_size: Batch size
            learning_rate: Learning rate
            num_epochs: Number of epochs
            seed: Random seed

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Store parameters
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.field_dims = field_dims
        self.dropout = dropout
        self.attention_dim = attention_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.seed = seed

        # Initialize device
        self.device = torch.device("cpu")

        # Initialize maps and features
        self.user_map = {}
        self.item_map = {}
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        self.feature_encoders = {}

        # Initialize model components
        self.is_fitted = False
        self.model = None

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize loss history
        self.loss_history = []

    def build_model(self):
        """
        Build the DIFM model architecture.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Create embeddings for each field
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, self.embed_dim) for dim in self.field_dims]
        )

        # Interest fusion layer
        self.interest_fusion = InterestFusionLayer(self.embed_dim, self.attention_dim)

        # MLP layers
        self.mlp = nn.ModuleList()
        input_dim = self.embed_dim

        for dim in self.mlp_dims:
            self.mlp.append(nn.Linear(input_dim, dim))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(self.dropout))
            input_dim = dim

        # Output layer
        self.output_layer = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DIFM model.

        Args:
            user_features: User features tensor
            item_features: Item features tensor

        Returns:
            Predicted probability

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Get embeddings
        user_embed = self.embeddings[0](user_features)
        item_embed = self.embeddings[1](item_features)

        # Interest fusion
        fused = self.interest_fusion(user_embed, item_embed)

        # MLP layers
        x = fused
        for layer in self.mlp:
            x = layer(x)

        # Output layer
        output = self.sigmoid(self.output_layer(x))

        return output

    def fit(self, interactions: List[Tuple]):
        """
        Fit the DIFM model to interactions.

        Args:
            interactions: List of (user, item, features) tuples

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Extract features and create mappings
        self._extract_features(interactions)

        # Build model
        self.build_model()

        # Move model to device
        self.to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Training loop
        self.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            batch_count = 0

            # Process in batches
            for i in range(0, len(interactions), self.batch_size):
                batch = interactions[i : i + self.batch_size]

                # Prepare batch data
                user_features, item_features, labels = self._prepare_batch(batch)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self(user_features, item_features)

                # Compute loss
                loss = F.binary_cross_entropy(outputs.squeeze(), labels.float())

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Update statistics
                total_loss += loss.item()
                batch_count += 1

            # Record epoch loss
            avg_loss = total_loss / batch_count
            self.loss_history.append(avg_loss)

            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

        self.is_fitted = True

    def predict(self, user: Any, item: Any, features: Dict[str, Any]) -> float:
        """
        Predict probability of interaction between user and item.

        Args:
            user: User ID
            item: Item ID
            features: Features dictionary

        Returns:
            Predicted probability

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")

        # Check if user and item exist
        if user not in self.user_map:
            raise ValueError(f"Unknown user: {user}")
        if item not in self.item_map:
            raise ValueError(f"Unknown item: {item}")

        # Prepare features
        user_features = torch.tensor([self.user_map[user]], device=self.device)
        item_features = torch.tensor([self.item_map[item]], device=self.device)

        # Make prediction
        self.eval()
        with torch.no_grad():
            prediction = self(user_features, item_features)

        return prediction.item()

    def recommend(
        self,
        user: Any,
        top_n: int = 10,
        exclude_seen: bool = True,
        features: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Generate recommendations for user.

        Args:
            user: User ID
            top_n: Number of recommendations
            exclude_seen: Whether to exclude seen items
            features: Optional features dictionary

        Returns:
            List of (item, score) tuples

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")

        if user not in self.user_map:
            return []

        # Get predictions for all items
        predictions = []
        for item in self.item_map:
            try:
                score = self.predict(user, item, features or {})
                predictions.append((item, score))
            except Exception:
                continue

        # Sort by score
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:top_n]

    def save(self, filepath: str):
        """
        Save model to file.

        Args:
            filepath: Path to save model

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")

        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model state
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "user_map": self.user_map,
                "item_map": self.item_map,
                "feature_names": self.feature_names,
                "categorical_features": self.categorical_features,
                "numerical_features": self.numerical_features,
                "feature_encoders": self.feature_encoders,
                "field_dims": self.field_dims,
                "config": {
                    "embed_dim": self.embed_dim,
                    "mlp_dims": self.mlp_dims,
                    "dropout": self.dropout,
                    "attention_dim": self.attention_dim,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "seed": self.seed,
                },
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str):
        """
        Load model from file.

        Args:
            filepath: Path to load model from

        Returns:
            Loaded model

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Load checkpoint
        checkpoint = torch.load(filepath)

        # Create new instance
        instance = cls(
            embed_dim=checkpoint["config"]["embed_dim"],
            mlp_dims=checkpoint["config"]["mlp_dims"],
            field_dims=checkpoint["field_dims"],
            dropout=checkpoint["config"]["dropout"],
            attention_dim=checkpoint["config"]["attention_dim"],
            batch_size=checkpoint["config"]["batch_size"],
            learning_rate=checkpoint["config"]["learning_rate"],
            num_epochs=checkpoint["config"]["num_epochs"],
            seed=checkpoint["config"]["seed"],
        )

        # Restore state
        instance.user_map = checkpoint["user_map"]
        instance.item_map = checkpoint["item_map"]
        instance.feature_names = checkpoint["feature_names"]
        instance.categorical_features = checkpoint["categorical_features"]
        instance.numerical_features = checkpoint["numerical_features"]
        instance.feature_encoders = checkpoint["feature_encoders"]

        # Build model and load state
        instance.build_model()
        instance.load_state_dict(checkpoint["model_state_dict"])

        # Create optimizer and load state
        instance.optimizer = torch.optim.Adam(instance.parameters(), lr=instance.learning_rate)
        instance.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        instance.is_fitted = True

        return instance

    def _extract_features(self, interactions: List[Tuple]):
        """
        Extract features from interactions.

        Args:
            interactions: List of (user, item, features) tuples

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Extract users and items
        users = set()
        items = set()
        for user, item, _ in interactions:
            users.add(user)
            items.add(item)

        # Create mappings
        self.user_map = {user: idx for idx, user in enumerate(sorted(users))}
        self.item_map = {item: idx for idx, item in enumerate(sorted(items))}

        # Set field dimensions
        self.field_dims = [len(users), len(items)]

    def _prepare_batch(self, batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for training.

        Args:
            batch: List of (user, item, features) tuples

        Returns:
            Tuple of (user_features, item_features, labels)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        user_features = []
        item_features = []
        labels = []

        for user, item, features in batch:
            user_features.append(self.user_map[user])
            item_features.append(self.item_map[item])
            labels.append(1)  # Assuming all interactions are positive

        return (
            torch.tensor(user_features, device=self.device),
            torch.tensor(item_features, device=self.device),
            torch.tensor(labels, device=self.device),
        )
