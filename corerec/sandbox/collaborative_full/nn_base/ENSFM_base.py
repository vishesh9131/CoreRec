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


class SimilarityEmbedding(nn.Module):
    """
    Similarity-based embedding module for ENSFM.

    Maps users and items to embedding vectors considering their similarity.

    Architecture:
    ┌─────────────────┐     ┌─────────────────┐
    │  User Features  │     │  Item Features  │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐
    │User Embeddings  │     │Item Embeddings  │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
    ┌─────────────────────────────────┐
    │ Similarity-based Representation │
    └─────────────────────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int):
        """
        Initialize similarity embedding layer.

        Args:
            num_users: Number of users
            num_items: Number of items
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim

        # User embeddings
        self.user_embedding = nn.Embedding(num_users, embed_dim)

        # Item embeddings
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(
        self, user_ids: torch.LongTensor, item_ids: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for similarity embedding.

        Args:
            user_ids: User ID tensor of shape (batch_size,)
            item_ids: Item ID tensor of shape (batch_size,)

        Returns:
            Tuple of (user_embed, item_embed) each of shape (batch_size, embed_dim)
        """
        # Get embeddings
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)

        return user_embed, item_embed


class FactorizationMachine(nn.Module):
    """
    Factorization Machine module for ENSFM.

    Models second-order feature interactions using factorized parameters.

    Architecture:
    ┌─────────────────┐
    │  Features       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │Linear + Quadratic│
    │    Terms        │
    └─────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initialize FM module.

        Args:
            input_dim: Input feature dimension
            latent_dim: Latent factor dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Linear term
        self.linear = nn.Linear(input_dim, 1, bias=True)

        # Factorized parameters for second-order interactions
        self.v = nn.Parameter(torch.zeros(input_dim, latent_dim))
        nn.init.xavier_uniform_(self.v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FM.

        Args:
            x: Input feature tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Linear term
        linear_term = self.linear(x)

        # Quadratic term
        square_of_sum = torch.mm(x, self.v) ** 2
        sum_of_square = torch.mm(x**2, self.v**2)
        quadratic_term = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        return linear_term + quadratic_term


class NeuralCF(nn.Module):
    """
    Neural Collaborative Filtering module for ENSFM.

    Models complex user-item interactions with deep neural networks.

    Architecture:
    ┌─────────────┐      ┌─────────────┐
    │User Embedding│      │Item Embedding│
    └──────┬──────┘      └──────┬──────┘
           │                    │
           └──────────┬─────────┘
                      │
                      ▼
    ┌───────────────────────────────────┐
    │           Concat Layer            │
    └───────────────┬───────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │          Hidden Layer 1           │
    └───────────────┬───────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │          Hidden Layer N           │
    └───────────────┬───────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │           Output Layer            │
    └───────────────────────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self, embed_dim: int, hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.1
    ):
        """
        Initialize Neural CF module.

        Args:
            embed_dim: Embedding dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = embed_dim * 2  # User and item embeddings concatenated

        # Create MLP layers
        layers = []
        input_size = self.input_dim

        for hidden_size in hidden_dims:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, user_embed: torch.Tensor, item_embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Neural CF.

        Args:
            user_embed: User embedding of shape (batch_size, embed_dim)
            item_embed: Item embedding of shape (batch_size, embed_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Concatenate user and item embeddings
        concat_embed = torch.cat([user_embed, item_embed], dim=1)

        # Pass through MLP
        output = self.mlp(concat_embed)

        return output


class SimilarityLayer(nn.Module):
    """
    Similarity calculation layer for ENSFM.

    Computes similarity between user and item embeddings.

    Architecture:
    ┌─────────────┐      ┌─────────────┐
    │User Embedding│      │Item Embedding│
    └──────┬──────┘      └──────┬──────┘
           │                    │
           └──────────┬─────────┘
                      │
                      ▼
    ┌───────────────────────────────────┐
    │      Cosine Similarity / Dot      │
    └───────────────────────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, similarity_type: str = "cosine"):
        """
        Initialize similarity layer.

        Args:
            similarity_type: Type of similarity to use ('cosine' or 'dot')
        """
        super().__init__()
        assert similarity_type in ["cosine", "dot"], "Similarity type must be 'cosine' or 'dot'"
        self.similarity_type = similarity_type

    def forward(self, user_embed: torch.Tensor, item_embed: torch.Tensor) -> torch.Tensor:
        """
        Calculate similarity between user and item embeddings.

        Args:
            user_embed: User embedding of shape (batch_size, embed_dim)
            item_embed: Item embedding of shape (batch_size, embed_dim)

        Returns:
            Similarity score of shape (batch_size, 1)
        """
        if self.similarity_type == "cosine":
            # Normalize embeddings
            user_embed_norm = F.normalize(user_embed, p=2, dim=1)
            item_embed_norm = F.normalize(item_embed, p=2, dim=1)

            # Calculate cosine similarity
            cosine_sim = torch.sum(user_embed_norm * item_embed_norm, dim=1, keepdim=True)
            return cosine_sim
        else:  # dot product
            # Calculate dot product
            dot_prod = torch.sum(user_embed * item_embed, dim=1, keepdim=True)
            return dot_prod


class ENSFMModel(nn.Module):
    """
    Ensemble Neural Similarity Factorization Machine model.

    Combines Neural CF and Similarity-based approaches for recommendation.

    Architecture:
    ┌─────────────┐      ┌─────────────┐
    │  User IDs   │      │  Item IDs   │
    └──────┬──────┘      └──────┬──────┘
           │                    │
           ▼                    ▼
    ┌─────────────┐      ┌─────────────┐
    │User Embedding│      │Item Embedding│
    └──────┬──────┘      └──────┬──────┘
           │                    │
           │                    │
           ▼                    ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │ Neural CF   │      │ Similarity  │      │ FM Component│
    │ Component   │      │ Component   │      │             │
    └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
           │                    │                    │
           └──────────┬─────────┴──────────┬────────┘
                      │                    │
                      ▼                    ▼
    ┌───────────────────────────────┐    ┌─────────────────┐
    │        Weighted Sum           │───►│ Final Prediction│
    └───────────────────────────────┘    └─────────────────┘

    References:
        - Chen, Chong, et al. "An ensemble learning approach for context-aware recommendation."
          IEEE Transactions on Knowledge and Data Engineering (2019).

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.1,
        similarity_type: str = "cosine",
    ):
        """
        Initialize the ENSFM model.

        Args:
            num_users: Number of users
            num_items: Number of items
            embed_dim: Embedding dimension
            hidden_dims: Hidden layer dimensions for Neural CF
            dropout: Dropout rate
            similarity_type: Type of similarity to use ('cosine' or 'dot')
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim

        # Similarity embedding layer
        self.embedding = SimilarityEmbedding(num_users, num_items, embed_dim)

        # Neural CF component
        self.ncf = NeuralCF(embed_dim, hidden_dims, dropout)

        # Similarity component
        self.similarity = SimilarityLayer(similarity_type)

        # FM component (for handling additional features)
        self.fm = FactorizationMachine(embed_dim * 2, embed_dim)

        # Component weights (learned during training)
        self.ncf_weight = nn.Parameter(torch.tensor(0.33))
        self.sim_weight = nn.Parameter(torch.tensor(0.33))
        self.fm_weight = nn.Parameter(torch.tensor(0.34))

    def forward(self, user_ids: torch.LongTensor, item_ids: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass of ENSFM.

        Args:
            user_ids: User ID tensor of shape (batch_size,)
            item_ids: Item ID tensor of shape (batch_size,)

        Returns:
            Prediction tensor of shape (batch_size, 1)
        """
        # Get embeddings
        user_embed, item_embed = self.embedding(user_ids, item_ids)

        # Neural CF component
        ncf_output = self.ncf(user_embed, item_embed)

        # Similarity component
        sim_output = self.similarity(user_embed, item_embed)

        # FM component
        concat_embed = torch.cat([user_embed, item_embed], dim=1)
        fm_output = self.fm(concat_embed)

        # Normalize weights using softmax to ensure they sum to 1
        weights = F.softmax(torch.stack([self.ncf_weight, self.sim_weight, self.fm_weight]), dim=0)

        # Weighted ensemble
        output = weights[0] * ncf_output + weights[1] * sim_output + weights[2] * fm_output

        # Apply sigmoid for final prediction
        return torch.sigmoid(output)


class ENSFM_base(BaseRecommender):
    """
    Ensemble Neural Similarity Factorization Machine implementation.

    ENSFM combines neural collaborative filtering and similarity approaches
    to achieve better recommendation performance.

    Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │                      ENSFM_base                           │
    │                                                           │
    │  ┌─────────────┐    ┌───────────────┐    ┌────────────┐  │
    │  │ User & Item │    │    Mapping    │    │Data Process│  │
    │  └──────┬──────┘    └───────┬───────┘    └──────┬─────┘  │
    │         │                   │                   │         │
    │         └───────────┬───────┴─────────┬─────────┘         │
    │                     │                 │                    │
    │                     ▼                 ▼                    │
    │            ┌────────────────┐  ┌─────────────┐            │
    │            │ ENSFM Model    │  │Training Loop│            │
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
        - Chen, Chong, et al. "An ensemble learning approach for context-aware recommendation."
          IEEE Transactions on Knowledge and Data Engineering (2019).

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        name: str = "ENSFM",
        embed_dim: int = 64,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.1,
        similarity_type: str = "cosine",
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
        Initialize the ENSFM model.

        Args:
            name: Model name
            embed_dim: Embedding dimension
            hidden_dims: Hidden layer dimensions for Neural CF
            dropout: Dropout rate
            similarity_type: Type of similarity ('cosine' or 'dot')
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
            self.similarity_type = config.get("similarity_type", similarity_type)
            self.learning_rate = config.get("learning_rate", learning_rate)
            self.batch_size = config.get("batch_size", batch_size)
            self.num_epochs = config.get("num_epochs", num_epochs)
            self.patience = config.get("patience", patience)
            self.shuffle = config.get("shuffle", shuffle)
        else:
            self.embed_dim = embed_dim
            self.hidden_dims = hidden_dims
            self.dropout = dropout
            self.similarity_type = similarity_type
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.patience = patience
            self.shuffle = shuffle

        # Validate parameters
        assert self.similarity_type in [
            "cosine",
            "dot",
        ], "Similarity type must be 'cosine' or 'dot'"

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
        self.user_map = {}
        self.item_map = {}

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
            data: List of dictionaries with user_id, item_id, and label
        """
        # Extract unique users and items
        user_ids = set()
        item_ids = set()

        for sample in data:
            if "user_id" in sample:
                user_ids.add(sample["user_id"])
            if "item_id" in sample:
                item_ids.add(sample["item_id"])

        # Create mappings
        self.user_map = {user_id: i for i, user_id in enumerate(sorted(list(user_ids)))}
        self.item_map = {item_id: i for i, item_id in enumerate(sorted(list(item_ids)))}

        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)

        if self.verbose:
            self.logger.info(f"Identified {self.num_users} users and {self.num_items} items")

    def _build_model(self):
        """Build the ENSFM model."""
        self.model = ENSFMModel(
            num_users=self.num_users,
            num_items=self.num_items,
            embed_dim=self.embed_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            similarity_type=self.similarity_type,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.verbose:
            self.logger.info(
                f"Built ENSFM model with {self.num_users} users and {self.num_items} items"
            )
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Total parameters: {total_params}")

    def _prepare_batch(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for training.

        Args:
            batch: List of dictionaries with user_id, item_id, and label

        Returns:
            Tuple of (user_ids, item_ids, labels)
        """
        batch_size = len(batch)

        # Initialize tensors
        user_ids = torch.zeros(batch_size, dtype=torch.long)
        item_ids = torch.zeros(batch_size, dtype=torch.long)
        labels = torch.zeros(batch_size, 1, dtype=torch.float)

        # Fill tensors with data
        for i, sample in enumerate(batch):
            # User ID
            if "user_id" in sample:
                user_id = sample["user_id"]
                user_idx = self.user_map.get(user_id, 0)
                user_ids[i] = user_idx

            # Item ID
            if "item_id" in sample:
                item_id = sample["item_id"]
                item_idx = self.item_map.get(item_id, 0)
                item_ids[i] = item_idx

            # Label
            if "label" in sample:
                labels[i, 0] = float(sample["label"])

        return (user_ids.to(self.device), item_ids.to(self.device), labels.to(self.device))

    def fit(self, data: List[Dict[str, Any]]) -> "ENSFM_base":
        """
        Fit the ENSFM model.

        Args:
            data: List of dictionaries with user_id, item_id, and label

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
                user_ids, item_ids, labels = self._prepare_batch(batch)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(user_ids, item_ids)

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

    def predict(self, user_id: Any, item_id: Any) -> float:
        """
        Predict rating for a user-item pair.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Predicted rating
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        # Get user and item indices
        user_idx = self.user_map.get(user_id, 0)
        item_idx = self.item_map.get(item_id, 0)

        # Convert to tensors
        user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
        item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(user_tensor, item_tensor).item()

        return prediction

    def recommend(
        self, user_id: Any, item_pool: List[Any] = None, top_n: int = 10
    ) -> List[Tuple[Any, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User ID
            item_pool: Pool of items to recommend from (default: all items)
            top_n: Number of recommendations to generate

        Returns:
            List of (item_id, score) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making recommendations")

        # If no item pool provided, use all items
        if item_pool is None:
            item_pool = list(self.item_map.keys())

        # Extract item_id from dicts if needed
        item_ids = []
        for item in item_pool:
            if isinstance(item, dict):
                if "item_id" in item:
                    item_ids.append(item["item_id"])
                else:
                    raise ValueError("item_pool dicts must contain 'item_id' key")
            else:
                item_ids.append(item)

        # Score each item in the pool
        scored_items = []
        for item_id in item_ids:
            score = self.predict(user_id, item_id)
            scored_items.append((item_id, score))

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
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "similarity_type": self.similarity_type,
                "name": self.name,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "patience": self.patience,
                "shuffle": self.shuffle,
                "seed": self.seed,
                "verbose": self.verbose,
            },
            "user_map": self.user_map,
            "item_map": self.item_map,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss_history": self.loss_history if hasattr(self, "loss_history") else [],
        }

        # Save to file
        torch.save(model_data, filepath)

        if self.verbose:
            self.logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "ENSFM_base":
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
            hidden_dims=model_data["model_config"]["hidden_dims"],
            dropout=model_data["model_config"]["dropout"],
            similarity_type=model_data["model_config"]["similarity_type"],
            learning_rate=model_data["model_config"]["learning_rate"],
            batch_size=model_data["model_config"]["batch_size"],
            num_epochs=model_data["model_config"]["num_epochs"],
            patience=model_data["model_config"]["patience"],
            shuffle=model_data["model_config"]["shuffle"],
            seed=model_data["model_config"]["seed"],
            verbose=model_data["model_config"]["verbose"],
            device=device,
        )

        # Restore model data
        instance.user_map = model_data["user_map"]
        instance.item_map = model_data["item_map"]
        instance.num_users = model_data["num_users"]
        instance.num_items = model_data["num_items"]
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
