from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import logging
import pickle
from pathlib import Path
import scipy.sparse as sp
from corerec.base_recommender import BaseCorerec


class HookManager:
    """Manager for model hooks to inspect internal states."""

    def __init__(self):
        """Initialize the hook manager."""
        self.hooks = {}
        self.activations = {}

    def _get_activation(self, name):
        """Get activation for a specific layer."""

        def hook(model, input, output):
            self.activations[name] = output.detach()

        return hook

    def register_hook(self, model, layer_name, callback=None):
        """Register a hook for a specific layer."""
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            if callback is None:
                callback = self._get_activation(layer_name)
            handle = layer.register_forward_hook(callback)
            self.hooks[layer_name] = handle
            return True

        # Try to find the layer in submodules
        for name, module in model.named_modules():
            if name == layer_name:
                if callback is None:
                    callback = self._get_activation(layer_name)
                handle = module.register_forward_hook(callback)
                self.hooks[layer_name] = handle
                return True

        return False

    def remove_hook(self, layer_name):
        """Remove a hook for a specific layer."""
        if layer_name in self.hooks:
            self.hooks[layer_name].remove()
            del self.hooks[layer_name]
            return True
        return False

    def get_activation(self, layer_name):
        """Get the activation for a specific layer."""
        return self.activations.get(layer_name, None)

    def clear_activations(self):
        """Clear all stored activations."""
        self.activations.clear()


class FeaturesEmbedding(nn.Module):
    """Embedding layer for categorical features."""

    def __init__(self, field_dims: List[int], embedding_dim: int):
        """
        Initialize the embedding layer.

        Args:
            field_dims: List of cardinalities of categorical features.
            embedding_dim: Dimension of embeddings.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.field_dims = field_dims
        self.num_fields = len(field_dims)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        self.embedding = nn.Embedding(sum(field_dims), embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding layer.

        Args:
            x: Input tensor of shape (batch_size, num_fields).
                Each value is the original feature value.

        Returns:
            Embedded features of shape (batch_size, num_fields, embedding_dim).
        """
        # Apply offsets to convert categorical features to global indices
        x = x + torch.tensor(self.offsets, device=x.device).unsqueeze(0)

        # Get embeddings
        return self.embedding(x)


class FeaturesLinear(nn.Module):
    """Linear part of the AutoInt model."""

    def __init__(self, field_dims: List[int], output_dim: int = 1):
        """
        Initialize the linear component.

        Args:
            field_dims: List of cardinalities of categorical features.
            output_dim: Output dimension.
        """
        super().__init__()
        self.field_dims = field_dims
        self.output_dim = output_dim
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        self.fc = nn.Embedding(sum(field_dims), output_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear component.

        Args:
            x: Input tensor of shape (batch_size, num_fields).

        Returns:
            Linear output of shape (batch_size, output_dim).
        """
        # Apply offsets to convert categorical features to global indices
        x = x + torch.tensor(self.offsets, device=x.device).unsqueeze(0)

        # Sum embeddings for each field
        return torch.sum(self.fc(x), dim=1) + self.bias


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer for feature interactions."""

    def __init__(
        self, embedding_dim: int, attention_dim: int, num_heads: int, dropout: float = 0.1
    ):
        """
        Initialize the multi-head attention layer.

        Args:
            embedding_dim: Dimension of input embeddings.
            attention_dim: Dimension of attention layer.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        # Ensure attention_dim is divisible by num_heads
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"
        self.head_dim = attention_dim // num_heads

        # Query, key, value projections
        self.W_query = nn.Linear(embedding_dim, attention_dim)
        self.W_key = nn.Linear(embedding_dim, attention_dim)
        self.W_value = nn.Linear(embedding_dim, attention_dim)

        # Output projection
        self.W_out = nn.Linear(attention_dim, embedding_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Initialize weights
        for layer in [self.W_query, self.W_key, self.W_value, self.W_out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim).

        Args:
            x: Input tensor of shape (batch_size, seq_len, attention_dim).

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the heads back into original shape.

        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim).

        Returns:
            Tensor of shape (batch_size, seq_len, attention_dim).
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        return x.contiguous().view(batch_size, seq_len, self.attention_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head attention layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            Attention output of shape (batch_size, seq_len, embedding_dim).
        """
        # Input shape: (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, _ = x.size()

        # Store residual for skip connection
        residual = x

        # Linear projections
        query = self.W_query(x)  # (batch_size, seq_len, attention_dim)
        key = self.W_key(x)  # (batch_size, seq_len, attention_dim)
        value = self.W_value(x)  # (batch_size, seq_len, attention_dim)

        # Split heads
        query = self.split_heads(query)  # (batch_size, num_heads, seq_len, head_dim)
        key = self.split_heads(key)  # (batch_size, num_heads, seq_len, head_dim)
        value = self.split_heads(value)  # (batch_size, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        # (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim**0.5)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        # (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attention_weights, value)

        # Combine heads
        context = self.combine_heads(context)  # (batch_size, seq_len, attention_dim)

        # Final linear projection
        output = self.W_out(context)  # (batch_size, seq_len, embedding_dim)

        # Add residual connection and apply layer normalization
        output = self.layer_norm(residual + output)

        return output


class InteractingLayer(nn.Module):
    """A single layer of feature interaction using multi-head attention."""

    def __init__(
        self, embedding_dim: int, attention_dim: int, num_heads: int, dropout: float = 0.1
    ):
        """
        Initialize the interacting layer.

        Args:
            embedding_dim: Dimension of input embeddings.
            attention_dim: Dimension of attention layer.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, attention_dim, num_heads, dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Initialize weights
        for layer in self.feedforward:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the interacting layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            Output tensor of shape (batch_size, seq_len, embedding_dim).
        """
        # Apply multi-head attention
        attention_output = self.attention(x)

        # Apply feedforward network
        feedforward_output = self.feedforward(attention_output)

        # Add residual connection and apply layer normalization
        output = self.layer_norm(attention_output + feedforward_output)

        return output


class AutoIntModel(nn.Module):
    """AutoInt model combining embedding, multi-head attention, and linear parts."""

    def __init__(
        self,
        field_dims: List[int],
        embedding_dim: int,
        attention_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        """
        Initialize the AutoInt model.

        Args:
            field_dims: List of cardinalities of categorical features.
            embedding_dim: Dimension of embedding vectors.
            attention_dim: Dimension of attention layer.
            num_heads: Number of attention heads.
            num_layers: Number of interacting layers.
            dropout: Dropout probability.
            output_dim: Output dimension.
        """
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embedding_dim)
        self.linear = FeaturesLinear(field_dims, output_dim)
        self.interacting_layers = nn.ModuleList(
            [
                InteractingLayer(embedding_dim, attention_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(len(field_dims) * embedding_dim, output_dim)

        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AutoInt model.

        Args:
            x: Input tensor of shape (batch_size, num_fields).

        Returns:
            Prediction of shape (batch_size, output_dim).
        """
        # Get embeddings and linear part
        embedded_features = self.embedding(x)  # (batch_size, num_fields, embedding_dim)
        linear_part = self.linear(x)  # (batch_size, output_dim)

        # Apply interacting layers
        interacting_part = embedded_features
        for layer in self.interacting_layers:
            interacting_part = layer(interacting_part)

        # Flatten and apply output layer
        batch_size = x.size(0)
        flattened = interacting_part.view(batch_size, -1)
        interacting_part = self.output_layer(flattened)

        # Combine linear and interacting parts
        output = linear_part + interacting_part

        return output.squeeze(-1)

    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention weights for visualization.

        Args:
            x: Input tensor of shape (batch_size, num_fields).

        Returns:
            List of attention weight tensors from each layer.
        """
        # Get embeddings
        embedded_features = self.embedding(x)

        attention_weights = []
        interacting_part = embedded_features

        # Get attention weights from each layer
        for layer in self.interacting_layers:
            # Register a temporary hook to capture attention weights
            weights = []

            def get_attention(module, input, output):
                # Retrieve attention weights from the module
                weights.append(module.attention_weights.detach().cpu())

            handle = layer.attention.register_forward_hook(get_attention)
            interacting_part = layer(interacting_part)
            handle.remove()

            if weights:
                attention_weights.append(weights[0])

        return attention_weights


class AutoInt_base(BaseCorerec):
    """Base class for AutoInt model implementation."""

    def __init__(
        self,
        name: str = "AutoInt",
        trainable: bool = True,
        verbose: bool = True,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        """
        Initialize the AutoInt base model.

        Args:
            name: Model name.
            trainable: Whether the model is trainable.
            verbose: Whether to print verbose information.
            config: Configuration dictionary.
            seed: Random seed for reproducibility.
        """
        super().__init__(name, trainable)
        self.verbose = verbose
        self.seed = seed

        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Default configuration
        default_config = {
            "embedding_dim": 16,
            "attention_dim": 32,
            "num_heads": 2,
            "num_layers": 2,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "weight_decay": 1e-6,
            "batch_size": 256,
            "num_epochs": 20,
            "patience": 5,
            "min_delta": 0.001,
            "validation_size": 0.1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        # Override defaults with provided config
        self.config = default_config.copy()
        if config is not None:
            self.config.update(config)

        # Set device
        self.device = torch.device(self.config["device"])

        # Initialize hook manager
        self.hooks = HookManager()

        # Version for tracking
        self.version = "1.0.0"

        # Model state
        self.is_fitted = False
        self.model = None
        self.num_users = 0
        self.num_items = 0
        self.field_dims = []
        self.user_ids = []
        self.item_ids = []
        self.uid_map = {}
        self.iid_map = {}
        self.interaction_matrix = None

    def _build_model(self):
        """Build the AutoInt model."""
        self.model = AutoIntModel(
            field_dims=self.field_dims,
            embedding_dim=self.config["embedding_dim"],
            attention_dim=self.config["attention_dim"],
            num_heads=self.config["num_heads"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
        ).to(self.device)

    def register_hook(self, layer_name, callback=None):
        """
        Register a hook to inspect model internals.

        Args:
            layer_name: Name of the layer to hook.
            callback: Optional callback function.

        Returns:
            Boolean indicating success.
        """
        if self.model is None:
            raise RuntimeError("Model is not built yet. Call fit() first.")

        return self.hooks.register_hook(self.model, layer_name, callback)

    def fit(
        self, interaction_matrix: sp.spmatrix, user_ids: List[str], item_ids: List[str]
    ) -> Dict[str, List[float]]:
        """
        Fit the model to the given interaction data.

        Args:
            interaction_matrix: User-item interaction matrix.
            user_ids: List of user IDs.
            item_ids: List of item IDs.

        Returns:
            Dictionary containing training history.
        """
        # Store the data
        self.interaction_matrix = interaction_matrix.copy()
        self.user_ids = user_ids.copy()
        self.item_ids = item_ids.copy()
        self.num_users = len(user_ids)
        self.num_items = len(item_ids)

        # Create mappings
        self.uid_map = {uid: i for i, uid in enumerate(user_ids)}
        self.iid_map = {iid: i for i, iid in enumerate(item_ids)}

        # Set field dimensions for the model
        self.field_dims = [self.num_users, self.num_items]

        # Build the model if not already built
        if self.model is None:
            self._build_model()

        if not self.trainable:
            if self.verbose:
                print("Model is not trainable. Skipping training.")
            self.is_fitted = True
            return {"loss": [], "val_loss": []}

        # Convert to PyTorch dataset
        interactions = []
        for u in range(self.num_users):
            for i in range(self.num_items):
                if interaction_matrix[u, i] > 0:
                    interactions.append((u, i, 1.0))  # Positive interaction

        if self.verbose:
            print(f"Number of positive interactions: {len(interactions)}")

        # Create training and validation sets
        np.random.shuffle(interactions)
        val_size = int(len(interactions) * self.config["validation_size"])
        train_interactions = interactions[val_size:]
        val_interactions = interactions[:val_size]

        # Create optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Loss function
        loss_fn = nn.BCEWithLogitsLoss()

        # Training loop
        batch_size = self.config["batch_size"]
        num_epochs = self.config["num_epochs"]
        patience = self.config["patience"]
        min_delta = self.config["min_delta"]
        best_val_loss = float("inf")
        epochs_no_improve = 0
        history = {"loss": [], "val_loss": []}

        if self.verbose:
            print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            np.random.shuffle(train_interactions)

            train_loss = 0.0
            num_batches = (len(train_interactions) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(train_interactions))
                batch = train_interactions[start_idx:end_idx]

                # Create positive examples
                pos_users = [interaction[0] for interaction in batch]
                pos_items = [interaction[1] for interaction in batch]
                pos_labels = [interaction[2] for interaction in batch]

                # Create negative examples
                neg_users = pos_users.copy()
                neg_items = []
                for u in pos_users:
                    while True:
                        i = np.random.randint(0, self.num_items)
                        if interaction_matrix[u, i] == 0:
                            neg_items.append(i)
                            break
                neg_labels = [0.0] * len(neg_users)

                # Combine positive and negative examples
                users = pos_users + neg_users
                items = pos_items + neg_items
                labels = pos_labels + neg_labels

                # Convert to tensors
                x = torch.tensor(
                    [[users[i], items[i]] for i in range(len(users))], dtype=torch.long
                ).to(self.device)
                y = torch.tensor(labels, dtype=torch.float32).to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = loss_fn(outputs, y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch)

            train_loss /= len(train_interactions)
            history["loss"].append(train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                num_val_batches = (len(val_interactions) + batch_size - 1) // batch_size

                for batch_idx in range(num_val_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(val_interactions))
                    val_batch = val_interactions[start_idx:end_idx]

                    # Create examples
                    val_users = [interaction[0] for interaction in val_batch]
                    val_items = [interaction[1] for interaction in val_batch]
                    val_labels = [interaction[2] for interaction in val_batch]

                    # Convert to tensors
                    val_x = torch.tensor(
                        [[val_users[i], val_items[i]] for i in range(len(val_users))],
                        dtype=torch.long,
                    ).to(self.device)
                    val_y = torch.tensor(val_labels, dtype=torch.float32).to(self.device)

                    # Forward pass
                    val_outputs = self.model(val_x)
                    val_batch_loss = loss_fn(val_outputs, val_y)

                    val_loss += val_batch_loss.item() * len(val_batch)

                val_loss /= len(val_interactions)
                history["val_loss"].append(val_loss)

            if self.verbose:
                print(
                    f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        self.is_fitted = True

        return history

    def recommend(
        self, user_id: str, top_n: int = 10, exclude_seen: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User ID.
            top_n: Number of top items to recommend.
            exclude_seen: Whether to exclude already seen items.

        Returns:
            List of (item_id, score) tuples.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if user_id not in self.uid_map:
            if self.verbose:
                print(f"User {user_id} not found in training data.")
            return []

        u = self.uid_map[user_id]

        # Get all items
        all_items = list(range(self.num_items))

        # Exclude seen items if required
        if exclude_seen:
            seen_items = set()
            for i in range(self.num_items):
                if self.interaction_matrix[u, i] > 0:
                    seen_items.add(i)
            candidate_items = [i for i in all_items if i not in seen_items]
        else:
            candidate_items = all_items

        if not candidate_items:
            return []

        # Create input tensor
        user_tensor = torch.tensor([[u, i] for i in candidate_items], dtype=torch.long).to(
            self.device
        )

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(self.model(user_tensor)).cpu().numpy()

        # Create recommendations
        item_scores = [
            (self.item_ids[candidate_items[i]], scores[i].item())
            for i in range(len(candidate_items))
        ]

        # Sort by score in descending order
        item_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-n recommendations
        return item_scores[:top_n]

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Prepare model state
        model_state = {
            "name": self.name,
            "trainable": self.trainable,
            "verbose": self.verbose,
            "config": self.config,
            "seed": self.seed,
            "version": self.version,
            "state_dict": self.model.state_dict(),
            "user_ids": self.user_ids,
            "item_ids": self.item_ids,
            "uid_map": self.uid_map,
            "iid_map": self.iid_map,
        }

        # Save model state
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(model_state, f)

        # Save metadata
        metadata = {
            "model_type": "AutoInt",
            "num_users": self.num_users,
            "num_items": self.num_items,
            "timestamp": str(datetime.now()),
            "version": self.version,
        }

        with open(f"{path}.meta", "w") as f:
            yaml.dump(metadata, f)

        if self.verbose:
            print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AutoInt_base":
        """
        Load a model from disk.

        Args:
            path: Path to load the model from.

        Returns:
            Loaded model.
        """
        with open(path, "rb") as f:
            model_state = pickle.load(f)

        # Create a new instance with the loaded configuration
        model = cls(
            name=model_state["name"],
            trainable=model_state["trainable"],
            verbose=model_state["verbose"],
            config=model_state["config"],
            seed=model_state["seed"],
        )

        # Restore model state
        model.version = model_state["version"]
        model.user_ids = model_state["user_ids"]
        model.item_ids = model_state["item_ids"]
        model.uid_map = model_state["uid_map"]
        model.iid_map = model_state["iid_map"]
        model.num_users = len(model.user_ids)
        model.num_items = len(model.item_ids)
        model.field_dims = [model.num_users, model.num_items]
        model.is_fitted = True
        model.interaction_matrix = sp.csr_matrix((model.num_users, model.num_items))

        # Build model and load state
        model._build_model()
        model.model.load_state_dict(model_state["state_dict"])

        return model

    def get_attention_weights(self, user_id: str, item_id: str) -> List[np.ndarray]:
        """
        Get attention weights for a specific user-item pair.

        Args:
            user_id: User ID.
            item_id: Item ID.

        Returns:
            List of attention weight matrices from each layer.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if user_id not in self.uid_map or item_id not in self.iid_map:
            raise ValueError("User or item not found in training data.")

        u = self.uid_map[user_id]
        i = self.iid_map[item_id]

        # Create input tensor
        x = torch.tensor([[u, i]], dtype=torch.long).to(self.device)

        # Get attention weights
        self.model.eval()
        with torch.no_grad():
            attention_weights = self.model.get_attention_weights(x)

        # Convert to numpy arrays
        return [w.numpy() for w in attention_weights]

    def update_incremental(self, new_interactions, new_user_ids=None, new_item_ids=None):
        """
        Update model incrementally with new data.

        Args:
            new_interactions: New interaction matrix.
            new_user_ids: New user IDs.
            new_item_ids: New item IDs.

        Returns:
            Updated model.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Check if we have new users or items
        if new_user_ids is not None:
            # Add new users to mappings
            new_users = [uid for uid in new_user_ids if uid not in self.uid_map]
            for uid in new_users:
                self.uid_map[uid] = len(self.uid_map)
                self.user_ids.append(uid)
            self.num_users = len(self.user_ids)

        if new_item_ids is not None:
            # Add new items to mappings
            new_items = [iid for iid in new_item_ids if iid not in self.iid_map]
            for iid in new_items:
                self.iid_map[iid] = len(self.iid_map)
                self.item_ids.append(iid)
            self.num_items = len(self.item_ids)

            # If new items were added, we need to rebuild the model
            if new_items:
                old_state_dict = self.model.state_dict()
                self.field_dims = [self.num_users, self.num_items]
                self._build_model()

                # Copy weights for existing parameters
                new_state_dict = self.model.state_dict()
                for name, param in old_state_dict.items():
                    if name in new_state_dict and new_state_dict[name].shape == param.shape:
                        new_state_dict[name] = param

                self.model.load_state_dict(new_state_dict)

        # Update interaction matrix
        self.interaction_matrix = new_interactions.copy()

        # Fine-tune on new data
        self.fit(self.interaction_matrix, self.user_ids, self.item_ids)

        return self
