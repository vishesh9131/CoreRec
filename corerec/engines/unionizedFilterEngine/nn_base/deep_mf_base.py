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
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from corerec.base_recommender import BaseCorerec


class HookManager:
    """
    Manager for model hooks to inspect internal states.

    This class provides functionality to register hooks on specific layers
    of PyTorch models to capture and analyze their activations during forward passes.

    Architecture:
    ┌───────────────┐
    │  HookManager  │
    ├───────────────┤
    │  hooks        │◄───── Stores hook handles
    │  activations  │◄───── Stores layer outputs
    └───────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self):
        """Initialize the hook manager."""
        self.hooks = {}
        self.activations = {}

    def _get_activation(self, name):
        """
        Get activation for a specific layer.

        Args:
            name: Name of the layer to capture activations from.

        Returns:
            A hook function that stores layer outputs.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """

        def hook(model, input, output):
            self.activations[name] = output.detach()

        return hook

    def register_hook(self, model, layer_name, callback=None):
        """
        Register a hook for a specific layer.

        Args:
            model: PyTorch model to register hook on.
            layer_name: Name of the layer to hook.
            callback: Custom hook function (optional). If None, will use default.

        Returns:
            bool: Whether the hook was successfully registered.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
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
        """
        Remove a hook for a specific layer.

        Args:
            layer_name: Name of the layer to remove hook from.

        Returns:
            bool: Whether the hook was successfully removed.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if layer_name in self.hooks:
            self.hooks[layer_name].remove()
            del self.hooks[layer_name]
            if layer_name in self.activations:
                del self.activations[layer_name]
            return True
        return False

    def clear_hooks(self):
        """
        Remove all hooks.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
        self.activations.clear()

    def get_activation(self, layer_name):
        """
        Get activation for a specific layer.

        Args:
            layer_name: Name of the layer.

        Returns:
            Activation tensor or None if not available.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return self.activations.get(layer_name)


class DeepMFModel(nn.Module):
    """
    Deep Matrix Factorization neural network model.

    This model implements a neural network-based matrix factorization approach
    for recommendation, allowing for more complex user-item interactions.

    Architecture:

    [User Embedding] [Item Embedding]
           │                │
           └────────┬───────┘
                    │
                    ▼
             [Concatenation]
                    │
                    ▼
         ┌──────────────────┐
         │   Hidden Layer 1 │
         └─────────┬────────┘
                   │
                   ▼
         ┌──────────────────┐
         │   Hidden Layer 2 │
         └─────────┬────────┘
                   │
                   ▼
         ┌──────────────────┐
         │   Output Layer   │
         └─────────┬────────┘
                   │
                   ▼
              [Prediction]

    Args:
        num_users: Number of users in the dataset.
        num_items: Number of items in the dataset.
        embedding_dim: Dimension of user and item embeddings.
        hidden_layers: Dimensions of hidden layers in the MLP.
        dropout: Dropout rate for the MLP.
        activation: Activation function to use ('relu', 'tanh', or 'sigmoid').

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_layers: List[int],
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super(DeepMFModel, self).__init__()

        # User and item embedding layers
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        # Setup activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()

        # MLP layers
        input_dim = embedding_dim * 2  # Concatenated user and item embeddings
        self.mlp_layers = nn.ModuleList()

        for hidden_dim in hidden_layers:
            self.mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            self.mlp_layers.append(self.activation)
            self.mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, user_ids, item_ids):
        """
        Forward pass of the model.

        Args:
            user_ids: Tensor of user IDs.
            item_ids: Tensor of item IDs.

        Returns:
            Predicted ratings or interaction probabilities.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)

        # Pass through MLP
        for layer in self.mlp_layers:
            x = layer(x)

        # Output layer
        logits = self.output_layer(x)

        # Apply sigmoid to get probabilities
        predictions = self.sigmoid(logits)

        return predictions.squeeze()


class DeepMF_base:
    """
    Deep Matrix Factorization base class for recommendation.

    This class implements a neural network-based matrix factorization approach,
    which models user-item interactions through a deep neural network rather than
    simple dot product of latent factors.

    Architecture:

    ┌───────────────┐    ┌───────────────┐
    │   User Data   │    │   Item Data   │
    └───────┬───────┘    └───────┬───────┘
            │                    │
    ┌───────▼───────┐    ┌───────▼───────┐
    │ User Embedding │    │ Item Embedding │
    └───────┬───────┘    └───────┬───────┘
            │                    │
            └────────┬───────────┘
                     │
             ┌───────▼───────┐
             │  Deep Neural  │
             │    Network    │
             └───────┬───────┘
                     │
             ┌───────▼───────┐
             │   Prediction  │
             └───────────────┘

    Args:
        name: Name of the model.
        embedding_dim: Dimension of user and item embeddings.
        hidden_layers: List of hidden layer dimensions.
        dropout: Dropout rate for regularization.
        activation: Activation function to use ('relu', 'tanh', or 'sigmoid').
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        num_epochs: Number of training epochs.
        seed: Random seed for reproducibility.
        device: Device to run the model on ('cpu' or 'cuda').
        verbose: Whether to show progress bars during training.

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        name: str = "DeepMF",
        embedding_dim: int = 32,
        hidden_layers: List[int] = None,
        dropout: float = 0.2,
        activation: str = "relu",
        batch_size: int = 64,
        learning_rate: float = 0.001,
        num_epochs: int = 20,
        seed: Optional[int] = None,
        device: str = None,
        verbose: bool = True,
    ):
        # Initialize model parameters
        self.name = name
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers or [64, 32, 16]
        self.dropout = dropout
        self.activation = activation
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.seed = seed if seed is not None else np.random.randint(1000)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.verbose = verbose

        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(self.seed)

        # Initialize logger
        self.logger = logging.getLogger(f"{self.name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        # Initialize hook manager for model introspection
        self.hook_manager = HookManager()

        # Initialize variables to be set during fit
        self.model = None
        self.user_map = {}  # Maps user IDs to indices
        self.item_map = {}  # Maps item IDs to indices
        self.reverse_user_map = {}  # Maps indices to user IDs
        self.reverse_item_map = {}  # Maps indices to item IDs
        self.num_users = 0
        self.num_items = 0
        self.loss_history = []
        self.is_fitted = False

    def _build_model(self):
        """
        Build the DeepMF model.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.model = DeepMFModel(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            activation=self.activation,
        ).to(self.device)

    def _create_user_item_mappings(self, interactions):
        """
        Create mappings between user/item IDs and indices.

        Args:
            interactions: List of (user_id, item_id, rating) tuples.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Extract unique user and item IDs
        user_ids = set()
        item_ids = set()

        for user_id, item_id, _ in interactions:
            user_ids.add(user_id)
            item_ids.add(item_id)

        # Create mappings (start from 1, reserve 0 for padding)
        self.user_map = {user_id: i + 1 for i, user_id in enumerate(user_ids)}
        self.item_map = {item_id: i + 1 for i, item_id in enumerate(item_ids)}

        # Create reverse mappings
        self.reverse_user_map = {i: user_id for user_id, i in self.user_map.items()}
        self.reverse_item_map = {i: item_id for item_id, i in self.item_map.items()}

        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)

        self.logger.info(f"Created mappings for {self.num_users} users and {self.num_items} items.")

    def _prepare_batches(self, interactions, batch_size=None):
        """
        Prepare batches for training.

        Args:
            interactions: List of (user_id, item_id, rating) tuples.
            batch_size: Batch size to use. If None, use self.batch_size.

        Returns:
            List of batches, where each batch is a tuple of (user_indices, item_indices, ratings).

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Shuffle interactions
        interaction_array = np.array(interactions, dtype=object)
        indices = np.arange(len(interactions))
        np.random.shuffle(indices)

        # Prepare batches
        batches = []

        for start_idx in range(0, len(interactions), batch_size):
            end_idx = min(start_idx + batch_size, len(interactions))
            batch_indices = indices[start_idx:end_idx]

            # Get batch interactions
            batch_interactions = interaction_array[batch_indices]

            # Extract user and item indices and ratings
            user_ids = [self.user_map[interaction[0]] for interaction in batch_interactions]
            item_ids = [self.item_map[interaction[1]] for interaction in batch_interactions]
            ratings = [float(interaction[2]) for interaction in batch_interactions]

            # Convert to tensors
            user_tensor = torch.tensor(user_ids, dtype=torch.long, device=self.device)
            item_tensor = torch.tensor(item_ids, dtype=torch.long, device=self.device)
            rating_tensor = torch.tensor(ratings, dtype=torch.float, device=self.device)

            batches.append((user_tensor, item_tensor, rating_tensor))

        return batches

    def fit(self, interactions, user_ids=None, item_ids=None):
        """
        Train the model with the provided interactions.

        Args:
            interactions: List of (user_id, item_id, rating) tuples.
            user_ids: Optional list of all user IDs. If None, will extract from interactions.
            item_ids: Optional list of all item IDs. If None, will extract from interactions.

        Returns:
            self

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.logger.info(f"Fitting {self.name} on {len(interactions)} interactions...")

        # Create user/item mappings
        self._create_user_item_mappings(interactions)

        # Build model
        self._build_model()

        # Prepare batches
        batches = self._prepare_batches(interactions)

        # Define optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        epoch_iterator = range(self.num_epochs)

        if self.verbose:
            epoch_iterator = tqdm(epoch_iterator, desc="Training", unit="epoch")

        for epoch in epoch_iterator:
            total_loss = 0.0

            batch_iterator = range(len(batches))
            if self.verbose > 1:
                batch_iterator = tqdm(
                    batch_iterator,
                    desc=f"Epoch {epoch+1}/{self.num_epochs}",
                    unit="batch",
                    leave=False,
                )

            for batch_idx in batch_iterator:
                user_ids, item_ids, ratings = batches[batch_idx]

                # Forward pass
                predictions = self.model(user_ids, item_ids)
                loss = criterion(predictions, ratings)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(user_ids)

            # Calculate average loss for the epoch
            avg_loss = total_loss / len(interactions)
            self.loss_history.append(avg_loss)

            # Log progress
            if self.verbose:
                epoch_iterator.set_postfix({"Loss": f"{avg_loss:.4f}"})

            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair.

        Args:
            user_id: ID of the user.
            item_id: ID of the item.

        Returns:
            Predicted rating.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Check if user and item exist in the mappings
        if user_id not in self.user_map:
            self.logger.warning(f"User {user_id} not in training data, using random prediction.")
            return np.random.random()

        if item_id not in self.item_map:
            self.logger.warning(f"Item {item_id} not in training data, using random prediction.")
            return np.random.random()

        # Get user and item indices
        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]

        # Convert to tensors
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)
        item_tensor = torch.tensor([item_idx], dtype=torch.long, device=self.device)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(user_tensor, item_tensor).item()

        return prediction

    def recommend(self, user_id, top_n=10, exclude_seen=True):
        """
        Recommend items for a user.

        Args:
            user_id: ID of the user.
            top_n: Number of recommendations to return.
            exclude_seen: Whether to exclude items the user has already interacted with.

        Returns:
            List of (item_id, score) tuples.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Check if user exists in the mappings
        if user_id not in self.user_map:
            self.logger.warning(
                f"User {user_id} not in training data, returning random recommendations."
            )
            return [
                (self.reverse_item_map[i], np.random.random())
                for i in np.random.randint(1, self.num_items + 1, size=top_n)
            ]

        # Get user index
        user_idx = self.user_map[user_id]

        # Make predictions for all items
        self.model.eval()
        predictions = []

        # Process in batches to avoid memory issues
        batch_size = 1024

        with torch.no_grad():
            for start_idx in range(1, self.num_items + 1, batch_size):
                end_idx = min(start_idx + batch_size, self.num_items + 1)
                item_indices = list(range(start_idx, end_idx))

                # Convert to tensors
                user_tensor = torch.tensor(
                    [user_idx] * len(item_indices), dtype=torch.long, device=self.device
                )
                item_tensor = torch.tensor(item_indices, dtype=torch.long, device=self.device)

                # Make predictions
                batch_predictions = self.model(user_tensor, item_tensor).cpu().numpy()

                # Store predictions
                for i, item_idx in enumerate(item_indices):
                    item_id = self.reverse_item_map[item_idx]
                    predictions.append((item_id, float(batch_predictions[i])))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Return top_n recommendations
        return predictions[:top_n]

    def get_similar_items(self, item_id, top_n=10):
        """
        Find similar items based on embedding similarity.

        Args:
            item_id: ID of the item.
            top_n: Number of similar items to return.

        Returns:
            List of (item_id, similarity_score) tuples.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Check if item exists in the mappings
        if item_id not in self.item_map:
            self.logger.warning(f"Item {item_id} not in training data, returning random items.")
            return [
                (self.reverse_item_map[i], np.random.random())
                for i in np.random.randint(1, self.num_items + 1, size=top_n)
                if i != item_id
            ]

        # Get item index
        item_idx = self.item_map[item_id]

        # Get item embeddings
        self.model.eval()
        with torch.no_grad():
            item_embeddings = self.model.item_embedding.weight[1:].cpu().numpy()  # Skip padding

            # Get target item embedding
            target_embedding = item_embeddings[item_idx - 1]  # -1 because indices start at 1

            # Compute similarities (cosine similarity)
            similarities = []
            for i in range(self.num_items):
                if i + 1 == item_idx:  # Skip target item
                    continue

                other_embedding = item_embeddings[i]
                similarity = np.dot(target_embedding, other_embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(other_embedding)
                )

                similarities.append((self.reverse_item_map[i + 1], float(similarity)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top_n similar items
        return similarities[:top_n]

    def save(self, filepath):
        """
        Save model to file.

        Args:
            filepath: Path to save the model to.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        # Prepare model data
        model_data = {
            "model_config": {
                "embedding_dim": self.embedding_dim,
                "hidden_layers": self.hidden_layers,
                "dropout": self.dropout,
                "activation": self.activation,
            },
            "user_map": self.user_map,
            "item_map": self.item_map,
            "reverse_user_map": self.reverse_user_map,
            "reverse_item_map": self.reverse_item_map,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "loss_history": self.loss_history,
            "state_dict": self.model.state_dict(),
            "seed": self.seed,
            "name": self.name,
        }

        # Save model data
        torch.save(model_data, filepath)

        # Save config separately for human readability
        config_path = f"{os.path.splitext(filepath)[0]}_config.yaml"
        config = {
            "name": self.name,
            "type": "DeepMF",
            "embedding_dim": self.embedding_dim,
            "hidden_layers": self.hidden_layers,
            "dropout": self.dropout,
            "activation": self.activation,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "seed": self.seed,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "saved_at": str(datetime.now()),
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        self.logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath, device=None):
        """
        Load model from file.

        Args:
            filepath: Path to load the model from.
            device: Device to load the model on. If None, use the same device as saved.

        Returns:
            Loaded model.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Load model data
        model_data = torch.load(filepath, map_location=device if device else "cpu")

        # Create new instance
        model = cls(
            name=model_data["name"],
            embedding_dim=model_data["model_config"]["embedding_dim"],
            hidden_layers=model_data["model_config"]["hidden_layers"],
            dropout=model_data["model_config"]["dropout"],
            activation=model_data["model_config"]["activation"],
            seed=model_data["seed"],
            device=device,
        )

        # Set attributes from saved model
        model.user_map = model_data["user_map"]
        model.item_map = model_data["item_map"]
        model.reverse_user_map = model_data["reverse_user_map"]
        model.reverse_item_map = model_data["reverse_item_map"]
        model.num_users = model_data["num_users"]
        model.num_items = model_data["num_items"]
        model.loss_history = model_data["loss_history"]

        # Build model and load state dictionary
        model._build_model()
        model.model.load_state_dict(model_data["state_dict"])
        model.is_fitted = True

        return model

    def register_hook(self, layer_name, callback=None):
        """
        Register a hook for a specific layer.

        Args:
            layer_name: Name of the layer to hook.
            callback: Custom hook function.

        Returns:
            Whether the hook was successfully registered.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not hasattr(self, "model") or self.model is None:
            self.logger.error("Model not initialized. Call fit() first.")
            return False

        return self.hook_manager.register_hook(self.model, layer_name, callback)

    def get_activation(self, layer_name):
        """
        Get activation for a specific layer.

        Args:
            layer_name: Name of the layer.

        Returns:
            Activation tensor.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return self.hook_manager.get_activation(layer_name)

    def get_user_embedding(self, user_id):
        """
        Get embedding for a specific user.

        Args:
            user_id: ID of the user.

        Returns:
            User embedding tensor.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if user_id not in self.user_map:
            self.logger.warning(f"User {user_id} not in training data.")
            return None

        user_idx = self.user_map[user_id]

        with torch.no_grad():
            embedding = self.model.user_embedding(
                torch.tensor([user_idx], dtype=torch.long, device=self.device)
            )

        return embedding.cpu().numpy()[0]

    def get_item_embedding(self, item_id):
        """
        Get embedding for a specific item.

        Args:
            item_id: ID of the item.

        Returns:
            Item embedding tensor.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if item_id not in self.item_map:
            self.logger.warning(f"Item {item_id} not in training data.")
            return None

        item_idx = self.item_map[item_id]

        with torch.no_grad():
            embedding = self.model.item_embedding(
                torch.tensor([item_idx], dtype=torch.long, device=self.device)
            )

        return embedding.cpu().numpy()[0]

    def export_user_embeddings(self):
        """
        Export all user embeddings.

        Returns:
            Dictionary mapping user IDs to embeddings.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        user_embeddings = {}

        with torch.no_grad():
            # Skip padding embedding at index 0
            embeddings = self.model.user_embedding.weight[1:].cpu().numpy()

            for user_id, user_idx in self.user_map.items():
                user_embeddings[user_id] = embeddings[user_idx - 1].tolist()

        return user_embeddings

    def export_item_embeddings(self):
        """
        Export all item embeddings.

        Returns:
            Dictionary mapping item IDs to embeddings.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        item_embeddings = {}

        with torch.no_grad():
            # Skip padding embedding at index 0
            embeddings = self.model.item_embedding.weight[1:].cpu().numpy()

            for item_id, item_idx in self.item_map.items():
                item_embeddings[item_id] = embeddings[item_idx - 1].tolist()

        return item_embeddings

    def update_incremental(self, new_interactions, new_user_ids=None, new_item_ids=None):
        """
        Update the model with new interactions.

        Args:
            new_interactions: List of (user_id, item_id, rating) tuples.
            new_user_ids: Optional list of new user IDs to add to the model.
            new_item_ids: Optional list of new item IDs to add to the model.

        Returns:
            self

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            return self.fit(new_interactions, new_user_ids, new_item_ids)

        # Add new users and items to mappings
        new_users_added = 0
        if new_user_ids is not None:
            for user_id in new_user_ids:
                if user_id not in self.user_map:
                    self.user_map[user_id] = self.num_users + 1 + new_users_added
                    self.reverse_user_map[self.num_users + 1 + new_users_added] = user_id
                    new_users_added += 1

        new_items_added = 0
        if new_item_ids is not None:
            for item_id in new_item_ids:
                if item_id not in self.item_map:
                    self.item_map[item_id] = self.num_items + 1 + new_items_added
                    self.reverse_item_map[self.num_items + 1 + new_items_added] = item_id
                    new_items_added += 1

        # Update counts
        old_num_users = self.num_users
        old_num_items = self.num_items

        self.num_users += new_users_added
        self.num_items += new_items_added

        # If we have new users or items, we need to rebuild the model
        if new_users_added > 0 or new_items_added > 0:
            self.logger.info(f"Added {new_users_added} new users and {new_items_added} new items.")

            # Save old model state_dict
            old_state_dict = self.model.state_dict()

            # Build new model
            self._build_model()

            # Copy weights from old model for existing users and items
            with torch.no_grad():
                # User embeddings
                if old_num_users > 0:
                    self.model.user_embedding.weight.data[1 : old_num_users + 1] = old_state_dict[
                        "user_embedding.weight"
                    ][1 : old_num_users + 1]

                # Item embeddings
                if old_num_items > 0:
                    self.model.item_embedding.weight.data[1 : old_num_items + 1] = old_state_dict[
                        "item_embedding.weight"
                    ][1 : old_num_items + 1]

                # MLP weights if they exist
                if hasattr(self.model, "mlp"):
                    for i, layer in enumerate(self.model.mlp):
                        if isinstance(layer, nn.Linear):
                            layer_name = f"mlp.{i}"
                            if layer_name in old_state_dict:
                                layer.weight.data = old_state_dict[f"{layer_name}.weight"]
                                layer.bias.data = old_state_dict[f"{layer_name}.bias"]

        # Process new interactions for fine-tuning
        if new_interactions:
            # Prepare data
            processed_interactions = []
            for user_id, item_id, rating in new_interactions:
                if user_id in self.user_map and item_id in self.item_map:
                    processed_interactions.append(
                        (self.user_map[user_id], self.item_map[item_id], rating)
                    )

            # Fine-tune with new interactions
            if processed_interactions:
                self._train(processed_interactions, num_epochs=min(5, self.num_epochs))

        return self

    def set_device(self, device):
        """
        Set the device to run the model on.

        Args:
            device: Device to run the model on ('cpu' or 'cuda').

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.device = torch.device(device)
        if hasattr(self, "model") and self.model is not None:
            self.model.to(self.device)
