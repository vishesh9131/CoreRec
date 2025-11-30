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
from corerec.api.base_recommender import BaseRecommender


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


class Encoder(nn.Module):
    """Encoder module for AutoencoderCF."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.0):
        """
        Initialize the encoder.

        Args:
            input_dim: Dimension of input (num_items).
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder."""
        return self.layers(x)


class Decoder(nn.Module):
    """Decoder module for AutoencoderCF."""

    def __init__(
        self, latent_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.0
    ):
        """
        Initialize the decoder.

        Args:
            latent_dim: Dimension of latent space.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Dimension of output (num_items).
            dropout: Dropout probability.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build layers
        layers = []
        dims = [latent_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        # For ratings, we might want a sigmoid at the end
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder."""
        return self.layers(x)


class AutoencoderCFModel(nn.Module):
    """Autoencoder model for collaborative filtering."""

    def __init__(
        self,
        num_items: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.0,
        activation: str = "sigmoid",
    ):
        """
        Initialize the AutoencoderCF model.

        Args:
            num_items: Number of items (input/output dimension).
            hidden_dims: List of hidden dimensions for encoder.
            latent_dim: Dimension of the latent space.
            dropout: Dropout probability.
            activation: Output activation ('sigmoid', 'identity').
        """
        super().__init__()

        self.num_items = num_items
        self.latent_dim = latent_dim
        self.activation = activation

        # Encoder and decoder
        encoder_dims = hidden_dims + [latent_dim]
        decoder_dims = hidden_dims[::-1]  # Reverse for decoder

        self.encoder = Encoder(num_items, encoder_dims, dropout)
        self.decoder = Decoder(latent_dim, decoder_dims, num_items, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        # Encode
        z = self.encoder(x)

        # Decode
        x_pred = self.decoder(z)

        # Apply activation
        if self.activation == "sigmoid":
            x_pred = torch.sigmoid(x_pred)

        return x_pred

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        x_pred = self.decoder(z)
        if self.activation == "sigmoid":
            x_pred = torch.sigmoid(x_pred)
        return x_pred


class AutoencoderCFBase(BaseRecommender):
    """Base class for Autoencoder-based Collaborative Filtering models.

    This class implements a general autoencoder architecture for collaborative filtering,
    which can be used for various tasks like recommendation, denoising, etc.

    References:
        - Sedhain, S., Menon, A. K., Sanner, S., & Xie, L. (2015).
          Autorec: Autoencoders meet collaborative filtering. WWW.
        - Wu, Y., DuBois, C., Zheng, A. X., & Ester, M. (2016).
          Collaborative denoising auto-encoders for top-n recommender systems. WSDM.
    """

    def __init__(
        self,
        name: str = "AutoencoderCF",
        trainable: bool = True,
        verbose: bool = False,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        """Initialize the AutoencoderCF model.

        Args:
            name: Name of the model.
            trainable: Whether the model can be trained.
            verbose: Whether to print verbose output.
            config: Configuration dictionary.
            seed: Random seed for reproducibility.
        """
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.config = self._set_default_config() if config is None else config
        self.seed = seed

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize components
        self.hooks = HookManager()
        self.version = "1.0.0"
        self.is_fitted = False
        self.model = None
        self.device = torch.device(self.config.get("device", "cpu"))

        # Initialize logger
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(name)

    def _set_default_config(self) -> Dict[str, Any]:
        """Set default configuration."""
        return {
            "hidden_dims": [256, 128],
            "latent_dim": 64,
            "dropout": 0.5,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 256,
            "num_epochs": 100,
            "early_stopping": True,
            "patience": 10,
            "min_delta": 0.0001,
            "mask_prob": 0.0,  # For denoising autoencoders
            "activation": "sigmoid",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

    def _create_dataset(self, interaction_matrix, user_ids, item_ids):
        """Create dataset from interaction matrix.

        Args:
            interaction_matrix: Sparse interaction matrix.
            user_ids: List of user IDs.
            item_ids: List of item IDs.
        """
        # Create mappings
        self.uid_map = {uid: idx for idx, uid in enumerate(user_ids)}
        self.iid_map = {iid: idx for idx, iid in enumerate(item_ids)}

        # Get number of users and items
        self.num_users = len(user_ids)
        self.num_items = len(item_ids)

        # Store user_ids and item_ids
        self.user_ids = user_ids
        self.item_ids = item_ids

        # Create dataset for training
        self.interaction_matrix = interaction_matrix.copy()

        # Set field dimensions for model
        self.field_dims = [self.num_items]

    def _build_model(self):
        """Build the AutoencoderCF model."""
        self.model = AutoencoderCFModel(
            num_items=self.num_items,
            hidden_dims=self.config["hidden_dims"],
            latent_dim=self.config["latent_dim"],
            dropout=self.config["dropout"],
            activation=self.config["activation"],
        ).to(self.device)

    def fit(self, interaction_matrix, user_ids, item_ids):
        """
        Fit the AutoencoderCF model.

        Args:
            interaction_matrix: Sparse matrix of shape (num_users, num_items).
            user_ids: List of user IDs.
            item_ids: List of item IDs.

        Returns:
            Dictionary with training history.
        """
        if not self.trainable:
            raise RuntimeError("This model is not trainable.")

        # Create dataset
        self._create_dataset(interaction_matrix, user_ids, item_ids)

        # Build model
        self._build_model()

        # Create optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Create loss function
        loss_fn = nn.MSELoss()

        # Training loop
        history = {"loss": [], "val_loss": []}

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config["num_epochs"]):
            # Set model to training mode
            self.model.train()

            # Shuffle users
            user_indices = np.arange(self.num_users)
            np.random.shuffle(user_indices)

            # Mini-batch training
            total_loss = 0.0
            num_batches = 0

            for i in range(0, len(user_indices), self.config["batch_size"]):
                batch_indices = user_indices[i : i + self.config["batch_size"]]

                # Get batch data
                batch_matrix = self.interaction_matrix[batch_indices].toarray()
                batch_tensor = torch.FloatTensor(batch_matrix).to(self.device)

                # Apply mask for denoising autoencoder
                if self.config["mask_prob"] > 0:
                    mask = torch.rand_like(batch_tensor) < self.config["mask_prob"]
                    batch_tensor_masked = batch_tensor.clone()
                    batch_tensor_masked[mask] = 0
                    inputs = batch_tensor_masked
                else:
                    inputs = batch_tensor

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss (only on non-zero entries)
                mask = (batch_tensor > 0).float()
                loss = loss_fn(outputs * mask, batch_tensor * mask)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Calculate average loss
            avg_loss = total_loss / max(1, num_batches)
            history["loss"].append(avg_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                # Get random validation users
                val_indices = np.random.choice(
                    self.num_users, min(1000, self.num_users), replace=False
                )
                val_matrix = self.interaction_matrix[val_indices].toarray()
                val_tensor = torch.FloatTensor(val_matrix).to(self.device)

                # Forward pass
                val_outputs = self.model(val_tensor)

                # Compute validation loss (only on non-zero entries)
                val_mask = (val_tensor > 0).float()
                val_loss = loss_fn(val_outputs * val_mask, val_tensor * val_mask).item()

                history["val_loss"].append(val_loss)

            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                    f"Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}"
                )

            # Early stopping
            if self.config["early_stopping"]:
                if val_loss < best_val_loss - self.config["min_delta"]:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config["patience"]:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        self.is_fitted = True
        return history

    def recommend(self, user_id, top_n=10, exclude_seen=True):
        """
        Recommend items for a user.

        Args:
            user_id: User ID.
            top_n: Number of recommendations to return.
            exclude_seen: Whether to exclude items the user has already interacted with.

        Returns:
            List of (item_id, score) tuples.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if user_id not in self.uid_map:
            self.logger.warning(f"User {user_id} not in training data")
            return []

        # Set model to eval mode
        self.model.eval()

        # Get user vector
        user_idx = self.uid_map[user_id]
        user_vector = self.interaction_matrix[user_idx].toarray().flatten()
        user_tensor = torch.FloatTensor(user_vector).unsqueeze(0).to(self.device)

        # Get predictions
        with torch.no_grad():
            predicted = self.model(user_tensor).squeeze().cpu().numpy()

        # Get seen items
        seen_indices = set() if not exclude_seen else set(np.where(user_vector > 0)[0])

        # Get top-n items
        candidate_indices = np.argsort(-predicted)

        # Filter out seen items
        recommended_indices = [idx for idx in candidate_indices if idx not in seen_indices][:top_n]

        # Convert to item IDs and scores
        recommendations = [
            (self.item_ids[idx], float(predicted[idx])) for idx in recommended_indices
        ]

        return recommendations

    def predict(self, user_ids, item_ids):
        """
        Predict ratings for a list of user-item pairs.

        Args:
            user_ids: List of user IDs.
            item_ids: List of item IDs.

        Returns:
            List of predicted ratings.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Set model to eval mode
        self.model.eval()

        # Convert user_ids and item_ids to indices
        user_indices = [self.uid_map.get(uid, -1) for uid in user_ids]
        item_indices = [self.iid_map.get(iid, -1) for iid in item_ids]

        # Check if any user or item is unknown
        unknown = [
            (i, uid, iid)
            for i, (uid, iid, u_idx, i_idx) in enumerate(
                zip(user_ids, item_ids, user_indices, item_indices)
            )
            if u_idx == -1 or i_idx == -1
        ]

        if unknown:
            self.logger.warning(f"Found {len(unknown)} unknown user-item pairs")

        # Get predictions for known pairs
        predictions = np.zeros(len(user_ids))

        with torch.no_grad():
            for i, (u_idx, i_idx) in enumerate(zip(user_indices, item_indices)):
                if u_idx == -1 or i_idx == -1:
                    continue

                # Get user vector
                user_vector = self.interaction_matrix[u_idx].toarray().flatten()
                user_tensor = torch.FloatTensor(user_vector).unsqueeze(0).to(self.device)

                # Get prediction
                predicted = self.model(user_tensor).squeeze().cpu().numpy()
                predictions[i] = predicted[i_idx]

        return predictions

    def register_hook(self, layer_name, callback=None):
        """
        Register a hook for a layer.

        Args:
            layer_name: Name of the layer.
            callback: Optional callback function.

        Returns:
            True if hook was registered, False otherwise.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        return self.hooks.register_hook(self.model, layer_name, callback)

    def get_user_embedding(self, user_id):
        """
        Get the latent representation of a user.

        Args:
            user_id: User ID.

        Returns:
            User embedding.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if user_id not in self.uid_map:
            self.logger.warning(f"User {user_id} not in training data")
            return None

        # Set model to eval mode
        self.model.eval()

        # Get user vector
        user_idx = self.uid_map[user_id]
        user_vector = self.interaction_matrix[user_idx].toarray().flatten()
        user_tensor = torch.FloatTensor(user_vector).unsqueeze(0).to(self.device)

        # Get embedding
        with torch.no_grad():
            embedding = self.model.encode(user_tensor).squeeze().cpu().numpy()

        return embedding

    def get_similar_users(self, user_id, top_n=10):
        """
        Find similar users based on latent representations.

        Args:
            user_id: User ID.
            top_n: Number of similar users to return.

        Returns:
            List of (user_id, similarity_score) tuples.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Get embedding for query user
        query_embedding = self.get_user_embedding(user_id)
        if query_embedding is None:
            return []

        # Compute embeddings for all users
        all_embeddings = []
        all_user_ids = []

        for uid, idx in self.uid_map.items():
            if uid == user_id:
                continue

            # Get user vector
            user_vector = self.interaction_matrix[idx].toarray().flatten()
            all_embeddings.append(user_vector)
            all_user_ids.append(uid)

        # Convert to tensor
        all_embeddings_tensor = torch.FloatTensor(np.vstack(all_embeddings)).to(self.device)

        # Get embeddings
        with torch.no_grad():
            embeddings = self.model.encode(all_embeddings_tensor).cpu().numpy()

        # Compute similarities
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )

        # Get top-n similar users
        top_indices = np.argsort(-similarities)[:top_n]
        similar_users = [(all_user_ids[idx], float(similarities[idx])) for idx in top_indices]

        return similar_users

    def save(self, path: Optional[str] = None) -> None:
        """
        Save model state.

        Args:
            path: Path to save model.
        """
        if path is None:
            path = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create directory if it doesn't exist
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_path = f"{path}.pkl"
        model_state = {
            "config": self.config,
            "state_dict": self.model.state_dict() if self.is_fitted else None,
            "user_ids": self.user_ids if self.is_fitted else None,
            "item_ids": self.item_ids if self.is_fitted else None,
            "uid_map": self.uid_map if self.is_fitted else None,
            "iid_map": self.iid_map if self.is_fitted else None,
            "version": self.version,
            "is_fitted": self.is_fitted,
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_state, f)

        # Save metadata
        meta_path = f"{path}.meta"
        metadata = {
            "name": self.name,
            "trainable": self.trainable,
            "verbose": self.verbose,
            "version": self.version,
            "is_fitted": self.is_fitted,
            "num_users": getattr(self, "num_users", 0),
            "num_items": getattr(self, "num_items", 0),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(meta_path, "w") as f:
            yaml.dump(metadata, f)

        if self.verbose:
            self.logger.info(f"Model saved to {model_path}")

    @classmethod
    def load(cls, path: str) -> "AutoencoderCFBase":
        """
        Load model state.

        Args:
            path: Path to load model from.

        Returns:
            Loaded model.
        """
        with open(path, "rb") as f:
            model_state = pickle.load(f)

        # Create model instance
        model = cls(name=Path(path).stem, config=model_state["config"])

        # Load model state
        if model_state["is_fitted"]:
            # Set attributes
            model.user_ids = model_state["user_ids"]
            model.item_ids = model_state["item_ids"]
            model.uid_map = model_state["uid_map"]
            model.iid_map = model_state["iid_map"]
            model.num_users = len(model.user_ids)
            model.num_items = len(model.item_ids)
            model.field_dims = [model.num_items]
            model.is_fitted = True
            model.interaction_matrix = sp.csr_matrix((model.num_users, model.num_items))

            # Build model and load state
            model._build_model()
            model.model.load_state_dict(model_state["state_dict"])
            model.version = model_state["version"]

        return model

    def monitor_value(self, train_set, val_set):
        """
        Calculate monitored value for early stopping.

        Args:
            train_set: Training set.
            val_set: Validation set.

        Returns:
            Monitored value.
        """
        # For AutoencoderCF, we monitor the negative validation loss
        self.model.eval()
        loss_fn = nn.MSELoss()

        with torch.no_grad():
            val_matrix = val_set.toarray()
            val_tensor = torch.FloatTensor(val_matrix).to(self.device)

            # Forward pass
            val_outputs = self.model(val_tensor)

            # Compute validation loss (only on non-zero entries)
            val_mask = (val_tensor > 0).float()
            val_loss = loss_fn(val_outputs * val_mask, val_tensor * val_mask).item()

        # Return negative loss (higher is better)
        return -val_loss

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
                self.field_dims = [self.num_items]
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
