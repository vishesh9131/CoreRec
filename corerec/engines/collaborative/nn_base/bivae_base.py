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
    """VAE Encoder network."""

    def __init__(
        self, input_dim: int, hidden_dims: List[int], latent_dim: int, dropout: float = 0.0
    ):
        """
        Initialize the encoder network.

        Args:
            input_dim: Input dimension.
            hidden_dims: List of hidden layer dimensions.
            latent_dim: Latent dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        # Build encoder layers
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)

        # Mean and variance layers
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (mu, logvar, z) where z is the sampled latent vector.
        """
        # Encode input
        h = self.encoder(x)

        # Get mean and log variance
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)

        # Sample from latent distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return mu, logvar, z


class Decoder(nn.Module):
    """VAE Decoder network."""

    def __init__(
        self, latent_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.0
    ):
        """
        Initialize the decoder network.

        Args:
            latent_dim: Latent dimension.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Output dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        # Build decoder layers
        layers = []
        dims = [latent_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())  # For binary data like interaction matrix

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            z: Latent tensor.

        Returns:
            Reconstructed output.
        """
        return self.decoder(z)


class BIVAE(nn.Module):
    """Bilateral Variational Autoencoder for Collaborative Filtering."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        latent_dim: int = 64,
        encoder_hidden_dims: List[int] = [256, 128],
        decoder_hidden_dims: List[int] = [128, 256],
        dropout: float = 0.2,
        beta: float = 0.2,  # Weight of KL divergence term
    ):
        """
        Initialize BIVAE model.

        Args:
            num_users: Number of users.
            num_items: Number of items.
            latent_dim: Dimension of latent space.
            encoder_hidden_dims: Hidden dimensions for encoder.
            decoder_hidden_dims: Hidden dimensions for decoder.
            dropout: Dropout probability.
            beta: Weight of KL divergence term.
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.beta = beta

        # User and item encoders
        self.user_encoder = Encoder(num_items, encoder_hidden_dims, latent_dim, dropout)
        self.item_encoder = Encoder(num_users, encoder_hidden_dims, latent_dim, dropout)

        # User and item decoders
        self.user_decoder = Decoder(latent_dim, decoder_hidden_dims, num_items, dropout)
        self.item_decoder = Decoder(latent_dim, decoder_hidden_dims, num_users, dropout)

    def encode_user(
        self, user_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode user data to latent space.

        Args:
            user_data: User interaction data.

        Returns:
            Tuple of (mu, logvar, z) for user.
        """
        return self.user_encoder(user_data)

    def encode_item(
        self, item_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode item data to latent space.

        Args:
            item_data: Item interaction data.

        Returns:
            Tuple of (mu, logvar, z) for item.
        """
        return self.item_encoder(item_data)

    def decode_user(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode user latent representation.

        Args:
            z: User latent vector.

        Returns:
            Reconstructed user interactions.
        """
        return self.user_decoder(z)

    def decode_item(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode item latent representation.

        Args:
            z: Item latent vector.

        Returns:
            Reconstructed item interactions.
        """
        return self.item_decoder(z)

    def forward(
        self, user_data: torch.Tensor = None, item_data: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of BIVAE.

        Args:
            user_data: User interaction data. Shape: (batch_size, num_items)
            item_data: Item interaction data. Shape: (batch_size, num_users)

        Returns:
            Dictionary with reconstruction, latent vectors, and KL divergence.
        """
        result = {}

        # Process user data
        if user_data is not None:
            user_mu, user_logvar, user_z = self.encode_user(user_data)
            user_recon = self.decode_user(user_z)

            result["user_mu"] = user_mu
            result["user_logvar"] = user_logvar
            result["user_z"] = user_z
            result["user_recon"] = user_recon

        # Process item data
        if item_data is not None:
            item_mu, item_logvar, item_z = self.encode_item(item_data)
            item_recon = self.decode_item(item_z)

            result["item_mu"] = item_mu
            result["item_logvar"] = item_logvar
            result["item_z"] = item_z
            result["item_recon"] = item_recon

        return result

    def calculate_loss(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate loss components.

        Args:
            batch_data: Dictionary with user_data and item_data.

        Returns:
            Dictionary with loss components.
        """
        user_data = batch_data.get("user_data")
        item_data = batch_data.get("item_data")

        outputs = self.forward(user_data, item_data)

        loss_dict = {}
        total_loss = 0.0

        # User reconstruction loss
        if user_data is not None:
            recon_user = F.binary_cross_entropy(outputs["user_recon"], user_data, reduction="sum")
            kl_user = -0.5 * torch.sum(
                1
                + outputs["user_logvar"]
                - outputs["user_mu"].pow(2)
                - outputs["user_logvar"].exp()
            )

            loss_dict["recon_user"] = recon_user
            loss_dict["kl_user"] = kl_user

            user_loss = recon_user + self.beta * kl_user
            loss_dict["user_loss"] = user_loss
            total_loss += user_loss

        # Item reconstruction loss
        if item_data is not None:
            recon_item = F.binary_cross_entropy(outputs["item_recon"], item_data, reduction="sum")
            kl_item = -0.5 * torch.sum(
                1
                + outputs["item_logvar"]
                - outputs["item_mu"].pow(2)
                - outputs["item_logvar"].exp()
            )

            loss_dict["recon_item"] = recon_item
            loss_dict["kl_item"] = kl_item

            item_loss = recon_item + self.beta * kl_item
            loss_dict["item_loss"] = item_loss
            total_loss += item_loss

        loss_dict["total_loss"] = total_loss

        return loss_dict

        def predict(self, user_id: int, item_id: int, **kwargs) -> float:
            """
            Predict rating/score for a user-item pair.

            Args:
                user_id: User ID
                item_id: Item ID
                **kwargs: Additional arguments

            Returns:
                Predicted score/rating
            """
            from corerec.api.exceptions import ModelNotFittedError

            if not self.is_fitted:
                raise ModelNotFittedError(f"{self.name} must be fitted before making predictions")

            # Check if user/item are known
            if hasattr(self, "user_map") and user_id not in self.user_map:
                return 0.0
            if hasattr(self, "item_map") and item_id not in self.item_map:
                return 0.0

            # Get internal indices
            if hasattr(self, "user_map"):
                user_idx = self.user_map.get(user_id, 0)
            else:
                user_idx = user_id

            if hasattr(self, "item_map"):
                item_idx = self.item_map.get(item_id, 0)
            else:
                item_idx = item_id

            # Model-specific prediction logic
            # This is a fallback - ideally should be customized per model
            try:
                if hasattr(self, "model") and self.model is not None:
                    import torch

                    if hasattr(self.model, "predict"):
                        # Use model's internal predict if available
                        with torch.no_grad():
                            self.model.eval()
                            score = self.model.predict(user_idx, item_idx)
                            if isinstance(score, torch.Tensor):
                                return float(score.item())
                            return float(score)

                # Fallback: return neutral score
                return 0.5

            except Exception as e:
                import logging

                logging.warning(f"Prediction failed for {self.name}: {e}")
                return 0.0


class BiVAE_base(BaseRecommender):
    """Bilateral Variational Autoencoder base class for recommendation."""

    def __init__(
        self,
        name: str = "BiVAE",
        config: Optional[Dict[str, Any]] = None,
        trainable: bool = True,
        verbose: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the BiVAE recommender.

        Args:
            name: Name of the recommender.
            config: Configuration dictionary.
            trainable: Whether the model is trainable.
            verbose: Whether to print verbose output.
            seed: Random seed.
        """
        super().__init__(name)

        self.config = config or {}
        self.trainable = trainable
        self.verbose = verbose
        self.seed = seed

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Set default device
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Initialize hook manager
        self.hooks = HookManager()

        # Default config
        self._set_default_config()

        # Model related
        self.model = None
        self.optimizer = None
        self.is_fitted = False

        # Data related
        self.user_ids = None
        self.item_ids = None
        self.uid_map = {}
        self.iid_map = {}
        self.num_users = 0
        self.num_items = 0
        self.interaction_matrix = None

    def _set_default_config(self):
        """Set default configuration."""
        defaults = {
            "latent_dim": 64,
            "encoder_hidden_dims": [256, 128],
            "decoder_hidden_dims": [128, 256],
            "dropout": 0.2,
            "beta": 0.2,
            "batch_size": 256,
            "num_epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "early_stopping_patience": 10,
            "train_ratio": 0.8,
            "validation_step": 1,
            "user_update_interval": 1,
            "item_update_interval": 1,
        }

        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    def _build_model(self):
        """Build the BIVAE model."""
        self.model = BIVAE(
            num_users=self.num_users,
            num_items=self.num_items,
            latent_dim=self.config["latent_dim"],
            encoder_hidden_dims=self.config["encoder_hidden_dims"],
            decoder_hidden_dims=self.config["decoder_hidden_dims"],
            dropout=self.config["dropout"],
            beta=self.config["beta"],
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

    def _prepare_data(self, interaction_matrix, user_ids, item_ids):
        """
        Prepare data for training.

        Args:
            interaction_matrix: Interaction matrix (scipy sparse matrix).
            user_ids: List of user IDs.
            item_ids: List of item IDs.
        """
        self.interaction_matrix = interaction_matrix
        self.user_ids = user_ids
        self.item_ids = item_ids

        # Create ID mappings
        self.uid_map = {uid: i for i, uid in enumerate(user_ids)}
        self.iid_map = {iid: i for i, iid in enumerate(item_ids)}

        self.num_users = len(user_ids)
        self.num_items = len(item_ids)

    def fit(self, interaction_matrix, user_ids=None, item_ids=None):
        """
        Fit the model to the interaction matrix.

        Args:
            interaction_matrix: Interaction matrix (scipy sparse matrix).
            user_ids: List of user IDs.
            item_ids: List of item IDs.

        Returns:
            Fitted model.
        """
        # Prepare data
        self._prepare_data(interaction_matrix, user_ids, item_ids)

        # Build model
        self._build_model()

        # Create data loaders
        user_data, item_data = self._create_training_data()

        train_users, val_users = self._train_val_split(user_data)
        train_items, val_items = self._train_val_split(item_data)

        train_user_loader = self._create_data_loader(train_users)
        val_user_loader = self._create_data_loader(val_users)

        train_item_loader = self._create_data_loader(train_items)
        val_item_loader = self._create_data_loader(val_items)

        # Train the model
        self._train(
            train_user_loader=train_user_loader,
            val_user_loader=val_user_loader,
            train_item_loader=train_item_loader,
            val_item_loader=val_item_loader,
        )

        # Set fitted flag
        self.is_fitted = True

        return self

    def _create_training_data(self):
        """
        Create training data from interaction matrix.

        Returns:
            Tuple of (user_data, item_data).
        """
        # Convert sparse matrix to dense
        user_data = self.interaction_matrix.toarray().astype(np.float32)

        # Transpose for item perspective
        item_data = self.interaction_matrix.T.toarray().astype(np.float32)

        return user_data, item_data

    def _train_val_split(self, data):
        """
        Split data into training and validation sets.

        Args:
            data: Data to split.

        Returns:
            Tuple of (train_data, val_data).
        """
        n = data.shape[0]
        indices = np.random.permutation(n)
        split = int(n * self.config["train_ratio"])

        train_indices = indices[:split]
        val_indices = indices[split:]

        train_data = data[train_indices]
        val_data = data[val_indices]

        return train_data, val_data

    def _create_data_loader(self, data):
        """
        Create a data loader for the given data.

        Args:
            data: Input data.

        Returns:
            PyTorch DataLoader.
        """
        tensor_data = torch.FloatTensor(data)
        dataset = torch.utils.data.TensorDataset(tensor_data)

        return torch.utils.data.DataLoader(
            dataset, batch_size=self.config["batch_size"], shuffle=True
        )

    def _train(
        self,
        train_user_loader,
        val_user_loader=None,
        train_item_loader=None,
        val_item_loader=None,
    ):
        """
        Train the model.

        Args:
            train_user_loader: DataLoader for user training data.
            val_user_loader: DataLoader for user validation data.
            train_item_loader: DataLoader for item training data.
            val_item_loader: DataLoader for item validation data.
        """
        if self.verbose:
            print(f"Starting training BiVAE with config: {self.config}")

        best_val_loss = float("inf")
        patience_counter = 0
        epochs_without_improvement = 0

        for epoch in range(self.config["num_epochs"]):
            # Training
            self.model.train()
            train_loss = self._train_epoch(
                train_user_loader=train_user_loader,
                train_item_loader=train_item_loader,
                epoch=epoch,
            )

            # Validation
            if (epoch + 1) % self.config["validation_step"] == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss = self._validate(
                        val_user_loader=val_user_loader, val_item_loader=val_item_loader
                    )

                if self.verbose:
                    print(
                        f"Epoch {epoch+1}/{self.config['num_epochs']}, "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.config["early_stopping_patience"]:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            elif self.verbose:
                print(f"Epoch {epoch+1}/{self.config['num_epochs']}, Train Loss: {train_loss:.4f}")

        # Load best model
        if "best_model_state" in locals():
            self.model.load_state_dict(best_model_state["model"])
            if self.verbose:
                print(f"Loaded best model from epoch {best_model_state['epoch']+1}")

    def _train_epoch(self, train_user_loader, train_item_loader=None, epoch=0):
        """
        Train for one epoch.

        Args:
            train_user_loader: DataLoader for user training data.
            train_item_loader: DataLoader for item training data.
            epoch: Current epoch number.

        Returns:
            Average training loss.
        """
        total_loss = 0.0
        batch_count = 0

        # Create item data iterator if available
        item_iter = iter(train_item_loader) if train_item_loader else None

        # Train on user data
        for user_batch in train_user_loader:
            batch_count += 1
            user_data = user_batch[0].to(self.device)

            # Get item batch if available and it's time to update
            item_data = None
            if item_iter and epoch % self.config["item_update_interval"] == 0:
                try:
                    item_batch = next(item_iter)
                    item_data = item_batch[0].to(self.device)
                except StopIteration:
                    item_iter = iter(train_item_loader)
                    item_batch = next(item_iter)
                    item_data = item_batch[0].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            batch_data = {"user_data": user_data, "item_data": item_data}
            loss_dict = self.model.calculate_loss(batch_data)

            # Backward pass
            loss = loss_dict["total_loss"]
            loss.backward()

            # Update parameters
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / batch_count if batch_count > 0 else float("inf")

    def _validate(self, val_user_loader, val_item_loader=None):
        """
        Validate the model.

        Args:
            val_user_loader: DataLoader for user validation data.
            val_item_loader: DataLoader for item validation data.

        Returns:
            Validation loss.
        """
        total_loss = 0.0
        batch_count = 0

        # Create item data iterator if available
        item_iter = iter(val_item_loader) if val_item_loader else None

        # Validate on user data
        for user_batch in val_user_loader:
            batch_count += 1
            user_data = user_batch[0].to(self.device)

            # Get item batch if available
            item_data = None
            if item_iter:
                try:
                    item_batch = next(item_iter)
                    item_data = item_batch[0].to(self.device)
                except StopIteration:
                    item_iter = iter(val_item_loader)
                    item_batch = next(item_iter)
                    item_data = item_batch[0].to(self.device)

            # Forward pass
            batch_data = {"user_data": user_data, "item_data": item_data}
            loss_dict = self.model.calculate_loss(batch_data)

            total_loss += loss_dict["total_loss"].item()

        return total_loss / batch_count if batch_count > 0 else float("inf")

        def recommend(self, user_id, top_n=10, exclude_rated=True):
            """
            Generate recommendations for a user.

            Args:
                user_id: User ID.
                top_n: Number of recommendations to generate.
                exclude_rated: Whether to exclude already rated items.

            Returns:
                List of (item_id, score) tuples.
            """
            if not self.is_fitted:
                raise RuntimeError("Model is not fitted yet. Call fit() first.")

            if user_id not in self.uid_map:
                raise ValueError(f"User {user_id} not found in training data.")

            user_idx = self.uid_map[user_id]

            # Get user interactions
            user_interactions = torch.FloatTensor(
                self.interaction_matrix[user_idx].toarray()
            ).squeeze()

            # Get user latent representation
            self.model.eval()
            with torch.no_grad():
                user_data = user_interactions.to(self.device).unsqueeze(0)
                _, _, user_z = self.model.encode_user(user_data)
                item_scores = self.model.decode_user(user_z).squeeze().cpu().numpy()

            # Exclude rated items if required
            if exclude_rated:
                for i in range(self.num_items):
                    if user_interactions[i] > 0:
                        item_scores[i] = -np.inf

            # Get top-N item indices
            top_item_indices = np.argsort(-item_scores)[:top_n]

            # Convert to item IDs and scores
            recommendations = []
            for idx in top_item_indices:
                item_id = self.item_ids[idx]
                score = float(item_scores[idx])
                recommendations.append((item_id, score))

            return recommendations

        def get_user_embedding(self, user_id):
            """
            Get latent embedding for a user.

            Args:
                user_id: User ID.

            Returns:
                User embedding vector.
            """
            if not self.is_fitted:
                raise RuntimeError("Model is not fitted yet. Call fit() first.")

            if user_id not in self.uid_map:
                raise ValueError(f"User {user_id} not found in training data.")

            user_idx = self.uid_map[user_id]

            # Get user interactions
            user_interactions = torch.FloatTensor(
                self.interaction_matrix[user_idx].toarray()
            ).squeeze()

            # Get user latent representation
            self.model.eval()
            with torch.no_grad():
                user_data = user_interactions.to(self.device).unsqueeze(0)
                user_mu, _, _ = self.model.encode_user(user_data)
                user_embedding = user_mu.squeeze().cpu().numpy()

            return user_embedding

        def get_item_embedding(self, item_id):
            """
            Get latent embedding for an item.

            Args:
                item_id: Item ID.

            Returns:
                Item embedding vector.
            """
            if not self.is_fitted:
                raise RuntimeError("Model is not fitted yet. Call fit() first.")

            if item_id not in self.iid_map:
                raise ValueError(f"Item {item_id} not found in training data.")

            item_idx = self.iid_map[item_id]

            # Get item interactions
            item_interactions = torch.FloatTensor(
                self.interaction_matrix.T[item_idx].toarray()
            ).squeeze()

            # Get item latent representation
            self.model.eval()
            with torch.no_grad():
                item_data = item_interactions.to(self.device).unsqueeze(0)
                item_mu, _, _ = self.model.encode_item(item_data)
                item_embedding = item_mu.squeeze().cpu().numpy()

            return item_embedding

        def save(self, path):
            """
            Save the model to the given path.

            Args:
                path: Path to save the model.
            """
            if not self.is_fitted:
                raise RuntimeError("Model is not fitted yet. Call fit() first.")

            # Create directory if it doesn't exist
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save model state
            model_state = {
                "config": self.config,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "user_ids": self.user_ids,
                "item_ids": self.item_ids,
                "uid_map": self.uid_map,
                "iid_map": self.iid_map,
                "interaction_matrix": self.interaction_matrix,
                "name": self.name,
                "trainable": self.trainable,
                "verbose": self.verbose,
                "seed": self.seed,
            }

            # Save model
            with open(f"{path}.pkl", "wb") as f:
                pickle.dump(model_state, f)

            # Save metadata
            with open(f"{path}.meta", "w") as f:
                yaml.dump(
                    {
                        "name": self.name,
                        "type": "BiVAE",
                        "version": "1.0",
                        "num_users": self.num_users,
                        "num_items": self.num_items,
                        "latent_dim": self.config["latent_dim"],
                        "created_at": str(datetime.now()),
                    },
                    f,
                )

        @classmethod
        def load(cls, path):
            """
            Load the model from the given path.

            Args:
                path: Path to load the model from.

            Returns:
                Loaded model.
            """
            # Load model state
            with open(path, "rb") as f:
                model_state = pickle.load(f)

            # Create new model
            model = cls(
                name=model_state["name"],
                config=model_state["config"],
                trainable=model_state["trainable"],
                verbose=model_state["verbose"],
                seed=model_state["seed"],
            )

            # Restore model state
            model.user_ids = model_state["user_ids"]
            model.item_ids = model_state["item_ids"]
            model.uid_map = model_state["uid_map"]
            model.iid_map = model_state["iid_map"]
            model.interaction_matrix = model_state["interaction_matrix"]
            model.num_users = len(model.user_ids)
            model.num_items = len(model.item_ids)

            # Rebuild model architecture
            model._build_model()

            # Load model weights
            model.model.load_state_dict(model_state["state_dict"])
            model.optimizer.load_state_dict(model_state["optimizer"])

            # Set fitted flag
            model.is_fitted = True

            return model

        def register_hook(self, layer_name, callback=None):
            """
            Register a hook for a specific layer.

            Args:
                layer_name: Name of the layer.
                callback: Optional callback function.

            Returns:
                True if hook was registered, False otherwise.
            """
            if self.model is None:
                raise RuntimeError("Model is not built yet. Call fit() first.")

            return self.hooks.register_hook(self.model, layer_name, callback)

        def remove_hook(self, layer_name):
            """
            Remove a hook for a specific layer.

            Args:
                layer_name: Name of the layer.

            Returns:
                True if hook was removed, False otherwise.
            """
            return self.hooks.remove_hook(layer_name)

        def export_embeddings(self):
            """
            Export user and item embeddings.

            Returns:
                Dictionary with user and item embeddings.
            """
            if not self.is_fitted:
                raise RuntimeError("Model is not fitted yet. Call fit() first.")

            # Get all user and item embeddings
            user_embeddings = {}
            item_embeddings = {}

            self.model.eval()
            with torch.no_grad():
                # User embeddings
                user_data = torch.FloatTensor(self.interaction_matrix.toarray())
                batch_size = self.config["batch_size"]

                for i in range(0, self.num_users, batch_size):
                    batch = user_data[i : i + batch_size].to(self.device)
                    user_mu, _, _ = self.model.encode_user(batch)

                    for j, uid in enumerate(self.user_ids[i : i + batch_size]):
                        user_embeddings[uid] = user_mu[j].cpu().numpy().tolist()

                # Item embeddings
                item_data = torch.FloatTensor(self.interaction_matrix.T.toarray())

                for i in range(0, self.num_items, batch_size):
                    batch = item_data[i : i + batch_size].to(self.device)
                    item_mu, _, _ = self.model.encode_item(batch)

                    for j, iid in enumerate(self.item_ids[i : i + batch_size]):
                        item_embeddings[iid] = item_mu[j].cpu().numpy().tolist()

            return {"user_embeddings": user_embeddings, "item_embeddings": item_embeddings}

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

        def set_device(self, device):
            """
            Set device for model.

            Args:
                device: Device to use ('cpu', 'cuda').
            """
            self.device = device
            if self.model is not None:
                self.model.to(device)
