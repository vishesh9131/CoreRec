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


class FeaturesEmbedding(nn.Module):
    """Embedding layer for categorical features."""

    def __init__(self, field_dims: List[int], embedding_dim: int):
        """
        Initialize the embedding layer.

        Args:
            field_dims: List of cardinalities of categorical features.
            embedding_dim: Dimension of embedding vectors.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.field_dims = field_dims
        self.num_fields = len(field_dims)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(sum(field_dims), embedding_dim) for _ in range(self.num_fields)]
        )

        # Initialize embeddings
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding layer.

        Args:
            x: Input tensor of shape (batch_size, num_fields).
                Each value is the original feature value + offset.

        Returns:
            Embedded tensor of shape (batch_size, num_fields, embedding_dim).
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        embeddings = [embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)]
        embeddings = torch.stack(embeddings, dim=1)
        return embeddings

    def get_embedding(self, field_idx: int, feature_idx: int) -> torch.Tensor:
        """
        Get embedding vector for a specific feature value.

        Args:
            field_idx: Index of the field.
            feature_idx: Index of the feature value within the field.

        Returns:
            Embedding vector of shape (embedding_dim,).
        """
        return self.embeddings[field_idx](torch.tensor(feature_idx + self.offsets[field_idx]))


class FeaturesLinear(nn.Module):
    """Linear layer for first-order feature importance."""

    def __init__(self, field_dims: List[int]):
        """
        Initialize the linear layer.

        Args:
            field_dims: List of cardinalities of categorical features.
        """
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims)
        self.fc = nn.Embedding(sum(field_dims), 1)
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear layer.

        Args:
            x: Input tensor of shape (batch_size, num_fields).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeatureInteractionDetector(nn.Module):
    """Module for detecting important feature interactions."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: List[int],
        num_interactions: int,
        dropout: float = 0.0,
    ):
        """
        Initialize the feature interaction detector.

        Args:
            embedding_dim: Dimension of embedding vectors.
            hidden_dims: List of hidden layer dimensions.
            num_interactions: Number of interactions to detect.
            dropout: Dropout probability.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_interactions = num_interactions

        # Interaction scoring network
        layers = []
        dims = [embedding_dim * 2] + hidden_dims + [1]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.interaction_scorer = nn.Sequential(*layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to detect important interactions.

        Args:
            embeddings: Input embeddings of shape (batch_size, num_fields, embedding_dim).

        Returns:
            Tuple of (interaction_scores, selected_pairs).
            interaction_scores: Scores for each interaction pair (batch_size, num_interactions).
            selected_pairs: Indices of selected interaction pairs (num_interactions, 2).
        """
        batch_size, num_fields, embedding_dim = embeddings.shape

        # Compute all pairwise concatenations
        pair_indices = []
        pair_embeddings = []

        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                pair_indices.append((i, j))
                # Concatenate embeddings of field i and j
                pair_emb = torch.cat([embeddings[:, i, :], embeddings[:, j, :]], dim=1)
                pair_embeddings.append(pair_emb)

        # Convert to tensor (batch_size, num_pairs, embedding_dim*2)
        pair_embeddings = torch.stack(pair_embeddings, dim=1)
        num_pairs = len(pair_indices)

        # Reshape for scoring (batch_size*num_pairs, embedding_dim*2)
        flat_embeddings = pair_embeddings.view(-1, embedding_dim * 2)

        # Score each pair
        scores = self.interaction_scorer(flat_embeddings).view(batch_size, num_pairs)

        # Select top-k interactions
        if self.num_interactions >= num_pairs:
            # If we want more interactions than available, use all
            selected_scores = scores
            selected_pairs = torch.tensor(pair_indices, device=embeddings.device)
        else:
            # Get top-k interaction pairs
            _, topk_indices = torch.topk(scores.mean(dim=0), k=self.num_interactions)
            selected_scores = torch.gather(
                scores, 1, topk_indices.unsqueeze(0).expand(batch_size, -1)
            )
            selected_pairs = torch.tensor(
                [pair_indices[i] for i in topk_indices], device=embeddings.device
            )

        return selected_scores, selected_pairs


class FeatureCrossing(nn.Module):
    """Module for crossing selected feature pairs."""

    def __init__(self, embedding_dim: int, num_interactions: int):
        """
        Initialize the feature crossing module.

        Args:
            embedding_dim: Dimension of embedding vectors.
            num_interactions: Number of interaction pairs to consider.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_interactions = num_interactions

        # Linear projection for crossed features
        self.projection = nn.Linear(embedding_dim, 1)

        # Initialize weights
        nn.init.xavier_uniform_(self.projection.weight.data)
        nn.init.zeros_(self.projection.bias.data)

    def forward(
        self,
        embeddings: torch.Tensor,
        selected_pairs: torch.Tensor,
        interaction_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to compute feature crossing.

        Args:
            embeddings: Input embeddings of shape (batch_size, num_fields, embedding_dim).
            selected_pairs: Indices of selected pairs of shape (num_interactions, 2).
            interaction_scores: Scores for each interaction (batch_size, num_interactions).

        Returns:
            Output of crossing features (batch_size, 1).
        """
        batch_size = embeddings.shape[0]
        crossed_features = []

        for i in range(self.num_interactions):
            idx1, idx2 = selected_pairs[i]
            # Element-wise product of embeddings
            crossed = embeddings[:, idx1, :] * embeddings[:, idx2, :]
            crossed_features.append(crossed)

        # Stack crossed features (batch_size, num_interactions, embedding_dim)
        crossed_features = torch.stack(crossed_features, dim=1)

        # Apply interaction scores as weights
        weighted_features = crossed_features * interaction_scores.unsqueeze(-1)

        # Sum over interactions
        summed_features = torch.sum(weighted_features, dim=1)

        # Project to output
        output = self.projection(summed_features)

        return output


class AutoFIModel(nn.Module):
    """AutoFI model that automatically detects and uses important feature interactions."""

    def __init__(
        self,
        field_dims: List[int],
        embedding_dim: int = 64,
        hidden_dims: List[int] = [64, 32],
        num_interactions: int = 10,
        dropout: float = 0.1,
    ):
        """
        Initialize the AutoFI model.

        Args:
            field_dims: List of cardinalities of categorical features.
            embedding_dim: Dimension of embedding vectors.
            hidden_dims: List of hidden layer dimensions for interaction detector.
            num_interactions: Number of interactions to consider.
            dropout: Dropout probability.
        """
        super().__init__()

        self.field_dims = field_dims
        self.embedding_dim = embedding_dim
        self.num_fields = len(field_dims)
        self.num_interactions = min(num_interactions, self.num_fields * (self.num_fields - 1) // 2)

        # First-order linear model
        self.linear = FeaturesLinear(field_dims)

        # Feature embedding
        self.embedding = FeaturesEmbedding(field_dims, embedding_dim)

        # Feature interaction detector
        self.interaction_detector = FeatureInteractionDetector(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            num_interactions=self.num_interactions,
            dropout=dropout,
        )

        # Feature crossing
        self.feature_crossing = FeatureCrossing(embedding_dim, self.num_interactions)

        # Output bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AutoFI model.

        Args:
            x: Input tensor of shape (batch_size, num_fields).

        Returns:
            Prediction of shape (batch_size, 1).
        """
        # First-order linear part
        linear_part = self.linear(x)

        # Embedding part
        embeddings = self.embedding(x)

        # Detect important interactions
        interaction_scores, selected_pairs = self.interaction_detector(embeddings)

        # Compute feature crossings
        crossed_part = self.feature_crossing(embeddings, selected_pairs, interaction_scores)

        # Combine and add bias
        output = linear_part + crossed_part + self.bias

        return output.squeeze(1)

    def get_selected_interactions(self, x: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Get the selected feature interactions for a batch of inputs.

        Args:
            x: Input tensor of shape (batch_size, num_fields).

        Returns:
            List of selected feature interaction pairs.
        """
        embeddings = self.embedding(x)
        _, selected_pairs = self.interaction_detector(embeddings)
        return [(int(pair[0]), int(pair[1])) for pair in selected_pairs.cpu().numpy()]

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


class AutoFI_base(BaseRecommender):
    """Base class for AutoFI recommendation model."""

    def __init__(
        self,
        name: str = "AutoFI",
        trainable: bool = True,
        verbose: bool = False,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        """
        Initialize the AutoFI base model.

        Args:
            name: Name of the model.
            trainable: Whether the model is trainable.
            verbose: Whether to print verbose output.
            config: Configuration dictionary.
            seed: Random seed for reproducibility.
        """
        super().__init__(name, trainable)

        self.verbose = verbose
        self.seed = seed

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Default configuration
        default_config = {
            "embedding_dim": 64,
            "hidden_dims": [64, 32],
            "num_interactions": 10,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 256,
            "num_epochs": 20,
            "early_stopping": True,
            "patience": 3,
            "min_delta": 1e-4,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "loss_function": "bce",  # Options: 'bce', 'mse'
        }

        # Update with user config
        self.config = default_config.copy()
        if config:
            self.config.update(config)

        # Initialize attributes
        self.is_fitted = False
        self.model = None
        self.hooks = HookManager()
        self.version = "1.0.0"

        # Store for item & user mappings
        self.user_ids = None
        self.item_ids = None
        self.uid_map = {}
        self.iid_map = {}
        self.num_users = 0
        self.num_items = 0
        self.interaction_matrix = None

        # For tracking field dimensions
        self.field_dims = None

    def _create_dataset(self, interaction_matrix, user_ids, item_ids):
        """
        Create dataset for training.

        Args:
            interaction_matrix: Sparse interaction matrix.
            user_ids: List of user IDs.
            item_ids: List of item IDs.

        Returns:
            DataLoader for training.
        """
        # Create mappings
        self.uid_map = {uid: idx for idx, uid in enumerate(user_ids)}
        self.iid_map = {iid: idx for idx, iid in enumerate(item_ids)}

        # Get positive interactions
        coo = interaction_matrix.tocoo()
        user_indices = coo.row
        item_indices = coo.col
        ratings = coo.data

        # Create training data
        train_data = []
        train_labels = []

        # Positive samples
        for u, i, r in zip(user_indices, item_indices, ratings):
            train_data.append([u, i])
            train_labels.append(1.0)

        # Negative sampling (1:1 ratio)
        num_negatives = len(train_data)
        for _ in range(num_negatives):
            u = np.random.randint(0, self.num_users)
            i = np.random.randint(0, self.num_items)
            while interaction_matrix[u, i] > 0:
                u = np.random.randint(0, self.num_users)
                i = np.random.randint(0, self.num_items)
            train_data.append([u, i])
            train_labels.append(0.0)

        # Convert to tensors
        train_data = torch.tensor(train_data, dtype=torch.long)
        train_labels = torch.tensor(train_labels, dtype=torch.float)

        # Create dataset
        dataset = torch.utils.data.TensorDataset(train_data, train_labels)

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=0
        )

        return dataloader

    def _build_model(self):
        """Build the AutoFI model."""
        if self.field_dims is None:
            self.field_dims = [self.num_users, self.num_items]

        self.model = AutoFIModel(
            field_dims=self.field_dims,
            embedding_dim=self.config["embedding_dim"],
            hidden_dims=self.config["hidden_dims"],
            num_interactions=self.config["num_interactions"],
            dropout=self.config["dropout"],
        )

        self.device = torch.device(self.config["device"])
        self.model.to(self.device)

        if self.verbose:
            print(f"Model built with {sum(p.numel() for p in self.model.parameters())} parameters")

    def fit(self, interaction_matrix, user_ids, item_ids):
        """
        Fit the AutoFI model.

        Args:
            interaction_matrix: Sparse interaction matrix.
            user_ids: List of user IDs.
            item_ids: List of item IDs.

        Returns:
            Training history.
        """
        # Store data
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.num_users = len(user_ids)
        self.num_items = len(item_ids)
        self.interaction_matrix = interaction_matrix.copy()

        # Build model
        self.field_dims = [self.num_users, self.num_items]
        self._build_model()

        # Create dataset
        train_loader = self._create_dataset(interaction_matrix, user_ids, item_ids)

        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Choose loss function
        if self.config["loss_function"] == "bce":
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:  # default to MSE
            loss_fn = torch.nn.MSELoss()

        # Training loop
        history = {"loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config["num_epochs"]):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass
                pred = self.model(x)
                loss = loss_fn(pred, y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["loss"].append(train_loss)

            # Validation
            self.model.eval()
            val_loss = self._validate(train_loader, loss_fn)
            history["val_loss"].append(val_loss)

            if self.verbose:
                print(
                    f"Epoch {epoch+1}/{self.config['num_epochs']} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
                )

            # Early stopping
            if self.config["early_stopping"]:
                if val_loss < best_val_loss - self.config["min_delta"]:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config["patience"]:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break

        self.is_fitted = True
        return history

    def _validate(self, dataloader, loss_fn):
        """
        Validate the model.

        Args:
            dataloader: DataLoader for validation.
            loss_fn: Loss function.

        Returns:
            Validation loss.
        """
        val_loss = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                loss = loss_fn(pred, y)

                val_loss += loss.item()

        val_loss /= len(dataloader)
        return val_loss

        def recommend(self, user_id, top_n=10, exclude_seen=True):
            """
            Generate recommendations for a user.

            Args:
                user_id: User ID.
                top_n: Number of recommendations to generate.
                exclude_seen: Whether to exclude seen items.

            Returns:
                List of (item_id, score) tuples.
            """
            if not self.is_fitted:
                raise RuntimeError("Model is not fitted yet. Call fit() first.")

            if user_id not in self.uid_map:
                raise ValueError(f"User {user_id} not found in training data.")

            user_idx = self.uid_map[user_id]

            # Get seen items
            seen_items = set()
            if exclude_seen:
                user_row = self.interaction_matrix[user_idx].nonzero()[1]
                seen_items = set(user_row)

            # Create candidate items
            candidate_items = [i for i in range(self.num_items) if i not in seen_items]

            if not candidate_items:
                return []

            # Prepare batch input
            user_tensor = torch.tensor([user_idx] * len(candidate_items), dtype=torch.long)
            item_tensor = torch.tensor(candidate_items, dtype=torch.long)
            batch = torch.stack([user_tensor, item_tensor], dim=1).to(self.device)

            # Get predictions
            self.model.eval()
            with torch.no_grad():
                scores = self.model(batch).cpu().numpy()

            # Create item-score pairs
            item_scores = [
                (self.item_ids[item_idx], float(score))
                for item_idx, score in zip(candidate_items, scores)
            ]

            # Sort by score and take top-n
            item_scores.sort(key=lambda x: x[1], reverse=True)
            return item_scores[:top_n]

        def register_hook(self, layer_name, callback=None):
            """
            Register a hook for a layer.

            Args:
                layer_name: Name of the layer.
                callback: Callback function.

            Returns:
                Whether hook was successfully registered.
            """
            if not self.is_fitted:
                raise RuntimeError("Model is not fitted yet. Call fit() first.")

            return self.hooks.register_hook(self.model, layer_name, callback)

        def save(self, path: str):
            """
            Save the model to disk.

            Args:
                path: Path to save the model.
            """
            if not self.is_fitted:
                raise RuntimeError("Model is not fitted yet. Call fit() first.")

            model_state = {
                "config": self.config,
                "state_dict": self.model.state_dict() if self.model else None,
                "user_ids": self.user_ids,
                "item_ids": self.item_ids,
                "uid_map": self.uid_map,
                "iid_map": self.iid_map,
                "is_fitted": self.is_fitted,
                "version": self.version,
            }

            # Save model state
            with open(f"{path}.pkl", "wb") as f:
                pickle.dump(model_state, f)

            # Save metadata
            metadata = {
                "model_type": self.__class__.__name__,
                "name": self.name,
                "version": self.version,
                "num_users": self.num_users,
                "num_items": self.num_items,
                "save_time": str(datetime.now()),
                "config": self.config,
            }

            with open(f"{path}.meta", "w") as f:
                yaml.dump(metadata, f)

            if self.verbose:
                print(f"Model saved to {path}")

        @classmethod
        def load(cls, path: str):
            """
            Load model from disk.

            Args:
                path: Path to the saved model.

            Returns:
                Loaded model.
            """
            # Load model state
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
                model.field_dims = [model.num_users, model.num_items]
                model.is_fitted = True
                model.interaction_matrix = sp.csr_matrix((model.num_users, model.num_items))

                # Build model and load state
                model._build_model()
                model.model.load_state_dict(model_state["state_dict"])
                model.version = model_state["version"]

            return model

        def get_important_interactions(self, batch_size=100):
            """
            Get the most important feature interactions learned by the model.

            Args:
                batch_size: Batch size for inference.

            Returns:
                List of feature interaction pairs.
            """
            if not self.is_fitted:
                raise RuntimeError("Model is not fitted yet. Call fit() first.")

            # Sample random user-item pairs
            sample_indices = []
            for _ in range(batch_size):
                u = np.random.randint(0, self.num_users)
                i = np.random.randint(0, self.num_items)
                sample_indices.append([u, i])

            # Convert to tensor
            sample_tensor = torch.tensor(sample_indices, dtype=torch.long).to(self.device)

            # Get selected interactions
            self.model.eval()
            with torch.no_grad():
                interactions = self.model.get_selected_interactions(sample_tensor)

            return interactions

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
