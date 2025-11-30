import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from corerec.api.base_recommender import BaseRecommender
from corerec.api.exceptions import ModelNotFittedError, InvalidParameterError
from corerec.utils.validation import (
    validate_fit_inputs,
    validate_user_id,
    validate_top_k,
    validate_model_fitted,
)
import pickle
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


class DeepFM(BaseRecommender):
    """
    Deep Factorization Machine (DeepFM)

    Combines factorization machines for recommendation with deep neural networks.
    It jointly learns a factorization machine for recommendation and deep representations
    of features through a neural network.

    Reference:
    Guo et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction" (IJCAI 2017)
    """

    def __init__(
        self,
        name: str = "DeepFM",
        embedding_dim: int = 16,
        hidden_layers: List[int] = [400, 400, 400],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        trainable: bool = True,
        verbose: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        self.feature_map = {}
        self.field_dims = []
        self.model = None
        self.is_fitted = False
        self.user_features = None
        self.item_features = None
        self.user_feature_types = []
        self.item_feature_types = []

    def _build_model(self, field_dims: List[int]):
        class FMLayer(nn.Module):
            def __init__(self, field_dims, embedding_dim):
                super().__init__()
                self.field_dims = field_dims
                self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
                self.embedding = nn.Embedding(sum(field_dims), 1)
                self.feature_embedding = nn.Embedding(sum(field_dims), embedding_dim)
                nn.init.xavier_uniform_(self.embedding.weight)
                nn.init.xavier_uniform_(self.feature_embedding.weight)

            def forward(self, x):
                # First-order term
                first_order = self.embedding(x + self.offsets.reshape(1, -1)).squeeze(-1).sum(dim=1)

                # Second-order term
                embeddings = self.feature_embedding(x + self.offsets.reshape(1, -1))
                square_sum = torch.sum(embeddings, dim=1) ** 2
                sum_square = torch.sum(embeddings**2, dim=1)
                second_order = 0.5 * (square_sum - sum_square).sum(1)

                return first_order, second_order, embeddings

        class DeepFMModel(nn.Module):
            def __init__(self, field_dims, embedding_dim, hidden_layers, dropout):
                super().__init__()
                self.fm_layer = FMLayer(field_dims, embedding_dim)

                # Deep component
                input_dim = len(field_dims) * embedding_dim
                self.deep_layers = nn.ModuleList()
                for hidden_dim in hidden_layers:
                    self.deep_layers.append(
                        nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            nn.BatchNorm1d(hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                        )
                    )
                    input_dim = hidden_dim

                self.output_layer = nn.Linear(hidden_layers[-1], 1)

            def forward(self, x):
                first_order, second_order, embeddings = self.fm_layer(x)

                # Deep component
                deep_input = embeddings.view(embeddings.size(0), -1)
                for layer in self.deep_layers:
                    deep_input = layer(deep_input)

                deep_output = self.output_layer(deep_input).squeeze(1)

                # Combine FM and Deep
                output = first_order + second_order + deep_output
                return torch.sigmoid(output)

        return DeepFMModel(field_dims, self.embedding_dim, self.hidden_layers, self.dropout).to(
            self.device
        )

    def fit(
        self,
        user_ids: List[int],
        item_ids: List[int],
        ratings: List[float],
        user_features: Optional[Dict[int, Dict[str, Any]]] = None,
        item_features: Optional[Dict[int, Dict[str, Any]]] = None,
        **kwargs,
    ) -> "DeepFM":
        # Validate inputs
        validate_fit_inputs(user_ids, item_ids, ratings)

        # Create feature mapping
        # First field: user IDs
        unique_users = sorted(set(user_ids))
        self.feature_map["user"] = {user: idx for idx, user in enumerate(unique_users)}
        self.field_dims.append(len(unique_users))

        # Second field: item IDs
        unique_items = sorted(set(item_ids))
        self.feature_map["item"] = {item: idx for idx, item in enumerate(unique_items)}
        self.field_dims.append(len(unique_items))

        # Additional fields for user features
        if user_features:
            user_feature_types = set()
            for features in user_features.values():
                user_feature_types.update(features.keys())

            for feature_type in sorted(user_feature_types):
                feature_values = set()
                for user, features in user_features.items():
                    if feature_type in features:
                        feature_values.add(features[feature_type])

                self.feature_map[f"user_{feature_type}"] = {
                    val: idx for idx, val in enumerate(sorted(feature_values))
                }
                self.field_dims.append(len(feature_values))

        # Additional fields for item features
        if item_features:
            item_feature_types = set()
            for features in item_features.values():
                item_feature_types.update(features.keys())

            for feature_type in sorted(item_feature_types):
                feature_values = set()
                for item, features in item_features.items():
                    if feature_type in features:
                        feature_values.add(features[feature_type])

                self.feature_map[f"item_{feature_type}"] = {
                    val: idx for idx, val in enumerate(sorted(feature_values))
                }
                self.field_dims.append(len(feature_values))

        # Build model
        self.model = self._build_model(self.field_dims)

        # Create training data
        X = []
        y = []

        for user, item, rating in zip(user_ids, item_ids, ratings):
            # Create feature vector
            x = [self.feature_map["user"][user], self.feature_map["item"][item]]

            # Add user features
            if user_features and user in user_features:
                for feature_type in sorted(user_feature_types):
                    if feature_type in user_features[user]:
                        value = user_features[user][feature_type]
                        x.append(self.feature_map[f"user_{feature_type}"][value])
                    else:
                        x.append(0)  # Default value for missing features

            # Add item features
            if item_features and item in item_features:
                for feature_type in sorted(item_feature_types):
                    if feature_type in item_features[item]:
                        value = item_features[item][feature_type]
                        x.append(self.feature_map[f"item_{feature_type}"][value])
                    else:
                        x.append(0)  # Default value for missing features

            X.append(x)
            y.append(1.0 if rating > 0 else 0.0)  # Convert to binary for implicit feedback

        # Convert to tensors
        X = torch.LongTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)

        # Define optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        # Train the model
        self.model.train()
        n_batches = len(X) // self.batch_size + (1 if len(X) % self.batch_size != 0 else 0)

        for epoch in range(self.epochs):
            total_loss = 0

            # Shuffle data
            indices = torch.randperm(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(X))

                batch_X = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]

                # Forward pass
                outputs = self.model(batch_X)

                # Compute loss
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if self.verbose:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/n_batches:.4f}")

        self.is_fitted = True
        self.user_features = user_features
        self.item_features = item_features
        if user_features:
            self.user_feature_types = sorted(set().union(*[f.keys() for f in user_features.values()]))
        if item_features:
            self.item_feature_types = sorted(set().union(*[f.keys() for f in item_features.values()]))

        return self

    def predict(self, user_id: Any, item_id: Any, **kwargs) -> float:
        """
        Predict the probability of interaction between user and item.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Predicted probability of interaction
        """
        validate_model_fitted(self.is_fitted, self.name)

        if user_id not in self.feature_map["user"]:
            return 0.0
        if item_id not in self.feature_map["item"]:
            return 0.0

        # Create feature vector
        x = [self.feature_map["user"][user_id], self.feature_map["item"][item_id]]

        # Add user features
        if self.user_features and user_id in self.user_features:
            for feature_type in self.user_feature_types:
                if feature_type in self.user_features[user_id]:
                    value = self.user_features[user_id][feature_type]
                    x.append(self.feature_map[f"user_{feature_type}"].get(value, 0))
                else:
                    x.append(0)

        # Add item features
        if self.item_features and item_id in self.item_features:
            for feature_type in self.item_feature_types:
                if feature_type in self.item_features[item_id]:
                    value = self.item_features[item_id][feature_type]
                    x.append(self.feature_map[f"item_{feature_type}"].get(value, 0))
                else:
                    x.append(0)

        # Convert to tensor
        x_tensor = torch.LongTensor([x]).to(self.device)

        # Get prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x_tensor).item()

        return float(prediction)

    def recommend(
        self,
        user_id: Any,
        top_n: int = 10,
        exclude_seen: bool = True,
        **kwargs,
    ) -> List[Any]:
        """
        Recommend top-N items for a user.

        Args:
            user_id: User ID
            top_n: Number of recommendations
            exclude_seen: Whether to exclude items the user has already interacted with

        Returns:
            List of recommended item IDs
        """
        validate_model_fitted(self.is_fitted, self.name)
        validate_user_id(user_id, self.feature_map.get("user", {}))
        validate_top_k(top_n)

        if user_id not in self.feature_map["user"]:
            return []

        # Get all items
        all_items = list(self.feature_map["item"].keys())

        # Get items the user has already interacted with
        seen_items = set()
        if exclude_seen and hasattr(self, "_user_item_interactions"):
            seen_items = self._user_item_interactions.get(user_id, set())

        # Generate predictions for all items
        user_idx = self.feature_map["user"][user_id]
        predictions = []

        # Process in batches for efficiency
        batch_size = 1024
        for i in range(0, len(all_items), batch_size):
            batch_items = all_items[i : i + batch_size]
            batch_X = []

            for item in batch_items:
                if item in seen_items:
                    continue

                # Create feature vector
                x = [user_idx, self.feature_map["item"][item]]

                # Add user features
                if self.user_features and user_id in self.user_features:
                    for feature_type in self.user_feature_types:
                        if feature_type in self.user_features[user_id]:
                            value = self.user_features[user_id][feature_type]
                            x.append(self.feature_map[f"user_{feature_type}"].get(value, 0))
                        else:
                            x.append(0)

                # Add item features
                if self.item_features and item in self.item_features:
                    for feature_type in self.item_feature_types:
                        if feature_type in self.item_features[item]:
                            value = self.item_features[item][feature_type]
                            x.append(self.feature_map[f"item_{feature_type}"].get(value, 0))
                        else:
                            x.append(0)

                batch_X.append(x)

            if not batch_X:
                continue

            # Convert to tensor
            batch_X = torch.LongTensor(batch_X).to(self.device)

            # Get predictions
            self.model.eval()
            with torch.no_grad():
                batch_preds = self.model(batch_X).cpu().detach().tolist()

            # Add to predictions
            for j, item in enumerate(batch_items):
                if item not in seen_items and j < len(batch_preds):
                    predictions.append((item, batch_preds[j]))

        # Sort predictions and get top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_items = [item for item, _ in predictions[:top_n]]

        return top_items

    def save(self, path: Union[str, Path], **kwargs) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ModelNotFittedError(f"{self.name} has not been fitted yet.")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model_state_dict": self.model.state_dict(),
            "feature_map": self.feature_map,
            "field_dims": self.field_dims,
            "embedding_dim": self.embedding_dim,
            "hidden_layers": self.hidden_layers,
            "dropout": self.dropout,
            "user_features": self.user_features,
            "item_features": self.item_features,
            "user_feature_types": self.user_feature_types,
            "item_feature_types": self.item_feature_types,
            "name": self.name,
            "verbose": self.verbose,
        }

        with open(path_obj, "wb") as f:
            pickle.dump(model_data, f)

        if self.verbose:
            logger.info(f"{self.name} model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "DeepFM":
        """
        Load the model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded DeepFM instance
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path_obj, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(
            name=model_data.get("name", "DeepFM"),
            embedding_dim=model_data.get("embedding_dim", 16),
            hidden_layers=model_data.get("hidden_layers", [400, 400, 400]),
            dropout=model_data.get("dropout", 0.3),
            verbose=model_data.get("verbose", False),
        )

        instance.feature_map = model_data["feature_map"]
        instance.field_dims = model_data["field_dims"]
        instance.user_features = model_data.get("user_features")
        instance.item_features = model_data.get("item_features")
        instance.user_feature_types = model_data.get("user_feature_types", [])
        instance.item_feature_types = model_data.get("item_feature_types", [])

        # Rebuild model
        instance.model = instance._build_model(instance.field_dims)
        instance.model.load_state_dict(model_data["model_state_dict"])
        instance.model.eval()

        instance.is_fitted = True

        if instance.verbose:
            logger.info(f"{instance.name} model loaded from {path}")

        return instance
