import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from corerec.api.base_recommender import BaseRecommender
from corerec.api.exceptions import ModelNotFittedError, InvalidParameterError, RecommendationError
from corerec.utils.validation import (
    validate_fit_inputs,
    validate_user_id,
    validate_top_k,
    validate_model_fitted,
)
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
        task: str = "auto",
        num_negatives: int = 4,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        if task not in ("auto", "implicit", "rating"):
            raise ValueError("task must be one of {'auto', 'implicit', 'rating'}")
        if num_negatives < 0:
            raise ValueError("num_negatives must be >= 0")
        # See DCN for the task contract; 'auto' -> implicit with negative sampling.
        self.task = task
        self.num_negatives = num_negatives
        self._fit_task = None
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

    def _build_model(self, field_dims: List[int], use_sigmoid: bool = True):
        class FMLayer(nn.Module):
            def __init__(self, field_dims, embedding_dim):
                super().__init__()
                self.field_dims = field_dims
                # self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
                offsets_np = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
                self.register_buffer('offsets', torch.tensor(offsets_np).long())
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
                return torch.sigmoid(output) if self.use_sigmoid else output

        model = DeepFMModel(field_dims, self.embedding_dim, self.hidden_layers, self.dropout)
        model.use_sigmoid = use_sigmoid
        return model.to(self.device)

    def fit(
        self,
        user_ids: List[int],
        item_ids: List[int],
        ratings: List[float],
        user_features: Optional[Dict[int, Dict[str, Any]]] = None,
        item_features: Optional[Dict[int, Dict[str, Any]]] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> "DeepFM":
        """
        Train the DeepFM model.

        Parameters:
            user_ids: List of user IDs
            item_ids: List of item IDs
            ratings: List of ratings
            user_features: Dictionary of user features
            item_features: Dictionary of item features
            batch_size: Batch size for training (overrides init param if provided)
        """
        (user_ids, item_ids, ratings), _ = self._unpack_fit_args(
            user_ids, item_ids, ratings, supported_modes=("triplet",)
        )
        # Validate inputs
        validate_fit_inputs(user_ids, item_ids, ratings)
        
        if batch_size is not None:
            self.batch_size = batch_size

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
            
            self.user_feature_types = sorted(user_feature_types)

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
            
            self.item_feature_types = sorted(item_feature_types)

        # Resolve task contract and build implicit negatives (see DCN). Training
        # BCE on observed-only data (all labels = 1) collapses the model; for the
        # implicit task we sample unobserved negatives so the loss has signal.
        task = self.task if self.task != "auto" else "implicit"
        self._fit_task = task
        from collections import defaultdict

        seen = defaultdict(set)
        for _u, _it in zip(user_ids, item_ids):
            seen[_u].add(_it)

        if task == "rating":
            train_users = list(user_ids)
            train_items = list(item_ids)
            train_labels = [float(r) for r in ratings]
        else:
            all_items = unique_items
            n_it = len(all_items)
            rng = np.random.RandomState(42)
            train_users, train_items, train_labels = [], [], []
            for _u, _it in zip(user_ids, item_ids):
                train_users.append(_u); train_items.append(_it); train_labels.append(1.0)
                us = seen[_u]
                for _ in range(self.num_negatives):
                    neg = all_items[rng.randint(n_it)]
                    for _t in range(10):
                        if neg not in us:
                            break
                        neg = all_items[rng.randint(n_it)]
                    train_users.append(_u); train_items.append(neg); train_labels.append(0.0)

        # Build model
        self.model = self._build_model(self.field_dims, use_sigmoid=(task != "rating"))

        # Define optimizer and loss (BCE for implicit ranking, MSE for rating)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss() if task != "rating" else nn.MSELoss()

        # Precompute the feature matrix ONCE. The previous implementation rebuilt
        # each row's feature vector inside Dataset.__getitem__, i.e. on every
        # access every epoch (millions of Python calls) -- the dominant training
        # cost. Here we build a single [N, num_fields] tensor and wrap it in a
        # on-device tensor sliced manually (see training loop below).
        umap = self.feature_map["user"]
        imap = self.feature_map["item"]
        has_feats = (self.user_feature_types or self.item_feature_types) and (
            user_features or item_features
        )

        if not has_feats:
            # id-only fast path (fully vectorized)
            X = np.empty((len(train_users), 2), dtype=np.int64)
            X[:, 0] = [umap[u] for u in train_users]
            X[:, 1] = [imap[it] for it in train_items]
        else:
            rows = []
            for u, it in zip(train_users, train_items):
                x = [umap[u], imap[it]]
                if user_features and u in user_features:
                    for ft in self.user_feature_types:
                        x.append(self.feature_map[f"user_{ft}"].get(user_features[u].get(ft), 0)
                                 if ft in user_features[u] else 0)
                if item_features and it in item_features:
                    for ft in self.item_feature_types:
                        x.append(self.feature_map[f"item_{ft}"].get(item_features[it].get(ft), 0)
                                 if ft in item_features[it] else 0)
                rows.append(x)
            X = np.asarray(rows, dtype=np.int64)

        # Move the whole dataset to the device ONCE and slice manually, rather
        # than streaming via a DataLoader that copies every batch host->device
        # (that per-batch transfer dominated training, esp. on GPU).
        X_t = torch.from_numpy(X).long().to(self.device)
        y_t = torch.as_tensor(np.asarray(train_labels, dtype=np.float32)).to(self.device)
        n_samples = X_t.shape[0]

        # Train the model
        self.model.train()
        bs = self.batch_size
        n_batches = (n_samples + bs - 1) // bs

        for epoch in range(self.epochs):
            total_loss = 0.0
            perm = torch.randperm(n_samples, device=self.device)
            for b in range(n_batches):
                idx = perm[b * bs:(b + 1) * bs]
                if idx.numel() < 2:
                    continue  # BatchNorm needs >1 sample
                batch_X = X_t[idx]
                batch_y = y_t[idx]

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if self.verbose:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/n_batches:.4f}")

        # Collapse guard: warn if the model produces near-constant scores.
        self.model.eval()
        with torch.no_grad():
            score_std = float(self.model(X_t[: min(2048, n_samples)]).std().item())
        if score_std < 1e-4:
            logger.warning(
                "DeepFM output collapsed (score std=%.2e): predictions are nearly "
                "constant. Check that labels match the task.", score_std,
            )

        self.is_fitted = True
        self.user_features = user_features
        self.item_features = item_features

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

    def _score_all_items(self, user_id) -> np.ndarray:
        """Score every item for a user in one batched forward pass.

        Builds a [num_items, num_fields] matrix (user/item ids in the first two
        fields, remaining feature fields zero-padded). Ids-only fast path.
        """
        items = list(self.feature_map["item"].keys())
        n_fields = len(self.field_dims)
        x = np.zeros((len(items), n_fields), dtype=np.int64)
        x[:, 0] = self.feature_map["user"][user_id]
        x[:, 1] = [self.feature_map["item"][it] for it in items]
        self.model.eval()
        with torch.no_grad():
            scores = self.model(torch.LongTensor(x).to(self.device)).detach().cpu().numpy().flatten()
        return scores

    def recommend(self, user_id, top_k: int = 10, exclude_items=None, **kwargs):
        """Top-K recommendations via a single batched forward pass."""
        validate_model_fitted(self.is_fitted, self.name)
        if user_id not in self.feature_map["user"]:
            return []
        exclude_items = set(exclude_items or [])
        scores = self._score_all_items(user_id)
        items = list(self.feature_map["item"].keys())
        out = []
        for idx in np.argsort(-scores):
            it = items[idx]
            if it in exclude_items:
                continue
            out.append(it)
            if len(out) >= top_k:
                break
        return out

    def recommend(
        self,
        user_id: Any,
        top_k: int = 10,
        exclude_items: Optional[List[Any]] = None,
        *,
        top_n: Optional[int] = None,
        exclude_seen: bool = True,
        **kwargs,
    ) -> List[Any]:
        """Recommend top-K items for a user."""
        top_k, exclude_items, _ = self._normalize_recommend(
            top_k=top_k,
            top_n=top_n,
            exclude_items=exclude_items,
            **kwargs,
        )
        validate_model_fitted(self.is_fitted, self.name)
        validate_top_k(top_k)

        if user_id not in self.feature_map.get("user", {}):
            raise RecommendationError(f"Unknown user_id: {user_id!r}")

        seen_items = set()
        if exclude_seen and hasattr(self, "_user_item_interactions"):
            seen_items = self._user_item_interactions.get(user_id, set())
        if exclude_items:
            seen_items = seen_items.union(set(exclude_items))

        all_items = list(self.feature_map["item"].keys())

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
        top_items = [item for item, _ in predictions[:top_k]]

        return top_items

    def save(self, path: Union[str, Path], safe: bool = True, **kwargs) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
            safe: Use corerec_safe_v1 bundle (default). Set False for legacy torch checkpoint.
        """
        if not self.is_fitted:
            raise ModelNotFittedError(f"{self.name} has not been fitted yet.")

        from corerec.api.bundle_helpers import load_map_state, save_feature_map, save_map_state
        from corerec.api.torch_bundle import save_torch_production

        path_obj = Path(path)
        config = {
            "name": self.name,
            "embedding_dim": self.embedding_dim,
            "hidden_layers": self.hidden_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "verbose": self.verbose,
            "device": self.device,
            "task": self.task,
            "num_negatives": self.num_negatives,
        }
        state = {
            "field_dims": self.field_dims,
            "_fit_task": self._fit_task,  # 'implicit'/'rating' -> sets the head
            "user_features": self.user_features,
            "item_features": self.item_features,
            "user_feature_types": self.user_feature_types,
            "item_feature_types": self.item_feature_types,
            "is_fitted": self.is_fitted,
            **save_feature_map(self.feature_map),
        }

        if save_torch_production(self, path_obj, config=config, state=state, safe=safe):
            if self.verbose:
                logger.info(f"{self.name} model saved (safe bundle) to {path}")
            return

        path_obj.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "config": config,
            "model_state_dict": self.model.state_dict(),
            **state,
        }

        torch.save(checkpoint, path_obj)

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
        from corerec.api.bundle_helpers import load_feature_map
        from corerec.api.torch_bundle import load_torch_production

        def _restore(instance, config, state, arrays, bundle):
            instance.feature_map = load_feature_map(state)
            instance.field_dims = state["field_dims"]
            instance.user_features = state.get("user_features")
            instance.item_features = state.get("item_features")
            instance.user_feature_types = state.get("user_feature_types", [])
            instance.item_feature_types = state.get("item_feature_types", [])
            instance.is_fitted = state.get("is_fitted", True)
            instance._fit_task = state.get("_fit_task", "implicit")

        def _build(instance, bundle):
            if bundle.get("state_dict") is not None:
                instance.model = instance._build_model(
                    instance.field_dims,
                    use_sigmoid=(getattr(instance, "_fit_task", "implicit") != "rating"),
                )

        loaded = load_torch_production(cls, path, build_model=_build, restore=_restore)
        if loaded is not None:
            if loaded.verbose:
                logger.info(f"{loaded.name} model loaded (safe bundle) from {path}")
            return loaded

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path_obj, weights_only=False)
        cfg = checkpoint["config"]

        instance = cls(
            name=cfg.get("name", "DeepFM"),
            embedding_dim=cfg.get("embedding_dim", 16),
            hidden_layers=cfg.get("hidden_layers", [400, 400, 400]),
            dropout=cfg.get("dropout", 0.3),
            learning_rate=cfg.get("learning_rate", 0.001),
            batch_size=cfg.get("batch_size", 256),
            epochs=cfg.get("epochs", 20),
            verbose=cfg.get("verbose", False),
            device=cfg.get("device", "cpu"),
            task=cfg.get("task", "auto"),
            num_negatives=cfg.get("num_negatives", 4),
        )
        instance._fit_task = checkpoint.get("_fit_task", cfg.get("task", "implicit"))
        if instance._fit_task == "auto":
            instance._fit_task = "implicit"

        instance.feature_map = checkpoint["feature_map"]
        instance.field_dims = checkpoint["field_dims"]
        instance.user_features = checkpoint.get("user_features")
        instance.item_features = checkpoint.get("item_features")
        instance.user_feature_types = checkpoint.get("user_feature_types", [])
        instance.item_feature_types = checkpoint.get("item_feature_types", [])

        instance.model = instance._build_model(
            instance.field_dims, use_sigmoid=(instance._fit_task != "rating"))
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.model.eval()

        instance.is_fitted = checkpoint.get("is_fitted", True)

        if instance.verbose:
            logger.info(f"{instance.name} model loaded from {path}")

        return instance
