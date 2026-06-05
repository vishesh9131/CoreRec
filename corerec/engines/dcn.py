import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from scipy.sparse import csr_matrix
from tqdm import tqdm

from corerec.api.base_recommender import BaseRecommender
from corerec.api.exceptions import ModelNotFittedError, InvalidParameterError
from corerec.utils.validation import (
    validate_fit_inputs,
    validate_user_id,
    validate_top_k,
    validate_model_fitted,
    validate_embeddings_dim,
    ValidationError,
)
from corerec.utils.training_utils import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


class DCN(BaseRecommender):
    """
    Deep & Cross Network for Feature-rich Recommendation

    Combines a cross network for explicit feature interactions with
    a deep neural network for implicit feature interactions. Especially
    effective for feature-rich recommendation scenarios.

    Reference:
    Wang et al. "Deep & Cross Network for Ad Click Predictions" (2017)

    Args:
        name: Model name for identification
        embedding_dim: Dimension of feature embeddings (default: 16)
        num_cross_layers: Number of cross network layers (default: 3)
        deep_layers: List of hidden layer sizes for deep network (default: [128, 64])
        dropout: Dropout rate (default: 0.2)
        learning_rate: Learning rate for optimizer (default: 0.001)
        batch_size: Training batch size (default: 256)
        epochs: Number of training epochs (default: 20)
        early_stopping_patience: Patience for early stopping, None to disable (default: 5)
        checkpoint_dir: Directory to save checkpoints, None to disable (default: None)
        trainable: Whether model is trainable (default: True)
        verbose: Whether to print training progress (default: False)
        device: Device for computation ('cuda' or 'cpu', default: auto-detect)

    Example:
        >>> model = DCN(embedding_dim=64, epochs=20, verbose=True)
        >>> model.fit(user_ids=[1,2,3], item_ids=[10,20,30], ratings=[5.0,4.0,3.0])
        >>> recommendations = model.recommend(user_id=1, top_k=10)
    """

    def __init__(
        self,
        name: str = "DCN",
        embedding_dim: int = 16,
        num_cross_layers: int = 3,
        deep_layers: List[int] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        early_stopping_patience: Optional[int] = 5,
        checkpoint_dir: Optional[str] = None,
        trainable: bool = True,
        verbose: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        task: str = "auto",
        num_negatives: int = 4,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        # Set default for deep_layers if None
        if deep_layers is None:
            deep_layers = [128, 64]

        # Validate parameters
        validate_embeddings_dim(embedding_dim)

        if num_cross_layers < 1:
            raise InvalidParameterError("num_cross_layers must be at least 1")

        if not deep_layers or len(deep_layers) == 0:
            raise InvalidParameterError("deep_layers must contain at least one layer")

        if not (0.0 <= dropout < 1.0):
            raise InvalidParameterError("dropout must be in range [0.0, 1.0)")

        if learning_rate <= 0:
            raise InvalidParameterError("learning_rate must be positive")

        if batch_size < 1:
            raise InvalidParameterError("batch_size must be at least 1")

        if epochs < 1:
            raise InvalidParameterError("epochs must be at least 1")

        if task not in ("auto", "implicit", "rating"):
            raise InvalidParameterError("task must be one of {'auto', 'implicit', 'rating'}")

        if num_negatives < 0:
            raise InvalidParameterError("num_negatives must be >= 0")

        # task contract:
        #   implicit -> binary relevance + negative sampling, sigmoid + BCE (ranking)
        #   rating   -> regress the supplied rating, linear head + MSE
        #   auto     -> implicit (top-K recommendation is the primary path)
        self.task = task
        self.num_negatives = num_negatives
        self._fit_task = None  # resolved task after fit

        self.embedding_dim = embedding_dim
        self.num_cross_layers = num_cross_layers
        self.deep_layers = deep_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        self.user_map = {}
        self.item_map = {}
        self.feature_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.user_features = {}
        self.item_features = {}
        self.model = None

    def _build_model(self, num_features: int, max_features: int = 2, use_sigmoid: bool = True):
        class CrossLayer(nn.Module):
            def __init__(self, input_dim: int):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(input_dim) * 0.01)
                self.bias = nn.Parameter(torch.zeros(input_dim))

            def forward(self, x0, x):
                # x0 is the input, x is the current layer's input
                # Cross network formula: x0 * (x^T w) + b + x
                xw = (x * self.weight).sum(dim=1, keepdim=True)  # [batch, 1]
                return x0 * xw + self.bias + x

        class DeepCrossNetworkModel(nn.Module):
            def __init__(
                self,
                num_features,
                embedding_dim,
                num_cross_layers,
                deep_layers,
                dropout,
                max_features,
            ):
                super().__init__()
                self.embedding = nn.Embedding(num_features, embedding_dim)

                # Input dimension after embedding (features per sample * embedding_dim)
                self.input_dim = max_features * embedding_dim

                # Cross Network
                self.cross_layers = nn.ModuleList(
                    [CrossLayer(self.input_dim) for _ in range(num_cross_layers)]
                )

                # Deep Network
                deep_input_dim = self.input_dim
                self.deep_layers = nn.ModuleList()
                for layer_size in deep_layers:
                    self.deep_layers.append(nn.Linear(deep_input_dim, layer_size))
                    self.deep_layers.append(nn.ReLU())
                    self.deep_layers.append(nn.Dropout(dropout))
                    deep_input_dim = layer_size

                # Combination Layer
                self.combination = nn.Linear(self.input_dim + deep_layers[-1], 1)

            def forward(self, feature_indices):
                # Get embeddings and flatten
                embeddings = self.embedding(feature_indices)
                x0 = embeddings.view(embeddings.size(0), -1)

                # Cross Network
                cross_output = x0
                for cross_layer in self.cross_layers:
                    cross_output = cross_layer(x0, cross_output)

                # Deep Network
                deep_output = x0
                for layer in self.deep_layers:
                    deep_output = layer(deep_output)

                # Combine outputs
                combined = torch.cat([cross_output, deep_output], dim=1)
                output = self.combination(combined)

                output = output.squeeze(1)
                # sigmoid only for the implicit (BCE) head; rating head is linear
                return torch.sigmoid(output) if self.use_sigmoid else output

        model = DeepCrossNetworkModel(
            num_features,
            self.embedding_dim,
            self.num_cross_layers,
            self.deep_layers,
            self.dropout,
            max_features,
        )
        model.use_sigmoid = use_sigmoid
        return model.to(self.device)

    def fit(
        self,
        user_ids: List[int],
        item_ids: List[int],
        ratings: List[float],
        user_features: Optional[Dict[int, Dict[str, Any]]] = None,
        item_features: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> "DCN":
        """Train the DCN model."""
        (user_ids, item_ids, ratings), _ = self._unpack_fit_args(
            user_ids, item_ids, ratings, supported_modes=("triplet",)
        )

        # Validate inputs
        validate_fit_inputs(user_ids, item_ids, ratings)

        # Create mappings
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))

        self.user_map = {user: idx for idx, user in enumerate(unique_users)}
        self.item_map = {item: idx + len(unique_users) for idx, item in enumerate(unique_items)}

        # Set uid_map and iid_map for BaseRecommender properties
        self.uid_map = self.user_map
        self.iid_map = self.item_map
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)

        self.reverse_user_map = {idx: user for user, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item for item, idx in self.item_map.items()}

        # Store features
        self.user_features = user_features or {}
        self.item_features = item_features or {}

        # Create feature map
        feature_values = set()

        # Add user and item IDs as features
        feature_values.update(self.user_map.values())
        feature_values.update(self.item_map.values())

        # Add user features
        if user_features:
            for user_id, features in user_features.items():
                for feature, value in features.items():
                    feature_key = f"user_{feature}_{value}"
                    if feature_key not in self.feature_map:
                        self.feature_map[feature_key] = (
                            len(self.feature_map) + len(unique_users) + len(unique_items)
                        )
                    feature_values.add(self.feature_map[feature_key])

        # Add item features
        if item_features:
            for item_id, features in item_features.items():
                for feature, value in features.items():
                    feature_key = f"item_{feature}_{value}"
                    if feature_key not in self.feature_map:
                        self.feature_map[feature_key] = (
                            len(self.feature_map) + len(unique_users) + len(unique_items)
                        )
                    feature_values.add(self.feature_map[feature_key])

        # Resolve the task contract. 'auto' -> implicit, because top-K
        # recommendation is the primary path and implicit training (positives
        # plus sampled negatives) is what makes ranking work. Training BCE on
        # observed-only data (all labels = 1) collapses the model to a constant.
        task = self.task if self.task != "auto" else "implicit"
        self._fit_task = task

        # Per-user seen items for negative sampling
        from collections import defaultdict

        seen = defaultdict(set)
        for u, it in zip(user_ids, item_ids):
            seen[u].add(it)
        all_item_ids = list(unique_items)
        n_items_total = len(all_item_ids)
        rng = np.random.RandomState(42)

        def build_feat(uid, iid):
            fi = [self.user_map[uid], self.item_map[iid]]
            if user_features and uid in user_features:
                for feature, value in user_features[uid].items():
                    fk = f"user_{feature}_{value}"
                    if fk in self.feature_map:
                        fi.append(self.feature_map[fk])
            if item_features and iid in item_features:
                for feature, value in item_features[iid].items():
                    fk = f"item_{feature}_{value}"
                    if fk in self.feature_map:
                        fi.append(self.feature_map[fk])
            return fi

        # Create training data first to determine max_features
        train_features = []
        train_labels = []

        for user_id, item_id, rating in zip(user_ids, item_ids, ratings):
            if task == "rating":
                train_features.append(build_feat(user_id, item_id))
                train_labels.append(float(rating))
                continue
            # implicit: observed interaction is a positive ...
            train_features.append(build_feat(user_id, item_id))
            train_labels.append(1.0)
            # ... plus sampled unobserved negatives so BCE has real signal
            user_seen = seen[user_id]
            for _ in range(self.num_negatives):
                neg = all_item_ids[rng.randint(n_items_total)]
                for _try in range(10):
                    if neg not in user_seen:
                        break
                    neg = all_item_ids[rng.randint(n_items_total)]
                train_features.append(build_feat(user_id, neg))
                train_labels.append(0.0)

        # Pad feature lists to the same length
        max_features = max(len(features) for features in train_features) if train_features else 2
        train_features = [
            features + [0] * (max_features - len(features)) for features in train_features
        ]

        # Build model with correct dimensions (after we know max_features)
        num_features = len(feature_values) + 1  # +1 for padding/unknown
        self._num_features = num_features
        self._max_features = max_features
        self.model = self._build_model(num_features, max_features, use_sigmoid=(task != "rating"))

        # Convert to tensors
        train_features = torch.LongTensor(train_features).to(self.device)
        train_labels = torch.FloatTensor(train_labels).to(self.device)

        # Define optimizer and loss (BCE for implicit ranking, MSE for rating)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss() if task != "rating" else nn.MSELoss()

        # Train the model
        self.model.train()
        n_batches = len(train_features) // self.batch_size + (
            1 if len(train_features) % self.batch_size != 0 else 0
        )

        for epoch in range(self.epochs):
            total_loss = 0

            # Shuffle data
            indices = torch.randperm(len(train_features))
            train_features = train_features[indices]
            train_labels = train_labels[indices]

            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(train_features))

                batch_features = train_features[start_idx:end_idx]
                batch_labels = train_labels[start_idx:end_idx]

                # Forward pass
                outputs = self.model(batch_features)

                # Compute loss
                loss = criterion(outputs, batch_labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if self.verbose:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/n_batches:.4f}")

        # Collapse guard: a healthy model produces varied scores across items.
        # If every score is (near) identical the model has degenerated (e.g. bad
        # label scale) and any ranking from it is meaningless.
        self.model.eval()
        with torch.no_grad():
            probe = self.model(train_features[: min(2048, len(train_features))])
            score_std = float(probe.std().item())
        if score_std < 1e-4:
            logger.warning(
                "DCN output collapsed (score std=%.2e): predictions are nearly "
                "constant, so rankings will be meaningless. Check that labels match "
                "the task ('implicit' expects relevance, 'rating' expects scores).",
                score_std,
            )

        self.is_fitted = True
        return self

    def predict(self, user_id: int, item_id: int, **kwargs) -> float:
        """Predict score for a user-item pair."""
        if not self.is_fitted:
            raise ModelNotFittedError()

        if user_id not in self.user_map or item_id not in self.item_map:
            return 0.0

        # Get indices
        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]

        # Build feature vector
        feature_indices = [user_idx, item_idx]

        # Add features
        if user_id in self.user_features:
            for feature, value in self.user_features[user_id].items():
                feature_key = f"user_{feature}_{value}"
                if feature_key in self.feature_map:
                    feature_indices.append(self.feature_map[feature_key])

        if item_id in self.item_features:
            for feature, value in self.item_features[item_id].items():
                feature_key = f"item_{feature}_{value}"
                if feature_key in self.feature_map:
                    feature_indices.append(self.feature_map[feature_key])

        # Pad
        max_len = self.model.input_dim // self.embedding_dim
        feature_indices = (feature_indices + [0] * max_len)[:max_len]

        # Predict
        feature_tensor = torch.LongTensor([feature_indices]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            score = self.model(feature_tensor).item()

        return score

    def recommend(
        self, user_id: int, top_k: int = 10, exclude_items: Optional[List[int]] = None, **kwargs
    ) -> List[int]:
        """Generate top-K recommendations for a user."""
        if not self.is_fitted:
            raise ModelNotFittedError()

        if user_id not in self.user_map:
            return []

        exclude_items = set(exclude_items or [])

        # Vectorized full ranking: score every item in ONE batched forward pass
        # instead of a Python loop of per-item predict() calls. This is the
        # production-critical path (turns ~hundreds of ms/user into a few ms).
        scores = self._score_all_items(user_id)  # np.ndarray over item ids
        all_items = list(self.item_map.keys())
        order = np.argsort(-scores)
        out = []
        for idx in order:
            item_id = all_items[idx]
            if item_id in exclude_items:
                continue
            out.append(item_id)
            if len(out) >= top_k:
                break
        return out

    def _score_all_items(self, user_id) -> np.ndarray:
        """Score every known item for a user in a single batched forward pass.

        Returns scores aligned with ``list(self.item_map.keys())`` order.
        """
        all_items = list(self.item_map.keys())
        max_len = self.model.input_dim // self.embedding_dim
        user_idx = self.user_map[user_id]

        # Build a [num_items, max_len] feature matrix: column 0 = user, column 1
        # = item, remaining columns padded. Per-item side features are omitted on
        # this fast path (ids only), matching the dominant id-based usage.
        feats = np.zeros((len(all_items), max_len), dtype=np.int64)
        feats[:, 0] = user_idx
        feats[:, 1] = [self.item_map[it] for it in all_items]

        self.model.eval()
        with torch.no_grad():
            t = torch.LongTensor(feats).to(self.device)
            scores = self.model(t).detach().cpu().numpy()
        return scores

    def save(self, path: Union[str, Path], safe: bool = True, **kwargs) -> None:
        """Save model to disk (safe bundle by default; legacy full checkpoint if safe=False)."""
        path_obj = Path(path)

        from corerec.api.bundle_helpers import pairs, save_map_state

        config = {
            "name": self.name,
            "embedding_dim": self.embedding_dim,
            "num_cross_layers": self.num_cross_layers,
            "deep_layers": self.deep_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "checkpoint_dir": self.checkpoint_dir,
            "verbose": self.verbose,
            "device": self.device,
            "task": self.task,
            "num_negatives": self.num_negatives,
        }
        state = {
            "_num_features": self._num_features,
            "_max_features": self._max_features,
            "_fit_task": self._fit_task,  # 'implicit'/'rating' -> sets the head
            "feature_map_pairs": pairs(self.feature_map),
            "user_features": self.user_features,
            "item_features": self.item_features,
            "is_fitted": self.is_fitted,
            **save_map_state(
                user_map=self.user_map,
                item_map=self.item_map,
                reverse_user_map=self.reverse_user_map,
                reverse_item_map=self.reverse_item_map,
            ),
        }

        from corerec.api.torch_bundle import save_torch_production

        if save_torch_production(self, path_obj, config=config, state=state, safe=safe):
            if self.verbose:
                logger.info(f"Model saved (safe bundle) to {path_obj}")
            return

        path_obj.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "config": config,
            "build_params": {
                "num_features": self._num_features,
                "max_features": self._max_features,
            },
            "model_state_dict": self.model.state_dict() if self.model else None,
            "user_map": self.user_map,
            "item_map": self.item_map,
            "feature_map": self.feature_map,
            "reverse_user_map": self.reverse_user_map,
            "reverse_item_map": self.reverse_item_map,
            "user_features": self.user_features,
            "item_features": self.item_features,
            "is_fitted": self.is_fitted,
        }

        torch.save(checkpoint, path_obj)

        if self.verbose:
            logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DCN":
        """Load model from disk (safe bundle or legacy checkpoint)."""
        from corerec.api.torch_bundle import load_torch_production
        from corerec.api.bundle_helpers import dict_from_pairs, load_map_state

        def _restore(instance, config, state, arrays, bundle):
            maps = load_map_state(
                state,
                "user_map",
                "item_map",
                "reverse_user_map",
                "reverse_item_map",
                int_key_names=("reverse_user_map", "reverse_item_map"),
            )
            instance.user_map = maps["user_map"]
            instance.item_map = maps["item_map"]
            instance.reverse_user_map = maps["reverse_user_map"]
            instance.reverse_item_map = maps["reverse_item_map"]
            instance.feature_map = dict_from_pairs(state.get("feature_map_pairs"))
            instance._num_features = state["_num_features"]
            instance._max_features = state["_max_features"]
            instance.user_features = state.get("user_features")
            instance.item_features = state.get("item_features")
            instance.is_fitted = state.get("is_fitted", True)
            instance._fit_task = state.get("_fit_task", "implicit")

        def _build(instance, bundle):
            if bundle.get("state_dict") is not None:
                instance.model = instance._build_model(
                    instance._num_features, instance._max_features,
                    use_sigmoid=(getattr(instance, "_fit_task", "implicit") != "rating"),
                )

        loaded = load_torch_production(cls, path, build_model=_build, restore=_restore)
        if loaded is not None:
            if loaded.verbose:
                logger.info(f"Model loaded (safe bundle) from {path}")
            return loaded

        checkpoint = torch.load(path, weights_only=False)
        cfg = checkpoint["config"]

        instance = cls(
            name=cfg["name"],
            embedding_dim=cfg["embedding_dim"],
            num_cross_layers=cfg["num_cross_layers"],
            deep_layers=cfg["deep_layers"],
            dropout=cfg["dropout"],
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            early_stopping_patience=cfg["early_stopping_patience"],
            checkpoint_dir=cfg["checkpoint_dir"],
            verbose=cfg["verbose"],
            device=cfg["device"],
            task=cfg.get("task", "auto"),
            num_negatives=cfg.get("num_negatives", 4),
        )
        instance._fit_task = checkpoint.get("_fit_task", cfg.get("task", "implicit"))
        if instance._fit_task == "auto":
            instance._fit_task = "implicit"

        instance.user_map = checkpoint["user_map"]
        instance.item_map = checkpoint["item_map"]
        instance.feature_map = checkpoint["feature_map"]
        instance.reverse_user_map = checkpoint["reverse_user_map"]
        instance.reverse_item_map = checkpoint["reverse_item_map"]
        instance.user_features = checkpoint["user_features"]
        instance.item_features = checkpoint["item_features"]
        instance.is_fitted = checkpoint["is_fitted"]

        bp = checkpoint["build_params"]
        instance._num_features = bp["num_features"]
        instance._max_features = bp["max_features"]

        if checkpoint["model_state_dict"] is not None:
            instance.model = instance._build_model(
                bp["num_features"], bp["max_features"],
                use_sigmoid=(instance._fit_task != "rating"),
            )
            instance.model.load_state_dict(checkpoint["model_state_dict"])
            instance.model.eval()

        if instance.verbose:
            logger.info(f"Model loaded from {path}")

        return instance
