# Copyright 2023 The UnionizedFilterEngine Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from typing import List, Optional, Dict, Any, Tuple
from corerec.api.base_recommender import BaseRecommender
from corerec.utils.validation import (
    validate_fit_inputs,
    validate_user_id,
    validate_top_k,
    validate_model_fitted,
)
from corerec.engines.collaborative.device_manager import DeviceManager
import logging

logger = logging.getLogger(__name__)

from corerec.api.exceptions import ModelNotFittedError
from typing import Union
from pathlib import Path


class MatrixFactorization(BaseRecommender):
    """Matrix Factorization for Unionized Filtering.

    Parameters
    ----------
    name: str, required
        Name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trainable.

    verbose: boolean, optional, default: False
        When True, running logs are displayed.
    k: int, optional, default: 10
        The dimension of the latent factors.
    learning_rate: float, optional, default: 0.01
        The learning rate.

    lambda_reg: float, optional, default: 0.02
        The regularization parameter.

    max_iter: int, optional, default: 20
        Maximum number of iterations.

    use_bias: bool, optional, default: True
        Whether to use user and item biases.

    verbose: bool, optional, default: False
        Whether to show training progress.

    seed: int, optional, default: None
        Random seed for reproducibility.

    device: str, optional, default: 'auto'
        Computation device ('cpu', 'gpu', or 'auto').

    batch_size: int, optional, default: 10000
        Batch size for mini-batch training.
    """

    def __init__(
        self,
        k: int = 10,
        learning_rate: float = 0.01,
        lambda_reg: float = 0.02,
        max_iter: int = 20,
        use_bias: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None,
        device: str = "auto",
        batch_size: int = 10000,
    ):
        super().__init__(name="MatrixFactorization", trainable=True, verbose=verbose)
        self.k = k
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.use_bias = use_bias
        self.seed = seed
        self.batch_size = batch_size

        # Model parameters
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = 0.0

        # Mappings
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}

        # User-item interaction matrix
        self.user_item_matrix = None

        # Device manager
        self.device_manager = DeviceManager()
        self.device_manager.set_device(device)
        self.xp = self.device_manager.get_framework_for_device()

    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """Create mappings between original IDs and matrix indices."""
        self.user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_map.items()}

    def fit(
        self, user_ids: List[int], item_ids: List[int], ratings: List[float], **kwargs
    ) -> "MatrixFactorization":
        """Train the matrix factorization model."""
        # Validate inputs
        validate_fit_inputs(user_ids, item_ids, ratings)

        self._create_mappings(user_ids, item_ids)

        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.user_item_matrix = lil_matrix((n_users, n_items), dtype=np.float32)

        for u_id, i_id, rating in zip(user_ids, item_ids, ratings):
            u_idx = self.user_map[u_id]
            i_idx = self.item_map[i_id]
            self.user_item_matrix[u_idx, i_idx] = rating

        self.user_item_matrix = self.user_item_matrix.tocsr()

        self.user_factors = np.random.normal(0, 0.01, (n_users, self.k))
        self.item_factors = np.random.normal(0, 0.01, (n_items, self.k))
        if self.use_bias:
            self.user_biases = np.zeros(n_users)
            self.item_biases = np.zeros(n_items)
        self.global_mean = np.mean(ratings)

        if self.seed is not None:
            np.random.seed(self.seed)

        for iteration in range(self.max_iter):
            total_loss = 0.0
            indices = np.arange(len(user_ids))
            np.random.shuffle(indices)

            for start in range(0, len(user_ids), self.batch_size):
                end = min(start + self.batch_size, len(user_ids))
                batch_indices = indices[start:end]

                batch_user_ids = [user_ids[i] for i in batch_indices]
                batch_item_ids = [item_ids[i] for i in batch_indices]
                batch_ratings = [ratings[i] for i in batch_indices]

                batch_user_indices = [self.user_map[u_id] for u_id in batch_user_ids]
                batch_item_indices = [self.item_map[i_id] for i_id in batch_item_ids]

                preds = self._predict_batch(batch_user_indices, batch_item_indices)
                errors = np.array(batch_ratings) - preds

                self.user_factors[batch_user_indices] += self.learning_rate * (
                    errors[:, np.newaxis] * self.item_factors[batch_item_indices]
                    - self.lambda_reg * self.user_factors[batch_user_indices]
                )
                self.item_factors[batch_item_indices] += self.learning_rate * (
                    errors[:, np.newaxis] * self.user_factors[batch_user_indices]
                    - self.lambda_reg * self.item_factors[batch_item_indices]
                )

                if self.use_bias:
                    self.user_biases[batch_user_indices] += self.learning_rate * (
                        errors - self.lambda_reg * self.user_biases[batch_user_indices]
                    )
                    self.item_biases[batch_item_indices] += self.learning_rate * (
                        errors - self.lambda_reg * self.item_biases[batch_item_indices]
                    )

                total_loss += np.sum(errors**2)

            if self.verbose:
                logger.info(
                    f"Iteration {iteration + 1}/{self.max_iter}, Loss: {total_loss / len(user_ids):.4f}"
                )

        self.is_fitted = True
        return self

    def _predict_batch(self, user_indices: List[int], item_indices: List[int]) -> np.ndarray:
        """Predict ratings for a batch of user-item pairs."""
        preds = np.zeros(len(user_indices))
        for i, (u_idx, i_idx) in enumerate(zip(user_indices, item_indices)):
            pred = self.global_mean
            if self.use_bias:
                pred += self.user_biases[u_idx] + self.item_biases[i_idx]
            pred += np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
            preds[i] = pred
        return preds

    def predict(self, user_id: int, item_id: int, **kwargs) -> float:
        """Predict rating for a user-item pair."""
        if not self.is_fitted:
            raise ModelNotFittedError(f"{self.name} must be fitted before making predictions")

        if user_id not in self.user_map or item_id not in self.item_map:
            return 0.0

        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]

        pred = self.global_mean
        if self.use_bias:
            pred += self.user_biases[user_idx] + self.item_biases[item_idx]
        pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return float(pred)

    def recommend(self, user_id: int, top_k: int = 10, **kwargs) -> List[int]:
        """Generate top-K recommendations for a user."""
        # Validate inputs
        validate_model_fitted(self.is_fitted, self.name)
        validate_user_id(user_id, self.user_map if hasattr(self, "user_map") else {})
        validate_top_k(top_k)

        if user_id not in self.user_map:
            return []

        user_idx = self.user_map[user_id]
        scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
        if self.use_bias:
            scores += self.item_biases

        # Exclude items user has already rated
        if self.user_item_matrix is not None:
            rated_items = self.user_item_matrix[user_idx].indices
            scores[rated_items] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.reverse_item_map[idx] for idx in top_indices if idx in self.reverse_item_map]

    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save model to disk."""
        import pickle

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.verbose:
            logger.info(f"{self.name} model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "MatrixFactorization":
        """Load model from disk."""
        import pickle

        with open(path, "rb") as f:
            model = pickle.load(f)

        if hasattr(model, "verbose") and model.verbose:
            logger.info(f"Model loaded from {path}")

        return model

    @classmethod
    def load_model(cls, filepath: str, device: str = "auto") -> "MatrixFactorization":
        """Load model from file using numpy save format."""
        model_data = np.load(filepath, allow_pickle=True).item()

        # Create instance with saved parameters
        instance = cls(
            k=model_data.get("k", 10), use_bias=model_data.get("use_bias", True), device=device
        )

        # Restore model state
        instance.user_factors = model_data["user_factors"]
        instance.item_factors = model_data["item_factors"]
        instance.user_biases = model_data.get("user_biases")
        instance.item_biases = model_data.get("item_biases")
        instance.global_mean = model_data.get("global_mean", 0.0)
        instance.user_map = model_data["user_map"]
        instance.item_map = model_data["item_map"]
        instance.reverse_user_map = model_data.get("reverse_user_map", {})
        instance.reverse_item_map = model_data.get("reverse_item_map", {})

        instance.is_fitted = True
        return instance
