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
from corerec.base_recommender import BaseCorerec
from corerec.engines.collaborative.device_manager import DeviceManager


class MatrixFactorization(BaseCorerec):
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

    def _create_mappings(self, user_ids: List[int], item_ids: List[int]):
        """Create mappings for user and item IDs."""
        unique_users = set(user_ids)
        self.user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_map.items()}

        unique_items = set(item_ids)
        self.item_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_map.items()}

    def fit(self, user_ids: List[int], item_ids: List[int], ratings: List[float]):
        """Fit the model to the data."""
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
                print(
                    f"Iteration {iteration + 1}/{self.max_iter}, Loss: {total_loss / len(user_ids):.4f}"
                )

    def _predict_batch(self, user_indices: List[int], item_indices: List[int]) -> np.ndarray:
        """Predict ratings for a batch of user-item pairs."""
        preds = self.global_mean
        if self.use_bias:
            preds += self.user_biases[user_indices] + self.item_biases[item_indices]
        preds += np.sum(self.user_factors[user_indices] * self.item_factors[item_indices], axis=1)
        return preds

    def _predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for a single user-item pair."""
        pred = self.global_mean
        if self.use_bias:
            pred += self.user_biases[user_idx] + self.item_biases[item_idx]
        pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return pred

    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """Generate top-N recommendations for a user."""
        if user_id not in self.user_map:
            return []

        user_idx = self.user_map[user_id]
        scores = (
            self.global_mean + self.item_biases if self.use_bias else np.zeros(len(self.item_map))
        )
        scores += np.dot(self.user_factors[user_idx], self.item_factors.T)

        if exclude_seen:
            user_items = set(
                [self.reverse_item_map[iid] for iid in self.user_item_matrix[user_idx].indices]
            )
            for item_id in user_items:
                if item_id in self.item_map:
                    scores[self.item_map[item_id]] = -np.inf

        top_item_indices = np.argsort(scores)[-top_n:][::-1]
        return [self.reverse_item_map[idx] for idx in top_item_indices]

    def save_model(self, filepath: str) -> None:
        """Save the model to a file."""
        model_data = {
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "user_biases": self.user_biases,
            "item_biases": self.item_biases,
            "global_mean": self.global_mean,
            "user_map": self.user_map,
            "item_map": self.item_map,
            "reverse_user_map": self.reverse_user_map,
            "reverse_item_map": self.reverse_item_map,
            "k": self.k,
            "use_bias": self.use_bias,
        }
        np.save(filepath, model_data, allow_pickle=True)

    @classmethod
    def load_model(cls, filepath: str, device: str = "auto") -> "MatrixFactorization":
        """Load a model from a file."""
        model_data = np.load(filepath, allow_pickle=True).item()
        instance = cls(k=model_data["k"], use_bias=model_data["use_bias"], device=device)

        instance.user_factors = model_data["user_factors"]
        instance.item_factors = model_data["item_factors"]
        instance.user_biases = model_data["user_biases"]
        instance.item_biases = model_data["item_biases"]
        instance.global_mean = model_data["global_mean"]
        instance.user_map = model_data["user_map"]
        instance.item_map = model_data["item_map"]
        instance.reverse_user_map = model_data["reverse_user_map"]
        instance.reverse_item_map = model_data["reverse_item_map"]

        return instance
