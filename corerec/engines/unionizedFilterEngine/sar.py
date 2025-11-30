import numpy as np
import torch
from typing import Union, List, Dict, Tuple, Optional
from scipy.sparse import csr_matrix
import os
import pickle
from pathlib import Path

from .base_recommender import BaseRecommender
from corerec.api.exceptions import ModelNotFittedError


class SAR(BaseRecommender):
    """
    Simple Algorithm for Recommendation (SAR)

    A neighborhood-based recommender system using item-item similarity.

    Supports similarity metrics:
    - jaccard
    - cosine
    - lift

    Supports time-decay on ratings.
    """

    def __init__(
        self,
        similarity_type: str = 'jaccard',
        time_decay_coefficient: Optional[float] = None,
        time_now: Optional[int] = None,
        timedecay_formula: str = 'exp',
    ):
        self.similarity_type = similarity_type
        self.time_decay_coefficient = time_decay_coefficient
        self.time_now = time_now
        self.timedecay_formula = timedecay_formula

        # mappings
        self.user_to_index: Dict = {}
        self.item_to_index: Dict = {}
        self.index_to_user: Dict = {}
        self.index_to_item: Dict = {}

        # model data
        self.similarity_matrix = None
        self.user_item_matrix = None

        self.is_fitted = False

    # ----------------------------------------------------------------------
    # MAPPINGS
    # ----------------------------------------------------------------------
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]):
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))

        self.user_to_index = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_index = {item: idx for idx, item in enumerate(unique_items)}

        self.index_to_user = {idx: user for user, idx in self.user_to_index.items()}
        self.index_to_item = {idx: item for item, idx in self.item_to_index.items()}

    # ----------------------------------------------------------------------
    # TIME DECAY
    # ----------------------------------------------------------------------
    def _apply_time_decay(self, ratings: List[float], timestamps: List[int]) -> List[float]:
        if self.time_decay_coefficient is None or self.time_now is None:
            return ratings

        decayed = []
        for r, t in zip(ratings, timestamps):
            diff = self.time_now - t
            if self.timedecay_formula == "exp":
                decay = np.exp(-self.time_decay_coefficient * diff)
            else:
                decay = 1.0 / (1.0 + self.time_decay_coefficient * diff)

            decayed.append(r * decay)

        return decayed

    # ----------------------------------------------------------------------
    # SIMILARITY
    # ----------------------------------------------------------------------
    def _compute_similarity(self, matrix: csr_matrix) -> np.ndarray:
        n_items = matrix.shape[1]

        if self.similarity_type in ["jaccard", "lift"]:
            binary_matrix = matrix.copy()
            binary_matrix.data = np.ones_like(binary_matrix.data)

        if self.similarity_type == "jaccard":
            intersection = binary_matrix.T @ binary_matrix  # co-occurrence
            item_pop = np.array(binary_matrix.sum(axis=0)).flatten()

            union = (
                item_pop.reshape(-1, 1)
                + item_pop.reshape(1, -1)
                - intersection.toarray()
            )

            similarity = intersection.toarray() / np.maximum(union, 1)

        elif self.similarity_type == "cosine":
            row_norms = np.sqrt(matrix.power(2).sum(axis=1)).A1
            row_idx, col_idx = matrix.nonzero()
            matrix.data = matrix.data / row_norms[row_idx]

            similarity = (matrix.T @ matrix).toarray()

        elif self.similarity_type == "lift":
            n_users = matrix.shape[0]

            item_prob = np.array(binary_matrix.sum(axis=0) / n_users).flatten()
            co_occur = (binary_matrix.T @ binary_matrix).toarray() / n_users

            similarity = np.zeros((n_items, n_items))
            for i in range(n_items):
                for j in range(n_items):
                    denom = item_prob[i] * item_prob[j]
                    if denom > 0:
                        similarity[i, j] = co_occur[i, j] / denom
        else:
            raise ValueError(f"Unsupported similarity type: {self.similarity_type}")

        np.fill_diagonal(similarity, 0)
        return similarity

    # ----------------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------------
    def fit(
        self,
        user_ids: List[int],
        item_ids: List[int],
        ratings: List[float],
        timestamps: Optional[List[int]] = None,
    ):
        self._create_mappings(user_ids, item_ids)

        if timestamps is not None:
            ratings = self._apply_time_decay(ratings, timestamps)

        rows = [self.user_to_index[u] for u in user_ids]
        cols = [self.item_to_index[i] for i in item_ids]

        n_users = len(self.user_to_index)
        n_items = len(self.item_to_index)

        self.user_item_matrix = csr_matrix(
            (ratings, (rows, cols)), shape=(n_users, n_items)
        )

        self.similarity_matrix = self._compute_similarity(self.user_item_matrix)

        self.is_fitted = True
        return self

    # ----------------------------------------------------------------------
    # RECOMMEND
    # ----------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_seen: bool = True,
    ) -> List[int]:

        if not self.is_fitted:
            raise ModelNotFittedError("Model not fitted. Call fit() first.")

        if user_id not in self.user_to_index:
            return []

        user_idx = self.user_to_index[user_id]
        user_vector = self.user_item_matrix[user_idx].toarray().flatten()

        seen = np.where(user_vector > 0)[0] if exclude_seen else []

        scores = user_vector @ self.similarity_matrix

        scores[seen] = -np.inf

        top_idx = np.argsort(scores)[::-1][:top_n]
        return [self.index_to_item[i] for i in top_idx]

    # ----------------------------------------------------------------------
    # SAVE
    # ----------------------------------------------------------------------
    def save_model(self, filepath: str):
        data = {
            "similarity_type": self.similarity_type,
            "time_decay_coefficient": self.time_decay_coefficient,
            "time_now": self.time_now,
            "timedecay_formula": self.timedecay_formula,
            "user_to_index": self.user_to_index,
            "item_to_index": self.item_to_index,
            "index_to_user": self.index_to_user,
            "index_to_item": self.index_to_item,
            "similarity_matrix": self.similarity_matrix,
            "user_item_matrix": self.user_item_matrix,
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    # ----------------------------------------------------------------------
    # LOAD
    # ----------------------------------------------------------------------
    @classmethod
    def load_model(cls, filepath: str) -> "SAR":
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        obj = cls(
            similarity_type=data["similarity_type"],
            time_decay_coefficient=data["time_decay_coefficient"],
            time_now=data["time_now"],
            timedecay_formula=data["timedecay_formula"],
        )

        obj.user_to_index = data["user_to_index"]
        obj.item_to_index = data["item_to_index"]
        obj.index_to_user = data["index_to_user"]
        obj.index_to_item = data["index_to_item"]
        obj.similarity_matrix = data["similarity_matrix"]
        obj.user_item_matrix = data["user_item_matrix"]

        obj.is_fitted = True
        return obj