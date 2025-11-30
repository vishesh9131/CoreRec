# position_aware_recommender.py

from corerec.engines.unionizedFilterEngine.mf_base.matrix_factorization_base import (
    MatrixFactorizationBase,
)
from scipy.sparse import csr_matrix
from typing import List
import numpy as np
import logging
from typing import List, Optional


class PositionAwareMFRecommender(MatrixFactorizationBase):
    def __init__(
        self,
        num_factors=20,
        learning_rate=0.01,
        reg_user=0.02,
        reg_item=0.02,
        epochs=20,
        early_stopping_rounds=5,
        n_threads=4,
    ):
        super().__init__(
            num_factors, learning_rate, reg_user, reg_item, epochs, early_stopping_rounds, n_threads
        )

    def fit(
        self,
        interaction_matrix: csr_matrix,
        user_ids: List[int],
        item_ids: List[int],
        positions: Optional[List[int]] = None,
    ):
        """
        Train the recommender with positional information.

        Args:
            interaction_matrix (csr_matrix): User-item interaction matrix.
            user_ids (List[int]): List of user IDs.
            item_ids (List[int]): List of item IDs.
            positions (Optional[List[int]]): List of positions for each interaction.
        """
        # Incorporate positions as additional features or modify the loss function accordingly
        # This is a placeholder for actual implementation
        super().fit(interaction_matrix, user_ids, item_ids)
