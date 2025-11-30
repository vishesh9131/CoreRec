# corerec/engines/unionizedFilterEngine/bpr_recommender.py

from corerec.engines.unionizedFilterEngine.mf_base.matrix_factorization_base import (
    MatrixFactorizationBase,
)
from scipy.sparse import csr_matrix
from typing import List
import numpy as np


class BPRRecommender(MatrixFactorizationBase):
    def __init__(
        self,
        num_factors: int = 20,
        learning_rate: float = 0.01,
        regularization: float = 0.02,
        epochs: int = 20,
    ):
        super().__init__(num_factors, learning_rate, regularization, epochs)

    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]):
        interaction_matrix = csr_matrix(interaction_matrix)
        num_users, num_items = interaction_matrix.shape
        self.initialize_factors(num_users, num_items)

        for epoch in range(self.epochs):
            for u in range(num_users):
                user_interactions = interaction_matrix[u].indices
                for i in user_interactions:
                    # Sample a negative item j not interacted by user u
                    j = self.sample_negative_item(u, interaction_matrix)
                    # Calculate prediction differences
                    x_ui = np.dot(self.user_factors[u], self.item_factors[i])
                    x_uj = np.dot(self.user_factors[u], self.item_factors[j])
                    x_uij = x_ui - x_uj
                    # BPR Optimization
                    gradient = 1 / (1 + np.exp(x_uij))
                    self.user_factors[u] += self.learning_rate * (
                        gradient * (self.item_factors[i] - self.item_factors[j])
                        - self.regularization * self.user_factors[u]
                    )
                    self.item_factors[i] += self.learning_rate * (
                        gradient * self.user_factors[u] - self.regularization * self.item_factors[i]
                    )
                    self.item_factors[j] -= self.learning_rate * (
                        gradient * self.user_factors[u] + self.regularization * self.item_factors[j]
                    )

            loss = self.compute_loss(interaction_matrix)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")

    def sample_negative_item(self, user_id: int, interaction_matrix: csr_matrix) -> int:
        num_items = interaction_matrix.shape[1]
        while True:
            j = np.random.randint(0, num_items)
            if j not in interaction_matrix[user_id].indices:
                return j

    def recommend(self, user_id: int, top_n: int = 10) -> List[int]:
        user_vector = self.user_factors[user_id]
        scores = self.item_factors.dot(user_vector)
        top_items = scores.argsort()[-top_n:][::-1]
        # Optionally, exclude already interacted items
        return top_items.tolist()
