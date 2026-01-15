import numpy as np
from scipy.sparse import csr_matrix
from typing import List
from corerec.engines.cr_content_based.base_recommender import BaseRecommender


class PairwiseRankingRecommender(BaseRecommender):
    def __init__(self, learning_rate: float = 0.01, regularization: float = 0.02, epochs: int = 20):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.user_factors = None
        self.item_factors = None

    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]):
        num_users, num_items = interaction_matrix.shape
        self.user_factors = np.random.normal(
            scale=1.0 / self.num_factors, size=(num_users, self.num_factors)
        )
        self.item_factors = np.random.normal(
            scale=1.0 / self.num_factors, size=(num_items, self.num_factors)
        )

        for epoch in range(self.epochs):
            for u in range(num_users):
                user_interactions = interaction_matrix[u].indices
                for i in user_interactions:
                    for j in range(num_items):
                        if j not in user_interactions:
                            # Calculate the difference in scores
                            x_ui = np.dot(self.user_factors[u], self.item_factors[i])
                            x_uj = np.dot(self.user_factors[u], self.item_factors[j])
                            x_uij = x_ui - x_uj

                            # Update factors
                            gradient = 1 / (1 + np.exp(x_uij))
                            self.user_factors[u] += self.learning_rate * (
                                gradient * (self.item_factors[i] - self.item_factors[j])
                                - self.regularization * self.user_factors[u]
                            )
                            self.item_factors[i] += self.learning_rate * (
                                gradient * self.user_factors[u]
                                - self.regularization * self.item_factors[i]
                            )
                            self.item_factors[j] -= self.learning_rate * (
                                gradient * self.user_factors[u]
                                + self.regularization * self.item_factors[j]
                            )

    def recommend(self, user_id: int, top_n: int = 10) -> List[int]:
        user_vector = self.user_factors[user_id]
        scores = np.dot(self.item_factors, user_vector)
        top_items = scores.argsort()[-top_n:][::-1]
        return top_items.tolist()
