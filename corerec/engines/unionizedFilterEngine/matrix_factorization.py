import numpy as np
from scipy.sparse import csr_matrix
from typing import List
from corerec.base_recommender import BaseRecommender

class MatrixFactorizationRecommender(BaseRecommender):
    def __init__(self, num_factors: int = 20, learning_rate: float = 0.01, regularization: float = 0.02, epochs: int = 20):
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.user_factors = None
        self.item_factors = None

    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]):
        interaction_matrix = csr_matrix(interaction_matrix)
        num_users, num_items = interaction_matrix.shape
        self.user_factors = np.random.normal(scale=1./self.num_factors, size=(num_users, self.num_factors))
        self.item_factors = np.random.normal(scale=1./self.num_factors, size=(num_items, self.num_factors))

        for epoch in range(self.epochs):
            for u in range(num_users):
                user_interactions = interaction_matrix[u].indices
                for i in user_interactions:
                    prediction = np.dot(self.user_factors[u], self.item_factors[i])
                    error = interaction_matrix[u, i] - prediction
                    # Update user and item factors
                    self.user_factors[u] += self.learning_rate * (error * self.item_factors[i] - self.regularization * self.user_factors[u])
                    self.item_factors[i] += self.learning_rate * (error * self.user_factors[u] - self.regularization * self.item_factors[i])
            loss = self.compute_loss(interaction_matrix)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")

    def compute_loss(self, interaction_matrix: csr_matrix) -> float:
        loss = 0
        for u in range(interaction_matrix.shape[0]):
            for i in interaction_matrix[u].indices:
                prediction = np.dot(self.user_factors[u], self.item_factors[i])
                error = interaction_matrix[u, i] - prediction
                loss += error ** 2
                loss += self.regularization * (np.linalg.norm(self.user_factors[u])**2 + np.linalg.norm(self.item_factors[i])**2)
        return loss

    def recommend(self, user_id: int, top_n: int = 10) -> List[int]:
        user_vector = self.user_factors[user_id]
        scores = np.dot(self.item_factors, user_vector)
        top_items = scores.argsort()[-top_n:][::-1]
        return top_items.tolist()