import numpy as np
from scipy.sparse import csr_matrix
from typing import List
from .base_recommender import BaseRecommender

class SVDRecommender(BaseRecommender):
    def __init__(self, num_factors: int = 20, learning_rate: float = 0.01, regularization: float = 0.02, epochs: int = 20):
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.U = None
        self.S = None
        self.Vt = None

    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]):
        interaction_matrix = csr_matrix(interaction_matrix)
        dense_matrix = interaction_matrix.toarray()
        U, S, Vt = np.linalg.svd(dense_matrix, full_matrices=False)
        self.U = U[:, :self.num_factors]
        self.S = np.diag(S[:self.num_factors])
        self.Vt = Vt[:self.num_factors, :]

    def recommend(self, user_id: int, top_n: int = 10) -> List[int]:
        user_vector = self.U[user_id].dot(self.S)
        scores = self.Vt.T.dot(user_vector)
        top_items = scores.argsort()[-top_n:][::-1]
        return top_items.tolist()