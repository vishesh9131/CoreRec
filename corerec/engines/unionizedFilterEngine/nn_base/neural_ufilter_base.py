# corerec/engines/unionizedFilterEngine/nn_base/neural_cf_base.py

from abc import ABC
from typing import List
from scipy.sparse import csr_matrix
from .base_recommender import BaseRecommender

class NeuralCollaborativeFilteringBase(BaseRecommender, ABC):
    def __init__(self, embedding_size: int = 50, learning_rate: float = 0.001, epochs: int = 10):
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Add additional initialization as needed

    def preprocess_data(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]):
        # Implement any preprocessing steps specific to neural CF
        pass