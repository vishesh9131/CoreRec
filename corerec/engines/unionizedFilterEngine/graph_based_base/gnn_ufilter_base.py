# corerec/engines/unionizedFilterEngine/graph_based_base/gnn_cf_base.py

from abc import ABC
from typing import List
from scipy.sparse import csr_matrix
from ..base_recommender import BaseRecommender

class GraphBasedCFBase(BaseRecommender, ABC):
    def __init__(self, graph_layers: int = 3, learning_rate: float = 0.01):
        self.graph_layers = graph_layers
        self.learning_rate = learning_rate
        # Initialize graph-specific parameters

    def build_graph(self, interaction_matrix: csr_matrix):
        # Implement graph construction from interaction matrix
        pass