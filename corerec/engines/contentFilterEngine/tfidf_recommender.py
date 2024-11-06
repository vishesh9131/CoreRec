import numpy as np
from .base_recommender import BaseRecommender

class TFIDFRecommender(BaseRecommender):
    def __init__(self, feature_matrix):
        if hasattr(feature_matrix, "toarray"):
            self.feature_matrix = feature_matrix.toarray()
        else:
            self.feature_matrix = np.array(feature_matrix)
        
        self.similarity_matrix = self.compute_similarity_matrix()

    def compute_similarity_matrix(self):
        norm = np.linalg.norm(self.feature_matrix, axis=1, keepdims=True)
        norm[norm == 0] = 1
        normalized_features = self.feature_matrix / norm
        similarity = np.dot(normalized_features, normalized_features.T)
        return similarity

    def fit(self, data):
        pass  # No additional training needed for TF-IDF-based recommender

    def recommend(self, item_indices, top_n=10):
        combined_scores = np.zeros(self.similarity_matrix.shape[1])
        for idx in item_indices:
            combined_scores += self.similarity_matrix[idx]
        top_items = combined_scores.argsort()[-top_n:][::-1]
        return top_items