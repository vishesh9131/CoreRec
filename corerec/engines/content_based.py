# # engines/content_based.py
# from typing import List, Optional
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# class ContentBasedFilteringEngine:
#     """
#     Content-Based Filtering Engine using Cosine Similarity.
#     """

#     def __init__(self, item_features: np.ndarray):
#         """
#         Initializes the content-based filtering engine.
        
#         Parameters:
#         - item_features (np.ndarray): Feature matrix for items.
#         """
#         self.item_features = item_features
#         self.item_ids = []

#     def fit(self, item_ids: List[int]):
#         """
#         Fit the content-based filtering engine with item features.

#         Parameters:
#         - item_ids (List[int]): List of item IDs corresponding to the columns of item_features.
#         """
#         self.item_ids = item_ids
#         # Add a small epsilon to avoid division by zero
#         epsilon = 1e-10
#         norms = np.linalg.norm(self.item_features, axis=1, keepdims=True) + epsilon
#         self.item_features = self.item_features / norms

#     def recommend(self, favorite_item_ids: List[int], top_n: int = 10) -> List[int]:
#         """
#         Recommend items based on content similarity to favorite items.

#         Parameters:
#         - favorite_item_ids (List[int]): List of item IDs that are favorites.
#         - top_n (int): Number of recommendations to generate.

#         Returns:
#         - List[int]: List of recommended item IDs.
#         """
#         # Compute similarity scores
#         favorite_indices = [self.item_ids.index(item_id) for item_id in favorite_item_ids]
#         favorite_features = self.item_features[favorite_indices]
#         scores = np.dot(self.item_features, favorite_features.T).sum(axis=1)

#         # Get top N items
#         top_indices = scores.argsort()[-top_n:][::-1]
#         return [self.item_ids[i] for i in top_indices]