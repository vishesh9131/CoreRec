import numpy as np
from scipy.sparse import csr_matrix
from typing import List
from corerec.base_recommender import BaseRecommender

class UserBasedUF(BaseRecommender):
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold
        self.user_similarity = None
        self.user_ids = []
        self.item_ids = []

    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]):
        interaction_matrix = csr_matrix(interaction_matrix)
        self.user_ids = user_ids
        self.item_ids = item_ids
        num_users = interaction_matrix.shape[0]
        self.user_similarity = np.zeros((num_users, num_users))

        # Compute cosine similarity between users
        for u in range(num_users):
            for v in range(u, num_users):
                if u == v:
                    self.user_similarity[u, v] = 1.0
                else:
                    vec_u = interaction_matrix[u].toarray().flatten()
                    vec_v = interaction_matrix[v].toarray().flatten()
                    norm_u = np.linalg.norm(vec_u)
                    norm_v = np.linalg.norm(vec_v)
                    if norm_u == 0 or norm_v == 0:
                        similarity = 0.0
                    else:
                        similarity = np.dot(vec_u, vec_v) / (norm_u * norm_v)
                    self.user_similarity[u, v] = similarity
                    self.user_similarity[v, u] = similarity

    def recommend(self, user_id: int, top_n: int = 10) -> List[int]:
        user_index = self.user_ids.index(user_id)
        similarities = self.user_similarity[user_index]
        similar_users = np.argsort(similarities)[::-1][1:]  # Exclude self

        scores = {}
        for similar_user in similar_users:
            if similarities[similar_user] < self.similarity_threshold:
                break
            for item in self.item_ids:
                item_index = self.item_ids.index(item)
                if interaction_matrix[user_index, item_index] == 0:
                    scores[item] = scores.get(item, 0) + similarities[similar_user]

        top_items = sorted(scores, key=scores.get, reverse=True)[:top_n]
        return top_items