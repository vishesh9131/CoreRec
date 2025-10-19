# cold_start implementation
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

class COLD_START:
    def __init__(self, method='content_based', n_neighbors=5):
        """
        Initialize cold start handling.
        
        Args:
            method (str): Cold start strategy ('content_based', 'popularity', 'hybrid')
            n_neighbors (int): Number of neighbors for similarity-based methods
        """
        self.method = method
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.item_features = None
        self.popularity_scores = None
        self.knn = None

    def fit(self, item_features, interaction_matrix=None):
        """
        Fit the cold start handler.
        
        Args:
            item_features: Content features of items
            interaction_matrix: User-item interaction matrix (optional)
        """
        self.item_features = self.scaler.fit_transform(item_features)
        
        if interaction_matrix is not None:
            self.popularity_scores = np.sum(interaction_matrix, axis=0)
        
        if self.method in ['content_based', 'hybrid']:
            self.knn = NearestNeighbors(n_neighbors=self.n_neighbors)
            self.knn.fit(self.item_features)

    def recommend_for_new_user(self, user_profile=None):
        """
        Generate recommendations for a new user.
        
        Args:
            user_profile: User preferences or features (optional)
        """
        if self.method == 'content_based' and user_profile is not None:
            user_profile = self.scaler.transform([user_profile])
            _, indices = self.knn.kneighbors(user_profile)
            return indices[0]
            
        elif self.method == 'popularity':
            if self.popularity_scores is None:
                raise ValueError("Popularity scores not available")
            return np.argsort(self.popularity_scores)[-self.n_neighbors:]
            
        elif self.method == 'hybrid':
            content_based_scores = self._get_content_based_scores(user_profile)
            popularity_weights = self._normalize(self.popularity_scores)
            
            # Ensure both arrays have the same shape
            if content_based_scores.shape != popularity_weights.shape:
                # Align the shapes by selecting the top items based on content-based scores
                top_indices = np.argsort(content_based_scores)[-len(popularity_weights):]
                content_based_scores = content_based_scores[top_indices]
                popularity_weights = popularity_weights[top_indices]
            
            hybrid_scores = 0.7 * content_based_scores + 0.3 * popularity_weights
            return np.argsort(hybrid_scores)[-self.n_neighbors:]

    def _get_content_based_scores(self, user_profile):
        """Calculate content-based similarity scores."""
        if user_profile is None:
            return np.zeros(self.item_features.shape[0])
        user_profile = self.scaler.transform([user_profile])
        distances, _ = self.knn.kneighbors(user_profile)
        return self._normalize(1 / (1 + distances[0]))

    def _normalize(self, scores):
        """Normalize scores to [0, 1] range."""
        min_score = np.min(scores)
        max_score = np.max(scores)
        if min_score == max_score:
            return np.zeros_like(scores)  # Avoid division by zero
        return (scores - min_score) / (max_score - min_score)
