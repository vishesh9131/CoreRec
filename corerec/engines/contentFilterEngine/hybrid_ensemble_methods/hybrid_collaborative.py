"""
Hybrid Collaborative Filtering Module

This module implements hybrid collaborative filtering techniques that combine multiple
recommendation strategies to improve accuracy and robustness. Hybrid methods leverage
the strengths of different algorithms, such as collaborative filtering, content-based
filtering, and others, to provide more personalized recommendations.

Key Features:
- Combines collaborative and content-based filtering methods.
- Utilizes ensemble techniques to enhance recommendation performance.
- Supports various hybridization strategies, including weighted, switching, and mixed
  hybrid approaches.

Classes:
- HYBRID_COLLABORATIVE: Main class implementing hybrid collaborative filtering logic.

Usage:
To use this module, instantiate the HYBRID_COLLABORATIVE class and call its methods
to train and generate recommendations based on your dataset.

Example:
    hybrid_cf = HYBRID_COLLABORATIVE()
    hybrid_cf.train(user_item_matrix, content_features)
    recommendations = hybrid_cf.recommend(user_id, top_n=10)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import warnings


class HYBRID_COLLABORATIVE:
    """
    Hybrid collaborative-content filtering recommender system.
    
    This class combines collaborative filtering (user-user or item-item) with 
    content-based filtering to leverage both interaction patterns and item features.
    """
    
    def __init__(
        self,
        hybrid_strategy: str = 'weighted',
        cf_weight: float = 0.6,
        content_weight: float = 0.4,
        cf_method: str = 'item_based',
        similarity_metric: str = 'cosine',
        k_neighbors: int = 20,
        min_similarity: float = 0.0,
        normalize: bool = True,
        random_state: int = 42
    ):
        """
        Initialize hybrid collaborative filtering model.
        
        Parameters:
        -----------
        hybrid_strategy : str
            Strategy for combining CF and content: 'weighted', 'switching', 
            'mixed', 'cascade', 'feature_augmented'
        cf_weight : float
            Weight for collaborative filtering component (0-1)
        content_weight : float
            Weight for content-based component (0-1)
        cf_method : str
            Collaborative filtering method: 'item_based' or 'user_based'
        similarity_metric : str
            Similarity metric: 'cosine', 'pearson', 'jaccard'
        k_neighbors : int
            Number of neighbors to consider
        min_similarity : float
            Minimum similarity threshold
        normalize : bool
            Whether to normalize predictions
        random_state : int
            Random seed for reproducibility
        """
        self.hybrid_strategy = hybrid_strategy
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.cf_method = cf_method
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        self.min_similarity = min_similarity
        self.normalize = normalize
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        # model state
        self.user_item_matrix = None
        self.item_features = None
        self.user_features = None
        self.item_similarity = None
        self.user_similarity = None
        self.content_similarity = None
        self.is_trained = False
        
        # statistics
        self.n_users = 0
        self.n_items = 0
        self.user_means = None
        self.item_means = None
        self.global_mean = 0.0
    
    def _compute_similarity(
        self, 
        matrix: np.ndarray, 
        metric: str = 'cosine'
    ) -> np.ndarray:
        """Compute similarity matrix between rows of input matrix"""
        if metric == 'cosine':
            # handle zero vectors
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1  # avoid division by zero
            normalized = matrix / norms
            similarity = np.dot(normalized, normalized.T)
        
        elif metric == 'pearson':
            # pearson correlation
            centered = matrix - np.mean(matrix, axis=1, keepdims=True)
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = centered / norms
            similarity = np.dot(normalized, normalized.T)
        
        elif metric == 'jaccard':
            # jaccard similarity for binary data
            binary_matrix = (matrix > 0).astype(float)
            intersection = np.dot(binary_matrix, binary_matrix.T)
            union = (np.sum(binary_matrix, axis=1, keepdims=True) + 
                    np.sum(binary_matrix, axis=1, keepdims=True).T - 
                    intersection)
            union[union == 0] = 1
            similarity = intersection / union
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
        
        return similarity
    
    def _collaborative_filtering_predict(
        self, 
        user_id: int, 
        item_id: int
    ) -> float:
        """Predict rating using collaborative filtering"""
        if self.cf_method == 'item_based':
            # item-based CF
            user_ratings = self.user_item_matrix[user_id, :]
            rated_items = np.where(user_ratings > 0)[0]
            
            if len(rated_items) == 0:
                return self.global_mean
            
            # get similarities to target item
            similarities = self.item_similarity[item_id, rated_items]
            ratings = user_ratings[rated_items]
            
            # filter by minimum similarity and get top-k
            valid_mask = similarities >= self.min_similarity
            similarities = similarities[valid_mask]
            ratings = ratings[valid_mask]
            
            if len(similarities) == 0:
                return self.item_means[item_id] if self.item_means is not None else self.global_mean
            
            # take top-k most similar
            if len(similarities) > self.k_neighbors:
                top_k_idx = np.argsort(similarities)[-self.k_neighbors:]
                similarities = similarities[top_k_idx]
                ratings = ratings[top_k_idx]
            
            # weighted average
            if np.sum(np.abs(similarities)) > 0:
                prediction = np.dot(similarities, ratings) / np.sum(np.abs(similarities))
            else:
                prediction = self.global_mean
        
        else:  # user-based CF
            # user-based CF
            item_ratings = self.user_item_matrix[:, item_id]
            rated_users = np.where(item_ratings > 0)[0]
            
            if len(rated_users) == 0:
                return self.global_mean
            
            # get similarities to target user
            similarities = self.user_similarity[user_id, rated_users]
            ratings = item_ratings[rated_users]
            
            # filter by minimum similarity
            valid_mask = similarities >= self.min_similarity
            similarities = similarities[valid_mask]
            ratings = ratings[valid_mask]
            
            if len(similarities) == 0:
                return self.user_means[user_id] if self.user_means is not None else self.global_mean
            
            # take top-k
            if len(similarities) > self.k_neighbors:
                top_k_idx = np.argsort(similarities)[-self.k_neighbors:]
                similarities = similarities[top_k_idx]
                ratings = ratings[top_k_idx]
            
            # weighted average with user mean centering
            if np.sum(np.abs(similarities)) > 0:
                # center ratings
                user_means_subset = self.user_means[rated_users][valid_mask]
                if len(similarities) > self.k_neighbors:
                    user_means_subset = user_means_subset[top_k_idx]
                
                centered_ratings = ratings - user_means_subset
                prediction = self.user_means[user_id] + (
                    np.dot(similarities, centered_ratings) / np.sum(np.abs(similarities))
                )
            else:
                prediction = self.global_mean
        
        return prediction
    
    def _content_based_predict(
        self, 
        user_id: int, 
        item_id: int
    ) -> float:
        """Predict rating using content-based filtering"""
        if self.item_features is None:
            return self.global_mean
        
        # get user's rated items
        user_ratings = self.user_item_matrix[user_id, :]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return self.global_mean
        
        # compute similarity between target item and rated items
        target_features = self.item_features[item_id]
        rated_features = self.item_features[rated_items]
        
        # cosine similarity
        similarities = cosine_similarity(
            target_features.reshape(1, -1), 
            rated_features
        ).flatten()
        
        ratings = user_ratings[rated_items]
        
        # weighted average by content similarity
        if np.sum(np.abs(similarities)) > 0:
            # take top-k most similar
            if len(similarities) > self.k_neighbors:
                top_k_idx = np.argsort(similarities)[-self.k_neighbors:]
                similarities = similarities[top_k_idx]
                ratings = ratings[top_k_idx]
            
            prediction = np.dot(similarities, ratings) / np.sum(np.abs(similarities))
        else:
            prediction = self.global_mean
        
        return prediction
    
    def _weighted_hybrid_predict(
        self, 
        user_id: int, 
        item_id: int
    ) -> float:
        """Weighted combination of CF and content-based predictions"""
        cf_pred = self._collaborative_filtering_predict(user_id, item_id)
        content_pred = self._content_based_predict(user_id, item_id)
        
        prediction = (self.cf_weight * cf_pred + 
                     self.content_weight * content_pred)
        
        return prediction
    
    def _switching_hybrid_predict(
        self, 
        user_id: int, 
        item_id: int
    ) -> float:
        """Switch between CF and content-based based on confidence"""
        # use CF if user has enough ratings, otherwise use content
        user_ratings = self.user_item_matrix[user_id, :]
        n_ratings = np.sum(user_ratings > 0)
        
        # threshold for switching (could be adaptive)
        threshold = 5
        
        if n_ratings >= threshold:
            # use collaborative filtering
            return self._collaborative_filtering_predict(user_id, item_id)
        else:
            # use content-based (cold start scenario)
            return self._content_based_predict(user_id, item_id)
    
    def _mixed_hybrid_predict(
        self, 
        user_id: int, 
        item_id: int
    ) -> float:
        """Mixed hybrid: combine different CF approaches with content"""
        # combine item-based and user-based CF
        old_method = self.cf_method
        
        self.cf_method = 'item_based'
        item_cf_pred = self._collaborative_filtering_predict(user_id, item_id)
        
        self.cf_method = 'user_based'
        user_cf_pred = self._collaborative_filtering_predict(user_id, item_id)
        
        self.cf_method = old_method
        
        content_pred = self._content_based_predict(user_id, item_id)
        
        # weighted combination of all three
        prediction = (0.4 * item_cf_pred + 
                     0.3 * user_cf_pred + 
                     0.3 * content_pred)
        
        return prediction
    
    def train(
        self,
        user_item_matrix: np.ndarray,
        content_features: Optional[np.ndarray] = None,
        user_features: Optional[np.ndarray] = None,
        compute_similarities: bool = True,
        verbose: bool = True
    ):
        """
        Train the hybrid model.
        
        Parameters:
        -----------
        user_item_matrix : np.ndarray
            User-item interaction matrix (n_users x n_items)
        content_features : np.ndarray, optional
            Item content feature matrix (n_items x n_features)
        user_features : np.ndarray, optional
            User feature matrix (n_users x n_features)
        compute_similarities : bool
            Whether to precompute similarity matrices
        verbose : bool
            Whether to print training progress
        """
        if verbose:
            print("Training hybrid collaborative-content model...")
        
        self.user_item_matrix = user_item_matrix
        self.item_features = content_features
        self.user_features = user_features
        
        self.n_users, self.n_items = user_item_matrix.shape
        
        # compute statistics
        self.global_mean = np.mean(user_item_matrix[user_item_matrix > 0])
        
        # user means (for user-based CF)
        user_sums = np.sum(user_item_matrix, axis=1)
        user_counts = np.sum(user_item_matrix > 0, axis=1)
        user_counts[user_counts == 0] = 1  # avoid division by zero
        self.user_means = user_sums / user_counts
        
        # item means (for item-based CF)
        item_sums = np.sum(user_item_matrix, axis=0)
        item_counts = np.sum(user_item_matrix > 0, axis=0)
        item_counts[item_counts == 0] = 1
        self.item_means = item_sums / item_counts
        
        if compute_similarities:
            if verbose:
                print("Computing similarity matrices...")
            
            # compute item similarity for item-based CF
            if self.cf_method in ['item_based', 'mixed']:
                if verbose:
                    print("  - Item-item similarity...")
                self.item_similarity = self._compute_similarity(
                    user_item_matrix.T, 
                    self.similarity_metric
                )
            
            # compute user similarity for user-based CF
            if self.cf_method in ['user_based', 'mixed']:
                if verbose:
                    print("  - User-user similarity...")
                self.user_similarity = self._compute_similarity(
                    user_item_matrix, 
                    self.similarity_metric
                )
            
            # compute content similarity if features provided
            if content_features is not None:
                if verbose:
                    print("  - Content similarity...")
                self.content_similarity = self._compute_similarity(
                    content_features, 
                    'cosine'
                )
        
        self.is_trained = True
        
        if verbose:
            print(f"Training complete! Model ready for {self.n_users} users and {self.n_items} items.")
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair.
        
        Parameters:
        -----------
        user_id : int
            User ID
        item_id : int
            Item ID
        
        Returns:
        --------
        prediction : float
            Predicted rating/score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # choose prediction method based on hybrid strategy
        if self.hybrid_strategy == 'weighted':
            prediction = self._weighted_hybrid_predict(user_id, item_id)
        elif self.hybrid_strategy == 'switching':
            prediction = self._switching_hybrid_predict(user_id, item_id)
        elif self.hybrid_strategy == 'mixed':
            prediction = self._mixed_hybrid_predict(user_id, item_id)
        elif self.hybrid_strategy == 'cf_only':
            prediction = self._collaborative_filtering_predict(user_id, item_id)
        elif self.hybrid_strategy == 'content_only':
            prediction = self._content_based_predict(user_id, item_id)
        else:
            # default to weighted
            prediction = self._weighted_hybrid_predict(user_id, item_id)
        
        # normalize if needed
        if self.normalize:
            # clip to reasonable range
            prediction = np.clip(prediction, 0, 5)
        
        return prediction
    
    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_known: bool = True,
        return_scores: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate top-N recommendations for a user.
        
        Parameters:
        -----------
        user_id : int
            User ID for recommendations
        top_n : int
            Number of recommendations to generate
        exclude_known : bool
            Whether to exclude items user has already interacted with
        return_scores : bool
            Whether to return predicted scores
        
        Returns:
        --------
        recommendations : np.ndarray or tuple
            Recommended item IDs (and scores if return_scores=True)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating recommendations")
        
        # predict scores for all items
        scores = np.zeros(self.n_items)
        for item_id in range(self.n_items):
            scores[item_id] = self.predict(user_id, item_id)
        
        # exclude known items
        if exclude_known:
            known_items = np.where(self.user_item_matrix[user_id, :] > 0)[0]
            scores[known_items] = -np.inf
        
        # get top-N
        top_indices = np.argsort(scores)[::-1][:top_n]
        top_scores = scores[top_indices]
        
        if return_scores:
            return top_indices, top_scores
        return top_indices
    
    def evaluate_components(
        self, 
        user_id: int, 
        item_id: int
    ) -> Dict[str, float]:
        """
        Evaluate individual component predictions for analysis.
        
        Parameters:
        -----------
        user_id : int
            User ID
        item_id : int
            Item ID
        
        Returns:
        --------
        component_scores : dict
            Dictionary with predictions from each component
        """
        return {
            'cf_prediction': self._collaborative_filtering_predict(user_id, item_id),
            'content_prediction': self._content_based_predict(user_id, item_id),
            'hybrid_prediction': self.predict(user_id, item_id),
            'global_mean': self.global_mean
        }
    
    def get_similar_items(
        self, 
        item_id: int, 
        top_n: int = 10,
        use_content: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get similar items based on CF or content similarity.
        
        Parameters:
        -----------
        item_id : int
            Target item ID
        top_n : int
            Number of similar items to return
        use_content : bool
            Whether to use content similarity (vs CF similarity)
        
        Returns:
        --------
        item_ids : np.ndarray
            Array of similar item IDs
        similarities : np.ndarray
            Array of similarity scores
        """
        if use_content and self.content_similarity is not None:
            similarities = self.content_similarity[item_id, :]
        elif self.item_similarity is not None:
            similarities = self.item_similarity[item_id, :]
        else:
            raise ValueError("No similarity matrix available")
        
        # exclude self
        similarities[item_id] = -np.inf
        
        # get top-N
        top_indices = np.argsort(similarities)[::-1][:top_n]
        top_similarities = similarities[top_indices]
        
        return top_indices, top_similarities
    
    def save_model(self, filepath: str):
        """Save model to file"""
        np.savez(
            filepath,
            user_item_matrix=self.user_item_matrix,
            item_features=self.item_features,
            user_features=self.user_features,
            item_similarity=self.item_similarity,
            user_similarity=self.user_similarity,
            content_similarity=self.content_similarity,
            user_means=self.user_means,
            item_means=self.item_means,
            global_mean=self.global_mean,
            cf_weight=self.cf_weight,
            content_weight=self.content_weight
        )
    
    def load_model(self, filepath: str):
        """Load model from file"""
        data = np.load(filepath, allow_pickle=True)
        self.user_item_matrix = data['user_item_matrix']
        self.item_features = data.get('item_features', None)
        self.user_features = data.get('user_features', None)
        self.item_similarity = data.get('item_similarity', None)
        self.user_similarity = data.get('user_similarity', None)
        self.content_similarity = data.get('content_similarity', None)
        self.user_means = data['user_means']
        self.item_means = data['item_means']
        self.global_mean = float(data['global_mean'])
        self.cf_weight = float(data['cf_weight'])
        self.content_weight = float(data['content_weight'])
        self.n_users, self.n_items = self.user_item_matrix.shape
        self.is_trained = True
