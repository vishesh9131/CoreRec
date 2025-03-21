import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from scipy.sparse import csr_matrix
import os
import pickle

from .base_recommender import BaseRecommender

class SAR(BaseRecommender):
    """
    Simple Algorithm for Recommendation (SAR)
    
    SAR is a neighborhood-based algorithm that computes item-to-item similarity
    and uses it to recommend items to users based on their interaction history.
    
    Parameters
    ----------
    similarity_type : str, optional
        Type of similarity to use. Options are 'jaccard', 'cosine', or 'lift'.
        Default is 'jaccard'.
    time_decay_coefficient : float, optional
        Coefficient for time decay. If None, no time decay is applied.
        Default is None.
    time_now : int, optional
        Current timestamp for time decay calculation. Required if time_decay_coefficient is not None.
        Default is None.
    timedecay_formula : str, optional
        Formula to use for time decay. Options are 'linear' or 'exp'.
        Default is 'exp'.
    """
    
    def __init__(
        self,
        similarity_type: str = 'jaccard',
        time_decay_coefficient: Optional[float] = None,
        time_now: Optional[int] = None,
        timedecay_formula: str = 'exp'
    ):
        self.similarity_type = similarity_type
        self.time_decay_coefficient = time_decay_coefficient
        self.time_now = time_now
        self.timedecay_formula = timedecay_formula
        
        self.user_to_index = {}
        self.item_to_index = {}
        self.index_to_user = {}
        self.index_to_item = {}
        
        self.similarity_matrix = None
        self.user_item_matrix = None
        
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Create mappings between original IDs and internal indices.
        
        Parameters
        ----------
        user_ids : List[int]
            List of user IDs.
        item_ids : List[int]
            List of item IDs.
        """
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))
        
        self.user_to_index = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_index = {item: idx for idx, item in enumerate(unique_items)}
        self.index_to_user = {idx: user for user, idx in self.user_to_index.items()}
        self.index_to_item = {idx: item for item, idx in self.item_to_index.items()}
        
    def _apply_time_decay(self, ratings: List[float], timestamps: List[int]) -> List[float]:
        """
        Apply time decay to ratings based on timestamps.
        
        Parameters
        ----------
        ratings : List[float]
            List of ratings.
        timestamps : List[int]
            List of timestamps corresponding to the ratings.
            
        Returns
        -------
        List[float]
            Time-decayed ratings.
        """
        if self.time_decay_coefficient is None or self.time_now is None:
            return ratings
        
        decayed_ratings = []
        for rating, timestamp in zip(ratings, timestamps):
            time_diff = self.time_now - timestamp
            if self.timedecay_formula == 'exp':
                decay = np.exp(-self.time_decay_coefficient * time_diff)
            else:  # linear
                decay = 1.0 / (1.0 + self.time_decay_coefficient * time_diff)
            decayed_ratings.append(rating * decay)
        
        return decayed_ratings
    
    def _compute_similarity(self, matrix: csr_matrix) -> np.ndarray:
        """
        Compute item-to-item similarity matrix.
        
        Parameters
        ----------
        matrix : csr_matrix
            User-item interaction matrix.
            
        Returns
        -------
        np.ndarray
            Item-to-item similarity matrix.
        """
        n_items = matrix.shape[1]
        
        # Convert to binary matrix for jaccard and lift
        if self.similarity_type in ['jaccard', 'lift']:
            binary_matrix = matrix.copy()
            binary_matrix.data = np.ones_like(binary_matrix.data)
        
        if self.similarity_type == 'jaccard':
            # Compute dot product for intersection
            intersection = binary_matrix.T @ binary_matrix
            
            # Compute item popularity (sum of interactions)
            item_popularity = np.array(binary_matrix.sum(axis=0)).flatten()
            
            # Compute union for each item pair
            union_matrix = np.zeros((n_items, n_items))
            for i in range(n_items):
                for j in range(n_items):
                    union_matrix[i, j] = item_popularity[i] + item_popularity[j] - intersection[i, j]
            
            # Compute Jaccard similarity
            similarity = intersection.toarray() / np.maximum(union_matrix, 1)
            
        elif self.similarity_type == 'cosine':
            # Normalize the matrix rows to unit length
            row_sums = np.sqrt(np.array(matrix.power(2).sum(axis=1)).flatten())
            row_indices, col_indices = matrix.nonzero()
            matrix.data = matrix.data / row_sums[row_indices]
            
            # Compute cosine similarity
            similarity = (matrix.T @ matrix).toarray()
            
        elif self.similarity_type == 'lift':
            # Compute item popularity (probability of item being chosen)
            n_users = matrix.shape[0]
            item_prob = np.array(binary_matrix.sum(axis=0) / n_users).flatten()
            
            # Compute co-occurrence matrix (probability of items being chosen together)
            co_occurrence = (binary_matrix.T @ binary_matrix).toarray() / n_users
            
            # Compute lift
            similarity = np.zeros((n_items, n_items))
            for i in range(n_items):
                for j in range(n_items):
                    if item_prob[i] * item_prob[j] > 0:
                        similarity[i, j] = co_occurrence[i, j] / (item_prob[i] * item_prob[j])
        else:
            raise ValueError(f"Unsupported similarity type: {self.similarity_type}")
        
        # Set diagonal to 0 to avoid self-recommendation
        np.fill_diagonal(similarity, 0)
        
        return similarity
    
    def fit(self, user_ids: List[int], item_ids: List[int], ratings: List[float], 
            timestamps: Optional[List[int]] = None) -> None:
        """
        Fit the SAR model.
        
        Parameters
        ----------
        user_ids : List[int]
            List of user IDs.
        item_ids : List[int]
            List of item IDs.
        ratings : List[float]
            List of ratings corresponding to the user-item pairs.
        timestamps : List[int], optional
            List of timestamps corresponding to the user-item pairs.
            Required if time_decay_coefficient is not None.
        """
        self._create_mappings(user_ids, item_ids)
        
        n_users = len(self.user_to_index)
        n_items = len(self.item_to_index)
        
        # Apply time decay if needed
        if self.time_decay_coefficient is not None and timestamps is not None:
            ratings = self._apply_time_decay(ratings, timestamps)
        
        # Create user-item matrix
        rows = [self.user_to_index[user] for user in user_ids]
        cols = [self.item_to_index[item] for item in item_ids]
        self.user_item_matrix = csr_matrix((ratings, (rows, cols)), shape=(n_users, n_items))
        
        # Compute similarity matrix
        self.similarity_matrix = self._compute_similarity(self.user_item_matrix)
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """
        Recommend items for a user.
        
        Parameters
        ----------
        user_id : int
            User ID.
        top_n : int, optional
            Number of recommendations to return. Default is 10.
        exclude_seen : bool, optional
            Whether to exclude items the user has already interacted with.
            Default is True.
            
        Returns
        -------
        List[int]
            List of recommended item IDs.
        """
        if user_id not in self.user_to_index:
            return []
        
        user_idx = self.user_to_index[user_id]
        user_vector = self.user_item_matrix[user_idx].toarray().flatten()
        
        # Get seen items
        seen_indices = np.where(user_vector > 0)[0] if exclude_seen else []
        
        # Compute scores
        scores = user_vector @ self.similarity_matrix
        
        # Set scores of seen items to -inf to exclude them
        scores[seen_indices] = -np.inf
        
        # Get top-n items
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        # Convert indices back to original item IDs
        recommendations = [self.index_to_item[idx] for idx in top_indices]
        
        return recommendations
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        model_data = {
            'similarity_type': self.similarity_type,
            'time_decay_coefficient': self.time_decay_coefficient,
            'time_now': self.time_now,
            'timedecay_formula': self.timedecay_formula,
            'user_to_index': self.user_to_index,
            'item_to_index': self.item_to_index,
            'index_to_user': self.index_to_user,
            'index_to_item': self.index_to_item,
            'similarity_matrix': self.similarity_matrix,
            'user_item_matrix': self.user_item_matrix
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SAR':
        """
        Load a model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model.
            
        Returns
        -------
        SAR
            Loaded model.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(
            similarity_type=model_data['similarity_type'],
            time_decay_coefficient=model_data['time_decay_coefficient'],
            time_now=model_data['time_now'],
            timedecay_formula=model_data['timedecay_formula']
        )
        
        instance.user_to_index = model_data['user_to_index']
        instance.item_to_index = model_data['item_to_index']
        instance.index_to_user = model_data['index_to_user']
        instance.index_to_item = model_data['index_to_item']
        instance.similarity_matrix = model_data['similarity_matrix']
        instance.user_item_matrix = model_data['user_item_matrix']
        
        return instance
