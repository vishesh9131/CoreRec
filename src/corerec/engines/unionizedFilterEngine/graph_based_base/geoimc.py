import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Optional, Dict, Any, Tuple
from ..base_recommender import BaseRecommender

class GeoIMC(BaseRecommender):
    """
    Geometric Matrix Completion (GeoIMC) for recommendation.
    
    This model uses geometric information to improve matrix completion
    by incorporating side information into the recommendation process.
    
    Parameters:
    -----------
    factors : int
        Number of latent factors
    learning_rate : float
        Learning rate for optimizer
    regularization : float
        Regularization strength
    iterations : int
        Number of training iterations
    batch_size : int
        Size of mini-batches
    seed : Optional[int]
        Random seed for reproducibility
    """
    def __init__(
        self,
        factors: int = 100,
        learning_rate: float = 0.01,
        regularization: float = 0.1,
        iterations: int = 100,
        batch_size: int = 256,
        seed: Optional[int] = None
    ):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.batch_size = batch_size
        self.seed = seed
        
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.user_factors = None
        self.item_factors = None
    
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """Create mappings between original IDs and matrix indices"""
        self.user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_map.items()}
    
    def _init_params(self, n_users: int, n_items: int) -> None:
        """Initialize model parameters"""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Initialize factors with small random values
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.factors))
    
    def _predict(self, user_idx: int, item_idx: int) -> float:
        """Make prediction for a user-item pair"""
        # Compute prediction using dot product
        return np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
    
    def fit(self, user_ids: List[int], item_ids: List[int], ratings: List[float]) -> None:
        """
        Train the GeoIMC model using the provided data.
        
        Parameters:
        -----------
        user_ids : List[int]
            List of user IDs
        item_ids : List[int]
            List of item IDs
        ratings : List[float]
            List of ratings
        """
        # Create mappings
        unique_user_ids = list(set(user_ids))
        unique_item_ids = list(set(item_ids))
        self._create_mappings(unique_user_ids, unique_item_ids)
        
        # Map IDs to indices
        user_indices = [self.user_map[user_id] for user_id in user_ids]
        item_indices = [self.item_map[item_id] for item_id in item_ids]
        
        # Create user-item matrix
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.user_item_matrix = csr_matrix((ratings, (user_indices, item_indices)), shape=(n_users, n_items))
        
        # Initialize parameters
        self._init_params(n_users, n_items)
        
        # Train model
        for _ in range(self.iterations):
            for start in range(0, len(ratings), self.batch_size):
                end = min(start + self.batch_size, len(ratings))
                batch_user_indices = user_indices[start:end]
                batch_item_indices = item_indices[start:end]
                batch_ratings = ratings[start:end]
                
                # Compute predictions and errors
                predictions = [self._predict(u, i) for u, i in zip(batch_user_indices, batch_item_indices)]
                errors = batch_ratings - predictions
                
                # Update factors
                for u, i, error in zip(batch_user_indices, batch_item_indices, errors):
                    user_factor_update = self.learning_rate * (error * self.item_factors[i] - self.regularization * self.user_factors[u])
                    item_factor_update = self.learning_rate * (error * self.user_factors[u] - self.regularization * self.item_factors[i])
                    
                    self.user_factors[u] += user_factor_update
                    self.item_factors[i] += item_factor_update
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """
        Generate top-N recommendations for a specific user.
        
        Parameters:
        -----------
        user_id : int
            ID of the user to generate recommendations for
        top_n : int
            Number of recommendations to generate
        exclude_seen : bool
            Whether to exclude items the user has already interacted with
            
        Returns:
        --------
        List[int] : List of recommended item IDs
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Map user_id to internal index
        if user_id not in self.user_map:
            raise ValueError(f"User ID {user_id} not found in training data")
            
        user_idx = self.user_map[user_id]
        
        # Calculate scores for all items
        scores = np.dot(self.item_factors, self.user_factors[user_idx])
        
        # If requested, exclude items the user has already interacted with
        if exclude_seen:
            seen_items = self.user_item_matrix[user_idx].indices
            scores[seen_items] = float('-inf')
        
        # Get top-n item indices
        top_item_indices = np.argsort(-scores)[:top_n]
        
        # Map indices back to original item IDs
        top_items = [self.reverse_item_map[idx] for idx in top_item_indices]
        
        return top_items
    
    def save_model(self, filepath: str) -> None:
        """Save the model to a file"""
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been trained. Call fit() first.")
            
        # Save model data
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'params': {
                'factors': self.factors,
                'learning_rate': self.learning_rate,
                'regularization': self.regularization,
                'iterations': self.iterations,
                'batch_size': self.batch_size,
                'seed': self.seed
            }
        }
        np.save(filepath, model_data, allow_pickle=True)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'GeoIMC':
        """Load a model from a file"""
        model_data = np.load(filepath, allow_pickle=True).item()
        
        # Create an instance with the saved parameters
        instance = cls(
            factors=model_data['params']['factors'],
            learning_rate=model_data['params']['learning_rate'],
            regularization=model_data['params']['regularization'],
            iterations=model_data['params']['iterations'],
            batch_size=model_data['params']['batch_size'],
            seed=model_data['params']['seed']
        )
        
        # Restore instance variables
        instance.user_factors = model_data['user_factors']
        instance.item_factors = model_data['item_factors']
        instance.user_map = model_data['user_map']
        instance.item_map = model_data['item_map']
        instance.reverse_user_map = model_data['reverse_user_map']
        instance.reverse_item_map = model_data['reverse_item_map']
        
        return instance 