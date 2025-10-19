import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Optional, Dict, Any, Tuple
from .base_recommender import BaseRecommender

class FASTRecommender(BaseRecommender):
    """
    FastAI Embedding Dot Bias (FAST) recommender.
    
    This is an implementation of the collaborative filtering approach used in the FastAI library,
    which combines embedding dot products with bias terms for efficient recommendation.
    
    Parameters:
    -----------
    factors : int
        Number of latent factors
    weight_decay : float
        Weight decay (L2 regularization)
    learning_rate : float
        Learning rate for optimizer
    iterations : int
        Number of training iterations
    batch_size : int
        Size of mini-batches
    seed : Optional[int]
        Random seed for reproducibility
    """
    def __init__(
        self,
        factors: int = 50,
        weight_decay: float = 0.01,
        learning_rate: float = 0.01,
        iterations: int = 100,
        batch_size: int = 1024,
        seed: Optional[int] = None
    ):
        self.factors = factors
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size
        self.seed = seed
        
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
    
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
        
        # Initialize biases
        self.global_bias = np.mean(self.user_item_matrix.data)
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
    
    def _predict(self, user_idx: int, item_idx: int) -> float:
        """Make prediction for a user-item pair"""
        # Compute prediction using dot product and biases
        pred = self.global_bias
        pred += self.user_bias[user_idx]
        pred += self.item_bias[item_idx]
        pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        return pred
    
    def _sgd_update(self, user_idx: int, item_idx: int, rating: float, pred: float) -> None:
        """Update parameters using SGD for a single sample"""
        # Calculate error
        error = rating - pred
        
        # Update biases
        self.user_bias[user_idx] += self.learning_rate * (error - self.weight_decay * self.user_bias[user_idx])
        self.item_bias[item_idx] += self.learning_rate * (error - self.weight_decay * self.item_bias[item_idx])
        
        # Update factors
        user_factors = self.user_factors[user_idx]
        item_factors = self.item_factors[item_idx]
        
        # Compute gradients
        user_grad = error * item_factors - self.weight_decay * user_factors
        item_grad = error * user_factors - self.weight_decay * item_factors
        
        # Update factors
        self.user_factors[user_idx] += self.learning_rate * user_grad
        self.item_factors[item_idx] += self.learning_rate * item_grad
    
    def fit(self, user_ids: List[int], item_ids: List[int], ratings: List[float]) -> None:
        """
        Train the model on the given data.
        
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
        unique_user_ids = sorted(set(user_ids))
        unique_item_ids = sorted(set(item_ids))
        self._create_mappings(unique_user_ids, unique_item_ids)
        
        # Create user-item matrix
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        
        # Map user and item IDs to indices
        user_indices = [self.user_map[user_id] for user_id in user_ids]
        item_indices = [self.item_map[item_id] for item_id in item_ids]
        
        # Create sparse matrix
        self.user_item_matrix = csr_matrix((ratings, (user_indices, item_indices)), shape=(n_users, n_items))
        
        # Initialize parameters
        self._init_params(n_users, n_items)
        
        # Convert to COO format for efficient iteration
        coo_matrix = self.user_item_matrix.tocoo()
        n_samples = len(coo_matrix.data)
        indices = np.arange(n_samples)
        
        # Training loop
        for iteration in range(self.iterations):
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Process in batches
            total_loss = 0.0
            for batch_start in range(0, n_samples, self.batch_size):
                batch_indices = indices[batch_start:batch_start + self.batch_size]
                batch_loss = 0.0
                
                for idx in batch_indices:
                    user_idx = coo_matrix.row[idx]
                    item_idx = coo_matrix.col[idx]
                    rating = coo_matrix.data[idx]
                    
                    # Make prediction
                    pred = self._predict(user_idx, item_idx)
                    
                    # Calculate loss
                    error = rating - pred
                    batch_loss += error ** 2
                    
                    # Update parameters
                    self._sgd_update(user_idx, item_idx, rating, pred)
                
                total_loss += batch_loss
            
            # Print progress
            avg_loss = total_loss / n_samples
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"Iteration {iteration+1}/{self.iterations}, Loss: {avg_loss:.4f}")
    
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
        user_vector = self.user_factors[user_idx]
        scores = self.global_bias + self.user_bias[user_idx]
        scores += self.item_bias
        scores += np.dot(self.item_factors, user_vector)
        
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
            'user_bias': self.user_bias,
            'item_bias': self.item_bias,
            'global_bias': self.global_bias,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'params': {
                'factors': self.factors,
                'weight_decay': self.weight_decay,
                'learning_rate': self.learning_rate,
                'iterations': self.iterations,
                'batch_size': self.batch_size,
                'seed': self.seed
            }
        }
        np.save(filepath, model_data, allow_pickle=True)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'FASTRecommender':
        """Load a model from a file"""
        model_data = np.load(filepath, allow_pickle=True).item()
        
        # Create an instance with the saved parameters
        instance = cls(
            factors=model_data['params']['factors'],
            weight_decay=model_data['params']['weight_decay'],
            learning_rate=model_data['params']['learning_rate'],
            iterations=model_data['params']['iterations'],
            batch_size=model_data['params']['batch_size'],
            seed=model_data['params']['seed']
        )
        
        # Restore instance variables
        instance.user_factors = model_data['user_factors']
        instance.item_factors = model_data['item_factors']
        instance.user_bias = model_data['user_bias']
        instance.item_bias = model_data['item_bias']
        instance.global_bias = model_data['global_bias']
        instance.user_map = model_data['user_map']
        instance.item_map = model_data['item_map']
        instance.reverse_user_map = model_data['reverse_user_map']
        instance.reverse_item_map = model_data['reverse_item_map']
        
        return instance 