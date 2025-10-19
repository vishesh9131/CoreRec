import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Optional, Dict, Any, Tuple
from ..base_recommender import BaseRecommender

class VMFBase(BaseRecommender):
    """
    Variational Matrix Factorization (VMF) for collaborative filtering.
    
    Based on the paper:
    "Variational Matrix Factorization via Stochastic Gradient Descent" by Sedhain et al.
    
    This model extends traditional matrix factorization with Bayesian inference
    using variational methods to handle uncertainty in recommendations.
    
    Parameters:
    -----------
    factors : int
        Number of latent factors
    learning_rate : float
        Learning rate for SGD
    regularization : float
        Regularization strength
    iterations : int
        Number of training iterations
    batch_size : int
        Size of mini-batches
    prior_mean : float
        Mean of prior distribution
    prior_std : float
        Standard deviation of prior distribution
    seed : Optional[int]
        Random seed for reproducibility
    """
    def __init__(
        self,
        factors: int = 100,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        iterations: int = 100,
        batch_size: int = 1000,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        seed: Optional[int] = None
    ):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.batch_size = batch_size
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.seed = seed
        
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.user_factors_mean = None
        self.user_factors_std = None
        self.item_factors_mean = None
        self.item_factors_std = None
        self.global_bias = None
        self.user_bias = None
        self.item_bias = None
    
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
        
        # Initialize factor means with small random values
        self.user_factors_mean = np.random.normal(0, 0.1, (n_users, self.factors))
        self.item_factors_mean = np.random.normal(0, 0.1, (n_items, self.factors))
        
        # Initialize factor standard deviations
        self.user_factors_std = np.ones((n_users, self.factors)) * 0.1
        self.item_factors_std = np.ones((n_items, self.factors)) * 0.1
        
        # Initialize biases
        self.global_bias = np.mean(self.user_item_matrix.data)
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
    
    def _sample_factors(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Sample factors from Gaussian distribution"""
        return mean + std * np.random.normal(0, 1, mean.shape)
    
    def _predict(self, user_idx: int, item_idx: int, sample: bool = False) -> float:
        """Make prediction for a user-item pair"""
        # Compute prediction using dot product and biases
        pred = self.global_bias
        pred += self.user_bias[user_idx]
        pred += self.item_bias[item_idx]
        
        if sample:
            # Sample factors for prediction
            user_factors = self._sample_factors(self.user_factors_mean[user_idx], self.user_factors_std[user_idx])
            item_factors = self._sample_factors(self.item_factors_mean[item_idx], self.item_factors_std[item_idx])
            pred += np.dot(user_factors, item_factors)
        else:
            # Use means for prediction
            pred += np.dot(self.user_factors_mean[user_idx], self.item_factors_mean[item_idx])
        
        return pred
    
    def _kl_divergence(self, mean: np.ndarray, std: np.ndarray) -> float:
        """Compute KL divergence between variational distribution and prior"""
        return 0.5 * np.sum(
            np.square(std) + np.square(mean - self.prior_mean) - 1 - 2 * np.log(std)
        ) / mean.size
    
    def _sgd_update(self, user_idx: int, item_idx: int, rating: float, pred: float) -> None:
        """Update parameters using SGD for a single sample"""
        # Calculate error
        error = rating - pred
        
        # Sample factors for gradient computation
        user_factors = self._sample_factors(self.user_factors_mean[user_idx], self.user_factors_std[user_idx])
        item_factors = self._sample_factors(self.item_factors_mean[item_idx], self.item_factors_std[item_idx])
        
        # Update biases
        self.user_bias[user_idx] += self.learning_rate * (error - self.regularization * self.user_bias[user_idx])
        self.item_bias[item_idx] += self.learning_rate * (error - self.regularization * self.item_bias[item_idx])
        
        # Compute gradients for factor means
        grad_user_mean = error * item_factors - self.regularization * self.user_factors_mean[user_idx]
        grad_item_mean = error * user_factors - self.regularization * self.item_factors_mean[item_idx]
        
        # Compute gradients for factor standard deviations
        grad_user_std = error * item_factors * np.random.normal(0, 1, self.factors)
        grad_item_std = error * user_factors * np.random.normal(0, 1, self.factors)
        
        # Add KL divergence gradients
        kl_grad_user_mean = (self.user_factors_mean[user_idx] - self.prior_mean) / self.factors
        kl_grad_user_std = (self.user_factors_std[user_idx] - 1.0 / self.user_factors_std[user_idx]) / self.factors
        
        kl_grad_item_mean = (self.item_factors_mean[item_idx] - self.prior_mean) / self.factors
        kl_grad_item_std = (self.item_factors_std[item_idx] - 1.0 / self.item_factors_std[item_idx]) / self.factors
        
        # Update factor means
        self.user_factors_mean[user_idx] += self.learning_rate * (grad_user_mean - kl_grad_user_mean)
        self.item_factors_mean[item_idx] += self.learning_rate * (grad_item_mean - kl_grad_item_mean)
        
        # Update factor standard deviations (ensure they remain positive)
        self.user_factors_std[user_idx] = np.maximum(
            0.01, 
            self.user_factors_std[user_idx] + self.learning_rate * (grad_user_std - kl_grad_user_std)
        )
        self.item_factors_std[item_idx] = np.maximum(
            0.01, 
            self.item_factors_std[item_idx] + self.learning_rate * (grad_item_std - kl_grad_item_std)
        )
    
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
        
        # Create training data
        train_data = list(zip(user_indices, item_indices, ratings))
        
        # Training loop
        for iteration in range(self.iterations):
            # Shuffle training data
            if self.seed is not None:
                np.random.seed(self.seed + iteration)
            np.random.shuffle(train_data)
            
            # Mini-batch training
            for i in range(0, len(train_data), self.batch_size):
                batch = train_data[i:i+self.batch_size]
                
                for user_idx, item_idx, rating in batch:
                    # Make prediction
                    pred = self._predict(user_idx, item_idx, sample=True)
                    
                    # Update parameters
                    self._sgd_update(user_idx, item_idx, rating, pred)
    
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
        if self.user_factors_mean is None or self.item_factors_mean is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Map user_id to internal index
        if user_id not in self.user_map:
            raise ValueError(f"User ID {user_id} not found in training data")
            
        user_idx = self.user_map[user_id]
        
        # Calculate scores for all items
        scores = np.zeros(len(self.item_map))
        
        # Use mean factors for prediction
        user_factors = self.user_factors_mean[user_idx]
        
        for item_idx in range(len(self.item_map)):
            scores[item_idx] = self._predict(user_idx, item_idx, sample=False)
        
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
        if self.user_factors_mean is None or self.item_factors_mean is None:
            raise ValueError("Model has not been trained. Call fit() first.")
            
        # Save model data
        model_data = {
            'user_factors_mean': self.user_factors_mean,
            'user_factors_std': self.user_factors_std,
            'item_factors_mean': self.item_factors_mean,
            'item_factors_std': self.item_factors_std,
            'global_bias': self.global_bias,
            'user_bias': self.user_bias,
            'item_bias': self.item_bias,
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
                'prior_mean': self.prior_mean,
                'prior_std': self.prior_std,
                'seed': self.seed
            }
        }
        np.save(filepath, model_data, allow_pickle=True)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'VMFBase':
        """Load a model from a file"""
        model_data = np.load(filepath, allow_pickle=True).item()
        
        # Create an instance with the saved parameters
        instance = cls(
            factors=model_data['params']['factors'],
            learning_rate=model_data['params']['learning_rate'],
            regularization=model_data['params']['regularization'],
            iterations=model_data['params']['iterations'],
            batch_size=model_data['params']['batch_size'],
            prior_mean=model_data['params']['prior_mean'],
            prior_std=model_data['params']['prior_std'],
            seed=model_data['params']['seed']
        )
        
        # Restore instance variables
        instance.user_factors_mean = model_data['user_factors_mean']
        instance.user_factors_std = model_data['user_factors_std']
        instance.item_factors_mean = model_data['item_factors_mean']
        instance.item_factors_std = model_data['item_factors_std']
        instance.global_bias = model_data['global_bias']
        instance.user_bias = model_data['user_bias']
        instance.item_bias = model_data['item_bias']
        instance.user_map = model_data['user_map']
        instance.item_map = model_data['item_map']
        instance.reverse_user_map = model_data['reverse_user_map']
        instance.reverse_item_map = model_data['reverse_item_map']
        
        return instance 