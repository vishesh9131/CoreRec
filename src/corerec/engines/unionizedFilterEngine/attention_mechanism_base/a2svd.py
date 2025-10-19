import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Optional, Dict, Any, Tuple
from ..base_recommender import BaseRecommender

class A2SVD(BaseRecommender):
    """
    Attentive Asynchronous Singular Value Decomposition (A2SVD) for collaborative filtering.
    
    This model extends traditional SVD by incorporating attention mechanisms to capture
    the varying importance of different user-item interactions.
    
    Parameters:
    -----------
    factors : int
        Number of latent factors
    iterations : int
        Number of iterations for optimization
    learning_rate : float
        Learning rate for gradient descent
    regularization : float
        Regularization strength
    seed : Optional[int]
        Random seed for reproducibility
    device : str
        Computation device ('cpu', 'cuda', 'mps', etc.)
    """
    def __init__(
        self,
        factors: int = 100,
        iterations: int = 20,
        learning_rate: float = 0.01,
        regularization: float = 0.1,
        seed: Optional[int] = None,
        device: str = 'auto'
    ):
        """
        Initialize the A2SVD model.
        
        Parameters:
        -----------
        factors : int
            Number of latent factors
        iterations : int
            Number of iterations for optimization
        learning_rate : float
            Learning rate for gradient descent
        regularization : float
            Regularization strength
        seed : Optional[int]
            Random seed for reproducibility
        device : str
            Computation device ('cpu', 'cuda', 'mps', etc.)
        """
        super().__init__(device=device)
        self.factors = factors
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.seed = seed
        
        # Model parameters (will be initialized during training)
        self.user_factors = None
        self.item_factors = None
        self.attention_weights = None
        self.user_map = None
        self.item_map = None
        self.reverse_user_map = None
        self.reverse_item_map = None
        self.user_item_matrix = None
        
        # Framework reference based on device
        self.xp = self.device_manager.get_framework_for_device()
    
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
        self.user_factors = self.create_tensor(
            np.random.normal(0, 0.1, (n_users, self.factors)), 
            dtype='float32'
        )
        self.item_factors = self.create_tensor(
            np.random.normal(0, 0.1, (n_items, self.factors)), 
            dtype='float32'
        )
        self.attention_weights = self.create_tensor(
            np.random.normal(0, 0.1, (n_users, self.factors)), 
            dtype='float32'
        )
    
    def _predict(self, user_idx: int, item_idx: int) -> float:
        """Make prediction for a user-item pair"""
        # Get framework reference (numpy, torch, etc.)
        xp = self.device_manager.get_framework_for_device()
        
        # Compute prediction using dot product and attention
        if hasattr(xp, 'dot'):  # NumPy-like API
            attention_score = xp.dot(self.attention_weights[user_idx], self.item_factors[item_idx])
            return xp.dot(self.user_factors[user_idx], self.item_factors[item_idx]) + attention_score
        else:  # PyTorch-like API
            attention_score = xp.sum(self.attention_weights[user_idx] * self.item_factors[item_idx])
            return xp.sum(self.user_factors[user_idx] * self.item_factors[item_idx]) + attention_score
    
    def fit(self, user_ids: List[int], item_ids: List[int], ratings: List[float]) -> None:
        """
        Train the A2SVD model using the provided data.
        
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
        self._create_mappings(user_ids, item_ids)
        
        # Convert IDs to indices
        user_indices = [self.user_map[user_id] for user_id in user_ids]
        item_indices = [self.item_map[item_id] for item_id in item_ids]
        
        # Create user-item matrix
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.user_item_matrix = csr_matrix((ratings, (user_indices, item_indices)), shape=(n_users, n_items))
        
        # Initialize parameters
        self._init_params(n_users, n_items)
        
        # Get framework reference (numpy, torch, etc.)
        xp = self.device_manager.get_framework_for_device()
        
        # Training loop
        for iteration in range(self.iterations):
            # Shuffle data
            if self.seed is not None:
                np.random.seed(self.seed + iteration)
            
            indices = np.random.permutation(len(user_indices))
            
            # Batch gradient descent
            for idx in indices:
                user_idx = user_indices[idx]
                item_idx = item_indices[idx]
                rating = ratings[idx]
                
                # Compute prediction and error
                pred = self._predict(user_idx, item_idx)
                
                # Convert to scalar if tensor
                if hasattr(pred, 'item'):
                    pred = pred.item()
                
                error = rating - pred
                
                # Compute gradients
                if hasattr(xp, 'dot'):  # NumPy-like API
                    # Update user factors
                    user_grad = error * self.item_factors[item_idx] - self.regularization * self.user_factors[user_idx]
                    self.user_factors[user_idx] += self.learning_rate * user_grad
                    
                    # Update item factors
                    item_grad = error * (self.user_factors[user_idx] + self.attention_weights[user_idx]) - self.regularization * self.item_factors[item_idx]
                    self.item_factors[item_idx] += self.learning_rate * item_grad
                    
                    # Update attention weights
                    attention_update = self.learning_rate * (error * self.item_factors[item_idx] - self.regularization * self.attention_weights[user_idx])
                    self.attention_weights[user_idx] += attention_update
                else:  # PyTorch-like API
                    # Update user factors
                    user_grad = error * self.item_factors[item_idx] - self.regularization * self.user_factors[user_idx]
                    self.user_factors[user_idx] += self.learning_rate * user_grad
                    
                    # Update item factors
                    item_grad = error * (self.user_factors[user_idx] + self.attention_weights[user_idx]) - self.regularization * self.item_factors[item_idx]
                    self.item_factors[item_idx] += self.learning_rate * item_grad
                    
                    # Update attention weights
                    attention_update = self.learning_rate * (error * self.item_factors[item_idx] - self.regularization * self.attention_weights[user_idx])
                    self.attention_weights[user_idx] += attention_update
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """
        Generate top-N recommendations for a specific user.
        
        Parameters:
        -----------
        user_id : int
            User ID
        top_n : int
            Number of recommendations to generate
        exclude_seen : bool
            Whether to exclude items the user has already interacted with
            
        Returns:
        --------
        List[int]
            List of recommended item IDs
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        if user_id not in self.user_map:
            raise ValueError(f"User ID {user_id} not found in training data")
        
        user_idx = self.user_map[user_id]
        n_items = len(self.item_map)
        
        # Get framework reference (numpy, torch, etc.)
        xp = self.device_manager.get_framework_for_device()
        
        # Compute scores for all items
        scores = []
        for item_idx in range(n_items):
            score = self._predict(user_idx, item_idx)
            
            # Convert to scalar if tensor
            if hasattr(score, 'item'):
                score = score.item()
                
            scores.append(score)
        
        # Convert to numpy array for sorting
        scores = np.array(scores)
        
        # Exclude seen items if requested
        if exclude_seen:
            seen_indices = self.user_item_matrix[user_idx].nonzero()[1]
            scores[seen_indices] = -np.inf
        
        # Get top-N items
        top_item_indices = np.argsort(-scores)[:top_n]
        
        # Map indices back to original item IDs
        top_items = [self.reverse_item_map[idx] for idx in top_item_indices]
        
        return top_items
    
    def save_model(self, filepath: str) -> None:
        """Save the model to a file"""
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been trained. Call fit() first.")
            
        # Convert tensors to numpy arrays if needed
        user_factors = self.user_factors
        item_factors = self.item_factors
        attention_weights = self.attention_weights
        
        if hasattr(user_factors, 'cpu'):
            user_factors = user_factors.cpu().numpy()
            item_factors = item_factors.cpu().numpy()
            attention_weights = attention_weights.cpu().numpy()
        
        # Save model data
        model_data = {
            'model_params': {
                'factors': self.factors,
                'iterations': self.iterations,
                'learning_rate': self.learning_rate,
                'regularization': self.regularization,
                'seed': self.seed
            },
            'user_factors': user_factors,
            'item_factors': item_factors,
            'attention_weights': attention_weights,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map
        }
        
        np.save(filepath, model_data, allow_pickle=True)
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'auto') -> 'A2SVD':
        """Load a model from a file"""
        model_data = np.load(filepath, allow_pickle=True).item()
        
        # Create an instance with the saved parameters
        instance = cls(
            factors=model_data['model_params']['factors'],
            iterations=model_data['model_params']['iterations'],
            learning_rate=model_data['model_params']['learning_rate'],
            regularization=model_data['model_params']['regularization'],
            seed=model_data['model_params']['seed'],
            device=device
        )
        
        # Load model data to the specified device
        instance.user_factors = instance.to_device(model_data['user_factors'])
        instance.item_factors = instance.to_device(model_data['item_factors'])
        instance.attention_weights = instance.to_device(model_data['attention_weights'])
        instance.user_map = model_data['user_map']
        instance.item_map = model_data['item_map']
        instance.reverse_user_map = model_data['reverse_user_map']
        instance.reverse_item_map = model_data['reverse_item_map']
        
        return instance 