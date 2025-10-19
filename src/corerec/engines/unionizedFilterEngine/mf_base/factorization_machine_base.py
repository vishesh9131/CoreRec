import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Optional, Dict, Any, Tuple
from ..base_recommender import BaseRecommender

class FactorizationMachineBase(BaseRecommender):
    """
    Factorization Machine (FM) for collaborative filtering.
    
    Based on the paper:
    "Factorization Machines" by Steffen Rendle.
    
    This implementation focuses on the 2-way FM model for collaborative filtering,
    which captures pairwise interactions between features.
    
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
    seed : Optional[int]
        Random seed for reproducibility
    """
    def __init__(
        self,
        factors: int = 10,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        iterations: int = 100,
        batch_size: int = 1000,
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
        self.w0 = None  # Global bias
        self.w = None   # Feature weights
        self.v = None   # Feature factor matrix
    
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
        
        # Number of features: user_id + item_id
        n_features = n_users + n_items
        
        # Initialize parameters
        self.w0 = 0.0  # Global bias
        self.w = np.zeros(n_features)  # Linear terms
        self.v = np.random.normal(0, 0.1, (n_features, self.factors))  # Interaction factors
    
    def _create_feature_vector(self, user_idx: int, item_idx: int, n_users: int) -> np.ndarray:
        """Create one-hot encoded feature vector for user-item pair"""
        # Feature vector: [user_1, user_2, ..., item_1, item_2, ...]
        x = np.zeros(n_users + self.user_item_matrix.shape[1])
        x[user_idx] = 1.0  # One-hot encode user
        x[n_users + item_idx] = 1.0  # One-hot encode item
        return x
    
    def _predict(self, x: np.ndarray) -> float:
        """Make prediction using FM model for a feature vector"""
        # First order term: w0 + sum(w_i * x_i)
        pred = self.w0 + np.dot(self.w, x)
        
        # Second order term: sum_f( (sum_i(v_i,f * x_i))^2 - sum_i(v_i,f^2 * x_i^2) ) / 2
        sum_square = np.zeros(self.factors)
        square_sum = np.zeros(self.factors)
        
        # Calculate interaction terms efficiently
        for i in range(len(x)):
            if x[i] != 0:
                sum_square += self.v[i] * x[i]
                square_sum += (self.v[i] * x[i]) ** 2
        
        # Add interaction term to prediction
        pred += 0.5 * np.sum(sum_square ** 2 - square_sum)
        
        return pred
    
    def _sgd_update(self, x: np.ndarray, y: float, pred: float) -> None:
        """Update parameters using SGD for a single sample"""
        # Calculate error
        error = pred - y
        
        # Update global bias
        self.w0 -= self.learning_rate * (error + self.regularization * self.w0)
        
        # Update feature weights and factors
        for i in range(len(x)):
            if x[i] == 0:
                continue
                
            # Update linear term
            grad_w = error * x[i] + self.regularization * self.w[i]
            self.w[i] -= self.learning_rate * grad_w
            
            # Calculate sum term for interaction factors
            sum_term = np.zeros(self.factors)
            for j in range(len(x)):
                if x[j] != 0 and j != i:
                    sum_term += self.v[j] * x[j]
            
            # Update interaction factors
            grad_v = error * x[i] * sum_term + self.regularization * self.v[i]
            self.v[i] -= self.learning_rate * grad_v
    
    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Train the FM model on the given interaction data.
        
        Parameters:
        -----------
        interaction_matrix : csr_matrix
            User-item interaction matrix where non-zero entries indicate interactions
        user_ids : List[int]
            List of user IDs corresponding to rows in the interaction matrix
        item_ids : List[int]
            List of item IDs corresponding to columns in the interaction matrix
        """
        # Create mappings
        self._create_mappings(user_ids, item_ids)
        
        # Store interaction matrix for later use
        self.user_item_matrix = interaction_matrix
        
        # Get dimensions
        n_users, n_items = interaction_matrix.shape
        
        # Initialize model parameters
        self._init_params(n_users, n_items)
        
        # Create training data
        user_indices, item_indices, ratings = [], [], []
        for user_idx in range(n_users):
            for item_idx in interaction_matrix[user_idx].indices:
                user_indices.append(user_idx)
                item_indices.append(item_idx)
                ratings.append(1.0)  # Assuming binary interactions
        
        # Training loop
        n_samples = len(user_indices)
        for iteration in range(self.iterations):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            user_indices_shuffled = [user_indices[i] for i in indices]
            item_indices_shuffled = [item_indices[i] for i in indices]
            ratings_shuffled = [ratings[i] for i in indices]
            
            total_loss = 0.0
            
            # Mini-batch training
            for batch_start in range(0, n_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_samples)
                
                batch_loss = 0.0
                
                # Process each sample in the batch
                for i in range(batch_start, batch_end):
                    user_idx = user_indices_shuffled[i]
                    item_idx = item_indices_shuffled[i]
                    rating = ratings_shuffled[i]
                    
                    # Create feature vector
                    x = self._create_feature_vector(user_idx, item_idx, n_users)
                    
                    # Make prediction
                    pred = self._predict(x)
                    
                    # Calculate loss
                    error = pred - rating
                    batch_loss += error ** 2
                    
                    # Update parameters
                    self._sgd_update(x, rating, pred)
                
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
        if self.w is None or self.v is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Map user_id to internal index
        if user_id not in self.user_map:
            raise ValueError(f"User ID {user_id} not found in training data")
            
        user_idx = self.user_map[user_id]
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        
        # Calculate scores for all items
        scores = np.zeros(n_items)
        for item_idx in range(n_items):
            # Create feature vector
            x = self._create_feature_vector(user_idx, item_idx, n_users)
            
            # Make prediction
            scores[item_idx] = self._predict(x)
        
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
        if self.w is None or self.v is None:
            raise ValueError("Model has not been trained. Call fit() first.")
            
        # Save model data
        model_data = {
            'w0': self.w0,
            'w': self.w,
            'v': self.v,
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
    def load_model(cls, filepath: str) -> 'FactorizationMachineBase':
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
        instance.w0 = model_data['w0']
        instance.w = model_data['w']
        instance.v = model_data['v']
        instance.user_map = model_data['user_map']
        instance.item_map = model_data['item_map']
        instance.reverse_user_map = model_data['reverse_user_map']
        instance.reverse_item_map = model_data['reverse_item_map']
        
        return instance 