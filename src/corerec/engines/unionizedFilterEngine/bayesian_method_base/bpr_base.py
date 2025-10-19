import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Optional, Dict, Any, Tuple
from ..base_recommender import BaseRecommender

class BPRBase(BaseRecommender):
    """
    Bayesian Personalized Ranking (BPR) for implicit feedback datasets.
    
    Implementation based on the paper:
    "BPR: Bayesian Personalized Ranking from Implicit Feedback" by Rendle et al.
    
    BPR optimizes for correct ranking of items using a pairwise loss function.
    It's specifically designed for implicit feedback scenarios where only positive
    interactions are observed.
    
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
        factors: int = 100,
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
        self.user_factors = None
        self.item_factors = None
        self.item_bias = None
    
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """Create mappings between original IDs and matrix indices"""
        self.user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_map.items()}
    
    def _init_factors(self, n_users: int, n_items: int) -> None:
        """Initialize model parameters"""
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Initialize factors with small random values
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.factors))
        self.item_bias = np.zeros(n_items)
    
    def _sample_triplet(self, user_item_matrix: csr_matrix, n_items: int) -> Tuple[int, int, int]:
        """Sample a (user, positive_item, negative_item) triplet for training"""
        # Sample a random user that has at least one interaction
        user_idx = np.random.randint(0, user_item_matrix.shape[0])
        while user_item_matrix[user_idx].nnz == 0:
            user_idx = np.random.randint(0, user_item_matrix.shape[0])
        
        # Sample a positive item for this user
        pos_items = user_item_matrix[user_idx].indices
        pos_idx = pos_items[np.random.randint(0, len(pos_items))]
        
        # Sample a negative item for this user
        neg_idx = np.random.randint(0, n_items)
        while neg_idx in pos_items:
            neg_idx = np.random.randint(0, n_items)
        
        return user_idx, pos_idx, neg_idx
    
    def _compute_loss(self, user_idx: int, pos_idx: int, neg_idx: int) -> float:
        """Compute BPR loss for a single triplet"""
        # Compute scores
        pos_score = np.dot(self.user_factors[user_idx], self.item_factors[pos_idx]) + self.item_bias[pos_idx]
        neg_score = np.dot(self.user_factors[user_idx], self.item_factors[neg_idx]) + self.item_bias[neg_idx]
        
        # Compute loss: -log(sigmoid(pos_score - neg_score))
        loss = -np.log(1.0 / (1.0 + np.exp(-(pos_score - neg_score))))
        
        # Add regularization
        loss += self.regularization * (
            np.sum(self.user_factors[user_idx]**2) +
            np.sum(self.item_factors[pos_idx]**2) +
            np.sum(self.item_factors[neg_idx]**2) +
            self.item_bias[pos_idx]**2 +
            self.item_bias[neg_idx]**2
        )
        
        return loss
    
    def _update_factors(self, user_idx: int, pos_idx: int, neg_idx: int) -> None:
        """Update model parameters using SGD for a single triplet"""
        # Compute scores
        pos_score = np.dot(self.user_factors[user_idx], self.item_factors[pos_idx]) + self.item_bias[pos_idx]
        neg_score = np.dot(self.user_factors[user_idx], self.item_factors[neg_idx]) + self.item_bias[neg_idx]
        
        # Compute sigmoid gradient
        sigmoid = 1.0 / (1.0 + np.exp(pos_score - neg_score))
        
        # Compute gradients
        grad_user = sigmoid * (self.item_factors[neg_idx] - self.item_factors[pos_idx]) + \
                    self.regularization * self.user_factors[user_idx]
                    
        grad_pos_item = sigmoid * (-self.user_factors[user_idx]) + \
                        self.regularization * self.item_factors[pos_idx]
                        
        grad_neg_item = sigmoid * self.user_factors[user_idx] + \
                        self.regularization * self.item_factors[neg_idx]
                        
        grad_pos_bias = sigmoid * (-1) + self.regularization * self.item_bias[pos_idx]
        grad_neg_bias = sigmoid * 1 + self.regularization * self.item_bias[neg_idx]
        
        # Update parameters
        self.user_factors[user_idx] -= self.learning_rate * grad_user
        self.item_factors[pos_idx] -= self.learning_rate * grad_pos_item
        self.item_factors[neg_idx] -= self.learning_rate * grad_neg_item
        self.item_bias[pos_idx] -= self.learning_rate * grad_pos_bias
        self.item_bias[neg_idx] -= self.learning_rate * grad_neg_bias
    
    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Train the BPR model on the given interaction data.
        
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
        self._init_factors(n_users, n_items)
        
        # Training loop
        for iteration in range(self.iterations):
            total_loss = 0.0
            
            # Mini-batch training
            for _ in range(self.batch_size):
                # Sample a triplet
                user_idx, pos_idx, neg_idx = self._sample_triplet(interaction_matrix, n_items)
                
                # Compute loss
                loss = self._compute_loss(user_idx, pos_idx, neg_idx)
                total_loss += loss
                
                # Update factors
                self._update_factors(user_idx, pos_idx, neg_idx)
            
            # Print progress
            avg_loss = total_loss / self.batch_size
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
        scores = np.dot(self.item_factors, user_vector) + self.item_bias
        
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
                'seed': self.seed
            }
        }
        np.save(filepath, model_data, allow_pickle=True)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BPRBase':
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
        instance.item_bias = model_data['item_bias']
        instance.user_map = model_data['user_map']
        instance.item_map = model_data['item_map']
        instance.reverse_user_map = model_data['reverse_user_map']
        instance.reverse_item_map = model_data['reverse_item_map']
        
        return instance 