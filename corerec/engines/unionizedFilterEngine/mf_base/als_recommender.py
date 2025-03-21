# # corerec/engines/unionizedFilterEngine/als_recommender.py
'''
This needs to be implemented
mf base class is implemented but extents a real branch
matrix_factorization_base class is not implemented and extends a fake branch.

real branch : mf_base
fake branch : matrix_factorization_base
'''


# from .matrix_factorization_base import MatrixFactorizationBase

# class ALSRecommender(MatrixFactorizationBase):
#     def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]):
#         interaction_matrix = csr_matrix(interaction_matrix)
#         num_users, num_items = interaction_matrix.shape
#         self.initialize_factors(num_users, num_items)

#         for epoch in range(self.epochs):
#             # Update user factors
#             for u in range(num_users):
#                 # Implement ALS user factor update
#                 pass

#             # Update item factors
#             for i in range(num_items):
#                 # Implement ALS item factor update
#                 pass

#             loss = self.compute_loss(interaction_matrix)
#             print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")

#     def recommend(self, user_id: int, top_n: int = 10) -> List[int]:
#         user_vector = self.user_factors[user_id]
#         scores = self.item_factors.dot(user_vector)
#         top_items = scores.argsort()[-top_n:][::-1]
#         return top_items.tolist()

import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Optional, Dict, Any
from ..base_recommender import BaseRecommender

class ALSRecommender(BaseRecommender):
    """
    Alternating Least Squares (ALS) algorithm for collaborative filtering.
    
    This implementation uses alternating least squares to learn latent factors for users and items.
    It works well with implicit feedback datasets and can scale to large datasets.
    
    Parameters:
    -----------
    num_factors : int
        Number of latent factors to use
    regularization : float
        Regularization parameter to prevent overfitting
    alpha : float
        Confidence parameter for implicit feedback
    iterations : int
        Number of iterations to run
    seed : Optional[int]
        Random seed for reproducibility
    """
    def __init__(
        self, 
        num_factors: int = 100, 
        regularization: float = 0.01, 
        alpha: float = 1.0, 
        iterations: int = 15,
        seed: Optional[int] = None
    ):
        self.num_factors = num_factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.user_factors = None
        self.item_factors = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """Create mappings between original IDs and matrix indices"""
        self.user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_map.items()}

    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Train the ALS model using the provided interaction matrix.
        
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
        
        # Initialize factors with small random values
        num_users, num_items = interaction_matrix.shape
        
        if self.seed is not None:
            np.random.seed(self.seed)
            
        self.user_factors = np.random.normal(0, 0.01, (num_users, self.num_factors))
        self.item_factors = np.random.normal(0, 0.01, (num_items, self.num_factors))
        
        # Convert to confidence matrix for implicit feedback
        confidence = 1.0 + self.alpha * interaction_matrix
        
        # Precompute transpose for efficiency
        confidence_T = confidence.T.tocsr()
        
        # Alternating least squares iterations
        for iteration in range(self.iterations):
            # Update user factors
            for u in range(num_users):
                # Get items this user has interacted with
                item_indices = interaction_matrix[u].indices
                if len(item_indices) == 0:
                    continue
                    
                # Get confidence values for this user
                conf_u = confidence[u, item_indices].A[0]
                
                # Get the associated item factors
                factors_i = self.item_factors[item_indices]
                
                # Build the right side of the equation
                A = factors_i.T.dot(np.diag(conf_u)).dot(factors_i) + \
                    np.eye(self.num_factors) * self.regularization
                    
                b = factors_i.T.dot(np.diag(conf_u)).dot(np.ones(len(item_indices)))
                
                # Solve the equation to get updated user factors
                self.user_factors[u] = np.linalg.solve(A, b)
            
            # Update item factors
            for i in range(num_items):
                # Get users who have interacted with this item
                user_indices = confidence_T[i].indices
                if len(user_indices) == 0:
                    continue
                    
                # Get confidence values for this item
                conf_i = confidence_T[i, user_indices].A[0]
                
                # Get the associated user factors
                factors_u = self.user_factors[user_indices]
                
                # Build the right side of the equation
                A = factors_u.T.dot(np.diag(conf_i)).dot(factors_u) + \
                    np.eye(self.num_factors) * self.regularization
                    
                b = factors_u.T.dot(np.diag(conf_i)).dot(np.ones(len(user_indices)))
                
                # Solve the equation to get updated item factors
                self.item_factors[i] = np.linalg.solve(A, b)
                
            print(f"Iteration {iteration+1}/{self.iterations} completed")
    
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
        user_vector = self.user_factors[user_idx]
        
        # Calculate scores for all items
        scores = self.item_factors.dot(user_vector)
        
        # If requested, exclude items the user has already interacted with
        if exclude_seen:
            # Find items the user has already interacted with
            seen_items = set()
            for item_id, item_idx in self.item_map.items():
                if self.user_item_matrix[user_idx, item_idx] > 0:
                    seen_items.add(item_idx)
            
            # Set scores of seen items to a very low value
            for item_idx in seen_items:
                scores[item_idx] = float('-inf')
        
        # Get top-n item indices
        top_item_indices = np.argsort(-scores)[:top_n]
        
        # Map indices back to original item IDs
        top_items = [self.reverse_item_map[idx] for idx in top_item_indices]
        
        return top_items
    
    def get_user_factors(self) -> np.ndarray:
        """Return the learned user factors"""
        return self.user_factors
    
    def get_item_factors(self) -> np.ndarray:
        """Return the learned item factors"""
        return self.item_factors
    
    def save_model(self, filepath: str) -> None:
        """Save the model to a file"""
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'params': {
                'num_factors': self.num_factors,
                'regularization': self.regularization,
                'alpha': self.alpha,
                'iterations': self.iterations,
                'seed': self.seed
            }
        }
        np.save(filepath, model_data, allow_pickle=True)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ALSRecommender':
        """Load a model from a file"""
        model_data = np.load(filepath, allow_pickle=True).item()
        
        # Create an instance with the saved parameters
        model = cls(
            num_factors=model_data['params']['num_factors'],
            regularization=model_data['params']['regularization'],
            alpha=model_data['params']['alpha'],
            iterations=model_data['params']['iterations'],
            seed=model_data['params']['seed']
        )
        
        # Restore model state
        model.user_factors = model_data['user_factors']
        model.item_factors = model_data['item_factors']
        model.user_map = model_data['user_map']
        model.item_map = model_data['item_map']
        model.reverse_user_map = model_data['reverse_user_map']
        model.reverse_item_map = model_data['reverse_item_map']
        
        return model