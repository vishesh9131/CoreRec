import numpy as np
from scipy.sparse import csr_matrix
from typing import Union, List, Optional, Dict
from pathlib import Path
from ..base_recommender import BaseRecommender
from corerec.api.exceptions import ModelNotFittedError
import pickle
import logging

logger = logging.getLogger(__name__)

        
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
        seed: Optional[int] = None,
        device: str = "auto"
    ):
        super().__init__(device=device)
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
        self.w0 = None
        self.w = None
        self.v = None
        
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Create mappings between user/item IDs and indices.
        
        Parameters:
        -----------
        user_ids : List[int]
            List of unique user IDs
        item_ids : List[int]
            List of unique item IDs
        """
        self.user_map = {user_id: i for i, user_id in enumerate(user_ids)}
        self.item_map = {item_id: i for i, item_id in enumerate(item_ids)}
        self.reverse_user_map = {i: user_id for user_id, i in self.user_map.items()}
        self.reverse_item_map = {i: item_id for item_id, i in self.item_map.items()}
    
    def _build_feature_vector(self, user_id: int, item_id: int) -> np.ndarray:
        """
        Build feature vector for a user-item pair.
        
        For collaborative filtering, features are one-hot encoded:
        - First n_users positions: user one-hot
        - Next n_items positions: item one-hot
        
        Parameters:
        -----------
        user_id : int
            User ID
        item_id : int
            Item ID
            
        Returns:
        --------
        np.ndarray
            Feature vector (sparse, mostly zeros)
        """
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        n_features = n_users + n_items
        
        x = np.zeros(n_features)
        
        if user_id in self.user_map:
            x[self.user_map[user_id]] = 1.0
        
        if item_id in self.item_map:
            x[n_users + self.item_map[item_id]] = 1.0
        
        return x
    
    def fit(
        self,
        interaction_matrix: csr_matrix,
        user_ids: List[int],
        item_ids: List[int]
    ) -> 'FactorizationMachineBase':
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
            
        Returns:
        --------
        self
        """
        self._create_mappings(user_ids, item_ids)
        
        n_users = len(user_ids)
        n_items = len(item_ids)
        n_features = n_users + n_items
        
        # Initialize parameters
        self.w0 = 0.0
        self.w = np.zeros(n_features)
        self.v = np.random.normal(0, 0.1, (n_features, self.factors))
        
        # Training loop using SGD
        for iteration in range(self.iterations):
            # Sample random interactions
            user_indices, item_indices = interaction_matrix.nonzero()
            
            if len(user_indices) == 0:
                break
            
            indices = np.random.choice(
                len(user_indices),
                size=min(self.batch_size, len(user_indices)),
                replace=False
            )
            
            batch_users = user_indices[indices]
            batch_items = item_indices[indices]
            
            # Process batch
            for u_idx, i_idx in zip(batch_users, batch_items):
                user_id = user_ids[u_idx]
                item_id = item_ids[i_idx]
                rating = interaction_matrix[u_idx, i_idx]
                
                x = self._build_feature_vector(user_id, item_id)
                pred = self._predict_internal(x)
                
                error = rating - pred
                
                # Update global bias
                self.w0 += self.learning_rate * error
                
                # Update linear terms
                for i in range(len(x)):
                    if x[i] != 0:
                        grad = error * x[i] - self.regularization * self.w[i]
                        self.w[i] += self.learning_rate * grad
                
                # Update interaction factors
                for f in range(self.factors):
                    sum_vf = np.sum(self.v[:, f] * x)
                    for i in range(len(x)):
                        if x[i] != 0:
                            grad = error * (sum_vf - self.v[i, f] * x[i]) * x[i] - self.regularization * self.v[i, f]
                            self.v[i, f] += self.learning_rate * grad
        
        if self.verbose:
            logger.info(f"FM training completed after {self.iterations} iterations")
        
        return self
    
    def _predict_internal(self, x: np.ndarray) -> float:
        """
        Internal prediction method using feature vector.
        
        Parameters:
        -----------
        x : np.ndarray
            Feature vector
            
            Returns:
        --------
        float
            Prediction score
        """
        if self.w is None or self.v is None:
            raise ModelNotFittedError("Model must be fitted before making predictions.")
        
        # First order term
            pred = self.w0 + np.dot(self.w, x)
        
        # Second order term
            sum_square = np.zeros(self.factors)
            square_sum = np.zeros(self.factors)
        
            for i in range(len(x)):
                if x[i] != 0:
                    sum_square += self.v[i] * x[i]
                    square_sum += (self.v[i] * x[i]) ** 2
        
            pred += 0.5 * np.sum(sum_square ** 2 - square_sum)
        
            return pred
    
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
        float
            Predicted rating
        """
        x = self._build_feature_vector(user_id, item_id)
        return self._predict_internal(x)
    
    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_seen: bool = True
    ) -> List[int]:
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
        List[int]
            List of recommended item IDs
        """
        if self.w is None or self.v is None:
            raise ModelNotFittedError("Model must be fitted before making recommendations.")
        
        if user_id not in self.user_map:
            return []
        
        scores = []
        
        for item_id in self.item_map.keys():
            score = self.predict(user_id, item_id)
            scores.append((item_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = [item_id for item_id, _ in scores[:top_n]]
        
        return recommendations
    
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """
        Save model to disk using pickle.
        
        Parameters:
        -----------
        path : Union[str, Path]
            File path to save the model
        **kwargs : dict
            Additional arguments (unused)
        """
        if self.w is None or self.v is None:
            raise ModelNotFittedError("Model must be fitted before saving.")
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if self.verbose:
            logger.info(f"FactorizationMachine model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> 'FactorizationMachineBase':
        """
        Load model from disk.
        
        Parameters:
        -----------
        path : Union[str, Path]
            File path to load the model from
        **kwargs : dict
            Additional arguments (unused)
            
        Returns:
        --------
        Loaded model instance
        """
        path_obj = Path(path)
        
        with open(path_obj, 'rb') as f:
            model = pickle.load(f)
        
        if not isinstance(model, cls):
            raise ValueError(f"Loaded object is not instance of {cls.__name__}")
        
        if model.verbose:
            logger.info(f"FactorizationMachine model loaded from {path}")
        
        return model
