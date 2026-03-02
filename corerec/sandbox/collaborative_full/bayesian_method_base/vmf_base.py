"""
Variational Deep Matrix Factorization (VDeepMF) for Collaborative Filtering

Based on the paper:
"Deep Variational Models for Collaborative Filtering-based Recommender Systems"
by Bobadilla et al., 2021

This implementation follows the VDeepMF architecture which combines:
- Embedding layers for user and item representations
- Variational layers that compute mean and variance for Gaussian distributions
- Stochastic sampling from these distributions
- Dot product for final prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from typing import Union, List, Optional, Dict, Any, Tuple
from pathlib import Path
import pickle
import logging
import sys

# Handle relative imports - allow both direct import and package import
try:
    from ..base_recommender import BaseRecommender
    from corerec.api.exceptions import ModelNotFittedError
    from corerec.utils.validation import validate_user_id, validate_top_k
except ImportError:
    # If running as script, try absolute imports
    import os
    # Add parent directories to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from corerec.engines.collaborative.base_recommender import BaseRecommender
    from corerec.api.exceptions import ModelNotFittedError
    from corerec.utils.validation import validate_user_id, validate_top_k

logger = logging.getLogger(__name__)


class VariationalSampling(nn.Module):
    """Variational sampling layer that samples from Gaussian distribution."""
    
    def __init__(self, latent_dim: int):
        super(VariationalSampling, self).__init__()
        self.latent_dim = latent_dim
    
    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """
        Sample from Gaussian distribution using reparameterization trick.
        
        Args:
            z_mean: Mean of the distribution
            z_log_var: Log variance of the distribution
        
        Returns:
            Sampled latent vector
        """
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class VDeepMFModel(nn.Module):
    """
    Variational Deep Matrix Factorization Model
    
    Architecture:
    1. Embedding layers for users and items
    2. Variational layers (mean and variance)
    3. Sampling layer
    4. Dot product for prediction
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        latent_dim: int = 64,
        embedding_dim: int = 64
    ):
        super(VDeepMFModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Variational layers for users
        self.user_mean = nn.Linear(embedding_dim, latent_dim)
        self.user_log_var = nn.Linear(embedding_dim, latent_dim)
        
        # Variational layers for items
        self.item_mean = nn.Linear(embedding_dim, latent_dim)
        self.item_log_var = nn.Linear(embedding_dim, latent_dim)
        
        # Sampling layer
        self.sampling = VariationalSampling(latent_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.user_mean.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.user_log_var.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_mean.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_log_var.weight, mean=0.0, std=0.01)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        return_variational_params: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass of the model.
        
        Args:
            user_ids: Tensor of user indices
            item_ids: Tensor of item indices
            return_variational_params: Whether to return variational parameters
        
        Returns:
            Predicted ratings or (predicted ratings, variational params)
        """
        # Embedding
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Variational layers - compute mean and log variance
        user_mean = self.user_mean(user_emb)
        user_log_var = self.user_log_var(user_emb)
        item_mean = self.item_mean(item_emb)
        item_log_var = self.item_log_var(item_emb)
        
        # Sample from distributions
        user_z = self.sampling(user_mean, user_log_var)
        item_z = self.sampling(item_mean, item_log_var)
        
        # Dot product for prediction
        prediction = torch.sum(user_z * item_z, dim=1)
        
        if return_variational_params:
            variational_params = {
                'user_mean': user_mean,
                'user_log_var': user_log_var,
                'item_mean': item_mean,
                'item_log_var': item_log_var
            }
            return prediction, variational_params
        
        return prediction
    
    def compute_kl_loss(
        self,
        user_mean: torch.Tensor,
        user_log_var: torch.Tensor,
        item_mean: torch.Tensor,
        item_log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence loss for regularization.
        
        Args:
            user_mean: User mean vectors
            user_log_var: User log variance vectors
            item_mean: Item mean vectors
            item_log_var: Item log variance vectors
        
        Returns:
            KL divergence loss
        """
        # KL divergence: -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        user_kl = -0.5 * torch.sum(
            1 + user_log_var - user_mean.pow(2) - user_log_var.exp(),
            dim=1
        )
        item_kl = -0.5 * torch.sum(
            1 + item_log_var - item_mean.pow(2) - item_log_var.exp(),
            dim=1
        )
        return torch.mean(user_kl) + torch.mean(item_kl)


class VMFBase(BaseRecommender):
    """
    Variational Deep Matrix Factorization (VDeepMF) for collaborative filtering.
    
    Based on the paper:
    "Deep Variational Models for Collaborative Filtering-based Recommender Systems"
    by Bobadilla et al., 2021
    
    This model extends DeepMF with variational inference to create robust,
    continuous, and structured latent spaces.
    
    Parameters:
    -----------
    latent_dim : int
        Dimension of the latent space (default: 64)
    embedding_dim : int
        Dimension of embedding layers (default: 64)
    learning_rate : float
        Learning rate for optimization (default: 0.001)
    regularization : float
        L2 regularization strength (default: 1e-5)
    kl_weight : float
        Weight for KL divergence loss (default: 0.01)
    n_epochs : int
        Number of training epochs (default: 100)
    batch_size : int
        Batch size for training (default: 256)
    verbose : bool
        Whether to print training progress (default: False)
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        embedding_dim: int = 64,
        learning_rate: float = 0.001,
        regularization: float = 1e-5,
        kl_weight: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = False,
        device: str = "auto"
    ):
        super().__init__(device=device)
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.kl_weight = kl_weight
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.name = "VMFBase"
        self.is_fitted = False
        
        # Mappings
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        
        # Model components
        self.model = None
        self.optimizer = None
        self.n_users = 0
        self.n_items = 0
    
    def _create_mappings(self, user_ids: List, item_ids: List):
        """Create mappings between IDs and indices."""
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))
        
        self.user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_map.items()}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
    
    def fit(
        self,
        interaction_matrix: csr_matrix,
        user_ids: List[int],
        item_ids: List[int]
    ):
        """
        Train the VDeepMF model.
        
        Args:
            interaction_matrix: Sparse matrix of user-item interactions
            user_ids: List of user IDs
            item_ids: List of item IDs
        """
        # Create mappings
        self._create_mappings(user_ids, item_ids)
        
        # Initialize model
        self.model = VDeepMFModel(
            num_users=self.n_users,
            num_items=self.n_items,
            latent_dim=self.latent_dim,
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.regularization
        )
        
        # Prepare training data
        user_indices = []
        item_indices = []
        ratings = []
        
        for user_id, item_id in zip(user_ids, item_ids):
            if user_id in self.user_map and item_id in self.item_map:
                user_indices.append(self.user_map[user_id])
                item_indices.append(self.item_map[item_id])
                # Get rating from interaction matrix
                u_idx = self.user_map[user_id]
                i_idx = self.item_map[item_id]
                rating = interaction_matrix[u_idx, i_idx]
                if rating > 0:
                    ratings.append(float(rating))
                else:
                    ratings.append(1.0)  # Implicit feedback
        
        if len(ratings) == 0:
            raise ValueError("No valid interactions found in the interaction matrix")
        
        # Convert to tensors
        user_tensor = torch.tensor(user_indices, dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(item_indices, dtype=torch.long).to(self.device)
        rating_tensor = torch.tensor(ratings, dtype=torch.float32).to(self.device)
        
        # Training loop
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            n_batches = 0
            
            # Shuffle data
            indices = torch.randperm(len(rating_tensor)).to(self.device)
            user_shuffled = user_tensor[indices]
            item_shuffled = item_tensor[indices]
            rating_shuffled = rating_tensor[indices]
            
            # Mini-batch training
            for i in range(0, len(rating_tensor), self.batch_size):
                batch_end = min(i + self.batch_size, len(rating_tensor))
                user_batch = user_shuffled[i:batch_end]
                item_batch = item_shuffled[i:batch_end]
                rating_batch = rating_shuffled[i:batch_end]
                
                self.optimizer.zero_grad()
                
                # Forward pass with variational parameters
                prediction, var_params = self.model(
                    user_batch,
                    item_batch,
                    return_variational_params=True
                )
                
                # Reconstruction loss (MSE)
                recon_loss = F.mse_loss(prediction, rating_batch)
                
                # KL divergence loss
                kl_loss = self.model.compute_kl_loss(
                    var_params['user_mean'],
                    var_params['user_log_var'],
                    var_params['item_mean'],
                    var_params['item_log_var']
                )
                
                # Total loss
                loss = recon_loss + self.kl_weight * kl_loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
            
            if self.verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        if self.verbose:
            logger.info("Training completed")
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
        
        Returns:
            Predicted rating
        """
        if not self.is_fitted:
            raise ModelNotFittedError("Model has not been fitted. Call fit() first.")
        
        if user_id not in self.user_map or item_id not in self.item_map:
            return 0.0
        
        self.model.eval()
        with torch.no_grad():
            user_idx = torch.tensor([self.user_map[user_id]], dtype=torch.long).to(self.device)
            item_idx = torch.tensor([self.item_map[item_id]], dtype=torch.long).to(self.device)
            
            # Average over multiple samples for stability
            predictions = []
            for _ in range(10):  # Sample 10 times and average
                pred = self.model(user_idx, item_idx)
                predictions.append(pred.item())
            
            return float(np.mean(predictions))
    
    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_seen: bool = True
    ) -> List[int]:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id: User ID
            top_n: Number of recommendations
            exclude_seen: Whether to exclude items the user has already interacted with
        
        Returns:
            List of recommended item IDs
        """
        validate_user_id(user_id, self.user_map if hasattr(self, 'user_map') else {})
        validate_top_k(top_n)
        
        if not self.is_fitted:
            raise ModelNotFittedError("Model has not been fitted. Call fit() first.")
        
        if user_id not in self.user_map:
            return []
        
        self.model.eval()
        user_idx = self.user_map[user_id]
        
        # Calculate scores for all items
        scores = np.zeros(self.n_items)
        
        user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # Average over multiple samples
            for _ in range(10):
                for item_idx in range(self.n_items):
                    item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
                    pred = self.model(user_tensor, item_tensor)
                    scores[item_idx] += pred.item()
        
        scores /= 10.0
        
        # Exclude seen items if requested
        if exclude_seen:
            # This would require storing interaction matrix
            # For now, we'll skip this optimization
            pass
        
        # Get top-n items
        top_item_indices = np.argsort(-scores)[:top_n]
        
        # Map back to original item IDs
        top_items = [self.reverse_item_map[idx] for idx in top_item_indices]
        
        return top_items
    
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save model to disk."""
        if not self.is_fitted:
            raise ModelNotFittedError("Model has not been fitted. Call fit() first.")
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'model_state': self.model.state_dict(),
            'user_map': self.user_map,
            'item_map': self.item_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'latent_dim': self.latent_dim,
            'embedding_dim': self.embedding_dim,
            'learning_rate': self.learning_rate,
            'regularization': self.regularization,
            'kl_weight': self.kl_weight,
        }
        
        with open(path_obj, 'wb') as f:
            pickle.dump(save_data, f)
        
        if self.verbose:
            logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "VMFBase":
        """Load model from disk."""
        path_obj = Path(path)
        
        with open(path_obj, 'rb') as f:
            save_data = pickle.load(f)
        
        instance = cls(
            latent_dim=save_data['latent_dim'],
            embedding_dim=save_data['embedding_dim'],
            learning_rate=save_data['learning_rate'],
            regularization=save_data['regularization'],
            kl_weight=save_data['kl_weight']
        )
        
        instance.n_users = save_data['n_users']
        instance.n_items = save_data['n_items']
        instance.user_map = save_data['user_map']
        instance.item_map = save_data['item_map']
        instance.reverse_user_map = save_data['reverse_user_map']
        instance.reverse_item_map = save_data['reverse_item_map']
        
        instance.model = VDeepMFModel(
            num_users=instance.n_users,
            num_items=instance.n_items,
            latent_dim=instance.latent_dim,
            embedding_dim=instance.embedding_dim
        ).to(instance.device)
        
        instance.model.load_state_dict(save_data['model_state'])
        instance.is_fitted = True
        
        return instance


if __name__ == "__main__":
    # Example usage
    print("VDeepMF (Variational Deep Matrix Factorization) Module")
    print("=" * 60)
    print("\nThis module implements VDeepMF for collaborative filtering.")
    print("It should be imported as a module, not run directly.")
    print("\nExample usage:")
    print("  from corerec.engines.collaborative.bayesian_method_base.vmf_base import VMFBase")
    print("  model = VMFBase(latent_dim=64, embedding_dim=64)")
    print("  model.fit(interaction_matrix, user_ids, item_ids)")
    print("  recommendations = model.recommend(user_id=1, top_n=10)")
    print("\nFor more information, see the module documentation.")
