# corerec/engines/collaborative/rlrmc.py
import numpy as np
import torch
from typing import Union, List, Dict, Tuple, Optional, Any
from pathlib import Path
import os
import pickle
from scipy.sparse import csr_matrix

from .base_recommender import BaseRecommender
import logging

logger = logging.getLogger(__name__)


import pickle
from pathlib import Path
        
class RLRMC(BaseRecommender):
    """
    Riemannian Low-rank Matrix Completion for Collaborative Filtering.
    
    This implementation uses PyTorch for efficient training and inference.
    
    Parameters
    ----------
    rank : int, optional
        Rank of the matrix factorization, by default 10
    learning_rate : float, optional
        Learning rate for training, by default 0.01
    max_iter : int, optional
        Maximum number of iterations, by default 100
    tol : float, optional
        Tolerance for convergence, by default 1e-5
    reg_param : float, optional
        Regularization parameter, by default 0.1
    momentum : float, optional
        Momentum for gradient updates, by default 0.9
"""
    def __init__(
        self,
        rank: int = 10,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-5,
        reg_param: float = 0.1,
        momentum: float = 0.9,
        device: str = "cpu",
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        """Initialize RLRMC model."""
        self.rank = rank
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.reg_param = reg_param
        self.momentum = momentum
        self.device = torch.device(device)
        self.seed = seed
        self.verbose = verbose
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Model parameters will be initialized in fit()
        self.U = None  # User factors
        self.V = None  # Item factors
        
        # Momentum buffers
        self.U_momentum = None
        self.V_momentum = None
        
        # Mapping dictionaries
        self.user_to_index = {}
        self.index_to_user = {}
        self.item_to_index = {}
        self.index_to_item = {}
        
        # ----------------------------------------------------------------------
        # Training history
        # ----------------------------------------------------------------------
        self.train_errors = []
        self.is_fitted = False

    # ----------------------------------------------------------------------
    # mappings
    # ----------------------------------------------------------------------
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]):
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))
        
        self.user_to_index = {user: idx for idx, user in enumerate(unique_users)}
        self.index_to_user = {idx: user for idx, user in enumerate(unique_users)}
        self.item_to_index = {item: idx for idx, item in enumerate(unique_items)}
        self.index_to_item = {idx: item for idx, item in enumerate(unique_items)}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
    # ----------------------------------------------------------------------
    # Riemannian gradients
    # ----------------------------------------------------------------------
    def _riemannian_gradient(self, user_indices: torch.Tensor, item_indices: torch.Tensor, values: torch.Tensor):
        """Compute Riemannian gradients."""
        predictions = (self.U[user_indices] * self.V[item_indices]).sum(dim=1)
        errors = values - predictions
        
        # Compute gradients
        grad_U = torch.zeros_like(self.U)
        grad_V = torch.zeros_like(self.V)
        
        for i, (u_idx, v_idx, err) in enumerate(zip(user_indices, item_indices, errors)):
            grad_U[u_idx] += err * self.V[v_idx]
            grad_V[v_idx] += err * self.U[u_idx]
        
        # Add regularization
        grad_U += self.reg_param * self.U
        grad_V += self.reg_param * self.V
        
        return grad_U, grad_V
    
    # ----------------------------------------------------------------------
    # retraction
    # ----------------------------------------------------------------------
    def _retraction(self, X: torch.Tensor, grad: torch.Tensor, step_size: float):
        return X + step_size * grad
    
    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    def _init_model_parameters(self):
        """Initialize model parameters."""
        self.U = torch.randn(self.n_users, self.rank, device=self.device, requires_grad=False) * 0.1
        self.V = torch.randn(self.n_items, self.rank, device=self.device, requires_grad=False) * 0.1
        self.U_momentum = torch.zeros_like(self.U)
        self.V_momentum = torch.zeros_like(self.V)
    
    def _prepare_data(self, interaction_matrix: csr_matrix):
        """Prepare data from interaction matrix."""
        coo = interaction_matrix.tocoo()
        user_indices = torch.tensor(coo.row, dtype=torch.long, device=self.device)
        item_indices = torch.tensor(coo.col, dtype=torch.long, device=self.device)
        values = torch.tensor(coo.data, dtype=torch.float32, device=self.device)
        return user_indices, item_indices, values
    
    def _compute_loss(self, user_indices: torch.Tensor, item_indices: torch.Tensor, values: torch.Tensor):
        """Compute reconstruction loss."""
        predictions = (self.U[user_indices] * self.V[item_indices]).sum(dim=1)
        mse_loss = torch.mean((predictions - values) ** 2)
        reg_loss = self.reg_param * (torch.norm(self.U) ** 2 + torch.norm(self.V) ** 2)
        return mse_loss.item() + reg_loss.item()
    
    # ----------------------------------------------------------------------
    # Train the RLRMC model.
    # ----------------------------------------------------------------------
    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]):
        """Fit the model on interaction matrix."""
        self._create_mappings(user_ids, item_ids)
        self._init_model_parameters()
        
        user_indices, item_indices, values = self._prepare_data(interaction_matrix)
        
        prev_loss = float("inf")
        
        for iteration in range(self.max_iter):
            # Compute loss
            loss = self._compute_loss(user_indices, item_indices, values)
            self.train_errors.append(loss)
            
            # Check convergence
            if prev_loss - loss < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1} with loss {loss:.6f}")
                break
            
            prev_loss = loss
            
            # Compute Riemannian gradients
            rgrad_U, rgrad_V = self._riemannian_gradient(user_indices, item_indices, values)
            
            # Update momentum buffers
            self.U_momentum = self.momentum * self.U_momentum + (1 - self.momentum) * rgrad_U
            self.V_momentum = self.momentum * self.V_momentum + (1 - self.momentum) * rgrad_V
            
            # Update parameters using retraction
            self.U = self._retraction(self.U, self.U_momentum, -self.learning_rate)
            self.V = self._retraction(self.V, self.V_momentum, -self.learning_rate)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Loss: {loss:.6f}")
        
        self.is_fitted = True

    # ----------------------------------------------------------------------
    # predict
    # ----------------------------------------------------------------------
    def predict(self, user_ids: List[int], item_ids: List[int]):
        """Predict ratings for user-item pairs."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        user_indices = torch.tensor([self.user_to_index[user] for user in user_ids], device=self.device)
        item_indices = torch.tensor([self.item_to_index[item] for item in item_ids], device=self.device)
        return (self.U[user_indices] @ self.V[item_indices].t()).diag().cpu().numpy()
    
    # ----------------------------------------------------------------------
    # recommend
    # ----------------------------------------------------------------------
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """Generate recommendations for a user."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before recommendation")
        if user_id not in self.user_to_index:
            return []
        
        user_idx = self.user_to_index[user_id]
        user_vector = self.U[user_idx].cpu().numpy()
        
        # Compute scores for all items
        item_vectors = self.V.cpu().numpy()
        scores = np.dot(item_vectors, user_vector)
        
        # Get top-n items
        top_indices = np.argsort(-scores)[:top_n]
        
        # Map indices back to original item IDs
        recommendations = [self.index_to_item[idx] for idx in top_indices]
        
        return recommendations

    # ----------------------------------------------------------------------
    # save model
    # ----------------------------------------------------------------------
    def save_model(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self.__dict__, f)