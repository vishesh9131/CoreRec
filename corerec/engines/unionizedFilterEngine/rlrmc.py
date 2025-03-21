# corerec/engines/unionizedFilterEngine/rlrmc.py
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
import os
import pickle
from scipy.sparse import csr_matrix

from .base_recommender import BaseRecommender


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
    device : str, optional
        Device to use for training ('cpu' or 'cuda'), by default 'cpu'
    seed : int, optional
        Random seed for reproducibility, by default 42
    verbose : bool, optional
        Whether to print progress during training, by default False
    """
    
    def __init__(
        self,
        rank: int = 10,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-5,
        reg_param: float = 0.1,
        momentum: float = 0.9,
        device: str = 'cpu',
        seed: int = 42,
        verbose: bool = False
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
        
        # Training history
        self.train_errors = []
    
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Create mappings between original IDs and internal indices.
        
        Parameters
        ----------
        user_ids : List[int]
            List of user IDs
        item_ids : List[int]
            List of item IDs
        """
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))
        
        self.user_to_index = {user: idx for idx, user in enumerate(unique_users)}
        self.index_to_user = {idx: user for idx, user in enumerate(unique_users)}
        self.item_to_index = {item: idx for idx, item in enumerate(unique_items)}
        self.index_to_item = {idx: item for idx, item in enumerate(unique_items)}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
    
    def _init_model_parameters(self) -> None:
        """Initialize model parameters."""
        # Initialize factors with random values on the Stiefel manifold
        # For Stiefel manifold, we need orthonormal matrices
        U_init = torch.randn(self.n_users, self.rank, device=self.device)
        V_init = torch.randn(self.n_items, self.rank, device=self.device)
        
        # Orthonormalize using QR decomposition
        self.U, _ = torch.linalg.qr(U_init)
        self.V, _ = torch.linalg.qr(V_init)
        
        # Initialize momentum buffers
        self.U_momentum = torch.zeros_like(self.U)
        self.V_momentum = torch.zeros_like(self.V)
    
    def _prepare_data(self, interaction_matrix: csr_matrix) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert sparse interaction matrix to tensors of indices and values.
        
        Parameters
        ----------
        interaction_matrix : csr_matrix
            User-item interaction matrix
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            User indices, item indices, and values
        """
        # Get non-zero entries
        user_indices, item_indices = interaction_matrix.nonzero()
        values = interaction_matrix.data
        
        # Convert to tensors
        user_indices_tensor = torch.tensor(user_indices, dtype=torch.long, device=self.device)
        item_indices_tensor = torch.tensor(item_indices, dtype=torch.long, device=self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        
        return user_indices_tensor, item_indices_tensor, values_tensor
    
    def _compute_loss(
        self, 
        user_indices: torch.Tensor, 
        item_indices: torch.Tensor, 
        values: torch.Tensor
    ) -> float:
        """
        Compute the loss function.
        
        Parameters
        ----------
        user_indices : torch.Tensor
            User indices
        item_indices : torch.Tensor
            Item indices
        values : torch.Tensor
            Interaction values
            
        Returns
        -------
        float
            Loss value
        """
        # Compute predictions
        u_factors = self.U[user_indices]
        v_factors = self.V[item_indices]
        predictions = torch.sum(u_factors * v_factors, dim=1)
        
        # Compute MSE loss
        mse_loss = torch.mean((predictions - values) ** 2)
        
        # Add regularization
        reg_loss = self.reg_param * (torch.norm(self.U, 'fro') ** 2 + torch.norm(self.V, 'fro') ** 2)
        
        # Total loss
        total_loss = mse_loss + reg_loss
        
        return total_loss.item()
    
    def _riemannian_gradient(
        self, 
        user_indices: torch.Tensor, 
        item_indices: torch.Tensor, 
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Riemannian gradients for U and V.
        
        Parameters
        ----------
        user_indices : torch.Tensor
            User indices
        item_indices : torch.Tensor
            Item indices
        values : torch.Tensor
            Interaction values
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Riemannian gradients for U and V
        """
        # Compute predictions
        u_factors = self.U[user_indices]
        v_factors = self.V[item_indices]
        predictions = torch.sum(u_factors * v_factors, dim=1)
        
        # Compute errors
        errors = predictions - values
        
        # Initialize Euclidean gradients
        grad_U = torch.zeros_like(self.U)
        grad_V = torch.zeros_like(self.V)
        
        # Compute gradients using scatter_add
        for i in range(len(user_indices)):
            u_idx = user_indices[i]
            v_idx = item_indices[i]
            error = errors[i]
            
            grad_U[u_idx] += error * v_factors[i] + self.reg_param * u_factors[i]
            grad_V[v_idx] += error * u_factors[i] + self.reg_param * v_factors[i]
        
        # Convert Euclidean gradients to Riemannian gradients
        # For Stiefel manifold: grad_R = grad_E - U @ (U.T @ grad_E + grad_E.T @ U) / 2
        rgrad_U = grad_U - self.U @ (self.U.t() @ grad_U + grad_U.t() @ self.U) / 2
        rgrad_V = grad_V - self.V @ (self.V.t() @ grad_V + grad_V.t() @ self.V) / 2
        
        return rgrad_U, rgrad_V
    
    def _retraction(self, X: torch.Tensor, grad: torch.Tensor, step_size: float) -> torch.Tensor:
        """
        Perform retraction on the Stiefel manifold.
        
        Parameters
        ----------
        X : torch.Tensor
            Current point on the manifold
        grad : torch.Tensor
            Riemannian gradient
        step_size : float
            Step size
            
        Returns
        -------
        torch.Tensor
            New point on the manifold
        """
        # Compute the new point using Cayley transform
        A = step_size * (X.t() @ grad - grad.t() @ X) / 2
        I = torch.eye(A.shape[0], device=self.device)
        Q = torch.inverse(I + A) @ (I - A)
        X_new = X @ Q
        
        return X_new
    
    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Train the RLRMC model.
        
        Parameters
        ----------
        interaction_matrix : csr_matrix
            User-item interaction matrix
        user_ids : List[int]
            List of user IDs
        item_ids : List[int]
            List of item IDs
        """
        # Create mappings
        self._create_mappings(user_ids, item_ids)
        
        # Initialize model parameters
        self._init_model_parameters()
        
        # Prepare data
        user_indices, item_indices, values = self._prepare_data(interaction_matrix)
        
        # Training loop
        prev_loss = float('inf')
        
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
# corerec/engines/unionizedFilterEngine/rlrmc.py (continued)
            # Update parameters using retraction
            self.U = self._retraction(self.U, self.U_momentum, -self.learning_rate)
            self.V = self._retraction(self.V, self.V_momentum, -self.learning_rate)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Loss: {loss:.6f}")
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict the rating for a user-item pair.
        
        Parameters
        ----------
        user_id : int
            User ID
        item_id : int
            Item ID
            
        Returns
        -------
        float
            Predicted rating
        """
        if user_id not in self.user_to_index or item_id not in self.item_to_index:
            # Return default prediction for unknown users or items
            return 0.0
        
        user_idx = self.user_to_index[user_id]
        item_idx = self.item_to_index[item_id]
        
        # Compute prediction
        prediction = torch.sum(self.U[user_idx] * self.V[item_idx]).item()
        
        return prediction
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """
        Recommend items for a user.
        
        Parameters
        ----------
        user_id : int
            User ID
        top_n : int, optional
            Number of recommendations to generate, by default 10
        exclude_seen : bool, optional
            Whether to exclude already seen items, by default True
            
        Returns
        -------
        List[int]
            List of recommended item IDs
        """
        if user_id not in self.user_to_index:
            # Return empty list for unknown users
            return []
        
        user_idx = self.user_to_index[user_id]
        
        # Compute predictions for all items
        user_factors = self.U[user_idx]
        predictions = torch.matmul(self.V, user_factors).cpu().numpy()
        
        # Create a list of (item_idx, prediction) tuples
        item_predictions = [(i, predictions[i]) for i in range(self.n_items)]
        
        # Exclude seen items if requested
        if exclude_seen:
            # Get the indices of items the user has interacted with
            seen_items = set()
            for item_id, item_idx in self.item_to_index.items():
                # Check if the user has interacted with this item
                # This is a placeholder - in a real implementation, you would check the interaction matrix
                if self.predict(user_id, item_id) > 0:
                    seen_items.add(item_idx)
            
            # Filter out seen items
            item_predictions = [(i, p) for i, p in item_predictions if i not in seen_items]
        
        # Sort by prediction in descending order and take top_n
        item_predictions.sort(key=lambda x: x[1], reverse=True)
        top_item_indices = [i for i, _ in item_predictions[:top_n]]
        
        # Convert indices back to original item IDs
        recommended_items = [self.index_to_item[idx] for idx in top_item_indices]
        
        return recommended_items
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        model_data = {
            'rank': self.rank,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'reg_param': self.reg_param,
            'momentum': self.momentum,
            'seed': self.seed,
            'verbose': self.verbose,
            'U': self.U.cpu().numpy() if self.U is not None else None,
            'V': self.V.cpu().numpy() if self.V is not None else None,
            'user_to_index': self.user_to_index,
            'index_to_user': self.index_to_user,
            'item_to_index': self.item_to_index,
            'index_to_item': self.index_to_item,
            'n_users': self.n_users if hasattr(self, 'n_users') else None,
            'n_items': self.n_items if hasattr(self, 'n_items') else None,
            'train_errors': self.train_errors
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'RLRMC':
        """
        Load a model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
            
        Returns
        -------
        RLRMC
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance with the saved hyperparameters
        instance = cls(
            rank=model_data['rank'],
            learning_rate=model_data['learning_rate'],
            max_iter=model_data['max_iter'],
            tol=model_data['tol'],
            reg_param=model_data['reg_param'],
            momentum=model_data['momentum'],
            seed=model_data['seed'],
            verbose=model_data['verbose']
        )
        
        # Restore model state
        if model_data['U'] is not None:
            instance.U = torch.tensor(model_data['U'], device=instance.device)
            instance.V = torch.tensor(model_data['V'], device=instance.device)
            
            # Initialize momentum buffers
            instance.U_momentum = torch.zeros_like(instance.U)
            instance.V_momentum = torch.zeros_like(instance.V)
        
        # Restore mappings
        instance.user_to_index = model_data['user_to_index']
        instance.index_to_user = model_data['index_to_user']
        instance.item_to_index = model_data['item_to_index']
        instance.index_to_item = model_data['index_to_item']
        
        # Restore other attributes
        if 'n_users' in model_data and model_data['n_users'] is not None:
            instance.n_users = model_data['n_users']
        if 'n_items' in model_data and model_data['n_items'] is not None:
            instance.n_items = model_data['n_items']
        
        instance.train_errors = model_data['train_errors']
        
        return instance
    
    def _project_to_stiefel(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project a matrix onto the Stiefel manifold.
        
        Parameters
        ----------
        X : torch.Tensor
            Matrix to project
            
        Returns
        -------
        torch.Tensor
            Projected matrix
        """
        # Perform QR decomposition
        Q, R = torch.linalg.qr(X)
        
        # Ensure the diagonal of R is positive
        d = torch.diag(torch.sign(torch.diag(R)))
        Q = Q @ d
        
        return Q
    
    def _parallel_transport(self, X: torch.Tensor, Y: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport a tangent vector from X to Y on the Stiefel manifold.
        
        Parameters
        ----------
        X : torch.Tensor
            Source point on the manifold
        Y : torch.Tensor
            Target point on the manifold
        V : torch.Tensor
            Tangent vector at X
            
        Returns
        -------
        torch.Tensor
            Transported tangent vector at Y
        """
        # Compute the projection of V onto the tangent space at X
        V_proj = V - X @ (X.t() @ V + V.t() @ X) / 2
        
        # Compute the projection of V_proj onto the tangent space at Y
        V_transported = V_proj - Y @ (Y.t() @ V_proj + V_proj.t() @ Y) / 2
        
        return V_transported
    
    def fit_from_interactions(self, user_ids: List[int], item_ids: List[int], ratings: Optional[List[float]] = None) -> None:
        """
        Train the model from user-item interactions.
        
        Parameters
        ----------
        user_ids : List[int]
            List of user IDs
        item_ids : List[int]
            List of item IDs
        ratings : Optional[List[float]], optional
            List of ratings, by default None
        """
        # Create mappings
        self._create_mappings(user_ids, item_ids)
        
        # Create interaction matrix
        if ratings is None:
            # If no ratings are provided, use binary interactions
            ratings = [1.0] * len(user_ids)
        
        # Convert to internal indices
        user_indices = [self.user_to_index[uid] for uid in user_ids]
        item_indices = [self.item_to_index[iid] for iid in item_ids]
        
        # Create sparse matrix
        interaction_matrix = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(self.n_users, self.n_items)
        )
        
        # Train the model
        self.fit(interaction_matrix, user_ids, item_ids)