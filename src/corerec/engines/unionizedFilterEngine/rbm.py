# corerec/engines/unionizedFilterEngine/rbm.py
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
import os
import pickle
from scipy.sparse import csr_matrix

from .base_recommender import BaseRecommender


class RBM(BaseRecommender):
    """
    Restricted Boltzmann Machine for Collaborative Filtering.
    
    This implementation uses PyTorch for efficient training and inference.
    
    Parameters
    ----------
    n_hidden : int, optional
        Number of hidden units, by default 100
    learning_rate : float, optional
        Learning rate for training, by default 0.01
    batch_size : int, optional
        Batch size for training, by default 100
    n_epochs : int, optional
        Number of training epochs, by default 20
    k : int, optional
        Number of Gibbs sampling steps for contrastive divergence, by default 1
    momentum : float, optional
        Momentum for gradient updates, by default 0.5
    weight_decay : float, optional
        Weight decay for regularization, by default 0.0001
    device : str, optional
        Device to use for training ('cpu' or 'cuda'), by default 'cpu'
    seed : int, optional
        Random seed for reproducibility, by default 42
    verbose : bool, optional
        Whether to print progress during training, by default False
    """
    
    def __init__(
        self,
        n_hidden: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 100,
        n_epochs: int = 20,
        k: int = 1,
        momentum: float = 0.5,
        weight_decay: float = 0.0001,
        device: str = 'cpu',
        seed: int = 42,
        verbose: bool = False
    ) -> None:
        """Initialize RBM model."""
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.k = k  # CD-k (contrastive divergence steps)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = torch.device(device)
        self.seed = seed
        self.verbose = verbose
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Model parameters will be initialized in fit()
        self.W = None  # Weights
        self.v_bias = None  # Visible bias
        self.h_bias = None  # Hidden bias
        
        # Momentum buffers
        self.W_momentum = None
        self.v_bias_momentum = None
        self.h_bias_momentum = None
        
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
        # Initialize weights and biases
        self.W = torch.randn(self.n_items, self.n_hidden, device=self.device) * 0.01
        self.v_bias = torch.zeros(self.n_items, device=self.device)
        self.h_bias = torch.zeros(self.n_hidden, device=self.device)
        
        # Initialize momentum buffers
        self.W_momentum = torch.zeros_like(self.W)
        self.v_bias_momentum = torch.zeros_like(self.v_bias)
        self.h_bias_momentum = torch.zeros_like(self.h_bias)
    
    def _prepare_data(self, interaction_matrix: csr_matrix) -> torch.Tensor:
        """
        Convert sparse interaction matrix to binary tensor.
        
        Parameters
        ----------
        interaction_matrix : csr_matrix
            User-item interaction matrix
            
        Returns
        -------
        torch.Tensor
            Binary tensor of user-item interactions
        """
        # Convert to binary interactions (1 if interaction exists, 0 otherwise)
        binary_matrix = (interaction_matrix > 0).astype(np.float32)
        
        # Convert to dense tensor
        return torch.tensor(binary_matrix.toarray(), dtype=torch.float32, device=self.device)
    
    def _sample_hidden(self, visible: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units given visible units.
        
        Parameters
        ----------
        visible : torch.Tensor
            Visible units
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Hidden probabilities and samples
        """
        # Calculate activation probabilities: p(h|v) = sigmoid(v * W + h_bias)
        hidden_probs = torch.sigmoid(torch.matmul(visible, self.W) + self.h_bias)
        # Sample binary states
        hidden_samples = torch.bernoulli(hidden_probs)
        return hidden_probs, hidden_samples
    
    def _sample_visible(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible units given hidden units.
        
        Parameters
        ----------
        hidden : torch.Tensor
            Hidden units
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Visible probabilities and samples
        """
        # Calculate activation probabilities: p(v|h) = sigmoid(h * W.T + v_bias)
        visible_probs = torch.sigmoid(torch.matmul(hidden, self.W.t()) + self.v_bias)
        # Sample binary states
        visible_samples = torch.bernoulli(visible_probs)
        return visible_probs, visible_samples
    
    def _contrastive_divergence(self, v_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform contrastive divergence to compute gradients.
        
        Parameters
        ----------
        v_data : torch.Tensor
            Input data (visible units)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Gradients for weights, visible bias, and hidden bias
        """
        # Positive phase
        h_prob_data, h_sample_data = self._sample_hidden(v_data)
        
        # Negative phase (CD-k)
        v_prob_model = v_data.clone()
        h_prob_model = h_prob_data.clone()
        
        for _ in range(self.k):
            _, h_sample_model = self._sample_hidden(v_prob_model)
            v_prob_model, v_sample_model = self._sample_visible(h_sample_model)
            h_prob_model, _ = self._sample_hidden(v_sample_model)
        
        # Compute gradients
        dW = (torch.matmul(v_data.t(), h_prob_data) - torch.matmul(v_prob_model.t(), h_prob_model)).t()
        dv_bias = torch.sum(v_data - v_prob_model, dim=0)
        dh_bias = torch.sum(h_prob_data - h_prob_model, dim=0)
        
        # Add weight decay
        dW -= self.weight_decay * self.W
        
        return dW, dv_bias, dh_bias
    
    def _update_parameters(self, dW: torch.Tensor, dv_bias: torch.Tensor, dh_bias: torch.Tensor) -> None:
        """
        Update model parameters using gradients and momentum.
        
        Parameters
        ----------
        dW : torch.Tensor
            Weight gradients
        dv_bias : torch.Tensor
            Visible bias gradients
        dh_bias : torch.Tensor
            Hidden bias gradients
        """
        # Update momentum buffers
        self.W_momentum = self.momentum * self.W_momentum + self.learning_rate * dW
        self.v_bias_momentum = self.momentum * self.v_bias_momentum + self.learning_rate * dv_bias
        self.h_bias_momentum = self.momentum * self.h_bias_momentum + self.learning_rate * dh_bias
        
        # Update parameters
        self.W += self.W_momentum
        self.v_bias += self.v_bias_momentum
        self.h_bias += self.h_bias_momentum
    
    def _compute_reconstruction_error(self, v_data: torch.Tensor) -> float:
        """
        Compute reconstruction error.
        
        Parameters
        ----------
        v_data : torch.Tensor
            Input data (visible units)
            
        Returns
        -------
        float
            Reconstruction error
        """
        h_prob, h_sample = self._sample_hidden(v_data)
        v_prob, _ = self._sample_visible(h_sample)
        error = torch.mean(torch.sum((v_data - v_prob) ** 2, dim=1))
        return error.item()
    
    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Train the RBM model.
        
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
        data = self._prepare_data(interaction_matrix)
        n_samples = data.shape[0]
        
        # Training loop
        for epoch in range(self.n_epochs):
            epoch_error = 0.0
            
            # Create random batches
            indices = torch.randperm(n_samples)
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_data = data[batch_indices]
                
                # Compute gradients using contrastive divergence
                dW, dv_bias, dh_bias = self._contrastive_divergence(batch_data)
                
                # Update parameters
                self._update_parameters(dW, dv_bias, dh_bias)
                
                # Compute reconstruction error
                batch_error = self._compute_reconstruction_error(batch_data)
                epoch_error += batch_error * len(batch_indices)
            
            # Average error over all samples
            epoch_error /= n_samples
            self.train_errors.append(epoch_error)
            
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Reconstruction Error: {epoch_error:.4f}")
    
    def _predict_user(self, user_id: int) -> torch.Tensor:
        """
        Generate predictions for a user.
        
        Parameters
        ----------
        user_id : int
            User ID
            
        Returns
        -------
        torch.Tensor
            Predicted ratings for all items
        """
        if user_id not in self.user_to_index:
            raise ValueError(f"User ID {user_id} not found in training data")
        
        user_idx = self.user_to_index[user_id]
        
        # Get user's interaction vector
        user_vector = torch.zeros(self.n_items, device=self.device)
        
        # Compute hidden activations
        h_prob, _ = self._sample_hidden(user_vector.unsqueeze(0))
        
        # Compute visible activations (predictions)
        v_prob, _ = self._sample_visible(h_prob)
        
        return v_prob.squeeze(0)
    
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
            raise ValueError(f"User ID {user_id} not found in training data")
        
        # Get user's predictions
        predictions = self._predict_user(user_id)
        
        # Convert to numpy for easier manipulation
        predictions = predictions.cpu().numpy()
        
        # Get user index
        user_idx = self.user_to_index[user_id]
        
        # Exclude seen items if requested
        if exclude_seen:
            # Create a mask for seen items
            seen_mask = np.zeros(self.n_items, dtype=bool)
            for item_id, item_idx in self.item_to_index.items():
                # Check if the user has interacted with this item
                if item_idx < len(seen_mask):
                    seen_mask[item_idx] = True
            
            # Set predictions for seen items to -inf
            predictions[seen_mask] = -np.inf
        
        # Get top-n item indices
        top_item_indices = np.argsort(predictions)[::-1][:top_n]
        
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
            'n_hidden': self.n_hidden,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'k': self.k,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'seed': self.seed,
            'verbose': self.verbose,
            'W': self.W.cpu().numpy() if self.W is not None else None,
            'v_bias': self.v_bias.cpu().numpy() if self.v_bias is not None else None,
            'h_bias': self.h_bias.cpu().numpy() if self.h_bias is not None else None,
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
    def load_model(cls, filepath: str) -> 'RBM':
        """
        Load a model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
            
        Returns
        -------
        RBM
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance with the saved hyperparameters
        instance = cls(
            n_hidden=model_data['n_hidden'],
            learning_rate=model_data['learning_rate'],
            batch_size=model_data['batch_size'],
            n_epochs=model_data['n_epochs'],
            k=model_data['k'],
            momentum=model_data['momentum'],
            weight_decay=model_data['weight_decay'],
            seed=model_data['seed'],
            verbose=model_data['verbose']
        )
        
        # Restore model state
        if model_data['W'] is not None:
            instance.W = torch.tensor(model_data['W'], device=instance.device)
            instance.v_bias = torch.tensor(model_data['v_bias'], device=instance.device)
            instance.h_bias = torch.tensor(model_data['h_bias'], device=instance.device)
            
            # Initialize momentum buffers
            instance.W_momentum = torch.zeros_like(instance.W)
            instance.v_bias_momentum = torch.zeros_like(instance.v_bias)
            instance.h_bias_momentum = torch.zeros_like(instance.h_bias)
        
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