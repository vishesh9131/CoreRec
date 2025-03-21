import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Union, Any
from scipy.sparse import csr_matrix
import os
import pickle
from ..base_recommender import BaseRecommender


class MultinomialVAENetwork(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int] = [600, 200], 
        latent_dim: int = 64, 
        dropout: float = 0.5
    ):
        super(MultinomialVAENetwork, self).__init__()
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.Tanh())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.Tanh())
            decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.output_layer = nn.Linear(hidden_dims[0], input_dim)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder(z)
        return self.output_layer(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class MultinomialVAE(BaseRecommender):
    """
    Multinomial Variational Autoencoder for collaborative filtering.
    
    This model uses a variational autoencoder with a multinomial likelihood
    for modeling implicit feedback data.
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [600, 200],
        latent_dim: int = 64,
        dropout: float = 0.5,
        lr: float = 0.001,
        batch_size: int = 100,
        epochs: int = 100,
        beta: float = 0.2,
        anneal_steps: int = 200000,
        anneal_cap: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        random_state: Optional[int] = None
    ):
        """
        Initialize the Multinomial VAE model.
        
        Args:
            hidden_dims: Dimensions of hidden layers
            latent_dim: Dimension of latent space
            dropout: Dropout rate
            lr: Learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            beta: Weight for KL divergence term
            anneal_steps: Number of steps for annealing beta
            anneal_cap: Maximum value for beta after annealing
            device: Device to run the model on ('cpu' or 'cuda')
            random_state: Random seed for reproducibility
        """
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta = beta
        self.anneal_steps = anneal_steps
        self.anneal_cap = anneal_cap
        self.device = device
        
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
        
        self.model = None
        self.optimizer = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Create mappings between original IDs and internal indices.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
        """
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))
        
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.idx_to_item = {idx: item_id for item_id, idx in self.item_to_idx.items()}
    
    def _prepare_data(self, user_ids: List[int], item_ids: List[int], ratings: Optional[List[float]] = None) -> torch.Tensor:
        """
        Prepare user-item interaction data for training.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            ratings: Optional list of ratings
            
        Returns:
            Sparse tensor of user-item interactions
        """
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        
        # Create interaction matrix
        interaction_matrix = np.zeros((n_users, n_items))
        
        for i, (user_id, item_id) in enumerate(zip(user_ids, item_ids)):
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[item_id]
            
            if ratings is not None:
                interaction_matrix[user_idx, item_idx] = ratings[i]
            else:
                interaction_matrix[user_idx, item_idx] = 1.0
        
        return torch.FloatTensor(interaction_matrix).to(self.device)
    
    def _compute_loss(self, x: torch.Tensor, x_pred: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Compute the loss function for the Multinomial VAE.
        
        Args:
            x: Input data
            x_pred: Reconstructed data
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            beta: Weight for KL divergence term
            
        Returns:
            Total loss
        """
        # Multinomial log likelihood
        log_softmax_var = F.log_softmax(x_pred, dim=1)
        neg_ll = -torch.sum(x * log_softmax_var, dim=1)
        
        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        return torch.mean(neg_ll + beta * kl_div)
    
    def fit(self, user_ids: List[int], item_ids: List[int], ratings: Optional[List[float]] = None) -> None:
        """
        Train the Multinomial VAE model.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            ratings: Optional list of ratings
        """
        # Create mappings and prepare data
        self._create_mappings(user_ids, item_ids)
        data = self._prepare_data(user_ids, item_ids, ratings)
        
        # Initialize model
        n_items = len(self.item_to_idx)
        self.model = MultinomialVAENetwork(
            input_dim=n_items,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        self.model.train()
        n_users = len(self.user_to_idx)
        update_count = 0
        
        for epoch in range(self.epochs):
            # Shuffle users
            idx_list = np.arange(n_users)
            np.random.shuffle(idx_list)
            
            total_loss = 0.0
            
            for batch_idx in range(0, n_users, self.batch_size):
                batch_indices = idx_list[batch_idx:min(batch_idx + self.batch_size, n_users)]
                batch_data = data[batch_indices]
                
                # Normalize input data (important for multinomial likelihood)
                batch_data_normalized = F.normalize(batch_data, p=1, dim=1)
                
                # Forward pass
                self.optimizer.zero_grad()
                x_pred, mu, logvar = self.model(batch_data_normalized)
                
                # Compute beta for annealing
                if self.anneal_steps > 0:
                    anneal = min(self.anneal_cap, 1. * update_count / self.anneal_steps)
                    update_count += 1
                else:
                    anneal = self.beta
                
                # Compute loss
                loss = self._compute_loss(batch_data_normalized, x_pred, mu, logvar, anneal)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(batch_indices)
            
            avg_loss = total_loss / n_users
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            top_n: Number of recommendations to generate
            exclude_seen: Whether to exclude items the user has already interacted with
            
        Returns:
            List of recommended item IDs
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        if user_id not in self.user_to_idx:
            # Return empty list for unknown users
            return []
        
        user_idx = self.user_to_idx[user_id]
        n_items = len(self.item_to_idx)
        
        # Get user's interaction history
        user_data = torch.zeros(1, n_items, device=self.device)
        
        # Find items the user has interacted with
        seen_items = set()
        for i, (uid, iid) in enumerate(zip(self.user_ids, self.item_ids)):
            if uid == user_id:
                item_idx = self.item_to_idx[iid]
                user_data[0, item_idx] = 1.0
                seen_items.add(item_idx)
        
        # Normalize input data
        user_data_normalized = F.normalize(user_data, p=1, dim=1)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            x_pred, _, _ = self.model(user_data_normalized)
            scores = F.softmax(x_pred, dim=1).cpu().numpy()[0]
            
            # Set scores of seen items to 0 if exclude_seen is True
            if exclude_seen:
                for item_idx in seen_items:
                    scores[item_idx] = 0
            
            # Get top-n item indices
            top_indices = np.argsort(-scores)[:top_n]
            
            # Convert indices back to original item IDs
            recommended_items = [self.idx_to_item[idx] for idx in top_indices]
            
            return recommended_items
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'hidden_dims': self.hidden_dims,
            'latent_dim': self.latent_dim,
            'dropout': self.dropout,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'beta': self.beta,
            'anneal_steps': self.anneal_steps,
            'anneal_cap': self.anneal_cap,
            'device': self.device
        }
        
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MultinomialVAE':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded MultinomialVAE model
        """
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        # Create a new instance with the saved hyperparameters
        instance = cls(
            hidden_dims=model_state['hidden_dims'],
            latent_dim=model_state['latent_dim'],
            dropout=model_state['dropout'],
            lr=model_state['lr'],
            batch_size=model_state['batch_size'],
            epochs=model_state['epochs'],
            beta=model_state['beta'],
            anneal_steps=model_state['anneal_steps'],
            anneal_cap=model_state['anneal_cap'],
            device=model_state['device']
        )
        
        # Restore mappings
        instance.user_to_idx = model_state['user_to_idx']
        instance.item_to_idx = model_state['item_to_idx']
        instance.idx_to_item = model_state['idx_to_item']
        
        # Initialize and restore model
        n_items = len(instance.item_to_idx)
        instance.model = MultinomialVAENetwork(
            input_dim=n_items,
            hidden_dims=instance.hidden_dims,
            latent_dim=instance.latent_dim,
            dropout=instance.dropout
        ).to(instance.device)
        
        instance.model.load_state_dict(model_state['model_state_dict'])
        
        # Initialize and restore optimizer
        instance.optimizer = optim.Adam(instance.model.parameters(), lr=instance.lr)
        instance.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        
        return instance
    
    def _compute_loss(self, x: torch.Tensor, x_pred: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, anneal: float) -> torch.Tensor:
        """
        Compute the loss function for the Multinomial VAE.
        
        Args:
            x: Input data
            x_pred: Reconstructed data
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            anneal: Annealing factor for KL divergence
            
        Returns:
            Total loss
        """
        # Multinomial log likelihood
        log_softmax_var = F.log_softmax(x_pred, dim=-1)
        neg_ll = -torch.sum(x * log_softmax_var, dim=-1).mean()
        
        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        # Total loss
        loss = neg_ll + anneal * kl_div
        
        return loss
    
    def _get_user_item_matrix(self) -> torch.Tensor:
        """
        Create a user-item interaction matrix.
        
        Returns:
            User-item interaction matrix as a torch tensor
        """
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        
        # Create a sparse matrix first
        rows, cols = [], []
        for user_id, item_id in zip(self.user_ids, self.item_ids):
            if user_id in self.user_to_idx and item_id in self.item_to_idx:
                rows.append(self.user_to_idx[user_id])
                cols.append(self.item_to_idx[item_id])
        
        data = np.ones(len(rows))
        sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        # Convert to dense tensor
        dense_matrix = torch.FloatTensor(sparse_matrix.toarray()).to(self.device)
        
        return dense_matrix