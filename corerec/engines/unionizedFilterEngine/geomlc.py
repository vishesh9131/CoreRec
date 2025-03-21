import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Union
import os
import pickle
from scipy.sparse import csr_matrix

from .base_recommender import BaseRecommender


class GeoMLC(BaseRecommender):
    """
    Geometric Matrix Completion for Recommendation Systems
    
    This model uses Riemannian geometry to model user-item interactions on a manifold,
    capturing the non-Euclidean nature of user preferences and item relationships.
    
    Parameters:
    -----------
    n_factors : int
        Number of latent factors
    learning_rate : float
        Learning rate for optimization
    regularization : float
        Regularization parameter
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    manifold_type : str
        Type of manifold to use ('hyperbolic', 'spherical', or 'euclidean')
    curvature : float
        Curvature parameter for the manifold (negative for hyperbolic, positive for spherical)
    init_range : float
        Range for random initialization of parameters
    device : str
        Device to use for computation ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        n_factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        n_epochs: int = 20,
        batch_size: int = 256,
        manifold_type: str = 'hyperbolic',
        curvature: float = -1.0,
        init_range: float = 0.1,
        device: str = 'cpu'
    ) -> None:
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.manifold_type = manifold_type
        self.curvature = curvature
        self.init_range = init_range
        self.device = torch.device(device)
        
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_bias = None
        
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
        self.model = None
        self.optimizer = None
        
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Create mappings between original IDs and internal indices
        
        Parameters:
        -----------
        user_ids : List[int]
            List of user IDs
        item_ids : List[int]
            List of item IDs
        """
        unique_user_ids = sorted(set(user_ids))
        unique_item_ids = sorted(set(item_ids))
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_item_ids)}
        
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        
        self.n_users = len(unique_user_ids)
        self.n_items = len(unique_item_ids)
        
    def _init_model(self) -> None:
        """Initialize the GeoMLC model with PyTorch"""
        self.model = GeoMLCModel(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            manifold_type=self.manifold_type,
            curvature=self.curvature,
            init_range=self.init_range
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.regularization
        )
        
    def _prepare_data(self, user_ids: List[int], item_ids: List[int], ratings: List[float]) -> torch.utils.data.DataLoader:
        """
        Prepare data for training
        
        Parameters:
        -----------
        user_ids : List[int]
            List of user IDs
        item_ids : List[int]
            List of item IDs
        ratings : List[float]
            List of ratings
            
        Returns:
        --------
        torch.utils.data.DataLoader
            DataLoader for training
        """
        # Map original IDs to internal indices
        mapped_user_ids = [self.user_mapping[user_id] for user_id in user_ids]
        mapped_item_ids = [self.item_mapping[item_id] for item_id in item_ids]
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(mapped_user_ids),
            torch.LongTensor(mapped_item_ids),
            torch.FloatTensor(ratings)
        )
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        return dataloader
    
    def fit(self, user_ids: List[int], item_ids: List[int], ratings: List[float]) -> None:
        """
        Train the GeoMLC model
        
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
        
        # Initialize model
        self._init_model()
        
        # Prepare data
        dataloader = self._prepare_data(user_ids, item_ids, ratings)
        
        # Train model
        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for batch_user_ids, batch_item_ids, batch_ratings in dataloader:
                # Move data to device
                batch_user_ids = batch_user_ids.to(self.device)
                batch_item_ids = batch_item_ids.to(self.device)
                batch_ratings = batch_ratings.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(batch_user_ids, batch_item_ids)
                loss = F.mse_loss(predictions, batch_ratings)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Project parameters back to the manifold if needed
                if self.manifold_type != 'euclidean':
                    self.model.project_to_manifold()
                
                epoch_loss += loss.item() * len(batch_ratings)
            
            avg_epoch_loss = epoch_loss / len(dataloader.dataset)
            print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_epoch_loss:.4f}")
    
    def fit_from_matrix(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Train the GeoMLC model from an interaction matrix
        
        Parameters:
        -----------
        interaction_matrix : csr_matrix
            Sparse matrix of user-item interactions
        user_ids : List[int]
            List of user IDs
        item_ids : List[int]
            List of item IDs
        """
        # Extract non-zero entries from the interaction matrix
        users, items, ratings = [], [], []
        for i, j, v in zip(interaction_matrix.row, interaction_matrix.col, interaction_matrix.data):
            users.append(user_ids[i])
            items.append(item_ids[j])
            ratings.append(float(v))
        
        # Train the model
        self.fit(users, items, ratings)
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """
        Generate recommendations for a user
        
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
        if user_id not in self.user_mapping:
            return []
        
        # Get internal user index
        user_idx = self.user_mapping[user_id]
        
        # Get user tensor
        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        
        # Get all items
        all_items = torch.arange(self.n_items).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(
                user_tensor.expand(self.n_items),
                all_items
            ).cpu().numpy()
        
        # Get seen items if needed
        if exclude_seen:
            seen_items = set()
            for i in range(self.n_users):
                if self.reverse_user_mapping[i] == user_id:
                    for j in range(self.n_items):
                        if self.model.is_interaction(i, j):
                            seen_items.add(j)
            
            # Set predictions for seen items to -inf
            for item_idx in seen_items:
                predictions[item_idx] = float('-inf')
        
        # Get top-n items
        top_item_indices = np.argsort(-predictions)[:top_n]
        
        # Map back to original item IDs
        recommended_items = [self.reverse_item_mapping[idx] for idx in top_item_indices]
        
        return recommended_items
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_item_mapping': self.reverse_item_mapping,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_factors': self.n_factors,
            'learning_rate': self.learning_rate,
            'regularization': self.regularization,
            'manifold_type': self.manifold_type,
            'curvature': self.curvature,
            'init_range': self.init_range
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'GeoMLC':
        """
        Load a model from a file
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        GeoMLC
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        # Create instance
        instance = cls(
            n_factors=model_state['n_factors'],
            learning_rate=model_state['learning_rate'],
            regularization=model_state['regularization'],
            manifold_type=model_state['manifold_type'],
            curvature=model_state['curvature'],
            init_range=model_state['init_range']
        )
        
        # Restore mappings
        instance.user_mapping = model_state['user_mapping']
        instance.item_mapping = model_state['item_mapping']
        instance.reverse_user_mapping = model_state['reverse_user_mapping']
        instance.reverse_item_mapping = model_state['reverse_item_mapping']
        instance.n_users = model_state['n_users']
        instance.n_items = model_state['n_items']
        
        # Initialize model
        instance._init_model()
        
        # Load state dictionaries
        instance.model.load_state_dict(model_state['model_state_dict'])
        instance.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        
        return instance


class GeoMLCModel(nn.Module):
    """
    PyTorch model for Geometric Matrix Completion
    
    Parameters:
    -----------
    n_users : int
        Number of users
    n_items : int
        Number of items
    n_factors : int
        Number of latent factors
    manifold_type : str
        Type of manifold to use ('hyperbolic', 'spherical', or 'euclidean')
    curvature : float
        Curvature parameter for the manifold
    init_range : float
        Range for random initialization of parameters
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        manifold_type: str = 'hyperbolic',
        curvature: float = -1.0,
        init_range: float = 0.1
    ) -> None:
        super(GeoMLCModel, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.manifold_type = manifold_type
        self.curvature = curvature
        
        # Initialize embeddings
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        
        # Initialize biases
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._init_weights(init_range)
        
        # Initialize interaction matrix for tracking seen items
        self.interaction_matrix = torch.zeros(n_users, n_items, dtype=torch.bool)
        
    def _init_weights(self, init_range: float) -> None:
        """
        Initialize weights with uniform distribution
        
        Parameters:
        -----------
        init_range : float
            Range for random initialization
        """
        nn.init.uniform_(self.user_embeddings.weight, -init_range, init_range)
        nn.init.uniform_(self.item_embeddings.weight, -init_range, init_range)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)
        
        # For hyperbolic space, ensure points are inside the manifold
        if self.manifold_type == 'hyperbolic':
            self.project_to_manifold()
    
    def project_to_manifold(self) -> None:
        """Project embeddings to the manifold"""
        with torch.no_grad():
            if self.manifold_type == 'hyperbolic':
                # Project to Poincaré ball
                user_norm = torch.norm(self.user_embeddings.weight, dim=1, keepdim=True)
                item_norm = torch.norm(self.item_embeddings.weight, dim=1, keepdim=True)
                
                # Ensure norm is less than 1 (with a small margin)
                max_norm = 0.999
                self.user_embeddings.weight.data = torch.where(
                    user_norm > max_norm,
                    self.user_embeddings.weight.data * (max_norm / user_norm),
                    self.user_embeddings.weight.data
                )
                self.item_embeddings.weight.data = torch.where(
                    item_norm > max_norm,
                    self.item_embeddings.weight.data * (max_norm / item_norm),
                    self.item_embeddings.weight.data
                )
            
            elif self.manifold_type == 'spherical':
                # Project to unit sphere
                self.user_embeddings.weight.data = F.normalize(self.user_embeddings.weight.data, p=2, dim=1)
                self.item_embeddings.weight.data = F.normalize(self.item_embeddings.weight.data, p=2, dim=1)
    
    def _manifold_distance(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute distance on the manifold
        
        Parameters:
        -----------
        user_emb : torch.Tensor
            User embeddings
        item_emb : torch.Tensor
            Item embeddings
            
        Returns:
        --------
        torch.Tensor
            Distances between users and items on the manifold
        """
        if self.manifold_type == 'euclidean':
            # Euclidean distance
            return torch.sum((user_emb - item_emb) ** 2, dim=-1)
        
        elif self.manifold_type == 'hyperbolic':
            # Poincaré distance
            # Formula: d(x,y) = arcosh(1 + 2 * ||x-y||^2 / ((1-||x||^2) * (1-||y||^2)))
            user_norm_sq = torch.sum(user_emb ** 2, dim=-1, keepdim=True)
            item_norm_sq = torch.sum(item_emb ** 2, dim=-1, keepdim=True)
            diff_norm_sq = torch.sum((user_emb - item_emb) ** 2, dim=-1)
            
            numerator = 2 * diff_norm_sq
            denominator = (1 - user_norm_sq) * (1 - item_norm_sq) + 1e-6  # Add epsilon to avoid division by zero
            
            # Compute distance
            distance = torch.acosh(1 + numerator / denominator.squeeze(-1))
            return distance
        
        elif self.manifold_type == 'spherical':
            # Spherical distance (geodesic on the unit sphere)
            # Formula: d(x,y) = arccos(x·y)
            dot_product = torch.sum(user_emb * item_emb, dim=-1)
            # Clamp to avoid numerical issues
            dot_product = torch.clamp(dot_product, -1 + 1e-6, 1 - 1e-6)
            distance = torch.acos(dot_product)
            return distance
        
        else:
            raise ValueError(f"Unsupported manifold type: {self.manifold_type}")
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Parameters:
        -----------
        user_ids : torch.Tensor
            User IDs
        item_ids : torch.Tensor
            Item IDs
            
        Returns:
        --------
        torch.Tensor
            Predicted ratings
        """
        # Get embeddings
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        
        # Get biases
        user_bias = self.user_biases(user_ids).squeeze(-1)
        item_bias = self.item_biases(item_ids).squeeze(-1)
        
        # Update interaction matrix
        if self.training:
            for u, i in zip(user_ids.cpu(), item_ids.cpu()):
                self.interaction_matrix[u, i] = True
        
        # Compute distance on the manifold
        distance = self._manifold_distance(user_emb, item_emb)
        
        # Convert distance to similarity (negative distance)
        similarity = -distance
        
        # Add biases
        prediction = similarity + user_bias + item_bias + self.global_bias
        
        return prediction
    
    def is_interaction(self, user_idx: int, item_idx: int) -> bool:
        """
        Check if a user has interacted with an item
        
        Parameters:
        -----------
        user_idx : int
            User index
        item_idx : int
            Item index
            
        Returns:
        --------
        bool
            True if the user has interacted with the item, False otherwise
        """
        return self.interaction_matrix[user_idx, item_idx].item()