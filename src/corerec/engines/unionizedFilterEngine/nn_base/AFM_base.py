from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import logging
from pathlib import Path
import scipy.sparse as sp
import sys
sys.path.append("..")   
from corerec.base_recommender import BaseCorerec

class HookManager:
    """Manager for model hooks to inspect internal states."""
    
    def __init__(self):
        """Initialize the hook manager."""
        self.hooks = {}
        self.activations = {}
    
    def _get_activation(self, name):
        """Get activation for a specific layer."""
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def register_hook(self, model, layer_name, callback=None):
        """Register a hook for a specific layer."""
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            if callback is None:
                callback = self._get_activation(layer_name)
            handle = layer.register_forward_hook(callback)
            self.hooks[layer_name] = handle
            return True
        
        # Try to find the layer in submodules
        for name, module in model.named_modules():
            if name == layer_name:
                if callback is None:
                    callback = self._get_activation(layer_name)
                handle = module.register_forward_hook(callback)
                self.hooks[layer_name] = handle
                return True
        
        return False
    
    def remove_hook(self, layer_name):
        """Remove a hook for a specific layer."""
        if layer_name in self.hooks:
            self.hooks[layer_name].remove()
            del self.hooks[layer_name]
            return True
        return False
    
    def clear_hooks(self):
        """Remove all hooks."""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
        self.activations.clear()
    
    def get_activation(self, layer_name):
        """Get the activation for a specific layer."""
        return self.activations.get(layer_name, None)


class AFM_base(BaseCorerec):
    """
    Base class for Attentional Factorization Machine (AFM) models.
    
    AFM extends Factorization Machines by introducing an attention mechanism
    to learn the importance of feature interactions.
    
    References:
        - Xiao, J., Ye, H., He, X., Zhang, H., Wu, F., & Chua, T. S. (2017).
          Attentional factorization machines: Learning the weight of feature
          interactions via attention networks. IJCAI.
    """
    
    def __init__(self, name: str = "AFM", trainable: bool = True, verbose: bool = False, 
                 config: Optional[Dict[str, Any]] = None, seed: int = 42):
        """
        Initialize the AFM module.
        
        Args:
            name: Name of the recommender model.
            trainable: When False, the model is not trainable.
            verbose: When True, running logs are displayed.
            config: Configuration dictionary (YAML/JSON-compatible).
            seed: Random seed for deterministic behavior.
        """
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        
        self.config = config or {}
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Initialize hooks for model inspection
        self.hooks = HookManager()
        
        # Set default configuration values
        self._set_default_config()
        
        # Initialize model components
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        # Initialize field dimensions
        self.field_dims = None
        
        # Set device
        self.device = self.config.get('device', 'cpu')
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        # Version tracking
        self.version = "1.0.0"
    
    def _set_default_config(self):
        """Set default configuration values if not provided."""
        defaults = {
            'embedding_dim': 64,
            'attention_dim': 32,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'weight_decay': 1e-6,
            'batch_size': 256,
            'num_epochs': 20,
            'early_stopping_patience': 5,
            'verbose': True,
            'device': 'cpu',
            'save_dir': './models'
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def register_hook(self, layer_name: str, callback: callable = None) -> bool:
        """
        Register a hook for a specific layer.
        
        Args:
            layer_name: Name of the layer.
            callback: Optional callback function.
            
        Returns:
            True if hook was registered successfully, False otherwise.
        """
        return self.hooks.register_hook(self.model, layer_name, callback)
    
    def _build_model(self, field_dims: List[int]):
        """
        Build the AFM model architecture.
        
        Args:
            field_dims: List of dimensions for each field.
        """
        self.field_dims = field_dims
        self.model = AFMModel(
            field_dims=field_dims,
            embedding_dim=self.config['embedding_dim'],
            attention_dim=self.config['attention_dim'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.criterion = nn.BCELoss()
    
    def _create_dataset(self, interaction_matrix, user_ids, item_ids):
        """
        Create a dataset from the interaction matrix.
        
        Args:
            interaction_matrix: User-item interaction matrix.
            user_ids: List of user IDs.
            item_ids: List of item IDs.
            
        Returns:
            Dataset for training.
        """
        # Create mappings
        self.uid_map = {uid: idx for idx, uid in enumerate(user_ids)}
        self.iid_map = {iid: idx for idx, iid in enumerate(item_ids)}
        
        # Get positive interactions
        coo = interaction_matrix.tocoo()
        user_indices = coo.row
        item_indices = coo.col
        ratings = coo.data
        
        # Create field dimensions
        field_dims = [len(user_ids), len(item_ids)]
        
        # Build model if not already built
        if self.model is None:
            self._build_model(field_dims)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(user_indices),
            torch.LongTensor(item_indices),
            torch.FloatTensor(ratings)
        )
        
        return dataset
    
    def fit(self, interaction_matrix, user_ids: List[int], item_ids: List[int]):
        """
        Train the AFM model on the provided interaction matrix.
        
        Args:
            interaction_matrix: User-item interaction matrix.
            user_ids: List of user IDs.
            item_ids: List of item IDs.
            
        Returns:
            Self.
        """
        if not self.trainable:
            return self
        
        # Set model attributes
        self.num_users = len(user_ids)
        self.num_items = len(item_ids)
        
        # Create dataset
        dataset = self._create_dataset(interaction_matrix, user_ids, item_ids)
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        # Training loop
        self.model.train()
        for epoch in range(self.config['num_epochs']):
            total_loss = 0
            for user_idx, item_idx, rating in dataloader:
                user_idx = user_idx.to(self.device)
                item_idx = item_idx.to(self.device)
                rating = rating.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                x = torch.stack([user_idx, item_idx], dim=1)
                prediction = self.model(x)
                
                # Binary classification loss
                loss = self.criterion(prediction, (rating > 0).float())
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(rating)
            
            avg_loss = total_loss / len(dataset)
            if self.verbose:
                self.logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']} - Loss: {avg_loss:.6f}")
        
        self.is_fitted = True
        return self
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Generate top-N item recommendations for a given user.
        
        Args:
            user_id: The ID of the user.
            top_n: The number of recommendations to generate.
            exclude_seen: Whether to exclude items the user has already interacted with.
            
        Returns:
            List of (item_id, score) tuples.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Convert user ID to index
        if user_id not in self.uid_map:
            return []
        
        user_idx = self.uid_map[user_id]
        
        # Get user's seen items
        seen_items = set()
        if exclude_seen:
            # Get items the user has interacted with
            for item_id, item_idx in self.iid_map.items():
                if self._has_interaction(user_idx, item_idx):
                    seen_items.add(item_id)
        
        # Generate predictions for all items
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx] * self.num_items).to(self.device)
            item_indices = list(range(self.num_items))
            item_tensor = torch.LongTensor(item_indices).to(self.device)
            
            x = torch.stack([user_tensor, item_tensor], dim=1)
            predictions = self.model(x).cpu().numpy().flatten()
        
        # Create (item_id, score) pairs
        item_scores = []
        for item_idx, score in enumerate(predictions):
            item_id = self.item_ids[item_idx]
            if exclude_seen and item_id in seen_items:
                continue
            item_scores.append((item_id, float(score)))
        
        # Sort by score and take top-n
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:top_n]
    
    def _has_interaction(self, user_idx: int, item_idx: int) -> bool:
        """
        Check if a user has interacted with an item.
        
        Args:
            user_idx: User index.
            item_idx: Item index.
            
        Returns:
            True if the user has interacted with the item, False otherwise.
        """
        # This is a placeholder. In a real implementation, you would check
        # the interaction matrix or another data structure.
        return False
    
    def save_weights(self, path: str) -> None:
        """
        Save model weights to disk.
        
        Args:
            path: Path to save the weights.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def load_weights(self, path: str) -> None:
        """
        Load model weights from disk.
        
        Args:
            path: Path to load the weights from.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert model configuration to JSON-serializable dictionary.
        
        Returns:
            Dictionary with model configuration.
        """
        return {
            'name': self.name,
            'version': self.version,
            'config': self.config,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'field_dims': self.field_dims
        }
    
    def from_json(self, json_data: Dict[str, Any]) -> None:
        """
        Load model configuration from JSON-serializable dictionary.
        
        Args:
            json_data: Dictionary with model configuration.
        """
        self.name = json_data.get('name', self.name)
        self.version = json_data.get('version', self.version)
        self.config = json_data.get('config', self.config)
        self.num_users = json_data.get('num_users', self.num_users)
        self.num_items = json_data.get('num_items', self.num_items)
        self.field_dims = json_data.get('field_dims', self.field_dims)
        
        # Rebuild model if field dimensions are available
        if self.field_dims is not None:
            self._build_model(self.field_dims)


class AFMModel(nn.Module):
    """PyTorch model for Attentional Factorization Machine."""
    
    def __init__(self, field_dims: List[int], embedding_dim: int = 64, 
                 attention_dim: int = 32, dropout: float = 0.1):
        """
        Initialize the AFM model.
        
        Args:
            field_dims: List of dimensions for each field.
            embedding_dim: Dimension of the embeddings.
            attention_dim: Dimension of the attention network.
            dropout: Dropout rate.
        """
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embedding_dim)
        self.attention = AttentionalInteraction(embedding_dim, attention_dim, dropout)
        self.prediction = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AFM model.
        
        Args:
            x: Input tensor of shape (batch_size, num_fields).
            
        Returns:
            Predicted scores.
        """
        # First-order term
        first_order = self.linear(x)
        
        # Get embeddings
        embeddings = self.embedding(x)  # (batch_size, num_fields, embedding_dim)
        
        # Attention-based interaction
        second_order = self.attention(embeddings)  # (batch_size, 1)
        
        # Final prediction
        y = first_order + second_order
        y = self.prediction(y)
        
        return torch.sigmoid(y.squeeze(1))
    
    def get_embeddings(self, field_idx: int, feature_idx: int) -> torch.Tensor:
        """
        Get embeddings for a specific field and feature.
        
        Args:
            field_idx: Field index.
            feature_idx: Feature index.
            
        Returns:
            Embedding tensor.
        """
        return self.embedding.get_embedding(field_idx, feature_idx)


class FeaturesLinear(nn.Module):
    """Linear part of the factorization machine model."""
    
    def __init__(self, field_dims: List[int]):
        """
        Initialize the linear part.
        
        Args:
            field_dims: List of dimensions for each field.
        """
        super().__init__()
        self.field_dims = field_dims
        self.offsets = self._compute_offsets()
        self.linear = nn.Embedding(sum(field_dims), 1)
        self.bias = nn.Parameter(torch.zeros(1))
    
    def _compute_offsets(self) -> torch.Tensor:
        """Compute offsets for each field."""
        offsets = torch.zeros(len(self.field_dims), dtype=torch.long)
        for i in range(1, len(self.field_dims)):
            offsets[i] = offsets[i-1] + self.field_dims[i-1]
        return offsets
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear part.
        
        Args:
            x: Input tensor of shape (batch_size, num_fields).
            
        Returns:
            Linear term output.
        """
        x = x + self.offsets.unsqueeze(0)
        return torch.sum(self.linear(x), dim=1) + self.bias


class FeaturesEmbedding(nn.Module):
    """Embedding layer for feature interactions."""
    
    def __init__(self, field_dims: List[int], embedding_dim: int):
        """
        Initialize the embedding layer.
        
        Args:
            field_dims: List of dimensions for each field.
            embedding_dim: Dimension of the embeddings.
        """
        super().__init__()
        self.field_dims = field_dims
        self.embedding_dim = embedding_dim
        self.offsets = self._compute_offsets()
        self.embedding = nn.Embedding(sum(field_dims), embedding_dim)
    
    def _compute_offsets(self) -> torch.Tensor:
        """Compute offsets for each field."""
        offsets = torch.zeros(len(self.field_dims), dtype=torch.long)
        for i in range(1, len(self.field_dims)):
            offsets[i] = offsets[i-1] + self.field_dims[i-1]
        return offsets
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding layer.
        
        Args:
            x: Input tensor of shape (batch_size, num_fields).
            
        Returns:
            Embedded features of shape (batch_size, num_fields, embedding_dim).
        """
        x = x + self.offsets.unsqueeze(0)
        return self.embedding(x)
    
    def get_embedding(self, field_idx: int, feature_idx: int) -> torch.Tensor:
        """
        Get embedding for a specific field and feature.
        
        Args:
            field_idx: Field index.
            feature_idx: Feature index.
            
        Returns:
            Embedding tensor.
        """
        idx = self.offsets[field_idx] + feature_idx
        return self.embedding(idx)


class AttentionalInteraction(nn.Module):
    """Attentional interaction layer for AFM."""
    
    def __init__(self, embedding_dim: int, attention_dim: int, dropout: float = 0.1):
        """
        Initialize the attentional interaction layer.
        
        Args:
            embedding_dim: Dimension of the feature embeddings.
            attention_dim: Dimension of the attention network.
            dropout: Dropout rate.
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1, bias=False),
            nn.Softmax(dim=1)
        )
        self.projection = nn.Linear(embedding_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attentional interaction layer.
        
        Args:
            embeddings: Embedded features of shape (batch_size, num_fields, embedding_dim).
            
        Returns:
            Attentional interaction output of shape (batch_size, 1).
        """
        # Create pairwise interactions
        num_fields = embeddings.size(1)
        row, col = list(), list()
        for i in range(num_fields):
            for j in range(i+1, num_fields):
                row.append(i)
                col.append(j)
        
        # Element-wise product of embedding pairs
        p = embeddings[:, row] * embeddings[:, col]  # (batch_size, num_interactions, embedding_dim)
        
        # Apply attention network
        attention_weights = self.attention(p)  # (batch_size, num_interactions, 1)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of interactions
        p = p * attention_weights  # (batch_size, num_interactions, embedding_dim)
        p = torch.sum(p, dim=1)  # (batch_size, embedding_dim)
        
        # Project to scalar output
        output = self.projection(p)  # (batch_size, 1)
        
        return output
