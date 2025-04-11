from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import logging
import pickle
from pathlib import Path
import scipy.sparse as sp
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from corerec.base_recommender import BaseCorerec


class FieldAwareEmbedding(nn.Module):
    """
    Field-aware embedding layer for DeepFEFM.
    
    This layer creates separate embeddings for each feature field,
    enabling field-aware feature interactions.
    
    Architecture:
    
    ┌──────────────────┐
    │ Feature Field 1  │───┐
    └──────────────────┘   │
    ┌──────────────────┐   │
    │ Feature Field 2  │───┼──► Field-Aware Embeddings
    └──────────────────┘   │
    ┌──────────────────┐   │
    │ Feature Field 3  │───┘
    └──────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, field_dims: List[int], embed_dim: int):
        """
        Initialize field-aware embedding layer.
        
        Args:
            field_dims: List of dimensions for each field.
            embed_dim: Embedding dimension.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        self.embedding = nn.ModuleList([
            nn.Embedding(sum(field_dims), embed_dim, padding_idx=0)
            for _ in range(len(field_dims))
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        self.field_dims = field_dims
        
        # Initialize embeddings
        for emb in self.embedding:
            nn.init.xavier_uniform_(emb.weight.data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the field-aware embedding layer.
        
        Args:
            x: Input tensor of field indices.
            
        Returns:
            Tensor of field-aware embeddings.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Add offsets to input indices
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        
        # Get embeddings for each field
        field_embs = [emb(x) for emb in self.embedding]
        
        return field_embs


class FEFMLayer(nn.Module):
    """
    Field-Embedded Factorization Machine Layer.
    
    This layer computes field-aware second-order feature interactions.
    
    Architecture:
    
    ┌──────────────────┐
    │ Field Embeddings │
    └─────────┬────────┘
              │
              ▼
    ┌──────────────────┐
    │ Field Pair       │
    │ Interactions     │
    └─────────┬────────┘
              │
              ▼
    ┌──────────────────┐
    │ Sum Pooling      │
    └─────────┬────────┘
              │
              ▼
    ┌──────────────────┐
    │ Output           │
    └──────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, field_dims: List[int], embed_dim: int):
        """
        Initialize FEFM layer.
        
        Args:
            field_dims: List of dimensions for each field.
            embed_dim: Embedding dimension.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)
        
        # Field-aware embeddings
        self.field_embeddings = FieldAwareEmbedding(field_dims, embed_dim)
        
        # Linear part
        self.linear = nn.Embedding(sum(field_dims), 1)
        self.bias = nn.Parameter(torch.zeros((1,)))
        
        # Initialize linear weights
        nn.init.xavier_uniform_(self.linear.weight.data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FEFM layer.
        
        Args:
            x: Input tensor of field indices.
            
        Returns:
            Tensor of field-aware feature interactions.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Get field-aware embeddings
        field_embs = self.field_embeddings(x)
        
        # Compute field-aware feature interactions
        interactions = []
        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                interaction = field_embs[i] * field_embs[j]
                interactions.append(interaction)
        
        # Sum all interactions
        sum_interactions = sum(interactions)
        
        # Linear part
        linear_part = self.linear(x)
        
        # Combine linear part and interactions
        output = linear_part + sum_interactions + self.bias
        
        return output


class DeepFEFM(nn.Module):
    """Neural network model for DeepFEFM."""
    
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.fefm = FEFMLayer(field_dims, embed_dim)
        
        # MLP layers
        layers = []
        input_dim = len(field_dims) * embed_dim
        for dim in mlp_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        fefm_out = self.fefm(x)
        return self.mlp(fefm_out).squeeze(-1)


class DeepFEFM_base:
    def __init__(self, 
                 embed_dim: int = 16,
                 mlp_dims: List[int] = [64, 32],
                 field_dims: Optional[List[int]] = None,
                 dropout: float = 0.2,
                 batch_size: int = 256,
                 learning_rate: float = 0.001,
                 num_epochs: int = 10,
                 seed: int = 42,
                 name: str = 'DeepFEFM',
                 device: Optional[str] = None):
        """
        Initialize DeepFEFM base model.
        
        Args:
            embed_dim: Embedding dimension
            mlp_dims: List of MLP layer dimensions
            field_dims: List of field dimensions (inferred from data if None)
            dropout: Dropout rate
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            seed: Random seed
            name: Model name
            device: Device to run model on ('cpu' or 'cuda')
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.field_dims = field_dims
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.seed = seed
        self.name = name
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Initialize other attributes
        self.model = None
        self.is_fitted = False
        self.feature_names = []
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.categorical_features = []
        self.numerical_features = []
        self.feature_encoders = {}
        self.numerical_means = {}
        self.numerical_stds = {}
        self.loss_history = []
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _build_model(self):
        """Build the neural network model."""
        if self.field_dims is None:
            raise ValueError("field_dims must be set before building model")
        
        self.model = DeepFEFM(
            field_dims=self.field_dims,
            embed_dim=self.embed_dim,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        self.criterion = nn.BCELoss()

    def _preprocess_features(self, interactions):
        """Preprocess features from interactions."""
        # Extract all features
        all_features = {}
        for _, _, features in interactions:
            for k, v in features.items():
                if k not in all_features:
                    all_features[k] = []
                all_features[k].append(v)
        
        # Determine feature types and create encoders
        self.feature_names = list(all_features.keys())
        self.categorical_features = []
        self.numerical_features = []
        
        for feature in self.feature_names:
            if isinstance(all_features[feature][0], (str, bool)):
                self.categorical_features.append(feature)
                unique_values = list(set(all_features[feature]))
                self.feature_encoders[feature] = {
                    val: idx + 1 for idx, val in enumerate(unique_values)
                }
            else:
                self.numerical_features.append(feature)
                values = np.array(all_features[feature])
                self.numerical_means[feature] = np.mean(values)
                self.numerical_stds[feature] = np.std(values) or 1.0

        # Create user and item mappings
        unique_users = set(user for user, _, _ in interactions)
        unique_items = set(item for _, item, _ in interactions)
        
        self.user_map = {user: idx + 1 for idx, user in enumerate(unique_users)}
        self.item_map = {item: idx + 1 for idx, item in enumerate(unique_items)}
        self.reverse_user_map = {v: k for k, v in self.user_map.items()}
        self.reverse_item_map = {v: k for k, v in self.item_map.items()}
        
        # Set field dimensions
        self.field_dims = [
            len(self.user_map) + 1,
            len(self.item_map) + 1
        ]
        for feature in self.feature_names:
            if feature in self.categorical_features:
                self.field_dims.append(len(self.feature_encoders[feature]) + 1)
            else:
                self.field_dims.append(1)  # Numerical features get 1 dimension

    def fit(self, interactions):
        """
        Fit the model to the given interactions.
        
        Args:
            interactions: List of (user, item, features) tuples
        """
        # Preprocess features and build model
        self._preprocess_features(interactions)
        self._build_model()
        
        # Convert interactions to tensors
        X = []
        y = []
        
        for user, item, features in interactions:
            # Create positive example
            x = [
                self.user_map[user],
                self.item_map[item]
            ]
            
            # Add features
            for feature in self.feature_names:
                value = features.get(feature)
                if feature in self.categorical_features:
                    x.append(self.feature_encoders[feature].get(value, 0))
                else:
                    if value is None:
                        value = 0
                    value = (value - self.numerical_means[feature]) / self.numerical_stds[feature]
                    x.append(value)
            
            X.append(x)
            y.append(1)
            
            # Create negative example with random item
            neg_item = item
            while neg_item == item:
                neg_item = np.random.choice(list(self.item_map.keys()))
            
            x = x.copy()
            x[1] = self.item_map[neg_item]
            X.append(x)
            y.append(0)
        
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        # Train model
        self.model.train()
        for epoch in range(self.num_epochs):
            indices = torch.randperm(len(X))
            total_loss = 0
            
            for start_idx in range(0, len(X), self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                
                self.optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss * self.batch_size / len(X)
            self.loss_history.append(avg_loss)
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True

    def train(self, interactions):
        """
        Train the model on given interactions.
        
        Args:
            interactions: List of (user, item, features) tuples
        """
        # TODO: Implement training logic
        pass

    def predict(self, user, item, features):
        """
        Make prediction for user-item pair with features.
        
        Args:
            user: User ID
            item: Item ID
            features: Dict of features
            
        Returns:
            float: Prediction score between 0 and 1
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        if user not in self.user_map:
            raise ValueError(f"Unknown user: {user}")
        if item not in self.item_map:
            raise ValueError(f"Unknown item: {item}")
        
        # Create feature vector
        x = [
            self.user_map[user],
            self.item_map[item]
        ]
        
        # Add features
        for feature in self.feature_names:
            value = features.get(feature)
            if feature in self.categorical_features:
                x.append(self.feature_encoders[feature].get(value, 0))
            else:
                if value is None:
                    value = 0
                value = (value - self.numerical_means[feature]) / self.numerical_stds[feature]
                x.append(value)
        
        # Convert to tensor and get prediction
        x = torch.tensor([x], dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)
        
        return float(pred[0])

    def recommend(self, user, top_n=5, exclude_seen=True, features=None):
        """
        Generate recommendations for user.
        
        Args:
            user: User ID
            top_n: Number of recommendations
            exclude_seen: Whether to exclude seen items
            features: Optional features for all items
            
        Returns:
            List of (item, score) tuples
        """
        # TODO: Implement recommendation logic
        pass

    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model to.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        save_data = {
            'config': {
                'name': self.name,
                'embed_dim': self.embed_dim,
                'mlp_dims': self.mlp_dims,
                'dropout': self.dropout,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'seed': self.seed,
                'device': self.device.type
            },
            'field_dims': self.field_dims,
            'feature_names': self.feature_names,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'feature_encoders': self.feature_encoders,
            'numerical_means': self.numerical_means,
            'numerical_stds': self.numerical_stds,
            'loss_history': self.loss_history,
            'model_state': self.model.state_dict()
        }
        
        # Save model
        torch.save(save_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> 'DeepFEFM_base':
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from.
            device: Device to load the model on.
            
        Returns:
            Loaded model.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load saved data
        save_data = torch.load(filepath, map_location=device)
        
        # Create new instance
        config = save_data['config']
        model = cls(
            name=config.get('name', 'DeepFEFM'),
            embed_dim=config.get('embed_dim', 16),
            mlp_dims=config.get('mlp_dims', [64, 32]),
            dropout=config.get('dropout', 0.2),
            batch_size=config.get('batch_size', 256),
            learning_rate=config.get('learning_rate', 0.001),
            num_epochs=config.get('num_epochs', 10),
            seed=save_data.get('seed', 42),
            device=device
        )
        
        # Restore model state
        model.field_dims = save_data['field_dims']
        model.feature_names = save_data['feature_names']
        model.user_map = save_data['user_map']
        model.item_map = save_data['item_map']
        model.reverse_user_map = save_data['reverse_user_map']
        model.reverse_item_map = save_data['reverse_item_map']
        model.categorical_features = save_data['categorical_features']
        model.numerical_features = save_data['numerical_features']
        model.feature_encoders = save_data['feature_encoders']
        model.numerical_means = save_data['numerical_means']
        model.numerical_stds = save_data['numerical_stds']
        model.loss_history = save_data['loss_history']
        
        # Build model
        model._build_model()
        model.model.load_state_dict(save_data['model_state'])
        model.is_fitted = True
        
        return model
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of feature importance scores.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        # Extract weights from the linear part
        linear_weights = self.model.fefm.linear.weight.data.cpu().numpy().flatten()
        
        # Get the index mapping for features
        feature_indices = {}
        start_idx = 0
        for i, field_dim in enumerate(self.field_dims):
            feature_indices[self.feature_names[i]] = slice(start_idx, start_idx + field_dim)
            start_idx += field_dim
        
        # Calculate importance for each feature
        importance = {}
        for feature, indices in feature_indices.items():
            importance[feature] = float(np.abs(linear_weights[indices]).mean())
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            for feature in importance:
                importance[feature] /= total
        
        return importance
    
    def set_device(self, device: str) -> None:
        """
        Set the device to run the model on.
        
        Args:
            device: Device to run the model on.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.device = torch.device(device)
        if hasattr(self, 'model') and self.model is not None:
            self.model.to(self.device)