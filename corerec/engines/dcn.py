import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from scipy.sparse import csr_matrix
from tqdm import tqdm

from corerec.base_recommender import BaseCorerec
from corerec.utils.validation import (
    validate_fit_inputs,
    validate_user_id,
    validate_top_k,
    validate_model_fitted,
    validate_embeddings_dim,
    ValidationError
)
from corerec.utils.training_utils import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)

class DCN(BaseCorerec):
    """
    Deep & Cross Network for Feature-rich Recommendation
    
    Combines a cross network for explicit feature interactions with
    a deep neural network for implicit feature interactions. Especially
    effective for feature-rich recommendation scenarios.
    
    Reference:
    Wang et al. "Deep & Cross Network for Ad Click Predictions" (2017)
    
    Args:
        name: Model name for identification
        embedding_dim: Dimension of feature embeddings (default: 16)
        num_cross_layers: Number of cross network layers (default: 3)
        deep_layers: List of hidden layer sizes for deep network (default: [128, 64])
        dropout: Dropout rate (default: 0.2)
        learning_rate: Learning rate for optimizer (default: 0.001)
        batch_size: Training batch size (default: 256)
        epochs: Number of training epochs (default: 20)
        early_stopping_patience: Patience for early stopping, None to disable (default: 5)
        checkpoint_dir: Directory to save checkpoints, None to disable (default: None)
        trainable: Whether model is trainable (default: True)
        verbose: Whether to print training progress (default: False)
        device: Device for computation ('cuda' or 'cpu', default: auto-detect)
        
    Example:
        >>> model = DCN(embedding_dim=64, epochs=20, verbose=True)
        >>> model.fit(user_ids=[1,2,3], item_ids=[10,20,30], ratings=[5.0,4.0,3.0])
        >>> recommendations = model.recommend(user_id=1, top_k=10)
    """
    
    def __init__(
        self,
        name: str = "DCN",
        embedding_dim: int = 16,
        num_cross_layers: int = 3,
        deep_layers: List[int] = [128, 64],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        early_stopping_patience: Optional[int] = 5,
        checkpoint_dir: Optional[str] = None,
        trainable: bool = True,
        verbose: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        
        # Validate parameters
        validate_embeddings_dim(embedding_dim)
        
        if num_cross_layers < 1:
            raise ValidationError("num_cross_layers must be at least 1")
        
        if not deep_layers or len(deep_layers) == 0:
            raise ValidationError("deep_layers must contain at least one layer")
        
        if not (0.0 <= dropout < 1.0):
            raise ValidationError("dropout must be in range [0.0, 1.0)")
        
        if learning_rate <= 0:
            raise ValidationError("learning_rate must be positive")
        
        if batch_size < 1:
            raise ValidationError("batch_size must be at least 1")
        
        if epochs < 1:
            raise ValidationError("epochs must be at least 1")
        
        self.embedding_dim = embedding_dim
        self.num_cross_layers = num_cross_layers
        self.deep_layers = deep_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        
        self.user_map = {}
        self.item_map = {}
        self.feature_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.user_features = {}
        self.item_features = {}
        self.model = None
        
    def _build_model(self, num_features: int, max_features: int = 2):
        class CrossLayer(nn.Module):
            def __init__(self, input_dim: int):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(input_dim) * 0.01)
                self.bias = nn.Parameter(torch.zeros(input_dim))
                
            def forward(self, x0, x):
                # x0 is the input, x is the current layer's input
                # Cross network formula: x0 * (x^T w) + b + x
                xw = (x * self.weight).sum(dim=1, keepdim=True)  # [batch, 1]
                return x0 * xw + self.bias + x
        
        class DeepCrossNetworkModel(nn.Module):
            def __init__(self, num_features, embedding_dim, num_cross_layers, deep_layers, dropout, max_features):
                super().__init__()
                self.embedding = nn.Embedding(num_features, embedding_dim)
                
                # Input dimension after embedding (features per sample * embedding_dim)
                self.input_dim = max_features * embedding_dim
                
                # Cross Network
                self.cross_layers = nn.ModuleList([
                    CrossLayer(self.input_dim) for _ in range(num_cross_layers)
                ])
                
                # Deep Network
                deep_input_dim = self.input_dim
                self.deep_layers = nn.ModuleList()
                for layer_size in deep_layers:
                    self.deep_layers.append(nn.Linear(deep_input_dim, layer_size))
                    self.deep_layers.append(nn.ReLU())
                    self.deep_layers.append(nn.Dropout(dropout))
                    deep_input_dim = layer_size
                
                # Combination Layer
                self.combination = nn.Linear(self.input_dim + deep_layers[-1], 1)
                
            def forward(self, feature_indices):
                # Get embeddings and flatten
                embeddings = self.embedding(feature_indices)
                x0 = embeddings.view(embeddings.size(0), -1)
                
                # Cross Network
                cross_output = x0
                for cross_layer in self.cross_layers:
                    cross_output = cross_layer(x0, cross_output)
                
                # Deep Network
                deep_output = x0
                for layer in self.deep_layers:
                    deep_output = layer(deep_output)
                
                # Combine outputs
                combined = torch.cat([cross_output, deep_output], dim=1)
                output = self.combination(combined)
                
                return torch.sigmoid(output).squeeze(1)
        
        return DeepCrossNetworkModel(num_features, self.embedding_dim, self.num_cross_layers, 
                                     self.deep_layers, self.dropout, max_features).to(self.device)
    
    def fit(self, 
            user_ids: List[int], 
            item_ids: List[int], 
            ratings: List[float], 
            user_features: Optional[Dict[int, Dict[str, Any]]] = None,
            item_features: Optional[Dict[int, Dict[str, Any]]] = None) -> None:
        
        # Validate inputs
        validate_fit_inputs(user_ids, item_ids, ratings)
        
        # Create mappings
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))
        
        self.user_map = {user: idx for idx, user in enumerate(unique_users)}
        self.item_map = {item: idx + len(unique_users) for idx, item in enumerate(unique_items)}
        
        self.reverse_user_map = {idx: user for user, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item for item, idx in self.item_map.items()}
        
        # Store features
        self.user_features = user_features or {}
        self.item_features = item_features or {}
        
        # Create feature map
        feature_values = set()
        
        # Add user and item IDs as features
        feature_values.update(self.user_map.values())
        feature_values.update(self.item_map.values())
        
        # Add user features
        if user_features:
            for user_id, features in user_features.items():
                for feature, value in features.items():
                    feature_key = f"user_{feature}_{value}"
                    if feature_key not in self.feature_map:
                        self.feature_map[feature_key] = len(self.feature_map) + len(unique_users) + len(unique_items)
                    feature_values.add(self.feature_map[feature_key])
        
        # Add item features
        if item_features:
            for item_id, features in item_features.items():
                for feature, value in features.items():
                    feature_key = f"item_{feature}_{value}"
                    if feature_key not in self.feature_map:
                        self.feature_map[feature_key] = len(self.feature_map) + len(unique_users) + len(unique_items)
                    feature_values.add(self.feature_map[feature_key])
        
        # Create training data first to determine max_features
        train_features = []
        train_labels = []
        
        for user_id, item_id, rating in zip(user_ids, item_ids, ratings):
            # Get user and item indices
            user_idx = self.user_map[user_id]
            item_idx = self.item_map[item_id]
            
            # Get feature indices
            feature_indices = [user_idx, item_idx]
            
            # Add user features
            if user_features and user_id in user_features:
                for feature, value in user_features[user_id].items():
                    feature_key = f"user_{feature}_{value}"
                    if feature_key in self.feature_map:
                        feature_indices.append(self.feature_map[feature_key])
            
            # Add item features
            if item_features and item_id in item_features:
                for feature, value in item_features[item_id].items():
                    feature_key = f"item_{feature}_{value}"
                    if feature_key in self.feature_map:
                        feature_indices.append(self.feature_map[feature_key])
            
            # Add to training data
            train_features.append(feature_indices)
            train_labels.append(1.0 if rating > 0 else 0.0)  # Convert to binary for implicit feedback
        
        # Pad feature lists to the same length
        max_features = max(len(features) for features in train_features) if train_features else 2
        train_features = [features + [0] * (max_features - len(features)) for features in train_features]
        
        # Build model with correct dimensions (after we know max_features)
        num_features = len(feature_values) + 1  # +1 for padding/unknown
        self.model = self._build_model(num_features, max_features)
        
        # Convert to tensors
        train_features = torch.LongTensor(train_features).to(self.device)
        train_labels = torch.FloatTensor(train_labels).to(self.device)
        
        # Define optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Train the model
        self.model.train()
        n_batches = len(train_features) // self.batch_size + (1 if len(train_features) % self.batch_size != 0 else 0)
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            # Shuffle data
            indices = torch.randperm(len(train_features))
            train_features = train_features[indices]
            train_labels = train_labels[indices]
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(train_features))
                
                batch_features = train_features[start_idx:end_idx]
                batch_labels = train_labels[start_idx:end_idx]
                
                # Forward pass
                outputs = self.model(batch_features)
                
                # Compute loss
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if self.verbose:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/n_batches:.4f}")
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        # Validate inputs
        validate_model_fitted(self.is_fitted, self.name)
        validate_user_id(user_id, self.user_map if hasattr(self, 'user_map') else {})
        validate_top_k(top_k if 'top_k' in locals() else 10)
        
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if user_id not in self.user_map:
            return []
        
        # Get user index
        user_idx = self.user_map[user_id]
        
        # Get all items
        all_items = list(self.item_map.keys())
        
        # Get items the user has already interacted with
        seen_items = set()
        if exclude_seen:
            # This would need to be implemented based on your data structure
            # For now, assume we have a list of seen items for each user
            pass
        
        # Generate predictions for all items
        predictions = []
        
        for item_id in all_items:
            if item_id in seen_items:
                predictions.append((item_id, float('-inf')))
                continue
            
            # Get item index
            item_idx = self.item_map[item_id]
            
            # Get feature indices
            feature_indices = [user_idx, item_idx]
            
            # Add user features
            if user_id in self.user_features:
                for feature, value in self.user_features[user_id].items():
                    feature_key = f"user_{feature}_{value}"
                    if feature_key in self.feature_map:
                        feature_indices.append(self.feature_map[feature_key])
            
            # Add item features
            if item_id in self.item_features:
                for feature, value in self.item_features[item_id].items():
                    feature_key = f"item_{feature}_{value}"
                    if feature_key in self.feature_map:
                        feature_indices.append(self.feature_map[feature_key])
            
            # Pad feature list
            feature_indices = feature_indices + [0] * (self.model.input_dim // self.embedding_dim - len(feature_indices))
            if len(feature_indices) > self.model.input_dim // self.embedding_dim:
                feature_indices = feature_indices[:self.model.input_dim // self.embedding_dim]
            
            # Convert to tensor
            feature_tensor = torch.LongTensor([feature_indices]).to(self.device)
            
            # Get prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(feature_tensor).item()
            
            predictions.append((item_id, prediction))
        
        # Sort predictions and get top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_items = [item_id for item_id, _ in predictions[:top_n]]
        
        return top_items