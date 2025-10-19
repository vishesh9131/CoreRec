import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
import os
import logging
import pickle
from pathlib import Path
from collections import defaultdict

from corerec.base_recommender import BaseCorerec


class BilinearInteraction(nn.Module):
    """
    Bilinear Interaction layer for FiBiNET.
    
    This module models feature interactions using a bilinear function with
    a 3D interaction matrix for each feature pair.
    
    Architecture:
    ┌─────────────────┐
    │  Embeddings     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Bilinear Layer │
    └─────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, num_fields: int, embed_dim: int, bilinear_type: str = 'field_all'):
        """
        Initialize bilinear interaction layer.
        
        Args:
            num_fields: Number of feature fields
            embed_dim: Embedding dimension
            bilinear_type: Type of bilinear interaction ('field_all', 'field_each', or 'field_interaction')
        """
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.bilinear_type = bilinear_type
        
        if bilinear_type == 'field_all':
            # One interaction matrix for all fields
            self.bilinear_W = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        elif bilinear_type == 'field_each':
            # One interaction matrix per field
            self.bilinear_W = nn.Parameter(torch.Tensor(num_fields, embed_dim, embed_dim))
        elif bilinear_type == 'field_interaction':
            # One interaction matrix per field pair
            self.bilinear_W = nn.Parameter(torch.Tensor(num_fields, num_fields, embed_dim, embed_dim))
        else:
            raise ValueError("bilinear_type must be 'field_all', 'field_each', or 'field_interaction'")
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize bilinear weights."""
        if self.bilinear_type == 'field_all':
            nn.init.xavier_normal_(self.bilinear_W)
        elif self.bilinear_type == 'field_each':
            for i in range(self.num_fields):
                nn.init.xavier_normal_(self.bilinear_W[i])
        elif self.bilinear_type == 'field_interaction':
            for i in range(self.num_fields):
                for j in range(self.num_fields):
                    nn.init.xavier_normal_(self.bilinear_W[i][j])
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply bilinear interaction to embeddings.
        
        Args:
            embeddings: Tensor of shape (batch_size, num_fields, embed_dim)
            
        Returns:
            Tensor of bilinear interactions of shape (batch_size, num_fields, embed_dim)
        """
        batch_size = embeddings.size(0)
        
        if self.bilinear_type == 'field_all':
            # Same matrix for all fields
            bilinear_out = torch.einsum('bnk,kl,bnl->bn', embeddings, self.bilinear_W, embeddings)
            return bilinear_out.unsqueeze(2)  # Add embed_dim dimension
            
        elif self.bilinear_type == 'field_each':
            # Different matrix for each field
            bilinear_out = torch.zeros(batch_size, self.num_fields, 1, device=embeddings.device)
            for i in range(self.num_fields):
                vi = embeddings[:, i, :].unsqueeze(1)  # (batch_size, 1, embed_dim)
                bilinear_out[:, i, 0] = torch.bmm(torch.bmm(vi, self.bilinear_W[i].unsqueeze(0).expand(batch_size, -1, -1)), vi.transpose(1, 2)).squeeze()
            return bilinear_out
            
        elif self.bilinear_type == 'field_interaction':
            # Different matrix for each field pair
            bilinear_out = torch.zeros(batch_size, self.num_fields, self.embed_dim, device=embeddings.device)
            for i in range(self.num_fields):
                for j in range(self.num_fields):
                    if i != j:  # Exclude self-interactions
                        vi = embeddings[:, i, :]  # (batch_size, embed_dim)
                        vj = embeddings[:, j, :]  # (batch_size, embed_dim)
                        bilinear_out[:, i, :] += torch.einsum('bi,ij,bj->bi', vi, self.bilinear_W[i][j], vj)
            return bilinear_out
        
        return embeddings  # Default fallback


class SENet(nn.Module):
    """
    Squeeze-Excitation Network layer for FiBiNET.
    
    This module implements the Squeeze-Excitation mechanism to dynamically
    recalibrate feature embeddings by modeling interdependencies between features.
    
    Architecture:
    ┌─────────────────┐
    │  Embeddings     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │    Squeeze      │ (Global pooling)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   FC + ReLU     │ (Dimensionality reduction)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   FC + Sigmoid  │ (Excitation)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Recalibration │ (Scale original features)
    └─────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, num_fields: int, reduction_ratio: int = 3):
        """
        Initialize SENet layer.
        
        Args:
            num_fields: Number of feature fields
            reduction_ratio: Reduction ratio for the bottleneck
        """
        super().__init__()
        self.num_fields = num_fields
        reduced_size = max(1, num_fields // reduction_ratio)
        
        # Squeeze and Excitation layers
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_size, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_size, num_fields, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply SE-Net to feature embeddings.
        
        Args:
            embeddings: Tensor of shape (batch_size, num_fields, embed_dim)
            
        Returns:
            Recalibrated embeddings of shape (batch_size, num_fields, embed_dim)
        """
        # Squeeze: (batch_size, num_fields, embed_dim) -> (batch_size, num_fields)
        Z = torch.mean(embeddings, dim=2)
        
        # Excitation: (batch_size, num_fields) -> (batch_size, num_fields)
        A = self.excitation(Z)
        
        # Recalibration: (batch_size, num_fields, 1) * (batch_size, num_fields, embed_dim)
        V_calibrated = embeddings * A.unsqueeze(2)
        
        return V_calibrated


class FeatureEmbedding(nn.Module):
    """
    Feature embedding module for FiBiNET.
    
    Maps categorical features to dense vectors.
    
    Architecture:
    ┌─────────────────┐
    │  Sparse Features│
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Embeddings    │
    └─────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, field_dims: List[int], embed_dim: int):
        """
        Initialize feature embedding layer.
        
        Args:
            field_dims: List of feature field dimensions
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        
        # Initialize embeddings with Xavier uniform
        for embed in self.embedding:
            nn.init.xavier_uniform_(embed.weight)
    
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass for feature embedding.
        
        Args:
            x: Sparse feature tensor of shape (batch_size, num_fields)
            
        Returns:
            Embedded features of shape (batch_size, num_fields, embed_dim)
        """
        return torch.stack([
            embedding(x[:, i]) for i, embedding in enumerate(self.embedding)
        ], dim=1)


class FiBiNETModel(nn.Module):
    """
    Feature Importance and Bilinear feature Interaction NETwork.
    
    FiBiNET uses SE-Net to dynamically learn feature importance and employs
    bilinear interaction to capture feature correlations.
    
    Architecture:
    ┌─────────────────┐
    │  Input Features │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Linear + Bias   │ (First-order terms)
    └───────┬─────────┘
            │
            └───────────┐
    ┌─────────────────┐ │
    │   Embeddings    │ │
    └────────┬────────┘ │
             │          │
      ┌──────┴───────┐  │
      │              │  │
      ▼              ▼  │
    ┌───────┐      ┌───────┐
    │ SE-Net│      │Original│
    └───┬───┘      └───┬───┘
        │              │
        ▼              ▼
    ┌───────┐      ┌───────┐
    │Bilinear│      │Bilinear│
    └───┬───┘      └───┬───┘
        │              │
        └──────┬───────┘
               │
               ▼
    ┌─────────────────────┐
    │Concat + MLP + Output│
    └─────────────────────┘
    
    References:
        - Huang, T., et al. "FiBiNET: Combining Feature Importance and Bilinear feature 
          Interaction for Click-Through Rate Prediction." RecSys 2019.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self, 
        field_dims: List[int], 
        embed_dim: int = 16, 
        bilinear_type: str = 'field_interaction',
        mlp_dims: List[int] = [128, 64],
        dropout: float = 0.1
    ):
        """
        Initialize the FiBiNET model.
        
        Args:
            field_dims: List of feature field dimensions
            embed_dim: Embedding dimension
            bilinear_type: Type of bilinear interaction ('field_all', 'field_each', or 'field_interaction')
            mlp_dims: Hidden layer dimensions for MLP
            dropout: Dropout rate
        """
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)
        
        # First-order linear terms (feature-level)
        self.linear = nn.ModuleList([
            nn.Embedding(dim, 1) for dim in field_dims
        ])
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Feature embedding
        self.embedding = FeatureEmbedding(field_dims, embed_dim)
        
        # Feature importance with SE-Net
        self.senet = SENet(self.num_fields)
        
        # Bilinear interaction layers for original features and calibrated features
        self.bilinear = BilinearInteraction(self.num_fields, embed_dim, bilinear_type)
        
        # Calculate input dimension for the combination layer
        if bilinear_type in ['field_all', 'field_each']:
            bilinear_out_dim = self.num_fields
        else:  # field_interaction
            bilinear_out_dim = self.num_fields * embed_dim
        
        self.combined_dim = bilinear_out_dim * 2  # Original + Senet pathways
        
        # Deep neural network for final prediction
        layers = []
        input_dim = self.combined_dim
        
        for hidden_dim in mlp_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        for embedding in self.linear:
            nn.init.xavier_uniform_(embedding.weight)
    
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass of the FiBiNET model.
        
        Args:
            x: Input tensor of shape (batch_size, num_fields)
            
        Returns:
            Prediction tensor of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # First-order term
        linear_sum = torch.full((batch_size,), self.bias.item(), device=x.device)
        first_order = torch.zeros_like(linear_sum)
        for i in range(self.num_fields):
            first_order = first_order + self.linear[i](x[:, i]).squeeze(1)
            
        # Field-wise embeddings
        embeddings = self.embedding(x)  # B x F x E
        
        # Feature extraction and interaction
        # Squeeze-Excitation enhancement layer
        se_embeddings = self.senet(embeddings)
        
        # Bilinear interaction
        bi_embeddings = self.bilinear(embeddings)
        se_bi_embeddings = self.bilinear(se_embeddings)
        
        # Combine features from original and Squeeze-Excitation enhanced embeddings
        combined_embeddings = torch.cat(
            [bi_embeddings.flatten(1), se_bi_embeddings.flatten(1)], dim=1
        )
        
        # Deep part
        output = self.mlp(combined_embeddings)
        
        # Combine linear part (1st order) and deep part
        result = linear_sum + first_order + output.squeeze(1)
        
        # Apply sigmoid for binary classification
        return torch.sigmoid(result.view(-1, 1))


class Fibinet_base(BaseCorerec):
    """
    Feature Importance and Bilinear feature Interaction NETwork implementation.
    
    FiBiNET enhances recommendation by dynamically learning feature importance
    with Squeeze-Excitation networks and modeling feature interactions through
    bilinear functions.
    
    Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │                      Fibinet_base                         │
    │                                                           │
    │  ┌─────────────┐    ┌───────────────┐    ┌────────────┐  │
    │  │ User & Item │    │Feature Mapping│    │Data Process│  │
    │  └──────┬──────┘    └───────┬───────┘    └──────┬─────┘  │
    │         │                   │                   │         │
    │         └───────────┬───────┴─────────┬─────────┘         │
    │                     │                 │                    │
    │                     ▼                 ▼                    │
    │            ┌────────────────┐  ┌─────────────┐            │
    │            │ FiBiNET Model  │  │Training Loop│            │
    │            └────────┬───────┘  └──────┬──────┘            │
    │                     │                 │                    │
    │                     └────────┬────────┘                    │
    │                              │                             │
    │                              ▼                             │
    │                    ┌─────────────────┐                     │
    │                    │Recommendation API│                     │
    │                    └─────────────────┘                     │
    └───────────────────────────────────────────────────────────┘
                           │         │
                           ▼         ▼
                    ┌─────────┐ ┌──────────┐
                    │Prediction│ │Recommend │
                    └─────────┘ └──────────┘
    
    References:
        - Huang, T., et al. "FiBiNET: Combining Feature Importance and Bilinear feature 
          Interaction for Click-Through Rate Prediction." RecSys 2019.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self,
        name: str = "FiBiNET",
        embed_dim: int = 16,
        bilinear_type: str = 'field_interaction',
        mlp_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        num_epochs: int = 20,
        patience: int = 5,
        shuffle: bool = True,
        device: str = None,
        seed: int = 42,
        verbose: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the FiBiNET model.
        
        Args:
            name: Model name
            embed_dim: Embedding dimension
            bilinear_type: Type of bilinear interaction ('field_all', 'field_each', or 'field_interaction')
            mlp_dims: Hidden layer dimensions for MLP
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Number of samples per batch
            num_epochs: Maximum number of training epochs
            patience: Early stopping patience
            shuffle: Whether to shuffle data during training
            device: Device to run model on ('cpu' or 'cuda')
            seed: Random seed for reproducibility
            verbose: Whether to display training progress
            config: Configuration dictionary that overrides the default parameters
        """
        super().__init__(name=name, verbose=verbose)
        self.seed = seed
        self.is_fitted = False
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Process config if provided
        if config is not None:
            self.embed_dim = config.get("embed_dim", embed_dim)
            self.bilinear_type = config.get("bilinear_type", bilinear_type)
            self.mlp_dims = config.get("mlp_dims", mlp_dims)
            self.dropout = config.get("dropout", dropout)
            self.learning_rate = config.get("learning_rate", learning_rate)
            self.batch_size = config.get("batch_size", batch_size)
            self.num_epochs = config.get("num_epochs", num_epochs)
            self.patience = config.get("patience", patience)
            self.shuffle = config.get("shuffle", shuffle)
        else:
            self.embed_dim = embed_dim
            self.bilinear_type = bilinear_type
            self.mlp_dims = mlp_dims
            self.dropout = dropout
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.patience = patience
            self.shuffle = shuffle
        
        # Validate bilinear type
        if self.bilinear_type not in ['field_all', 'field_each', 'field_interaction']:
            raise ValueError("bilinear_type must be 'field_all', 'field_each', or 'field_interaction'")
            
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Setup logger
        self._setup_logger()
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.BCELoss()
        
        # Initialize data structures
        self.field_names = []
        self.field_dims = []
        self.field_mapping = {}
        
        if self.verbose:
            self.logger.info(f"Initialized {self.name} model with {self.embed_dim} embedding dimensions")
    
    def _setup_logger(self):
        """Setup logger for the model."""
        self.logger = logging.getLogger(f"{self.name}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False
    
    def _preprocess_data(self, data: List[Dict[str, Any]]):
        """
        Preprocess data for training.
        
        Args:
            data: List of dictionaries with features and label
        """
        # Extract field names
        all_fields = set()
        for sample in data:
            for field in sample.keys():
                if field != 'label':
                    all_fields.add(field)
        
        self.field_names = sorted(list(all_fields))
        if self.verbose:
            self.logger.info(f"Identified {len(self.field_names)} fields: {self.field_names}")
        
        # Create field mappings
        for field in self.field_names:
            self.field_mapping[field] = {}
            values = set()
            
            # Collect all values for this field
            for sample in data:
                if field in sample:
                    values.add(sample[field])
            
            # Map values to indices
            for i, value in enumerate(sorted(list(values))):
                self.field_mapping[field][value] = i + 1  # Reserve 0 for unknown/missing
            
            # Set field dimension
            self.field_dims.append(len(self.field_mapping[field]) + 1)  # +1 for unknown/missing
        
        if self.verbose:
            self.logger.info(f"Field dimensions: {self.field_dims}")
    
    def _build_model(self):
        """Build the FiBiNET model."""
        self.model = FiBiNETModel(
            field_dims=self.field_dims,
            embed_dim=self.embed_dim,
            bilinear_type=self.bilinear_type,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        if self.verbose:
            self.logger.info(f"Built FiBiNET model with {len(self.field_dims)} fields")
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Total parameters: {total_params}")
    
    def _prepare_batch(self, batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for training.
        
        Args:
            batch: List of dictionaries with features and label
            
        Returns:
            Tuple of (features, labels)
        """
        batch_size = len(batch)
        
        # Initialize tensors
        features = torch.zeros((batch_size, len(self.field_names)), dtype=torch.long)
        labels = torch.zeros((batch_size, 1), dtype=torch.float)
        
        # Fill tensors with data
        for i, sample in enumerate(batch):
            # Features
            for j, field in enumerate(self.field_names):
                if field in sample:
                    value = sample[field]
                    field_idx = self.field_mapping[field].get(value, 0)  # Use 0 for unknown values
                    features[i, j] = field_idx
            
            # Label
            if 'label' in sample:
                labels[i, 0] = float(sample['label'])
        
        return features.to(self.device), labels.to(self.device)
    
    def fit(self, data: List[Dict[str, Any]]) -> 'Fibinet_base':
        """
        Fit the FiBiNET model.
        
        Args:
            data: List of dictionaries with features and label
            
        Returns:
            Fitted model
        """
        if self.verbose:
            self.logger.info(f"Fitting {self.name} model on {len(data)} samples")
        
        # Preprocess data
        self._preprocess_data(data)
        
        # Build model
        self._build_model()
        
        # Training loop
        num_batches = (len(data) + self.batch_size - 1) // self.batch_size
        best_loss = float('inf')
        patience_counter = 0
        self.loss_history = []
        
        for epoch in range(self.num_epochs):
            if self.shuffle:
                np.random.shuffle(data)
            
            epoch_loss = 0
            
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i+self.batch_size]
                
                # Prepare data
                features, labels = self._prepare_batch(batch)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(features)
                
                # Compute loss
                loss = self.loss_fn(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / num_batches
            self.loss_history.append(avg_epoch_loss)
            
            if self.verbose:
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_epoch_loss:.4f}")
            
            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.is_fitted = True
        return self
    
    def predict(self, features: Dict[str, Any]) -> float:
        """
        Predict probability for a single sample.
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            Predicted probability
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Convert features to tensor
        feature_tensor = torch.zeros(1, len(self.field_names), dtype=torch.long)
        
        for i, field in enumerate(self.field_names):
            if field in features:
                value = features[field]
                field_idx = self.field_mapping.get(field, {}).get(value, 0)
                feature_tensor[0, i] = field_idx
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            feature_tensor = feature_tensor.to(self.device)
            prediction = self.model(feature_tensor).item()
        
        return prediction
    
    def recommend(self, user_features: Dict[str, Any], item_pool: List[Dict[str, Any]], 
                  top_n: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_features: Dictionary with user features
            item_pool: List of dictionaries with item features
            top_n: Number of recommendations to generate
            
        Returns:
            List of (item, score) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making recommendations")
        
        # Score each item in the pool
        scored_items = []
        for item in item_pool:
            # Merge user and item features
            features = {**user_features, **item}
            
            # Make prediction
            score = self.predict(features)
            scored_items.append((item, score))
        
        # Sort by score in descending order
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-n items
        return scored_items[:top_n]
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data to save
        model_data = {
            'model_config': {
                'embed_dim': self.embed_dim,
                'bilinear_type': self.bilinear_type,
                'mlp_dims': self.mlp_dims,
                'dropout': self.dropout,
                'name': self.name,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'patience': self.patience,
                'shuffle': self.shuffle,
                'seed': self.seed,
                'verbose': self.verbose
            },
            'field_data': {
                'field_names': self.field_names,
                'field_dims': self.field_dims,
                'field_mapping': self.field_mapping
            },
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss_history': self.loss_history if hasattr(self, 'loss_history') else []
        }
        
        # Save to file
        torch.save(model_data, filepath)
        
        if self.verbose:
            self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> 'Fibinet_base':
        """
        Load model from file.
        
        Args:
            filepath: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Loaded model
        """
        # Load model data
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
            
        model_data = torch.load(filepath, map_location=device)
        
        # Create model instance with saved config
        instance = cls(
            name=model_data['model_config']['name'],
            embed_dim=model_data['model_config']['embed_dim'],
            bilinear_type=model_data['model_config']['bilinear_type'],
            mlp_dims=model_data['model_config']['mlp_dims'],
            dropout=model_data['model_config']['dropout'],
            learning_rate=model_data['model_config']['learning_rate'],
            batch_size=model_data['model_config']['batch_size'],
            num_epochs=model_data['model_config']['num_epochs'],
            patience=model_data['model_config']['patience'],
            shuffle=model_data['model_config']['shuffle'],
            seed=model_data['model_config']['seed'],
            verbose=model_data['model_config']['verbose'],
            device=device
        )
        
        # Restore field data
        instance.field_names = model_data['field_data']['field_names']
        instance.field_dims = model_data['field_data']['field_dims']
        instance.field_mapping = model_data['field_data']['field_mapping']
        instance.loss_history = model_data.get('loss_history', [])
        
        # Build and load model
        instance._build_model()
        instance.model.load_state_dict(model_data['model_state'])
        instance.optimizer.load_state_dict(model_data['optimizer_state'])
        
        instance.is_fitted = True
        return instance
    
    def train(self):
        """Required by base class but implemented as fit."""
        pass