import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
import os
import yaml
import logging
import pickle
from pathlib import Path
import pandas as pd
from collections import defaultdict

from corerec.base_recommender import BaseCorerec


class FeatureEmbedding(nn.Module):
    """
    Feature embedding module for DLRM.
    
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
        Initialize the feature embedding module.
        
        Args:
            field_dims: List of feature field dimensions (cardinalities).
            embed_dim: Embedding dimension.
        """
        super().__init__()
        self.embedding = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=embed_dim)
            for dim in field_dims
        ])
        
        # Initialize embeddings with Xavier uniform
        for embed in self.embedding:
            nn.init.xavier_uniform_(embed.weight)
    
    def forward(self, x_sparse: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sparse feature embedding.
        
        Args:
            x_sparse: Sparse features tensor of shape (batch_size, num_sparse_fields)
            
        Returns:
            Tensor of shape (batch_size, num_sparse_fields, embed_dim)
        """
        sparse_embeds = [
            self.embedding[i](x_sparse[:, i]) for i in range(x_sparse.size(1))
        ]
        return torch.stack(sparse_embeds, dim=1)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron module for DLRM.
    
    Used for both bottom MLP (dense features) and top MLP (combined features).
    
    Architecture:
    ┌─────────────┐
    │Input Features│
    └──────┬──────┘
           │
           ▼
    ┌─────────────────┐
    │ Linear Layer 1  │───► BatchNorm ───► ReLU ───► Dropout
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Linear Layer 2  │───► BatchNorm ───► ReLU ───► Dropout
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Linear Layer N  │
    └─────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self,
        input_dim: int,
        layer_dims: List[int],
        dropout: float = 0.1,
        batchnorm: bool = True,
        output_layer: bool = True
    ):
        """
        Initialize the MLP module.
        
        Args:
            input_dim: Input dimension.
            layer_dims: Dimensions of hidden and output layers.
            dropout: Dropout probability.
            batchnorm: Whether to use batch normalization.
            output_layer: Whether to add an output layer (without activation).
        """
        super().__init__()
        layers = []
        
        # Build hidden layers
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(input_dim if i == 0 else layer_dims[i-1], layer_dims[i]))
            if batchnorm:
                layers.append(nn.BatchNorm1d(layer_dims[i]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Add output layer if specified
        if output_layer:
            layers.append(nn.Linear(layer_dims[-2] if len(layer_dims) > 1 else input_dim, layer_dims[-1]))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        return self.mlp(x)


class DotInteraction(nn.Module):
    """
    Dot interaction module for feature crossing in DLRM.
    
    Computes pairwise dot products between embeddings.
    
    Architecture:
    ┌─────────────────┐
    │   Embeddings    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────┐
    │ Pairwise Dot Products   │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │      Flattened Output   │
    └─────────────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self):
        """Initialize the dot interaction module."""
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise dot product interactions.
        
        Args:
            x: Input tensor of shape (batch_size, num_fields, embed_dim)
            
        Returns:
            Tensor of shape (batch_size, num_fields * (num_fields - 1) // 2)
            containing all pairwise interactions.
        """
        batch_size, num_fields, embed_dim = x.size()
        
        # Compute pairwise dot products
        dot_products = []
        for i in range(num_fields):
            for j in range(i+1, num_fields):
                dot_products.append(torch.sum(x[:, i] * x[:, j], dim=1))
        
        # Stack and reshape to (batch_size, num_pairs)
        interactions = torch.stack(dot_products, dim=1)
        
        return interactions


class DLRMModel(nn.Module):
    """
    Deep Learning Recommendation Model (DLRM).
    
    Combines sparse and dense features with dot product interactions.
    
    Architecture:
    ┌─────────────────┐  ┌─────────────────┐
    │  Sparse Features│  │  Dense Features │
    └────────┬────────┘  └────────┬────────┘
             │                    │
             ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐
    │   Embeddings    │  │   Bottom MLP    │
    └────────┬────────┘  └────────┬────────┘
             │                    │
             └──────────┬─────────┘
                        │
                        ▼
    ┌─────────────────────────────────────┐
    │       Feature Interaction Layer     │
    │     (pairwise embedding dot product)│
    └───────────────────┬─────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────┐
    │              Top MLP                │
    └───────────────────┬─────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────┐
    │           Output Layer              │
    └─────────────────────────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self,
        field_dims: List[int],
        dense_dim: int,
        embed_dim: int = 16,
        bottom_mlp_dims: List[int] = [64, 32],
        top_mlp_dims: List[int] = [64, 32, 1],
        dropout: float = 0.1,
        batchnorm: bool = True
    ):
        """
        Initialize the DLRM model.
        
        Args:
            field_dims: List of categorical field dimensions.
            dense_dim: Dimension of dense features.
            embed_dim: Embedding dimension for categorical features.
            bottom_mlp_dims: Dimensions of bottom MLP layers.
            top_mlp_dims: Dimensions of top MLP layers.
            dropout: Dropout probability.
            batchnorm: Whether to use batch normalization.
        """
        super().__init__()
        
        # Feature embedding for sparse features
        self.sparse_embedding = FeatureEmbedding(field_dims, embed_dim)
        self.num_sparse_fields = len(field_dims)
        
        # Bottom MLP for dense features
        self.bottom_mlp = MLP(
            input_dim=dense_dim,
            layer_dims=bottom_mlp_dims + [embed_dim],  # Output dim matches embedding dim
            dropout=dropout,
            batchnorm=batchnorm
        )
        
        # Feature interaction layer
        self.interaction = DotInteraction()
        
        # Calculate top MLP input dimension
        num_interactions = self.num_sparse_fields * (self.num_sparse_fields + 1) // 2
        top_input_dim = num_interactions + embed_dim  # interactions + bottom MLP output
        
        # Top MLP
        self.top_mlp = MLP(
            input_dim=top_input_dim,
            layer_dims=top_mlp_dims,
            dropout=dropout,
            batchnorm=batchnorm
        )
    
    def forward(self, x_sparse: torch.Tensor, x_dense: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DLRM model.
        
        Args:
            x_sparse: Sparse features tensor of shape (batch_size, num_sparse_fields)
            x_dense: Dense features tensor of shape (batch_size, dense_dim)
            
        Returns:
            Predictions of shape (batch_size, 1)
        """
        # Process sparse features
        sparse_embeds = self.sparse_embedding(x_sparse)  # (batch_size, num_sparse_fields, embed_dim)
        
        # Process dense features
        dense_out = self.bottom_mlp(x_dense)  # (batch_size, embed_dim)
        
        # Combine dense output with sparse embeddings
        combined = torch.cat([sparse_embeds, dense_out.unsqueeze(1)], dim=1)  # (batch_size, num_fields+1, embed_dim)
        
        # Feature interactions
        interactions = self.interaction(combined)  # (batch_size, num_interactions)
        
        # Concatenate with dense output
        concat_features = torch.cat([interactions, dense_out], dim=1)
        
        # Top MLP
        output = self.top_mlp(concat_features)
        
        # Apply sigmoid for binary classification
        output = torch.sigmoid(output)
        
        return output


class DLRM_base(BaseCorerec):
    """
    Deep Learning Recommendation Model (DLRM) for CTR prediction.
    
    DLRM combines embeddings for sparse features, MLP for dense features,
    and feature interactions through dot products for effective CTR prediction.
    
    Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │                      DLRM_base                             │
    │                                                           │
    │  ┌─────────────┐    ┌───────────────┐    ┌────────────┐  │
    │  │Sparse & Dense│    │Feature Mapping│    │Data Process│  │
    │  └──────┬──────┘    └───────┬───────┘    └──────┬─────┘  │
    │         │                   │                   │         │
    │         └───────────┬───────┴─────────┬─────────┘         │
    │                     │                 │                    │
    │                     ▼                 ▼                    │
    │            ┌────────────────┐  ┌─────────────┐            │
    │            │ DLRM Model     │  │Training Loop│            │
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
        - Naumov, M., et al. "Deep Learning Recommendation Model for Personalization and Recommendation Systems." arXiv:1906.00091, 2019.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self,
        name: str = "DLRM",
        embed_dim: int = 16,
        bottom_mlp_dims: List[int] = [64, 32],
        top_mlp_dims: List[int] = [64, 32, 1],
        dropout: float = 0.1,
        batchnorm: bool = True,
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
        Initialize the DLRM model.
        
        Args:
            name: Model name.
            embed_dim: Embedding dimension.
            bottom_mlp_dims: Dimensions of hidden layers for bottom MLP.
            top_mlp_dims: Dimensions of hidden layers for top MLP.
            dropout: Dropout probability.
            batchnorm: Whether to use batch normalization.
            learning_rate: Learning rate for optimizer.
            batch_size: Number of samples per batch.
            num_epochs: Maximum number of training epochs.
            patience: Early stopping patience.
            shuffle: Whether to shuffle data during training.
            device: Device to run model on ('cpu' or 'cuda').
            seed: Random seed for reproducibility.
            verbose: Whether to display training progress.
            config: Configuration dictionary that overrides the default parameters.
        """
        super().__init__()
        self.name = name
        self.seed = seed
        self.verbose = verbose
        self.is_fitted = False
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Process config if provided
        if config is not None:
            self.embed_dim = config.get("embed_dim", embed_dim)
            self.bottom_mlp_dims = config.get("bottom_mlp_dims", bottom_mlp_dims)
            self.top_mlp_dims = config.get("top_mlp_dims", top_mlp_dims)
            self.dropout = config.get("dropout", dropout)
            self.batchnorm = config.get("batchnorm", batchnorm)
            self.learning_rate = config.get("learning_rate", learning_rate)
            self.batch_size = config.get("batch_size", batch_size)
            self.num_epochs = config.get("num_epochs", num_epochs)
            self.patience = config.get("patience", patience)
            self.shuffle = config.get("shuffle", shuffle)
        else:
            self.embed_dim = embed_dim
            self.bottom_mlp_dims = bottom_mlp_dims
            self.top_mlp_dims = top_mlp_dims
            self.dropout = dropout
            self.batchnorm = batchnorm
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.patience = patience
            self.shuffle = shuffle
        
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
        
        # Initialize data structures for users, items, and features
        self.categorical_map = {}
        self.categorical_names = []
        self.field_dims = []
        self.dense_features = []
        self.dense_dim = 0
        
        # Initialize hook manager for model introspection
        self.hook_manager = None
        
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
        Preprocess data for DLRM.
        
        Args:
            data: List of dictionaries containing features and labels.
        """
        # Extract feature names
        categorical_features = set()
        dense_features = set()
        
        for sample in data:
            for feature, value in sample.items():
                if feature == 'label':
                    continue
                
                if isinstance(value, (int, str)):
                    categorical_features.add(feature)
                elif isinstance(value, (float, np.float32, np.float64)):
                    dense_features.add(feature)
        
        self.categorical_names = sorted(list(categorical_features))
        self.dense_features = sorted(list(dense_features))
        self.dense_dim = len(self.dense_features)
        
        if self.verbose:
            self.logger.info(f"Categorical features: {self.categorical_names}")
            self.logger.info(f"Dense features: {self.dense_features}")
        
        # Create mappings for categorical features
        for feature in self.categorical_names:
            self.categorical_map[feature] = {}
            
            # Extract unique values for this feature
            values = set()
            for sample in data:
                if feature in sample:
                    values.add(sample[feature])
            
            # Create mapping
            for i, value in enumerate(sorted(values)):
                self.categorical_map[feature][value] = i
            
            # Add field dimension
            self.field_dims.append(len(self.categorical_map[feature]) + 1)  # +1 for unknown values
        
        if self.verbose:
            self.logger.info(f"Field dimensions: {self.field_dims}")
    
    def _build_model(self):
        """Build the DLRM model."""
        self.model = DLRMModel(
            field_dims=self.field_dims,
            dense_dim=self.dense_dim,
            embed_dim=self.embed_dim,
            bottom_mlp_dims=self.bottom_mlp_dims,
            top_mlp_dims=self.top_mlp_dims,
            dropout=self.dropout,
            batchnorm=self.batchnorm
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        if self.verbose:
            self.logger.info(f"Built DLRM model with {len(self.field_dims)} categorical fields and {self.dense_dim} dense features")
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Total parameters: {total_params}")
    
    def _prepare_batch(self, batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for training.
        
        Args:
            batch: List of dictionaries containing features and labels.
            
        Returns:
            Tuple of (sparse_features, dense_features, labels).
        """
        batch_size = len(batch)
        
        # Initialize tensors
        sparse_features = torch.zeros((batch_size, len(self.categorical_names)), dtype=torch.long)
        dense_features = torch.zeros((batch_size, self.dense_dim), dtype=torch.float)
        labels = torch.zeros((batch_size, 1), dtype=torch.float)
        
        # Fill tensors
        for i, sample in enumerate(batch):
            # Sparse features
            for j, feature_name in enumerate(self.categorical_names):
                if feature_name in sample:
                    feature_value = sample[feature_name]
                    feature_idx = self.categorical_map[feature_name].get(feature_value, 0)
                    sparse_features[i, j] = feature_idx
            
            # Dense features
            for j, feature_name in enumerate(self.dense_features):
                if feature_name in sample:
                    dense_features[i, j] = float(sample[feature_name])
            
            # Label
            if 'label' in sample:
                labels[i, 0] = float(sample['label'])
            
        return sparse_features.to(self.device), dense_features.to(self.device), labels.to(self.device)
    
    def fit(self, data: List[Dict[str, Any]]) -> 'DLRM_base':
        """
        Fit the DLRM model on the given data.
        
        Args:
            data: List of dictionaries containing features and labels.
            
        Returns:
            The fitted model.
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
                sparse_features, dense_features, labels = self._prepare_batch(batch)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(sparse_features, dense_features)
                
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
            features: Dictionary containing feature values.
            
        Returns:
            Predicted probability.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Prepare sparse features
        sparse_features = torch.zeros((1, len(self.categorical_names)), dtype=torch.long)
        for i, feature_name in enumerate(self.categorical_names):
            if feature_name in features:
                feature_value = features[feature_name]
                feature_idx = self.categorical_map.get(feature_name, {}).get(feature_value, 0)
                sparse_features[0, i] = feature_idx
        
        # Prepare dense features
        dense_features = torch.zeros((1, self.dense_dim), dtype=torch.float)
        for i, feature_name in enumerate(self.dense_features):
            if feature_name in features:
                dense_features[0, i] = float(features[feature_name])
        
        # Move to device
        sparse_features = sparse_features.to(self.device)
        dense_features = dense_features.to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(sparse_features, dense_features).item()
        
        return prediction
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            'name': self.name,
            'embed_dim': self.embed_dim,
            'bottom_mlp_dims': self.bottom_mlp_dims,
            'top_mlp_dims': self.top_mlp_dims,
            'dropout': self.dropout,
            'batchnorm': self.batchnorm,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'shuffle': self.shuffle,
            'seed': self.seed,
            'verbose': self.verbose,
            'categorical_map': self.categorical_map,
            'categorical_names': self.categorical_names,
            'field_dims': self.field_dims,
            'dense_features': self.dense_features,
            'dense_dim': self.dense_dim,
            'loss_history': self.loss_history if hasattr(self, 'loss_history') else []
        }
        
        # Save model state
        model_state = {
            'config': config,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        torch.save(model_state, filepath)
        
        if self.verbose:
            self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> 'DLRM_base':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from.
            device: Device to load the model on.
            
        Returns:
            Loaded model.
        """
        # Load model state
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        model_state = torch.load(filepath, map_location=device)
        config = model_state['config']
        
        # Create new model
        model = cls(
            name=config['name'],
            embed_dim=config['embed_dim'],
            bottom_mlp_dims=config['bottom_mlp_dims'],
            top_mlp_dims=config['top_mlp_dims'],
            dropout=config['dropout'],
            batchnorm=config['batchnorm'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            num_epochs=config['num_epochs'],
            patience=config['patience'],
            shuffle=config['shuffle'],
            seed=config['seed'],
            verbose=config['verbose'],
            device=device
        )
        
        # Restore mappings
        model.categorical_map = config['categorical_map']
        model.categorical_names = config['categorical_names']
        model.field_dims = config['field_dims']
        model.dense_features = config['dense_features']
        model.dense_dim = config['dense_dim']
        
        if 'loss_history' in config:
            model.loss_history = config['loss_history']
        
        # Build and restore model
        model._build_model()
        model.model.load_state_dict(model_state['model_state_dict'])
        model.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        
        model.is_fitted = True
        
        return model
    
    def train(self):
        """This method is required by the base class but is implemented as fit."""
        pass