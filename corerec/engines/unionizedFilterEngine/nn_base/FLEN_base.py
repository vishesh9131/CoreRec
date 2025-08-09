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


class FieldEmbedding(nn.Module):
    """
    Field-wise Embedding for FLEN.
    
    Creates embeddings for each field and handles categorical features.
    
    Architecture:
    ┌─────────────────┐
    │  Sparse Features│
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Field Embeddings│
    └─────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, field_dims: List[int], embed_dim: int):
        """
        Initialize field-wise embedding.
        
        Args:
            field_dims: List of dimensions for each field
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
        Forward pass for field embedding.
        
        Args:
            x: Input tensor of shape (batch_size, num_fields)
            
        Returns:
            Embedded features of shape (batch_size, num_fields, embed_dim)
        """
        return torch.stack([
            embedding(x[:, i]) for i, embedding in enumerate(self.embedding)
        ], dim=1)


class FieldWiseBiInteraction(nn.Module):
    """
    Field-Wise Bi-Interaction Layer for FLEN.
    
    This layer implements field-wise bi-interaction and produces field-wise
    interaction features.
    
    Architecture:
    ┌─────────────────┐
    │  Field Features │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Field-Wise      │ 
    │ Bi-Interaction  │
    └─────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, num_fields: int, embed_dim: int):
        """
        Initialize field-wise bi-interaction layer.
        
        Args:
            num_fields: Number of fields
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply field-wise bi-interaction to embeddings.
        
        Args:
            embeddings: Tensor of shape (batch_size, num_fields, embed_dim)
            
        Returns:
            Field-wise interaction features of shape (batch_size, embed_dim)
        """
        # Calculate the square of sum
        sum_embed = torch.sum(embeddings, dim=1)  # (batch_size, embed_dim)
        sum_square = torch.square(sum_embed)      # (batch_size, embed_dim)
        
        # Calculate the sum of square
        square_embed = torch.square(embeddings)   # (batch_size, num_fields, embed_dim)
        square_sum = torch.sum(square_embed, dim=1)  # (batch_size, embed_dim)
        
        # Bi-interaction: 0.5 * (sum_square - square_sum)
        field_wise_interaction = 0.5 * (sum_square - square_sum)
        
        return field_wise_interaction


class MLPModule(nn.Module):
    """
    Multi-Layer Perceptron Module for FLEN.
    
    Simple MLP with configurable layers, activation functions, and dropout.
    
    Architecture:
    ┌─────────────────┐
    │     Input       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Linear + BatchNorm │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │    Activation   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │     Dropout     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ More hidden layers│
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │    Output Layer │
    └─────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, input_dim: int, layer_dims: List[int], dropout: float = 0.1, 
                 output_dim: int = 1, use_batch_norm: bool = True):
        """
        Initialize MLP module.
        
        Args:
            input_dim: Input dimension
            layer_dims: List of hidden layer dimensions
            dropout: Dropout rate
            output_dim: Output dimension
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for dim in layer_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create sequential model
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.mlp(x)


class MaskBlock(nn.Module):
    """
    Field Mask Block for FLEN.
    
    This block implements field masking to learn the importance of different field
    combinations for different groups of field embeddings.
    
    Architecture:
    ┌─────────────────┐
    │  Embeddings     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Field Masks    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │Masked Embeddings│
    └─────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, num_fields: int, embed_dim: int, mask_mode: str = 'field'):
        """
        Initialize mask block.
        
        Args:
            num_fields: Number of fields
            embed_dim: Embedding dimension
            mask_mode: Masking mode ('field' or 'element')
        """
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.mask_mode = mask_mode
        
        if mask_mode == 'field':
            # Field-level masking: one mask per field
            mask_shape = (num_fields, 1)
        elif mask_mode == 'element':
            # Element-level masking: one mask per field and embedding dimension
            mask_shape = (num_fields, embed_dim)
        else:
            raise ValueError("mask_mode must be 'field' or 'element'")
        
        # Initialize field masks as learnable parameters
        self.field_masks = nn.Parameter(torch.ones(mask_shape))
        
        # Initialize with ones (start with equal importance)
        nn.init.ones_(self.field_masks)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply field masking to embeddings.
        
        Args:
            embeddings: Tensor of shape (batch_size, num_fields, embed_dim)
            
        Returns:
            Masked embeddings of shape (batch_size, num_fields, embed_dim)
        """
        # Apply sigmoid to normalize masks between 0 and 1
        masks = torch.sigmoid(self.field_masks)
        
        # Apply masking (element-wise multiplication)
        return embeddings * masks


class FLENModel(nn.Module):
    """
    Field-wise Logistic Embedding Network Model.
    
    FLEN is designed for click-through rate prediction with special handling
    of field embedding groups and interactions.
    
    Architecture:
          ┌───────────────────────────────────────┐
          │            Input Features             │
          └───────────────┬───────────────────────┘
                          │
                          ▼
        ┌────────────────────────────────────────────┐
        │              Field Embedding               │
        └───┬─────────────────────┬──────────────────┘
            │                     │
            ▼                     ▼
    ┌───────────────┐     ┌───────────────┐
    │ User Group    │     │  Item Group   │
    │  Embedding    │     │   Embedding   │
    └───────┬───────┘     └───────┬───────┘
            │                     │
            ▼                     ▼
    ┌───────────────┐     ┌───────────────┐
    │ Group Masking │     │ Group Masking │
    └───────┬───────┘     └───────┬───────┘
            │                     │
            ▼                     ▼
    ┌───────────────┐     ┌───────────────┐
    │ Bi-Interaction│     │ Bi-Interaction│
    └───────┬───────┘     └───────┬───────┘
            │                     │
            └─────────┬───────────┘
                      │
                      ▼
            ┌────────────────────┐
            │ Concat + MLP + Out │
            └────────────────────┘
    
    References:
        - Liu, Z., et al. "Field-wise Learning for Multi-field Categorical Data." 
          NeurIPS 2021.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self, 
        field_dims: List[int], 
        embed_dim: int = 16,
        mlp_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        mask_mode: str = 'field',
        field_groups: Dict[str, List[int]] = None
    ):
        """
        Initialize the FLEN model.
        
        Args:
            field_dims: List of feature field dimensions
            embed_dim: Embedding dimension
            mlp_dims: Hidden layer dimensions for MLP
            dropout: Dropout rate
            mask_mode: Masking mode ('field' or 'element')
            field_groups: Dictionary mapping group names to field indices
        """
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)
        
        # First-order linear terms
        self.linear = nn.ModuleList([
            nn.Embedding(dim, 1) for dim in field_dims
        ])
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Field embedding layer
        self.embedding = FieldEmbedding(field_dims, embed_dim)
        
        # Define field groups if not provided
        if field_groups is None:
            # Default: split fields evenly into two groups
            mid_point = self.num_fields // 2
            self.field_groups = {
                'user_group': list(range(mid_point)),
                'item_group': list(range(mid_point, self.num_fields))
            }
        else:
            self.field_groups = field_groups
        
        # Create mask blocks for each field group
        self.mask_blocks = nn.ModuleDict({
            group_name: MaskBlock(len(indices), embed_dim, mask_mode)
            for group_name, indices in self.field_groups.items()
        })
        
        # Bi-interaction layers for each field group
        self.bi_interaction_layers = nn.ModuleDict({
            group_name: FieldWiseBiInteraction(len(indices), embed_dim)
            for group_name, indices in self.field_groups.items()
        })
        
        # Calculate input dimension for the final MLP
        self.mlp_input_dim = len(self.field_groups) * embed_dim
        
        # Final MLP for prediction
        self.mlp = MLPModule(
            input_dim=self.mlp_input_dim, 
            layer_dims=mlp_dims,
            dropout=dropout,
            output_dim=1,
            use_batch_norm=True
        )
        
        # Initialize linear embeddings
        for embedding in self.linear:
            nn.init.xavier_uniform_(embedding.weight)
    
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass of the FLEN model.
        
        Args:
            x: Input tensor of shape (batch_size, num_fields)
            
        Returns:
            Prediction tensor of shape (batch_size, 1)
        """
        # First-order terms
        batch_size = x.size(0)
        linear_output = torch.full((batch_size,), self.bias.item(), device=x.device)
        first_order = torch.zeros_like(linear_output)
        for i in range(self.num_fields):
            first_order = first_order + self.linear[i](x[:, i]).squeeze(1)
        
        # Get embeddings
        embeddings = self.embedding(x)  # (batch_size, num_fields, embed_dim)
        
        # Process embeddings for each field group
        group_outputs = []
        for group_name, field_indices in self.field_groups.items():
            # Extract field embeddings for this group
            if not field_indices:  # Skip empty groups
                continue
                
            # Get embeddings for fields in this group
            group_embeds = embeddings[:, field_indices, :]  # (batch_size, group_size, embed_dim)
            
            # Apply field masking for this group
            masked_embeds = self.mask_blocks[group_name](group_embeds)
            
            # Apply bi-interaction for this group
            group_output = self.bi_interaction_layers[group_name](masked_embeds)
            group_outputs.append(group_output)
        
        # Concatenate outputs from all field groups
        if group_outputs:
            combined = torch.cat(group_outputs, dim=1)
            
            # Final MLP for prediction
            output = self.mlp(combined).squeeze(1)
            
            # Combine first-order and higher-order interactions
            result = linear_output + first_order + output
            
            return torch.sigmoid(result.view(-1, 1))
        else:
            # Fallback if no groups (shouldn't happen)
            result = linear_output + first_order
            return torch.sigmoid(result.view(-1, 1))


class FLEN_base(BaseCorerec):
    """
    Field-wise Logistic Embedding Network for recommendation.
    
    FLEN enhances CTR prediction by grouping fields and applying field-wise
    masking and bi-interaction to capture complex feature relationships.
    
    Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │                       FLEN_base                           │
    │                                                           │
    │  ┌─────────────┐    ┌───────────────┐    ┌────────────┐  │
    │  │ User & Item │    │ Field Grouping│    │Field Process│  │
    │  └──────┬──────┘    └───────┬───────┘    └──────┬─────┘  │
    │         │                   │                   │         │
    │         └───────────┬───────┴─────────┬─────────┘         │
    │                     │                 │                    │
    │                     ▼                 ▼                    │
    │            ┌────────────────┐  ┌─────────────┐            │
    │            │   FLEN Model   │  │Training Loop│            │
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
        - Liu, Z., et al. "Field-wise Learning for Multi-field Categorical Data." 
          NeurIPS 2021.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self,
        name: str = "FLEN",
        embed_dim: int = 16,
        mlp_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        mask_mode: str = 'field',
        learning_rate: float = 0.001,
        batch_size: int = 256,
        num_epochs: int = 20,
        patience: int = 5,
        shuffle: bool = True,
        field_groups: Dict[str, List[int]] = None,
        device: str = None,
        seed: int = 42,
        verbose: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the FLEN model.
        
        Args:
            name: Model name
            embed_dim: Embedding dimension
            mlp_dims: Hidden layer dimensions for MLP
            dropout: Dropout rate
            mask_mode: Masking mode ('field' or 'element')
            learning_rate: Learning rate for optimizer
            batch_size: Number of samples per batch
            num_epochs: Maximum number of training epochs
            patience: Early stopping patience
            shuffle: Whether to shuffle data during training
            field_groups: Dictionary mapping group names to field indices
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
            self.mlp_dims = config.get("mlp_dims", mlp_dims)
            self.dropout = config.get("dropout", dropout)
            self.mask_mode = config.get("mask_mode", mask_mode)
            self.learning_rate = config.get("learning_rate", learning_rate)
            self.batch_size = config.get("batch_size", batch_size)
            self.num_epochs = config.get("num_epochs", num_epochs)
            self.patience = config.get("patience", patience)
            self.shuffle = config.get("shuffle", shuffle)
            self.field_groups = config.get("field_groups", field_groups)
        else:
            self.embed_dim = embed_dim
            self.mlp_dims = mlp_dims
            self.dropout = dropout
            self.mask_mode = mask_mode
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.patience = patience
            self.shuffle = shuffle
            self.field_groups = field_groups
        
        # Validate mask mode
        if self.mask_mode not in ['field', 'element']:
            raise ValueError("mask_mode must be 'field' or 'element'")
            
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
        
        # Create field groups if not provided
        if self.field_groups is None:
            # Try to infer user and item fields from field names
            user_fields = []
            item_fields = []
            
            for i, field in enumerate(self.field_names):
                if any(user_term in field.lower() for user_term in ['user', 'uid', 'customer']):
                    user_fields.append(i)
                elif any(item_term in field.lower() for item_term in ['item', 'iid', 'product']):
                    item_fields.append(i)
            
            # If we couldn't identify user/item fields, split evenly
            if not user_fields and not item_fields:
                mid_point = len(self.field_names) // 2
                user_fields = list(range(mid_point))
                item_fields = list(range(mid_point, len(self.field_names)))
            
            # Assign remaining fields
            all_assigned = set(user_fields + item_fields)
            remaining = [i for i in range(len(self.field_names)) if i not in all_assigned]
            
            # Add remaining fields to user or item group based on proximity
            if remaining:
                mid_point = len(self.field_names) // 2
                user_fields.extend([i for i in remaining if i < mid_point])
                item_fields.extend([i for i in remaining if i >= mid_point])
            
            self.field_groups = {
                'user_group': sorted(user_fields),
                'item_group': sorted(item_fields)
            }
        
        if self.verbose:
            self.logger.info(f"Field dimensions: {self.field_dims}")
            self.logger.info(f"Field groups: {self.field_groups}")
    
    def _build_model(self):
        """Build the FLEN model."""
        self.model = FLENModel(
            field_dims=self.field_dims,
            embed_dim=self.embed_dim,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout,
            mask_mode=self.mask_mode,
            field_groups=self.field_groups
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        if self.verbose:
            self.logger.info(f"Built FLEN model with {len(self.field_dims)} fields")
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
    
    def fit(self, data: List[Dict[str, Any]]) -> 'FLEN_base':
        """
        Fit the FLEN model.
        
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
                'mlp_dims': self.mlp_dims,
                'dropout': self.dropout,
                'mask_mode': self.mask_mode,
                'name': self.name,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'patience': self.patience,
                'shuffle': self.shuffle,
                'field_groups': self.field_groups,
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
    def load(cls, filepath: str, device: Optional[str] = None) -> 'FLEN_base':
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
            mlp_dims=model_data['model_config']['mlp_dims'],
            dropout=model_data['model_config']['dropout'],
            mask_mode=model_data['model_config']['mask_mode'],
            learning_rate=model_data['model_config']['learning_rate'],
            batch_size=model_data['model_config']['batch_size'],
            num_epochs=model_data['model_config']['num_epochs'],
            patience=model_data['model_config']['patience'],
            shuffle=model_data['model_config']['shuffle'],
            field_groups=model_data['model_config']['field_groups'],
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