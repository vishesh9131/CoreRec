from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import yaml
import logging
from pathlib import Path

class AttentionLayer(nn.Module):
    """
    Attention Layer for DIN that computes attention weights between user behaviors and target item.
    
    Architecture:
    
    ┌───────────┐   ┌───────────┐
    │  User     │   │  Target   │
    │ Behaviors │   │   Item    │
    └─────┬─────┘   └─────┬─────┘
          │               │
          └───────┬───────┘
                  │
            ┌─────▼─────┐
            │ Attention │
            │   Layer   │
            └─────┬─────┘
                  │
            ┌─────▼─────┐
            │  Output   │
            └───────────┘
            
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    def __init__(self, embed_dim: int, attention_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 4, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, behaviors: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights between behaviors and target item.
        
        Args:
            behaviors: User behavior embeddings [batch_size, seq_len, embed_dim]
            target: Target item embedding [batch_size, embed_dim]
            
        Returns:
            Weighted sum of behavior embeddings [batch_size, embed_dim]
        """
        # Target is already expanded to [batch_size, seq_len, embed_dim]
        # No need to unsqueeze if already 3D
        if target.dim() == 2:
            target = target.unsqueeze(1)  # [batch_size, 1, embed_dim]
            target = target.repeat(1, behaviors.size(1), 1)  # [batch_size, seq_len, embed_dim]
        
        # Compute attention weights
        attention_input = torch.cat([
            behaviors,
            target,
            behaviors * target,
            behaviors - target
        ], dim=-1)
        
        attention_weights = self.attention(attention_input)
        
        # Apply attention weights
        weighted_sum = torch.sum(attention_weights * behaviors, dim=1)
        
        return weighted_sum

class DIN_base(nn.Module):
    """
    Deep Interest Network (DIN) base implementation.
    
    Architecture:
    
    ┌───────────┐
    │  Input    │
    │  Layer    │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │ Embedding │
    │  Layer    │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │ Attention │
    │  Layer    │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │   MLP     │
    │  Layer    │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │  Output   │
    │  Layer    │
    └───────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        mlp_dims: List[int] = [128, 64],
        field_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        attention_dim: int = 32,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        seed: int = 42
    ):
        """
        Initialize DIN model.
        
        Args:
            embed_dim: Embedding dimension
            mlp_dims: List of MLP layer dimensions
            field_dims: List of field dimensions (optional)
            dropout: Dropout rate
            attention_dim: Attention layer dimension
            batch_size: Batch size
            learning_rate: Learning rate
            num_epochs: Number of epochs
            seed: Random seed
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Store parameters
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.field_dims = field_dims
        self.dropout = dropout
        self.attention_dim = attention_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.seed = seed
        
        # Initialize device
        self.device = torch.device('cpu')
        
        # Initialize maps and features
        self.user_map = {}
        self.item_map = {}
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        self.feature_encoders = {}
        
        # Initialize model components
        self.is_fitted = False
        self.model = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize loss history
        self.loss_history = []
    
    def build_model(self):
        """
        Build the DIN model architecture.
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Create embeddings for each field
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, self.embed_dim)
            for dim in self.field_dims
        ])
        
        # Attention layer
        self.attention = AttentionLayer(
            self.embed_dim,
            self.attention_dim
        )
        
        # MLP layers
        self.mlp = nn.ModuleList()
        input_dim = self.embed_dim * 2  # Concatenated user and item features
        
        for dim in self.mlp_dims:
            self.mlp.append(nn.Linear(input_dim, dim))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(self.dropout))
            input_dim = dim
        
        # Output layer
        self.output_layer = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, user_behaviors: torch.Tensor, target_item: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DIN model.
        
        Args:
            user_behaviors: User behavior indices [batch_size, seq_len]
            target_item: Target item indices [batch_size]
            
        Returns:
            Predicted probability
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Get embeddings
        user_embed = self.embeddings[0](user_behaviors)  # [batch_size, seq_len, embed_dim]
        item_embed = self.embeddings[1](target_item)     # [batch_size, embed_dim]
        
        # Expand item_embed to match sequence length
        item_embed_expanded = item_embed.unsqueeze(1).expand(-1, user_embed.size(1), -1)  # [batch_size, seq_len, embed_dim]
        
        # Apply attention
        attended_user = self.attention(user_embed, item_embed_expanded)  # [batch_size, embed_dim]
        
        # Concatenate features
        combined = torch.cat([attended_user, item_embed], dim=-1)  # [batch_size, embed_dim * 2]
        
        # MLP layers
        x = combined
        for layer in self.mlp:
            x = layer(x)
        
        # Output layer
        output = self.sigmoid(self.output_layer(x))
        
        return output
    
    def fit(self, interactions: List[Tuple]):
        """
        Fit the DIN model to interactions.
        
        Args:
            interactions: List of (user, item, features) tuples
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Extract features and create mappings
        self._extract_features(interactions)
        
        # Build model
        self.build_model()
        
        # Move model to device
        self.to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        
        # Training loop
        self.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            batch_count = 0
            
            # Process in batches
            for i in range(0, len(interactions), self.batch_size):
                batch = interactions[i:i + self.batch_size]
                
                # Prepare batch data
                user_behaviors, target_item, labels = self._prepare_batch(batch)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self(user_behaviors, target_item)
                
                # Compute loss
                loss = F.binary_cross_entropy(
                    outputs.squeeze(),
                    labels.float()
                )
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                batch_count += 1
            
            # Record epoch loss
            avg_loss = total_loss / batch_count
            self.loss_history.append(avg_loss)
            
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
    
    def predict(self, user: Any, item: Any, features: Dict[str, Any]) -> float:
        """
        Predict probability of interaction between user and item.
        
        Args:
            user: User ID
            item: Item ID
            features: Features dictionary
            
        Returns:
            Predicted probability
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")
        
        # Check if user and item exist
        if user not in self.user_map:
            raise ValueError(f"Unknown user: {user}")
        if item not in self.item_map:
            raise ValueError(f"Unknown item: {item}")
        
        # Prepare features
        user_behaviors = torch.tensor([self.user_map[user]], device=self.device)
        target_item = torch.tensor([self.item_map[item]], device=self.device)
        
        # Make prediction
        self.eval()
        with torch.no_grad():
            prediction = self(user_behaviors, target_item)
        
        return prediction.item()
    
    def recommend(
        self,
        user: Any,
        top_n: int = 10,
        exclude_seen: bool = True,
        features: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Any, float]]:
        """
        Generate recommendations for user.
        
        Args:
            user: User ID
            top_n: Number of recommendations
            exclude_seen: Whether to exclude seen items
            features: Optional features dictionary
            
        Returns:
            List of (item, score) tuples
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")
        
        if user not in self.user_map:
            return []
        
        # Get predictions for all items
        predictions = []
        for item in self.item_map:
            try:
                score = self.predict(user, item, features or {})
                predictions.append((item, score))
            except Exception:
                continue
        
        # Sort by score
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_n]
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'user_map': self.user_map,
            'item_map': self.item_map,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'feature_encoders': self.feature_encoders,
            'field_dims': self.field_dims,
            'config': {
                'embed_dim': self.embed_dim,
                'mlp_dims': self.mlp_dims,
                'dropout': self.dropout,
                'attention_dim': self.attention_dim,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'seed': self.seed
            }
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded model
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Load checkpoint
        checkpoint = torch.load(filepath)
        
        # Create new instance
        instance = cls(
            embed_dim=checkpoint['config']['embed_dim'],
            mlp_dims=checkpoint['config']['mlp_dims'],
            field_dims=checkpoint['field_dims'],
            dropout=checkpoint['config']['dropout'],
            attention_dim=checkpoint['config']['attention_dim'],
            batch_size=checkpoint['config']['batch_size'],
            learning_rate=checkpoint['config']['learning_rate'],
            num_epochs=checkpoint['config']['num_epochs'],
            seed=checkpoint['config']['seed']
        )
        
        # Restore state
        instance.user_map = checkpoint['user_map']
        instance.item_map = checkpoint['item_map']
        instance.feature_names = checkpoint['feature_names']
        instance.categorical_features = checkpoint['categorical_features']
        instance.numerical_features = checkpoint['numerical_features']
        instance.feature_encoders = checkpoint['feature_encoders']
        
        # Build model and load state
        instance.build_model()
        instance.load_state_dict(checkpoint['model_state_dict'])
        
        # Create optimizer and load state
        instance.optimizer = torch.optim.Adam(
            instance.parameters(),
            lr=instance.learning_rate
        )
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        instance.is_fitted = True
        
        return instance
    
    def _extract_features(self, interactions: List[Tuple]):
        """
        Extract features from interactions.
        
        Args:
            interactions: List of (user, item, features) tuples
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Extract users and items
        users = set()
        items = set()
        for user, item, _ in interactions:
            users.add(user)
            items.add(item)
        
        # Create mappings
        self.user_map = {user: idx for idx, user in enumerate(sorted(users))}
        self.item_map = {item: idx for idx, item in enumerate(sorted(items))}
        
        # Set field dimensions
        self.field_dims = [len(users), len(items)]
    
    def _prepare_batch(self, batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for training.
        
        Args:
            batch: List of (user, item, features) tuples
            
        Returns:
            Tuple of (user_behaviors, target_item, labels)
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        user_behaviors = []
        target_items = []
        labels = []
        
        # Create sequence of user behaviors
        for user, item, features in batch:
            # Create a sequence of user behaviors (repeating the same user for now)
            user_seq = [self.user_map[user]] * 10  # Fixed sequence length of 10
            user_behaviors.append(user_seq)
            target_items.append(self.item_map[item])
            labels.append(1)  # Assuming all interactions are positive
        
        return (
            torch.tensor(user_behaviors, device=self.device),  # [batch_size, seq_len]
            torch.tensor(target_items, device=self.device),    # [batch_size]
            torch.tensor(labels, device=self.device)           # [batch_size]
        )
