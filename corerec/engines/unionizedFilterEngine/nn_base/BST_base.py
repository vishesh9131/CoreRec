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
import random
from tqdm import tqdm
from datetime import datetime
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
    
    def get_activation(self, layer_name):
        """Get the activation for a specific layer."""
        return self.activations.get(layer_name, None)
    
    def clear_activations(self):
        """Clear all stored activations."""
        self.activations.clear()


class TokenEmbedding(nn.Module):
    """Embedding layer for tokens (items, users, etc.)."""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize the token embedding layer.
        
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Embedding dimension.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the token embedding layer.
        
        Args:
            x: Input tensor of token indices.
        
        Returns:
            Embedded tensor.
        """
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """Embedding layer for positions in a sequence."""
    
    def __init__(self, max_seq_len: int, embedding_dim: int):
        """
        Initialize the positional embedding layer.
        
        Args:
            max_seq_len: Maximum sequence length.
            embedding_dim: Embedding dimension.
        """
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional embedding layer.
        
        Args:
            x: Input tensor of any shape. Position indices are created based on the second dimension.
        
        Returns:
            Positional embedding tensor.
        """
        batch_size, seq_len = x.size()[:2]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.embedding(positions)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize the multi-head attention module.
        
        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor = None, value: torch.Tensor = None, 
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the multi-head attention module.
        
        Args:
            query: Query tensor.
            key: Key tensor. If None, uses query.
            value: Value tensor. If None, uses query.
            mask: Optional mask tensor.
        
        Returns:
            Tuple of (output, attention_weights).
        """
        batch_size = query.size(0)
        
        if key is None:
            key = query
        if value is None:
            value = query
        
        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to multi-head format
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape back to original format
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """Feed-forward neural network."""
    
    def __init__(self, hidden_dim: int, feed_forward_dim: int, dropout: float = 0.1):
        """
        Initialize the feed-forward network.
        
        Args:
            hidden_dim: Hidden dimension.
            feed_forward_dim: Feed-forward layer dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.linear1 = nn.Linear(hidden_dim, feed_forward_dim)
        self.linear2 = nn.Linear(feed_forward_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.
        
        Args:
            x: Input tensor.
        
        Returns:
            Processed tensor.
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, hidden_dim: int, num_heads: int, feed_forward_dim: int, dropout: float = 0.1):
        """
        Initialize the transformer block.
        
        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            feed_forward_dim: Feed-forward layer dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = FeedForward(hidden_dim, feed_forward_dim, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer block.
        
        Args:
            x: Input tensor.
            mask: Optional attention mask.
        
        Returns:
            Tuple of (output, attention_weights).
        """
        # Self-attention with residual connection and normalization
        attn_output, attn_weights = self.attention(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


class FeatureEmbedding(nn.Module):
    """Embedding layer for categorical features."""
    
    def __init__(self, field_dims: List[int], embedding_dim: int):
        """
        Initialize the feature embedding layer.
        
        Args:
            field_dims: List of feature vocabulary sizes.
            embedding_dim: Embedding dimension.
        """
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0)
            for dim in field_dims
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature embedding layer.
        
        Args:
            x: Input tensor of categorical feature indices.
        
        Returns:
            Embedded tensor.
        """
        return torch.cat([
            embedding(x[:, :, i])
            for i, embedding in enumerate(self.embeddings)
        ], dim=-1)


class BehaviorSequenceTransformer(nn.Module):
    """Behavior Sequence Transformer model."""
    
    def __init__(self, 
                 item_vocab_size: int,
                 feature_field_dims: List[int],
                 max_seq_len: int, 
                 hidden_dim: int, 
                 num_heads: int, 
                 num_layers: int, 
                 feed_forward_dim: int,
                 dropout: float = 0.1):
        """
        Initialize the BST model.
        
        Args:
            item_vocab_size: Size of the item vocabulary.
            feature_field_dims: List of feature field dimensions.
            max_seq_len: Maximum sequence length.
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            feed_forward_dim: Feed-forward dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        
        # Embedding layers
        self.item_embedding = TokenEmbedding(item_vocab_size, hidden_dim)
        self.pos_embedding = PositionalEmbedding(max_seq_len, hidden_dim)
        
        # Feature embeddings for target item
        self.feature_embedding = FeatureEmbedding(feature_field_dims, hidden_dim // len(feature_field_dims))
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim, 
                num_heads=num_heads, 
                feed_forward_dim=feed_forward_dim, 
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Prediction layer
        self.pred_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, seq_items: torch.Tensor, target_item: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BST model.
        
        Args:
            seq_items: Sequence of item IDs (batch_size, seq_len).
            target_item: Target item IDs (batch_size).
            target_features: Target item features (batch_size, num_features).
        
        Returns:
            Prediction scores.
        """
        # Get sequence length
        batch_size, seq_len = seq_items.size()
        
        # Embed sequence items
        seq_emb = self.item_embedding(seq_items)
        pos_emb = self.pos_embedding(seq_items)
        seq_emb = seq_emb + pos_emb
        
        # Apply transformer blocks
        attention_weights_list = []
        for block in self.transformer_blocks:
            seq_emb, attn_weights = block(seq_emb)
            attention_weights_list.append(attn_weights)
        
        # Get sequence representation (last position)
        seq_rep = seq_emb[:, -1, :]
        
        # Embed target item
        target_item_emb = self.item_embedding(target_item.unsqueeze(1)).squeeze(1)
        target_feature_emb = self.feature_embedding(target_features)
        target_rep = torch.cat([target_item_emb, target_feature_emb], dim=-1)
        
        # Concatenate sequence and target representations
        combined_rep = torch.cat([seq_rep, target_rep], dim=1)
        
        # Predict score
        pred = self.pred_layer(combined_rep)
        
        return pred, attention_weights_list
    
    def get_attention_weights(self, seq_items: torch.Tensor, target_item: torch.Tensor, target_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention weights for visualization.
        
        Args:
            seq_items: Sequence of item IDs.
            target_item: Target item ID.
            target_features: Target item features.
        
        Returns:
            List of attention weight tensors.
        """
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(seq_items, target_item, target_features)
        return attention_weights


class BST_base(BaseCorerec):
    """Base class for Behavior Sequence Transformer recommender."""
    
    def __init__(self, 
                 name: str = "BST", 
                 config: Dict[str, Any] = None,
                 trainable: bool = True,
                 verbose: bool = True,
                 seed: int = 42):
        """
        Initialize the BST base recommender.
        
        Args:
            name: Name of the recommender.
            config: Configuration dictionary.
            trainable: Whether the model is trainable.
            verbose: Whether to print verbose output.
            seed: Random seed.
        """
        super().__init__(name)
        
        self.trainable = trainable
        self.verbose = verbose
        self.seed = seed
        self.is_fitted = False
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        # Set up hook manager
        self.hooks = HookManager()
        
        # Default configuration
        default_config = {
            'hidden_dim': 64,
            'num_heads': 2,
            'num_layers': 2,
            'feed_forward_dim': 256,
            'max_seq_len': 20,
            'dropout': 0.1,
            'batch_size': 64,
            'num_epochs': 10,
            'learning_rate': 0.001,
            'l2_reg': 0.0,
            'early_stopping_patience': 3,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 4
        }
        
        # Update with user configuration
        self.config = default_config.copy()
        if config is not None:
            self.config.update(config)
        
        # Set device
        self.device = self.config['device']
        
        # Initialize user and item mappings
        self.user_ids = None
        self.item_ids = None
        self.uid_map = {}
        self.iid_map = {}
        self.num_users = 0
        self.num_items = 0
        
        # Initialize sequence and feature data
        self.user_sequences = []
        self.item_features = {}
        self.feature_field_dims = []
    
    def _build_model(self):
        """Build the BST model."""
        self.model = BehaviorSequenceTransformer(
            item_vocab_size=self.num_items + 1,  # +1 for padding
            feature_field_dims=self.feature_field_dims,
            max_seq_len=self.config['max_seq_len'],
            hidden_dim=self.config['hidden_dim'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            feed_forward_dim=self.config['feed_forward_dim'],
            dropout=self.config['dropout']
        ).to(self.device)
    
    def fit(self, interactions, user_ids=None, item_ids=None, item_features=None):
        """
        Fit the model to the given data.
        
        Args:
            interactions: List of (user_id, item_id, timestamp, [features]) tuples or similar structure.
            user_ids: List of user IDs.
            item_ids: List of item IDs.
            item_features: Dictionary mapping item_id to feature values.
        
        Returns:
            Fitted model.
        """
        # Process user and item IDs
        if user_ids is not None:
            self.user_ids = list(user_ids)
            self.uid_map = {uid: i for i, uid in enumerate(self.user_ids)}
            self.num_users = len(self.user_ids)
        
        if item_ids is not None:
            self.item_ids = list(item_ids)
            self.iid_map = {iid: i + 1 for i, iid in enumerate(self.item_ids)}  # +1 for padding
            self.num_items = len(self.item_ids)
        
        # Process item features if provided
        if item_features is not None:
            self.item_features = item_features
            
            # Extract feature dimensions
            first_item = next(iter(item_features.values()))
            if isinstance(first_item, dict):
                feature_names = list(first_item.keys())
                all_values = defaultdict(set)
                
                for features in item_features.values():
                    for name, value in features.items():
                        all_values[name].add(value)
                
                self.feature_field_dims = [len(all_values[name]) + 1 for name in feature_names]  # +1 for padding
                
                # Create feature value mapping
                self.feature_value_maps = {}
                for name in feature_names:
                    self.feature_value_maps[name] = {val: i + 1 for i, val in enumerate(all_values[name])}  # +1 for padding
            else:
                # Assume features are already encoded as integers
                max_values = defaultdict(int)
                for i, features in enumerate(item_features.values()):
                    for j, val in enumerate(features):
                        max_values[j] = max(max_values[j], val)
                
                self.feature_field_dims = [max_val + 1 for max_val in max_values.values()]
        else:
            # Default to a single binary feature
            self.feature_field_dims = [2]
            self.item_features = {iid: [1] for iid in self.item_ids}
        
        # Process interactions into sequences
        self._process_sequences(interactions)
        
        # Build the model
        self._build_model()
        
        # Create dataset and dataloader
        train_dataset = self._create_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['l2_reg']
        )
        
        # Initialize loss function
        criterion = nn.BCELoss()
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            total_loss = 0
            
            if self.verbose:
                pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            for batch in train_loader:
                seq_items, target_item, target_features, label = batch
                seq_items = seq_items.to(self.device)
                target_item = target_item.to(self.device)
                target_features = target_features.to(self.device)
                label = label.to(self.device)
                
                optimizer.zero_grad()
                
                pred, _ = self.model(seq_items, target_item, target_features)
                loss = criterion(pred, label)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if self.verbose:
                    pbar.update(1)
            
            avg_loss = total_loss / len(train_loader)
            
            if self.verbose:
                pbar.close()
                print(f"Epoch {epoch+1}/{self.config['num_epochs']}, Loss: {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.is_fitted = True
        return self
    
    def _process_sequences(self, interactions):
        """
        Process interactions into user sequences.
        
        Args:
            interactions: List of (user_id, item_id, timestamp) tuples.
        """
        # Sort interactions by user and timestamp
        sorted_interactions = sorted(interactions, key=lambda x: (x[0], x[2]))
        
        # Group by user
        user_sequences = defaultdict(list)
        for user_id, item_id, timestamp in sorted_interactions:
            if user_id in self.uid_map and item_id in self.iid_map:
                user_sequences[self.uid_map[user_id]].append(self.iid_map[item_id])
        
        # Convert to list of sequences
        self.user_sequences = [user_sequences.get(i, []) for i in range(self.num_users)]
    
    def _create_dataset(self):
        """
        Create a dataset for training.
        
        Returns:
            PyTorch dataset.
        """
        samples = []
        max_seq_len = self.config['max_seq_len']
        
        for user_idx, sequence in enumerate(self.user_sequences):
            if len(sequence) < 2:
                continue
            
            # Create positive samples
            for i in range(1, len(sequence)):
                # Determine sequence length
                start_idx = max(0, i - max_seq_len)
                seq = sequence[start_idx:i]
                
                # Pad sequence if needed
                if len(seq) < max_seq_len:
                    seq = [0] * (max_seq_len - len(seq)) + seq
                
                target_item = sequence[i]
                
                # Get item features
                target_iid = self.item_ids[target_item - 1]  # -1 to adjust for padding
                if target_iid in self.item_features:
                    target_features = self.item_features[target_iid]
                    if isinstance(target_features, dict):
                        # Convert feature dict to indices
                        feature_indices = []
                        for name, value_map in self.feature_value_maps.items():
                            feature_indices.append(value_map.get(target_features.get(name, 0), 0))
                        target_features = feature_indices
                else:
                    target_features = [0] * len(self.feature_field_dims)
                
                # Reshape features for transformer input
                target_features = np.array([target_features])
                
                # Add positive sample
                samples.append((seq, target_item, target_features, 1.0))
                
                # Create negative sample (random item)
                neg_item = np.random.randint(1, self.num_items + 1)
                while neg_item in sequence:
                    neg_item = np.random.randint(1, self.num_items + 1)
                
                # Get negative item features
                neg_iid = self.item_ids[neg_item - 1]  # -1 to adjust for padding
                if neg_iid in self.item_features:
                    neg_features = self.item_features[neg_iid]
                    if isinstance(neg_features, dict):
                        # Convert feature dict to indices
                        feature_indices = []
                        for name, value_map in self.feature_value_maps.items():
                            feature_indices.append(value_map.get(neg_features.get(name, 0), 0))
                        neg_features = feature_indices
                else:
                    neg_features = [0] * len(self.feature_field_dims)
                
                # Reshape features for transformer input
                neg_features = np.array([neg_features])
                
                # Add negative sample
                samples.append((seq, neg_item, neg_features, 0.0))
        
        # Create dataset
        class BSTDataset(torch.utils.data.Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                seq_items, target_item, target_features, label = self.samples[idx]
                return (
                    torch.tensor(seq_items, dtype=torch.long),
                    torch.tensor(target_item, dtype=torch.long),
                    torch.tensor(target_features, dtype=torch.long),
                    torch.tensor(label, dtype=torch.float32).unsqueeze(0)
                )
        
        return BSTDataset(samples)
    
    def recommend(self, user_id, top_n=10, exclude_seen=True):
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID.
            top_n: Number of recommendations.
            exclude_seen: Whether to exclude seen items.
        
        Returns:
            List of (item_id, score) tuples.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        if user_id not in self.uid_map:
            raise ValueError(f"User {user_id} not found in training data.")
        
        user_idx = self.uid_map[user_id]
        
        # Get user sequence
        sequence = self.user_sequences[user_idx][-self.config['max_seq_len']:]
        
        # Pad sequence
        if len(sequence) < self.config['max_seq_len']:
            sequence = [0] * (self.config['max_seq_len'] - len(sequence)) + sequence
        
        # Convert to tensor
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
        
        # Get scores for all items
        self.model.eval()
        with torch.no_grad():
            scores = []
            
            # Get seen items for exclusion
            seen_items = set(self.user_sequences[user_idx]) if exclude_seen else set()
            
            # Evaluate items in batches
            batch_size = 100
            item_batches = [list(range(1, self.num_items + 1))[i:i+batch_size] 
                           for i in range(0, self.num_items, batch_size)]
            
            for item_batch in item_batches:
                batch_scores = []
                
                for item_idx in item_batch:
                    if item_idx in seen_items:
                        batch_scores.append(-float('inf'))
                        continue
                    
                    # Get item features
                    item_id = self.item_ids[item_idx - 1]  # -1 to adjust for padding
                    if item_id in self.item_features:
                        item_features = self.item_features[item_id]
                        if isinstance(item_features, dict):
                            # Convert feature dict to indices
                            feature_indices = []
                            for name, value_map in self.feature_value_maps.items():
                                feature_indices.append(value_map.get(item_features.get(name, 0), 0))
                            item_features = feature_indices
                    else:
                        item_features = [0] * len(self.feature_field_dims)
                    
                    # Reshape features for transformer input
                    item_features = np.array([item_features])
                    item_features_tensor = torch.tensor(item_features, dtype=torch.long).to(self.device)
                    
                    item_idx_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
                    pred, _ = self.model(sequence_tensor, item_idx_tensor, item_features_tensor)
                    batch_scores.append(pred.item())
                
                scores.extend(batch_scores)
        
        # Get top items
        item_scores = [(self.item_ids[i-1], score) for i, score in enumerate(scores, 1)]
        item_scores = [(item, score) for item, score in item_scores if score > -float('inf')]
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return item_scores[:top_n]
    
    def register_hook(self, layer_name, callback=None):
        """
        Register a hook for a specific layer.
        
        Args:
            layer_name: Name of the layer.
            callback: Callback function.
        
        Returns:
            True if hook was registered, False otherwise.
        """
        return self.hooks.register_hook(self.model, layer_name, callback)
    
    def remove_hook(self, layer_name):
        """
        Remove a hook for a specific layer.
        
        Args:
            layer_name: Name of the layer.
        
        Returns:
            True if hook was removed, False otherwise.
        """
        return self.hooks.remove_hook(layer_name)
    
    def get_attention_weights(self, user_id, item_id=None):
        """
        Get attention weights for visualization.
        
        Args:
            user_id: User ID.
            item_id: Optional target item ID. If None, uses the last item in the sequence.
        
        Returns:
            List of attention weight matrices.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        if user_id not in self.uid_map:
            raise ValueError(f"User {user_id} not found in training data.")
        
        user_idx = self.uid_map[user_id]
        
        # Get user sequence
        sequence = self.user_sequences[user_idx][-self.config['max_seq_len']:]
        
        # Pad sequence
        if len(sequence) < self.config['max_seq_len']:
            sequence = [0] * (self.config['max_seq_len'] - len(sequence)) + sequence
        
        # Convert to tensor
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
        
        # Get target item
        if item_id is None:
            # Use the next item in the user's sequence
            if len(self.user_sequences[user_idx]) > 0:
                target_idx = self.user_sequences[user_idx][-1]
            else:
                raise ValueError(f"User {user_id} has no items in their sequence.")
        else:
            if item_id not in self.iid_map:
                raise ValueError(f"Item {item_id} not found in training data.")
            target_idx = self.iid_map[item_id]
        
        # Get item features
        target_iid = self.item_ids[target_idx - 1]  # -1 to adjust for padding
        if target_iid in self.item_features:
            target_features = self.item_features[target_iid]
            if isinstance(target_features, dict):
                # Convert feature dict to indices
                feature_indices = []
                for name, value_map in self.feature_value_maps.items():
                    feature_indices.append(value_map.get(target_features.get(name, 0), 0))
                target_features = feature_indices
        else:
            target_features = [0] * len(self.feature_field_dims)
        
        # Reshape features for transformer input
        target_features = np.array([target_features])
        target_features_tensor = torch.tensor(target_features, dtype=torch.long).to(self.device)
        
        target_idx_tensor = torch.tensor([target_idx], dtype=torch.long).to(self.device)
        
        # Get attention weights
        self.model.eval()
        with torch.no_grad():
            attention_weights = self.model.get_attention_weights(
                sequence_tensor, target_idx_tensor, target_features_tensor
            )
        
        # Convert to numpy arrays
        return [w.cpu().numpy() for w in attention_weights]
    
    def save(self, path):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        # Create directory if it doesn't exist
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_state = {
            'config': self.config,
            'state_dict': self.model.state_dict(),
            'user_ids': self.user_ids,
            'item_ids': self.item_ids,
            'uid_map': self.uid_map,
            'iid_map': self.iid_map,
            'user_sequences': self.user_sequences,
            'item_features': self.item_features,
            'feature_field_dims': self.feature_field_dims,
            'name': self.name,
            'trainable': self.trainable,
            'verbose': self.verbose,
            'seed': self.seed
        }
        
        if hasattr(self, 'feature_value_maps'):
            model_state['feature_value_maps'] = self.feature_value_maps
        
        # Save model
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(model_state, f)
        
        # Save metadata
        with open(f"{path}.meta", 'w') as f:
            yaml.dump({
                'name': self.name,
                'type': 'BST',
                'version': '1.0',
                'num_users': self.num_users,
                'num_items': self.num_items,
                'hidden_dim': self.config['hidden_dim'],
                'num_layers': self.config['num_layers'],
                'num_heads': self.config['num_heads'],
                'max_seq_len': self.config['max_seq_len'],
                'created_at': str(datetime.now())
            }, f)
    
    @classmethod
    def load(self, model_state):
        # Restore model state
        self.user_ids = model_state['user_ids']
        self.item_ids = model_state['item_ids']
        self.uid_map = model_state['uid_map']
        self.iid_map = model_state['iid_map']
        self.user_sequences = model_state['user_sequences']
        self.item_features = model_state['item_features']
        self.feature_field_dims = model_state['feature_field_dims']
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        
        if 'feature_value_maps' in model_state:
            self.feature_value_maps = model_state['feature_value_maps']
        
        # Build model architecture
        self._build_model()
        
        # Load model weights
        self.model.load_state_dict(model_state['state_dict'])
        
        # Set fitted flag
        self.is_fitted = True
        
        return self
    
    def update_incremental(self, new_interactions, new_user_ids=None, new_item_ids=None, new_item_features=None):
        """
        Update model incrementally with new interactions.
        
        Args:
            new_interactions: New interaction data (user-item-timestamp).
            new_user_ids: New user IDs (if any).
            new_item_ids: New item IDs (if any).
            new_item_features: New item features (if any).
        
        Returns:
            Updated model.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        # Check if we have new users or items
        if new_user_ids is not None:
            # Add new users to mappings
            new_users = [uid for uid in new_user_ids if uid not in self.uid_map]
            for uid in new_users:
                self.uid_map[uid] = len(self.uid_map)
                self.user_ids.append(uid)
            self.num_users = len(self.user_ids)
        
        if new_item_ids is not None:
            # Add new items to mappings
            new_items = [iid for iid in new_item_ids if iid not in self.iid_map]
            for iid in new_items:
                self.iid_map[iid] = len(self.iid_map)
                self.item_ids.append(iid)
            self.num_items = len(self.item_ids)
            
            # If we have new item features, update item features
            if new_item_features is not None:
                for iid, features in new_item_features.items():
                    self.item_features[iid] = features
            
            # If new items were added, rebuild the model
            if new_items and hasattr(self, 'model'):
                old_state_dict = self.model.state_dict()
                self._build_model()
                
                # Copy weights for existing parameters
                new_state_dict = self.model.state_dict()
                for name, param in old_state_dict.items():
                    if name in new_state_dict and new_state_dict[name].shape == param.shape:
                        new_state_dict[name] = param
                
                self.model.load_state_dict(new_state_dict)
        
        # Update user sequences with new interactions
        self._update_user_sequences(new_interactions)
        
        # Fine-tune on new data
        self.fit(new_interactions, epochs=self.config.get('incremental_epochs', 5))
        
        return self
    
    def _update_user_sequences(self, interactions):
        """
        Update user sequences with new interactions.
        
        Args:
            interactions: New interactions (user-item-timestamp).
        """
        # Process interactions into timestamped sequences
        timestamp_data = []
        for u, i, t in interactions:
            if u in self.uid_map and i in self.iid_map:
                user_idx = self.uid_map[u]
                item_idx = self.iid_map[i]
                timestamp_data.append((user_idx, item_idx, t))
        
        # Sort by user_idx and timestamp
        timestamp_data.sort(key=lambda x: (x[0], x[2]))
        
        # Update user sequences
        current_user = -1
        for user_idx, item_idx, _ in timestamp_data:
            if user_idx != current_user:
                current_user = user_idx
                if user_idx >= len(self.user_sequences):
                    # Add new user sequence
                    self.user_sequences.append([])
            
            # Add item to sequence if not already there
            if not self.user_sequences[user_idx] or self.user_sequences[user_idx][-1] != item_idx:
                self.user_sequences[user_idx].append(item_idx)
    
    def export_embeddings(self):
        """
        Export item embeddings.
        
        Returns:
            Dict mapping item IDs to embeddings.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        # Get item embeddings
        item_embeddings = {}
        
        # Create a tensor with item indices
        item_indices = torch.arange(1, self.num_items + 1, dtype=torch.long).to(self.device)
        
        # Get item embeddings from the model
        with torch.no_grad():
            embeddings = self.model.item_embedding(item_indices).cpu().numpy()
        
        # Create mapping from item ID to embedding
        for i, iid in enumerate(self.item_ids):
            item_embeddings[iid] = embeddings[i].tolist()
        
        return item_embeddings
    
    def set_device(self, device):
        """
        Set the device to run the model on.
        
        Args:
            device: Device to run the model on.
        """
        self.device = device
        if hasattr(self, 'model') and self.model is not None:
            self.model.to(device)