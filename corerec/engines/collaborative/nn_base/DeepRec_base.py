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
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from corerec.api.base_recommender import BaseRecommender


class AttentionLayer(nn.Module):
    """
    Attention Layer for DeepRec model.
    
    Architecture:
    
    ┌───────────┐   ┌───────────┐
    │  Query    │   │   Key     │
    └─────┬─────┘   └─────┬─────┘
          │               │
          └───────┬───────┘
                  │
                  ▼
           ┌─────────────┐
           │ Attention   │
           │  Scores     │
           └──────┬──────┘
                  │
                  ▼
          ┌───────────────┐
          │    Softmax    │
          └───────┬───────┘
                  │
                  ▼
         ┌─────────────────┐
    ┌────┤ Weighted Values │
    │    └─────────────────┘
    │
    ▼
    Output
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, embed_dim: int, attention_dim: int = 64):
        super().__init__()
        self.query_layer = nn.Linear(embed_dim, attention_dim)
        self.key_layer = nn.Linear(embed_dim, attention_dim)
        self.energy_layer = nn.Linear(attention_dim, 1, bias=False)
        
        # Initialize weights
        nn.init.xavier_normal_(self.query_layer.weight)
        nn.init.xavier_normal_(self.key_layer.weight)
        nn.init.xavier_normal_(self.energy_layer.weight)
    
    def forward(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Reshape query to match batch dimension
        query = query.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # Project query and keys
        query_proj = self.query_layer(query)  # (batch_size, 1, attention_dim)
        key_proj = self.key_layer(keys)  # (batch_size, seq_len, attention_dim)
        
        # Calculate attention scores
        query_key = torch.tanh(query_proj + key_proj)  # (batch_size, seq_len, attention_dim)
        scores = self.energy_layer(query_key).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1).unsqueeze(1)  # (batch_size, 1, seq_len)
        
        # Apply attention weights to values
        context = torch.bmm(attention_weights, values).squeeze(1)  # (batch_size, embed_dim)
        
        return context


class SequenceEncoder(nn.Module):
    """
    Sequence Encoder for DeepRec model.
    
    Architecture:
    
    ┌───────────┐
    │ Sequences │
    └─────┬─────┘
          │
          ▼
    ┌─────────────┐
    │     GRU     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Attention  │
    └──────┬──────┘
           │
           ▼
     ┌───────────┐
     │  Encoded  │
     │ Sequences │
     └───────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int = 128, num_layers: int = 1, attention_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = AttentionLayer(embed_dim=hidden_dim, attention_dim=attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor, item_embed: torch.Tensor) -> torch.Tensor:
        # Pack padded sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Pass through GRU
        outputs, _ = self.gru(packed_x)
        
        # Unpack sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # Create mask based on sequence lengths
        batch_size, max_len = x.size(0), x.size(1)
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        
        # Apply attention
        attended = self.attention(item_embed, outputs, outputs, mask)
        
        # Apply dropout
        attended = self.dropout(attended)
        
        return attended


class DeepRecModel(nn.Module):
    """
    Deep Recommendation Model with Attention and Sequence Modeling.
    
    Architecture:
    
    ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
    │   User    │ │   Item    │ │ Sequence  │ │ Context   │
    │ Embedding │ │ Embedding │ │  History  │ │ Features  │
    └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
          │             │             │             │
          │             │             ▼             │
          │             │       ┌──────────┐        │
          │             │       │ Sequence │        │
          │             │       │ Encoder  │        │
          │             │       └────┬─────┘        │
          │             │            │              │
          └─────────────┼────────────┼──────────────┘
                        │            │
                        ▼            ▼
                  ┌─────────────────────────┐
                  │          MLP            │
                  └────────────┬────────────┘
                               │
                               ▼
                          ┌─────────┐
                          │ Output  │
                          └─────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self, 
        field_dims: List[int], 
        embed_dim: int = 64,
        hidden_dim: int = 128,
        mlp_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        attention_dim: int = 64,
        num_gru_layers: int = 1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Embedding layers
        self.embedding = nn.ModuleList([
            nn.Embedding(field_dim, embed_dim, padding_idx=0)
            for field_dim in field_dims
        ])
        
        # Initialize embeddings
        for embedding in self.embedding:
            nn.init.normal_(embedding.weight, std=0.01)
        
        # Sequence encoder
        self.seq_encoder = SequenceEncoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gru_layers,
            attention_dim=attention_dim,
            dropout=dropout
        )
        
        # Calculate input dimension for MLP
        # User embed + Item embed + Seq encoder output + Context features
        num_categorical = len(field_dims) - 2  # Exclude user and item fields
        num_numerical = 0  # Will be set by DeepRec_base
        mlp_input_dim = embed_dim * 2 + hidden_dim + num_numerical
        
        # MLP layers
        self.mlp = nn.Sequential()
        input_dim = mlp_input_dim
        for i, dim in enumerate(mlp_dims):
            self.mlp.add_module(f'linear_{i}', nn.Linear(input_dim, dim))
            self.mlp.add_module(f'relu_{i}', nn.ReLU())
            self.mlp.add_module(f'dropout_{i}', nn.Dropout(dropout))
            input_dim = dim
        
        # Output layer
        self.output_layer = nn.Linear(input_dim, 1)
        
        # Initialize MLP weights
        for name, module in self.mlp.named_children():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
        nn.init.xavier_normal_(self.output_layer.weight)
        
        # Hooks for model introspection
        self.hooks = {}
        self.activations = {}
    
    def forward(
        self, 
        x: torch.Tensor, 
        seq_x: torch.Tensor, 
        seq_lengths: torch.Tensor,
        numerical_x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get user and item embeddings
        user_idx = x[:, 0]
        item_idx = x[:, 1]
        
        user_embed = self.embedding[0](user_idx)
        item_embed = self.embedding[1](item_idx)
        
        # Get sequence embeddings - convert indices to embeddings
        batch_size, seq_len = seq_x.size()
        seq_embed = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device)
        
        for i in range(batch_size):
            length = seq_lengths[i]
            if length > 0:
                seq_embed[i, :length] = self.embedding[1](seq_x[i, :length])
        
        # Encode sequence with attention
        seq_encoded = self.seq_encoder(seq_embed, seq_lengths, item_embed)
        
        # Concatenate all inputs for MLP
        mlp_input = [user_embed, item_embed, seq_encoded]
        
        # Add numerical features if provided
        if numerical_x is not None:
            mlp_input.append(numerical_x)
        
        mlp_input = torch.cat(mlp_input, dim=1)
        
        # Apply MLP
        mlp_output = self.mlp(mlp_input)
        
        # Apply output layer and sigmoid
        pred = torch.sigmoid(self.output_layer(mlp_output))
        
        return pred
    
    def register_hook(self, layer_name: str) -> bool:
        # Get layer by name
        layer = None
        if layer_name == 'embedding_0':
            layer = self.embedding[0]
        elif layer_name == 'embedding_1':
            layer = self.embedding[1]
        elif layer_name == 'seq_encoder':
            layer = self.seq_encoder
        elif layer_name == 'mlp':
            layer = self.mlp
        elif layer_name == 'output_layer':
            layer = self.output_layer
        else:
            # Try to find the layer in MLP
            for name, module in self.mlp.named_children():
                if layer_name == name:
                    layer = module
                    break
        
        if layer is None:
            return False
        
        # Define hook function
        def hook_fn(module, input, output):
            self.activations[layer_name] = output.detach()
        
        # Register hook
        handle = layer.register_forward_hook(hook_fn)
        self.hooks[layer_name] = handle
        
        return True
    
    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        return self.activations.get(layer_name, None)
    
    def remove_hook(self, layer_name: str) -> bool:
        if layer_name in self.hooks:
            self.hooks[layer_name].remove()
            del self.hooks[layer_name]
            if layer_name in self.activations:
                del self.activations[layer_name]
            return True
        return False


class DeepRec_base(BaseRecommender):
    """
    Deep Recommendation Model with Attention and Sequence Modeling.
    
    This model combines embeddings, sequence modeling with GRU,
    attention mechanism, and MLP for generating recommendations.
    
    Architecture:
    
    ┌─────────────────────────────────────────────────────┐
    │                   DeepRec Model                     │
    │ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────┐ │
    │ │   User    │ │   Item    │ │ Sequence  │ │Context│ │
    │ │ Embedding │ │ Embedding │ │   Data    │ │ Data  │ │
    │ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └───┬───┘ │
    │       │             │             │             │   │
    │       │             │             │             │   │
    │       │             │             ▼             │   │
    │       │             │       ┌──────────┐        │   │
    │       │             │       │ Sequence │        │   │
    │       │             │       │ Encoder  │        │   │
    │       │             │       └────┬─────┘        │   │
    │       │             │            │              │   │
    │       └─────────────┼────────────┼──────────────┘   │
    │                     │            │                  │
    │                     ▼            ▼                  │
    │               ┌─────────────────────────┐           │
    │               │          MLP            │           │
    │               └────────────┬────────────┘           │
    │                            │                        │
    │                            ▼                        │
    │                       ┌─────────┐                   │
    │                       │ Output  │                   │
    │                       └─────────┘                   │
    └─────────────────────────────────────────────────────┘
    
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self, 
        name: str = "DeepRec",
        embed_dim: int = 64,
        mlp_dims: List[int] = [128, 64],
        attention_dim: int = 64,
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 1,
        dropout: float = 0.1,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        weight_decay: float = 1e-6,
        max_seq_length: int = 50,
        device: str = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        """
            Initialize DeepRec model.
        
            Args:
                name: Model name.
                embed_dim: Dimension of embeddings.
                mlp_dims: Dimensions of MLP layers.
                attention_dim: Dimension of attention layer.
                gru_hidden_dim: Dimension of GRU hidden state.
                gru_num_layers: Number of GRU layers.
                dropout: Dropout probability.
                batch_size: Training batch size.
                learning_rate: Learning rate.
                num_epochs: Number of training epochs.
                weight_decay: Weight decay for L2 regularization.
                max_seq_length: Maximum sequence length.
                device: Device to run the model on.
                seed: Random seed.
                verbose: Whether to output verbose logs.
            
            Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.name = name
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.attention_dim = attention_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.verbose = verbose
        
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        
        self.is_fitted = False
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        self.feature_encoders = {}
        self.numerical_means = {}
        self.numerical_stds = {}
        self.field_dims = []
        self.user_sequences = defaultdict(list)
        self.loss_history = []
        self.model = None
        self.optimizer = None
    
    def fit(self, interactions: List[Tuple]) -> "DeepRec_base":
        """
            Fit the model with interactions.
        
            Args:
                interactions: List of (user, item, features) tuples.
            
            Returns:
                Fitted model.
            
            Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.logger.info(f"Fitting {self.name} model with {len(interactions)} interactions")
        
        self._process_interactions(interactions)
        self._build_model()
        self._train(interactions)
        
        self.is_fitted = True
        return self
    
    def _process_interactions(self, interactions: List[Tuple]) -> None:
        """Process interactions to build user and item maps and extract features."""
        users = set()
        items = set()
        
        for user, item, _ in interactions:
            users.add(user)
            items.add(item)
        
        self.user_map = {user: i + 1 for i, user in enumerate(sorted(users))}
        self.item_map = {item: i + 1 for i, item in enumerate(sorted(items))}
        self.reverse_user_map = {v: k for k, v in self.user_map.items()}
        self.reverse_item_map = {v: k for k, v in self.item_map.items()}
        
        self._extract_features(interactions)
        self._update_user_sequences(interactions)
    
    def _extract_features(self, interactions: List[Tuple]) -> None:
        """Extract features from interactions."""
        feature_values = defaultdict(set)
        numerical_values = defaultdict(list)
        
        for user, item, features, _ in interactions:
            for feature, value in features.items():
                if isinstance(value, (str, bool, int)):
                    feature_values[feature].add(value)
                elif isinstance(value, (float, np.floating)):
                    numerical_values[feature].append(value)
        
        self.categorical_features = sorted(feature_values.keys())
        self.numerical_features = sorted(numerical_values.keys())
        self.feature_names = self.categorical_features + self.numerical_features
        
        for feature, values in feature_values.items():
            if feature not in self.feature_encoders:
                self.feature_encoders[feature] = {}
            
            values = sorted(values)
            for i, value in enumerate(values):
                if value not in self.feature_encoders[feature]:
                    self.feature_encoders[feature][value] = i + 1
        
        for feature, values in numerical_values.items():
            if len(values) > 0:
                self.numerical_means[feature] = float(np.mean(values))
                self.numerical_stds[feature] = float(np.std(values) or 1.0)
    
    def _update_user_sequences(self, interactions: List[Tuple]) -> None:
        """Build user sequences from interactions for sequential modeling."""
        for user, item, _, _ in interactions:
            if user not in self.user_map or item not in self.item_map:
                continue
            
            user_idx = self.user_map[user]
            item_idx = self.item_map[item]
            
            if user not in self.user_sequences:
                self.user_sequences[user] = []
            
            if item_idx not in self.user_sequences[user]:
                self.user_sequences[user].append(item_idx)
            
            if len(self.user_sequences[user]) > self.max_seq_length:
                self.user_sequences[user] = self.user_sequences[user][-self.max_seq_length:]
    
    def _build_model(self) -> None:
        """Build the DeepRec model."""
        self._set_field_dims()
        
        self.model = DeepRecModel(
            field_dims=self.field_dims,
            embed_dim=self.embed_dim,
            hidden_dim=self.gru_hidden_dim,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout,
            attention_dim=self.attention_dim,
            num_gru_layers=self.gru_num_layers
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        
        self.logger.info(f"Built model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _train(self, interactions: List[Tuple]) -> None:
        """Train the model with interactions."""
        self.model.train()
        
        positive_samples = []
        for user, item, features in interactions:
            if user in self.user_map and item in self.item_map:
                positive_samples.append((user, item, features, 1))
        
        negative_samples = self._generate_negative_samples(positive_samples)
        all_samples = positive_samples + negative_samples
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            np.random.shuffle(all_samples)
            
            num_batches = len(all_samples) // self.batch_size + (
                1 if len(all_samples) % self.batch_size != 0 else 0
            )
            
            pbar = tqdm(
                range(num_batches),
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                disable=not self.verbose,
            )
            
            for batch_idx in pbar:
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, len(all_samples))
                batch = all_samples[batch_start:batch_end]
                
                user_indices = []
                item_indices = []
                seq_indices = []
                seq_lengths = []
                numerical_features = []
                labels = []
                
                for user, item, features, label in batch:
                    user_idx = self.user_map[user]
                    item_idx = self.item_map[item]
                    
                    seq = self.user_sequences.get(user, [])
                    if item_idx in seq:
                        seq = [idx for idx in seq if idx != item_idx]
                    
                    padded_seq = seq + [0] * (self.max_seq_length - len(seq))
                    
                    encoded_features = self._encode_features(features)
                    numerical_vals = [encoded_features.get(f, 0.0) for f in self.numerical_features]
                    
                    user_indices.append(user_idx)
                    item_indices.append(item_idx)
                    seq_indices.append(padded_seq)
                    seq_lengths.append(len(seq))
                    numerical_features.append(numerical_vals)
                    labels.append(label)
                
                user_indices = torch.tensor(user_indices, dtype=torch.long, device=self.device)
                item_indices = torch.tensor(item_indices, dtype=torch.long, device=self.device)
                seq_indices = torch.tensor(seq_indices, dtype=torch.long, device=self.device)
                seq_lengths = torch.tensor(seq_lengths, dtype=torch.long, device=self.device)
                numerical_features = torch.tensor(
                    numerical_features, dtype=torch.float, device=self.device
                )
                labels = torch.tensor(labels, dtype=torch.float, device=self.device).view(-1, 1)
                
                categorical_input = torch.zeros(
                    len(batch), len(self.field_dims), dtype=torch.long, device=self.device
                )
                categorical_input[:, 0] = user_indices
                categorical_input[:, 1] = item_indices
                
                self.optimizer.zero_grad()
                
                outputs = self.model(
                    categorical_input, seq_indices, seq_lengths, numerical_features
                )
                
                loss = F.binary_cross_entropy(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": epoch_loss / (batch_idx + 1)})
            
            avg_epoch_loss = epoch_loss / num_batches
            self.loss_history.append(avg_epoch_loss)
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {avg_epoch_loss:.6f}")
    
    def _generate_negative_samples(self, positive_samples: List[Tuple]) -> List[Tuple]:
        """Generate negative samples for training."""
        negative_samples = []
        all_items = list(self.item_map.keys())
        user_item_set = {(user, item) for user, item, _, _ in positive_samples}
        
        for user, _, features, _ in positive_samples:
            while True:
                item = np.random.choice(all_items)
                if (user, item) not in user_item_set:
                    break
            negative_samples.append((user, item, features, 0))
        
        return negative_samples
    
    def predict(self, user: Any, item: Any, features: Dict[str, Any] = None) -> float:
        """Predict the probability of interaction between user and item."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        if user not in self.user_map:
            raise ValueError(f"User {user} not found in training data")
        
        if item not in self.item_map:
            raise ValueError(f"Item {item} not found in training data")
        
        self.model.eval()
        
        user_idx = self.user_map[user]
        item_idx = self.item_map[item]
        
        x = torch.zeros(1, len(self.field_dims), dtype=torch.long, device=self.device)
        x[0, 0] = user_idx
        x[0, 1] = item_idx
        
        seq = self.user_sequences.get(user, [])
        if item_idx in seq:
            seq = [idx for idx in seq if idx != item_idx]
        
        padded_seq = seq + [0] * (self.max_seq_length - len(seq))
        seq_tensor = torch.tensor([padded_seq], dtype=torch.long, device=self.device)
        seq_length = torch.tensor([len(seq)], dtype=torch.long, device=self.device)
        
        encoded_features = self._encode_features(features or {})
        numerical_vals = [encoded_features.get(f, 0.0) for f in self.numerical_features]
        numerical_tensor = torch.tensor([numerical_vals], dtype=torch.float, device=self.device)
        
        with torch.no_grad():
            output = self.model(x, seq_tensor, seq_length, numerical_tensor)
            prediction = output.item()
        
        return prediction
    
    def recommend(
        self, user: Any, top_n: int = 10, exclude_seen: bool = True, features: Dict[str, Any] = None
    ) -> List[Tuple[Any, float]]:
        """Recommend items for a user."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        if user not in self.user_map:
            raise ValueError(f"User {user} not found in training data")
        
        seen_items = set()
        if exclude_seen:
            for seq in self.user_sequences.get(user, []):
                if seq != 0:
                    seen_items.add(self.reverse_item_map[seq])
        
        self.model.eval()
        
        candidate_items = [item for item in self.item_map.keys() if item not in seen_items]
        
        scores = []
        for item in candidate_items:
            try:
                score = self.predict(user, item, features)
                scores.append((item, score))
            except Exception as e:
                self.logger.warning(f"Error predicting for {user}-{item}: {e}")
        
        scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = scores[:top_n]
        
        return recommendations
    
    def get_user_embeddings(self) -> Dict[Any, np.ndarray]:
        """Get embeddings for all users."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        self.model.eval()
        
        embeddings = {}
        with torch.no_grad():
            for user_id, user_idx in self.user_map.items():
                x = torch.zeros(1, len(self.field_dims), dtype=torch.long, device=self.device)
                x[0, 0] = user_idx
                embed = self.model.embedding[0](x[:, 0]).squeeze(0).cpu().numpy()
                embeddings[user_id] = embed
        
        return embeddings
    
    def get_item_embeddings(self) -> Dict[Any, np.ndarray]:
        """Get embeddings for all items."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        self.model.eval()
        
        embeddings = {}
        with torch.no_grad():
            for item_id, item_idx in self.item_map.items():
                x = torch.zeros(1, len(self.field_dims), dtype=torch.long, device=self.device)
                x[0, 1] = item_idx
                embed = self.model.embedding[1](x[:, 1]).squeeze(0).cpu().numpy()
                embeddings[item_id] = embed
        
        return embeddings
    
    def export_feature_importance(self) -> Dict[str, float]:
        """Export importance of different features."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            user_weight = torch.norm(self.model.embedding[0].weight).item()
            item_weight = torch.norm(self.model.embedding[1].weight).item()
            
            categorical_weights = {}
            for i, feature in enumerate(self.categorical_features):
                field_idx = i + 2
                weight = torch.norm(self.model.embedding[field_idx].weight).item()
                categorical_weights[feature] = weight
            
            all_weights = {
                "user": user_weight,
                "item": item_weight,
                **categorical_weights,
            }
            
            total_weight = sum(all_weights.values())
            normalized_weights = {k: v / total_weight for k, v in all_weights.items()}
            
            return normalized_weights
    
    def _set_field_dims(self) -> None:
        """Set field dimensions for embedding layers."""
        self.field_dims = [len(self.user_map) + 1, len(self.item_map) + 1]
        
        for feature in self.categorical_features:
            self.field_dims.append(
                len(self.feature_encoders.get(feature, {})) + 1
            )
    
    def _encode_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Encode categorical and numerical features."""
        encoded = {}
        
        for feature in self.categorical_features:
            value = features.get(feature)
            if value is not None:
                encoded[feature] = self.feature_encoders.get(feature, {}).get(value, 0)
        
        for feature in self.numerical_features:
            value = features.get(feature)
            if value is not None:
                mean = self.numerical_means.get(feature, 0)
                std = self.numerical_stds.get(feature, 1)
                if std == 0:
                    std = 1
                encoded[feature] = (value - mean) / std
        
        return encoded