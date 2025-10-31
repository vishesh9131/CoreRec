import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from corerec.base_recommender import BaseCorerec
import logging

logger = logging.getLogger(__name__)

class NASRec(BaseCorerec):
    """
    Neural Architecture Search for Recommendation (NASRec)
    
    Automatically discovers optimal neural network architectures for recommendation tasks.
    This implementation uses a pre-defined architecture discovered through NAS instead of
    performing the search during training.
    
    Reference:
    Cheng et al. "NASRec: Weight Sharing Neural Architecture Search for Recommender Systems" (WWW 2021)
    """
    
    def __init__(
        self,
        name: str = "NASRec",
        embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        trainable: bool = True,
        verbose: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.model = None
        
    def _build_model(self, num_users: int, num_items: int):
        class NASRecCell(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                
                # Define operations discovered by NAS
                self.op1 = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU()
                )
                
                self.op2 = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Sigmoid()
                )
                
                self.op3 = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh()
                )
                
                # Attention weights for combining operations
                self.attention = nn.Parameter(torch.ones(3) / 3)
                
            def forward(self, x):
                # Apply operations
                out1 = self.op1(x)
                out2 = self.op2(x)
                out3 = self.op3(x)
                
                # Normalize attention weights
                weights = F.softmax(self.attention, dim=0)
                
                # Combine outputs using attention
                outputs = weights[0] * out1 + weights[1] * out2 + weights[2] * out3
                
                return outputs
        
        class NASRecModel(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim, hidden_dims, dropout):
                super().__init__()
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                
                # Initialize weights
                nn.init.normal_(self.user_embedding.weight, std=0.01)
                nn.init.normal_(self.item_embedding.weight, std=0.01)
                
                # NAS-discovered architecture
                self.layers = nn.ModuleList()
                input_dim = embedding_dim * 2  # Concatenated user and item embeddings
                
                for hidden_dim in hidden_dims:
                    self.layers.append(NASRecCell(input_dim, hidden_dim))
                    self.layers.append(nn.Dropout(dropout))
                    input_dim = hidden_dim
                
                self.output_layer = nn.Linear(input_dim, 1)
                
            def forward(self, user_indices, item_indices):
                # Get embeddings
                user_emb = self.user_embedding(user_indices)
                item_emb = self.item_embedding(item_indices)
                
                # Concatenate embeddings
                x = torch.cat([user_emb, item_emb], dim=1)
                
                # Apply NAS-discovered layers
                for layer in self.layers:
                    x = layer(x)
                
                # Output layer
                output = self.output_layer(x)
                
                return torch.sigmoid(output).squeeze(1)
        
        return NASRecModel(num_users, num_items, self.embedding_dim, self.hidden_dims, 
                           self.dropout).to(self.device)
    
    def fit(self, user_ids: List[int], item_ids: List[int], ratings: List[float]) -> None:
        # Create mappings
        # Validate inputs
        validate_fit_inputs(user_ids, item_ids, ratings)
        
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))
        
        self.user_map = {user: idx for idx, user in enumerate(unique_users)}
        self.item_map = {item: idx for idx, item in enumerate(unique_items)}
        
        self.reverse_user_map = {idx: user for user, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item for item, idx in self.item_map.items()}
        
        # Build model
        self.model = self._build_model(len(unique_users), len(unique_items))
        
        # Create training data
        user_indices = [self.user_map[user] for user in user_ids]
        item_indices = [self.item_map[item] for item in item_ids]
        
        # Convert to tensors
        X_users = torch.LongTensor(user_indices).to(self.device)
        X_items = torch.LongTensor(item_indices).to(self.device)
        y = torch.FloatTensor(ratings).to(self.device)
        
        # Define optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Train the model
        self.model.train()
        n_batches = len(X_users) // self.batch_size + (1 if len(X_users) % self.batch_size != 0 else 0)
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            # Shuffle data
            indices = torch.randperm(len(X_users))
            X_users_shuffled = X_users[indices]
            X_items_shuffled = X_items[indices]
            y_shuffled = y[indices]
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(X_users))
                
                batch_users = X_users_shuffled[start_idx:end_idx]
                batch_items = X_items_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                outputs = self.model(batch_users, batch_items)
                
                # Compute loss
                loss = criterion(outputs, batch_y)
                
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
            # For demonstration purposes, we'll assume we have this information
            pass
        
        # Generate predictions for all items
        predictions = []
        
        # Process in batches for efficiency
        batch_size = 1024
        for i in range(0, len(all_items), batch_size):
            batch_items = all_items[i:i+batch_size]
            batch_users = [user_id] * len(batch_items)
            
            # Filter out seen items
            if exclude_seen:
                filtered_items = []
                filtered_users = []
                for user, item in zip(batch_users, batch_items):
                    if item not in seen_items:
                        filtered_items.append(item)
                        filtered_users.append(user)
                batch_items = filtered_items
                batch_users = filtered_users
            
            if not batch_items:
                continue
            
            # Map to indices
            user_indices = [self.user_map[user] for user in batch_users]
            item_indices = [self.item_map[item] for item in batch_items]
            
            # Convert to tensors
            user_tensor = torch.LongTensor(user_indices).to(self.device)
            item_tensor = torch.LongTensor(item_indices).to(self.device)
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                batch_preds = self.model(user_tensor, item_tensor).cpu().numpy()
            
            # Add to predictions
            for item, pred in zip(batch_items, batch_preds):
                predictions.append((item, pred))
        
        # Sort predictions and get top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_items = [item for item, _ in predictions[:top_n]]
        
        return top_items