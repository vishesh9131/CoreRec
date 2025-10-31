import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.sparse import csr_matrix
from corerec.base_recommender import BaseCorerec
import logging

logger = logging.getLogger(__name__)

class GNNRec(BaseCorerec):
    """
    Graph Neural Network for Recommendation (GNNRec)
    
    Leverages graph neural networks to capture higher-order connectivity patterns
    in the user-item interaction graph. It models the recommendation problem as a 
    link prediction task on a bipartite graph.
    
    Reference:
    Wang et al. "Neural Graph Collaborative Filtering" (SIGIR 2019)
    """
    
    def __init__(
        self,
        name: str = "GNNRec",
        embedding_dim: int = 64,
        num_gnn_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 1024,
        epochs: int = 20,
        trainable: bool = True,
        verbose: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.embedding_dim = embedding_dim
        self.num_gnn_layers = num_gnn_layers
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
        self.user_item_matrix = None
        
    def _build_model(self, num_users: int, num_items: int):
        class GraphConvLayer(nn.Module):
            def __init__(self, in_dim: int, out_dim: int):
                super().__init__()
                self.linear = nn.Linear(in_dim, out_dim)
                
            def forward(self, x, adj):
                # Normalize adjacency matrix
                rowsum = adj.sum(1)
                d_inv_sqrt = torch.pow(rowsum, -0.5)
                d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
                normalized_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
                
                # Graph convolution
                return self.linear(normalized_adj @ x)
        
        class GNNModel(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim, num_layers, dropout):
                super().__init__()
                self.num_users = num_users
                self.num_items = num_items
                self.embedding_dim = embedding_dim
                
                # User and item embeddings
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                
                # GNN layers
                self.gnn_layers = nn.ModuleList()
                for i in range(num_layers):
                    self.gnn_layers.append(GraphConvLayer(embedding_dim, embedding_dim))
                
                # Prediction layer
                self.predictor = nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embedding_dim, 1)
                )
                
                # Initialize weights
                self.init_weights()
                
            def init_weights(self):
                # Initialize embeddings
                nn.init.normal_(self.user_embedding.weight, std=0.1)
                nn.init.normal_(self.item_embedding.weight, std=0.1)
                
                # Initialize GNN layers
                for gnn in self.gnn_layers:
                    nn.init.xavier_uniform_(gnn.linear.weight)
                    nn.init.zeros_(gnn.linear.bias)
                
            def forward(self, user_indices, item_indices, adj_matrix):
                # Get initial embeddings
                all_embeddings = torch.cat([
                    self.user_embedding.weight,
                    self.item_embedding.weight
                ], dim=0)
                
                # Apply GNN layers
                embeddings_list = [all_embeddings]
                
                for gnn in self.gnn_layers:
                    all_embeddings = gnn(all_embeddings, adj_matrix)
                    all_embeddings = F.relu(all_embeddings)
                    all_embeddings = F.dropout(all_embeddings, p=0.1, training=self.training)
                    embeddings_list.append(all_embeddings)
                
                # Final embeddings (mean pooling across layers)
                all_embeddings = torch.stack(embeddings_list, dim=1)
                all_embeddings = torch.mean(all_embeddings, dim=1)
                
                # Split into user and item embeddings
                user_embeddings = all_embeddings[:self.num_users]
                item_embeddings = all_embeddings[self.num_users:]
                
                # Get embeddings for specific users and items
                users_emb = user_embeddings[user_indices]
                items_emb = item_embeddings[item_indices]
                
                # Predict scores
                concat_embeddings = torch.cat([users_emb, items_emb], dim=1)
                scores = self.predictor(concat_embeddings)
                
                return torch.sigmoid(scores).squeeze(1)
        
        return GNNModel(num_users, num_items, self.embedding_dim, self.num_gnn_layers, self.dropout).to(self.device)
    
    def fit(self, user_ids: List[int], item_ids: List[int], ratings: Optional[List[float]] = None) -> None:
        # Create mappings
        # Validate inputs
        validate_fit_inputs(user_ids, item_ids, ratings)
        
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))
        
        self.user_map = {user: idx for idx, user in enumerate(unique_users)}
        self.item_map = {item: idx for idx, item in enumerate(unique_items)}
        
        self.reverse_user_map = {idx: user for user, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item for item, idx in self.item_map.items()}
        
        num_users = len(unique_users)
        num_items = len(unique_items)
        
        # Convert to internal indices
        user_indices = [self.user_map[user] for user in user_ids]
        item_indices = [self.item_map[item] for item in item_ids]
        
        # Create user-item matrix
        if ratings is None:
            ratings = [1.0] * len(user_ids)  # Default to implicit feedback
            
        self.user_item_matrix = csr_matrix((ratings, (user_indices, item_indices)), 
                                           shape=(num_users, num_items))
        
        # Create adjacency matrix for the bipartite graph
        user_item_edges = torch.tensor([user_indices, item_indices], dtype=torch.long)
        item_user_edges = torch.tensor([item_indices, user_indices], dtype=torch.long)
        
        # Create adjacency matrix for user-item graph
        # Use sparse representation
        indices = torch.cat([
            # User -> Item
            torch.stack([
                torch.tensor(user_indices, dtype=torch.long),
                torch.tensor(item_indices, dtype=torch.long) + num_users  # Offset for items
            ], dim=0),
            # Item -> User
            torch.stack([
                torch.tensor(item_indices, dtype=torch.long) + num_users,  # Offset for items
                torch.tensor(user_indices, dtype=torch.long)
            ], dim=0)
        ], dim=1)
        values = torch.ones(indices.size(1))
        adj_size = num_users + num_items
        adj_matrix = torch.sparse.FloatTensor(indices, values, (adj_size, adj_size)).to_dense().to(self.device)
        
        # Build model
        self.model = self._build_model(num_users, num_items)
        
        # Create training data
        train_user_indices = torch.tensor(user_indices, dtype=torch.long).to(self.device)
        train_item_indices = torch.tensor(item_indices, dtype=torch.long).to(self.device)
        train_labels = torch.tensor(ratings, dtype=torch.float).to(self.device)
        
        # Define optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Train the model
        self.model.train()
        n_batches = len(train_user_indices) // self.batch_size + (1 if len(train_user_indices) % self.batch_size != 0 else 0)
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            # Shuffle data
            indices = torch.randperm(len(train_user_indices))
            batch_user_indices = train_user_indices[indices]
            batch_item_indices = train_item_indices[indices]
            batch_labels = train_labels[indices]
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(batch_user_indices))
                
                # Get batch data
                users = batch_user_indices[start_idx:end_idx]
                items = batch_item_indices[start_idx:end_idx]
                labels = batch_labels[start_idx:end_idx]
                
                # Forward pass
                outputs = self.model(users, items, adj_matrix)
                
                # Compute loss
                loss = criterion(outputs, labels)
                
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
        
        # Get items the user has already interacted with
        seen_items = set()
        if exclude_seen:
            seen_items = set(self.user_item_matrix[user_idx].indices)
        
        # Create adjacency matrix
        num_users = len(self.user_map)
        num_items = len(self.item_map)
        adj_size = num_users + num_items
        
        # Reconstruct adjacency matrix (this could be cached in fit method)
        user_indices, item_indices = self.user_item_matrix.nonzero()
        indices = torch.cat([
            # User -> Item
            torch.stack([
                torch.tensor(user_indices, dtype=torch.long),
                torch.tensor(item_indices, dtype=torch.long) + num_users  # Offset for items
            ], dim=0),
            # Item -> User
            torch.stack([
                torch.tensor(item_indices, dtype=torch.long) + num_users,  # Offset for items
                torch.tensor(user_indices, dtype=torch.long)
            ], dim=0)
        ], dim=1)
        values = torch.ones(indices.size(1))
        adj_matrix = torch.sparse.FloatTensor(indices, values, (adj_size, adj_size)).to_dense().to(self.device)
        
        # Generate predictions for all items
        self.model.eval()
        predictions = []
        
        # Use batched prediction for efficiency
        batch_size = 1024
        all_items = list(range(num_items))
        num_batches = (num_items + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_items)
            
            batch_items = all_items[start_idx:end_idx]
            batch_users = [user_idx] * len(batch_items)
            
            # Convert to tensors
            batch_users = torch.tensor(batch_users, dtype=torch.long).to(self.device)
            batch_items = torch.tensor(batch_items, dtype=torch.long).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                batch_preds = self.model(batch_users, batch_items, adj_matrix).cpu().numpy()
            
            # Add to predictions
            for item_idx, pred in zip(batch_items.cpu().numpy(), batch_preds):
                if exclude_seen and item_idx in seen_items:
                    continue
                predictions.append((item_idx, pred))
        
        # Sort predictions and get top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_item_indices = [item_idx for item_idx, _ in predictions[:top_n]]
        
        # Map indices back to original item IDs
        top_items = [self.reverse_item_map[idx] for idx in top_item_indices]
        
        return top_items