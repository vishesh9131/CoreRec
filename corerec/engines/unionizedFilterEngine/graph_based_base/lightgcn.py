# corerec/engines/unionizedFilterEngine/lightgcn.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import pickle
import os
from scipy.sparse import csr_matrix

from ..base_recommender import BaseRecommender


class LightGCN(BaseRecommender):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    
    This implementation follows the paper:
    "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
    by Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang
    """
    
    def __init__(self, 
                 n_factors: int = 64, 
                 n_layers: int = 3,
                 learning_rate: float = 0.001,
                 regularization: float = 1e-5,
                 batch_size: int = 1024,
                 epochs: int = 100,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 dropout: float = 0.0,
                 early_stopping_patience: int = 10,
                 verbose: bool = True):
        """
        Initialize LightGCN model
        
        Args:
            n_factors: Embedding size
            n_layers: Number of graph convolution layers
            learning_rate: Learning rate for optimizer
            regularization: L2 regularization coefficient
            batch_size: Training batch size
            epochs: Number of training epochs
            device: Computing device ('cpu' or 'cuda')
            dropout: Dropout rate
            early_stopping_patience: Number of epochs to wait before early stopping
            verbose: Whether to print training progress
        """
        super().__init__()
        self.n_factors = n_factors
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.dropout = dropout
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        
        # These will be initialized during fitting
        self.n_users = None
        self.n_items = None
        self.user_embedding = None
        self.item_embedding = None
        self.model = None
        self.optimizer = None
        
        # For mapping between original IDs and internal indices
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        
        # For storing user interactions
        self.user_interactions = {}
        
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Create mappings between original IDs and internal indices
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
        """
        unique_user_ids = sorted(set(user_ids))
        unique_item_ids = sorted(set(item_ids))
        
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
        self.item_id_map = {item_id: idx for idx, item_id in enumerate(unique_item_ids)}
        
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_id_map.items()}
        
        self.n_users = len(unique_user_ids)
        self.n_items = len(unique_item_ids)
        
    def _build_model(self) -> None:
        """
        Build the LightGCN model
        """
        self.model = LightGCNModel(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            n_layers=self.n_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.regularization
        )
        
    def _create_adjacency_matrix(self, interaction_matrix: csr_matrix) -> torch.Tensor:
        """
        Create normalized adjacency matrix for graph convolution
        
        Args:
            interaction_matrix: User-item interaction matrix
            
        Returns:
            Normalized adjacency matrix as a sparse tensor
        """
        # Convert to PyTorch sparse tensor
        indices = interaction_matrix.nonzero()
        values = interaction_matrix.data
        
        # Create indices for the sparse tensor
        user_indices = torch.LongTensor(indices[0])
        item_indices = torch.LongTensor(indices[1])
        
        # Create edge index for the graph
        edge_index = torch.stack([
            torch.cat([user_indices, item_indices + self.n_users]),
            torch.cat([item_indices + self.n_users, user_indices])
        ])
        
        # Create adjacency matrix
        adj = torch.sparse.FloatTensor(
            edge_index,
            torch.ones(edge_index.size(1)),
            torch.Size([self.n_users + self.n_items, self.n_users + self.n_items])
        ).to(self.device)
        
        # Calculate degree matrix
        rowsum = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # Calculate normalized adjacency matrix: D^(-0.5) * A * D^(-0.5)
        norm_adj = torch.sparse.mm(torch.sparse.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        
        return norm_adj
    
    def _store_user_interactions(self, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Store user interactions for recommendation filtering
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
        """
        self.user_interactions = {}
        for user_id, item_id in zip(user_ids, item_ids):
            user_idx = self.user_id_map.get(user_id)
            item_idx = self.item_id_map.get(item_id)
            
            if user_idx is not None and item_idx is not None:
                if user_idx not in self.user_interactions:
                    self.user_interactions[user_idx] = set()
                self.user_interactions[user_idx].add(item_idx)
    
    def _sample_negative_items(self, user_idx: int, n_neg: int = 1) -> List[int]:
        """
        Sample negative items for a user
        
        Args:
            user_idx: User index
            n_neg: Number of negative items to sample
            
        Returns:
            List of negative item indices
        """
        pos_items = self.user_interactions.get(user_idx, set())
        neg_items = []
        
        while len(neg_items) < n_neg:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in pos_items and neg_item not in neg_items:
                neg_items.append(neg_item)
                
        return neg_items
    
    def _create_bpr_loss(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor) -> torch.Tensor:
        """
        Create BPR loss for training
        
        Args:
            users: User indices
            pos_items: Positive item indices
            neg_items: Negative item indices
            
        Returns:
            BPR loss
        """
        # Get embeddings
        user_emb, item_emb = self.model()
        
        # Get user embeddings
        user_embeddings = user_emb[users]
        
        # Get positive and negative item embeddings
        pos_item_embeddings = item_emb[pos_items]
        neg_item_embeddings = item_emb[neg_items]
        
        # Calculate scores
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)
        
        # Calculate BPR loss
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # Add L2 regularization
        l2_loss = self.regularization * (
            torch.norm(user_embeddings) ** 2 + 
            torch.norm(pos_item_embeddings) ** 2 + 
            torch.norm(neg_item_embeddings) ** 2
        ) / len(users)
        
        return loss + l2_loss
    
    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Fit the LightGCN model
        
        Args:
            interaction_matrix: User-item interaction matrix
            user_ids: List of user IDs
            item_ids: List of item IDs
        """
        # Create mappings
        self._create_mappings(user_ids, item_ids)
        
        # Store user interactions
        self._store_user_interactions(user_ids, item_ids)
        
        # Build model
        self._build_model()
        
        # Create adjacency matrix
        norm_adj = self._create_adjacency_matrix(interaction_matrix)
        self.model.set_adj_matrix(norm_adj)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            
            # Sample training data
            users = []
            pos_items = []
            neg_items = []
            
            for user_idx in range(self.n_users):
                if user_idx in self.user_interactions:
                    for pos_item in self.user_interactions[user_idx]:
                        users.append(user_idx)
                        pos_items.append(pos_item)
                        neg_items.append(self._sample_negative_items(user_idx)[0])
            
            # Create batches
            n_batches = len(users) // self.batch_size + (1 if len(users) % self.batch_size != 0 else 0)
            
            total_loss = 0
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(users))
                
                batch_users = torch.LongTensor(users[start_idx:end_idx]).to(self.device)
                batch_pos_items = torch.LongTensor(pos_items[start_idx:end_idx]).to(self.device)
                batch_neg_items = torch.LongTensor(neg_items[start_idx:end_idx]).to(self.device)
                
                self.optimizer.zero_grad()
                loss = self._create_bpr_loss(batch_users, batch_pos_items, batch_neg_items)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Get final embeddings
        self.model.eval()
        self.user_embedding, self.item_embedding = self.model()
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """
        Recommend items for a user
        
        Args:
            user_id: User ID
            top_n: Number of recommendations to return
            exclude_seen: Whether to exclude seen items
            
        Returns:
            List of recommended item IDs
        """
        if self.user_embedding is None or self.item_embedding is None:
            raise ValueError("Model has not been trained yet")
        
        if user_id not in self.user_id_map:
            return []
        
        user_idx = self.user_id_map[user_id]
        user_emb = self.user_embedding[user_idx].detach()
        
        # Calculate scores for all items
        scores = torch.matmul(user_emb, self.item_embedding.t()).detach().cpu().numpy()
        
        # Exclude seen items if needed
        if exclude_seen and user_idx in self.user_interactions:
            seen_items = self.user_interactions[user_idx]
            for item_idx in seen_items:
                scores[item_idx] = -np.inf
        
        # Get top-N items
        top_item_indices = np.argsort(-scores)[:top_n]
        
        # Convert indices back to original IDs
        top_items = [self.reverse_item_map[idx] for idx in top_item_indices]
        
        return top_items
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        if self.user_embedding is None or self.item_embedding is None:
            raise ValueError("Model has not been trained yet")
        
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            return 0.0
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[item_id]
        
        user_emb = self.user_embedding[user_idx]
        item_emb = self.item_embedding[item_idx]
        
        score = torch.dot(user_emb, item_emb).detach().cpu().numpy()
        
        return float(score)
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'n_factors': self.n_factors,
            'n_layers': self.n_layers,
            'learning_rate': self.learning_rate,
            'regularization': self.regularization,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': self.device,
            'dropout': self.dropout,
            'early_stopping_patience': self.early_stopping_patience,
            'verbose': self.verbose,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'user_interactions': self.user_interactions,
            'user_embedding': self.user_embedding.detach().cpu().numpy() if self.user_embedding is not None else None,
            'item_embedding': self.item_embedding.detach().cpu().numpy() if self.item_embedding is not None else None
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'LightGCN':
        """
        Load model from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded LightGCN model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(
            n_factors=model_data['n_factors'],
            n_layers=model_data['n_layers'],
            learning_rate=model_data['learning_rate'],
            regularization=model_data['regularization'],
            batch_size=model_data['batch_size'],
            epochs=model_data['epochs'],
            device=model_data['device'],
            dropout=model_data['dropout'],
            early_stopping_patience=model_data['early_stopping_patience'],
            verbose=model_data['verbose']
        )
        
        instance.n_users = model_data['n_users']
        instance.n_items = model_data['n_items']
        instance.user_id_map = model_data['user_id_map']
        instance.item_id_map = model_data['item_id_map']
        instance.reverse_user_map = model_data['reverse_user_map']
        instance.reverse_item_map = model_data['reverse_item_map']
        instance.user_interactions = model_data['user_interactions']
        
        if model_data['user_embedding'] is not None and model_data['item_embedding'] is not None:
            instance.user_embedding = torch.tensor(model_data['user_embedding']).to(instance.device)
            instance.item_embedding = torch.tensor(model_data['item_embedding']).to(instance.device)
        
        return instance


class LightGCNModel(nn.Module):
    """
    LightGCN Model implementation
    """
    
    def __init__(self, n_users: int, n_items: int, n_factors: int, n_layers: int, dropout: float = 0.0):
        """
        Initialize LightGCN model
        
        Args:
            n_users: Number of users
            n_items: Number of items
            n_factors: Embedding size
            n_layers: Number of graph convolution layers
            dropout: Dropout rate
        """
        super(LightGCNModel, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        
        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # Adjacency matrix
        self.adj_matrix = None
    
    def set_adj_matrix(self, adj_matrix: torch.Tensor) -> None:
        """
        Set adjacency matrix for graph convolution
        
        Args:
            adj_matrix: Normalized adjacency matrix
        """
        self.adj_matrix = adj_matrix
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            Tuple of user and item embeddings
        """
        if self.adj_matrix is None:
            raise ValueError("Adjacency matrix has not been set")
        
        # Get initial embeddings
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        
        # Concatenate user and item embeddings
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        
        # List to store embeddings from each layer
        embeddings_list = [all_embeddings]
        
        # Graph convolution
        for _ in range(self.n_layers):
            # Dropout if needed
            if self.dropout > 0 and self.training:
                all_embeddings = F.dropout(all_embeddings, p=self.dropout)
            
            # Perform graph convolution: A * H
            all_embeddings = torch.sparse.mm(self.adj_matrix, all_embeddings)
            
            # Add to list
            embeddings_list.append(all_embeddings)
        
        # Sum embeddings from all layers
        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        # Split user and item embeddings
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        return user_embeddings, item_embeddings