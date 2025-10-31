import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import logging
import os
import pickle
import pandas as pd

from corerec.base_recommender import BaseCorerec
from corerec.vish_graphs import draw_graph

class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer for GNN-based recommendation
    """
    def __init__(self, in_dim, out_dim, bias=True, activation=F.relu, residual=True):
        super(GraphConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
            
        self.activation = activation
        self.residual = residual
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights with Glorot uniform and bias with zeros"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
    def forward(self, features, adj_norm):
        """
        Perform graph convolution
        
        Parameters:
        -----------
        features: torch.Tensor
            Node features (N x in_dim)
        adj_norm: torch.Tensor
            Normalized adjacency matrix (N x N)
            
        Returns:
        --------
        torch.Tensor
            Updated node features (N x out_dim)
        """
        # Propagate features
        support = torch.mm(features, self.weight)
        output = torch.sparse.mm(adj_norm, support)
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias
            
        # Apply activation
        if self.activation is not None:
            output = self.activation(output)
            
        # Add residual connection if needed
        if self.residual and self.in_dim == self.out_dim:
            output += features
            
        return output

class GNNRecommender(BaseCorerec):
    """
    Graph Neural Network for Recommendation (GNNRec)
    
    This model represents users and items as nodes in a bipartite graph,
    and uses graph neural networks to learn representations that capture
    collaborative signals through higher-order connectivity patterns.
    
    Features:
    - Message passing between users and items
    - Capture higher-order user-item connections
    - Multi-hop propagation of information
    - Optional side information integration
    - Scalable implementation for large graphs
    
    Reference:
    Wang et al. "Neural Graph Collaborative Filtering" (SIGIR 2019)
    """
    
    def __init__(
        self,
        name: str = "GNNRec",
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 1024,
        epochs: int = 20,
        weight_decay: float = 0.0001,
        node_dropout: float = 0.1,
        message_dropout: float = 0.1,
        aggregation: str = "mean",  # Options: mean, sum, max, lstm
        add_self_loops: bool = True,
        normalization: str = "symmetric",  # Options: symmetric, random_walk, none
        loss_type: str = "bpr",  # Options: bpr, ce, bce
        trainable: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        
        # Model hyperparameters
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.node_dropout = node_dropout
        self.message_dropout = message_dropout
        self.aggregation = aggregation
        self.add_self_loops = add_self_loops
        self.normalization = normalization
        self.loss_type = loss_type
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Initialize logging
        self.logger = self._setup_logger()
        
        # These will be set during fit
        self.graph = None
        self.nx_graph = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.num_users = 0
        self.num_items = 0
        self.adj_matrix = None
        self.model = None
        self.is_fitted = False
        
    def _setup_logger(self):
        """Set up a logger for the model"""
        logger = logging.getLogger(f"{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger
        
    def _build_graph(self, interactions):
        """
        Build a user-item interaction graph
        
        Parameters:
        -----------
        interactions: DataFrame or list of tuples
            User-item interactions (user_id, item_id, [rating])
            
        Returns:
        --------
        nx.Graph
            NetworkX graph representing interactions
        """
        # Create mappings for users and items
        unique_users = sorted(np.unique([u for u, _, *_ in interactions]))
        unique_items = sorted(np.unique([i for _, i, *_ in interactions]))
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx + len(unique_users) for idx, item in enumerate(unique_items)}
        
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)
        
        # Create bipartite graph
        graph = nx.Graph()
        
        # Add nodes
        user_nodes = [(self.user_mapping[user], {'type': 'user', 'original_id': user}) 
                      for user in unique_users]
        item_nodes = [(self.item_mapping[item], {'type': 'item', 'original_id': item}) 
                      for item in unique_items]
        
        graph.add_nodes_from(user_nodes)
        graph.add_nodes_from(item_nodes)
        
        # Add edges
        edges = []
        for user, item, *rest in interactions:
            user_idx = self.user_mapping[user]
            item_idx = self.item_mapping[item]
            
            # Use rating as edge weight if available
            if rest:
                rating = rest[0]
                edges.append((user_idx, item_idx, {'weight': float(rating)}))
            else:
                edges.append((user_idx, item_idx))
        
        graph.add_edges_from(edges)
        
        self.logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        return graph
    
    def _build_model(self):
        """Build the GNN model architecture"""
        
        class GNNModel(nn.Module):
            def __init__(
                self, 
                num_users, 
                num_items, 
                embedding_dim, 
                num_layers,
                dropout=0.1,
                node_dropout=0.1,
                message_dropout=0.1,
                aggregation='mean'
            ):
                super(GNNModel, self).__init__()
                
                self.num_users = num_users
                self.num_items = num_items
                self.embedding_dim = embedding_dim
                self.num_layers = num_layers
                
                # Initial embeddings
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                
                # GNN layers
                self.layers = nn.ModuleList()
                for i in range(num_layers):
                    self.layers.append(GraphConvLayer(
                        embedding_dim, 
                        embedding_dim,
                        residual=(i > 0)
                    ))
                
                # Dropouts
                self.dropout = nn.Dropout(dropout)
                self.node_dropout = node_dropout
                self.message_dropout = message_dropout
                
                # Prediction layer
                self.prediction = nn.Linear(embedding_dim * 2, 1)
                
                # Initialize weights
                self.reset_parameters()
                
            def reset_parameters(self):
                """Initialize embeddings and layers"""
                nn.init.normal_(self.user_embedding.weight, std=0.1)
                nn.init.normal_(self.item_embedding.weight, std=0.1)
                
            def forward(self, adj_matrix):
                """
                Full forward pass through the GNN
                
                Parameters:
                -----------
                adj_matrix: torch.Tensor
                    Normalized adjacency matrix (N x N)
                    
                Returns:
                --------
                Tuple of embeddings:
                    user_embeddings: torch.Tensor
                        Updated user embeddings (num_users x embedding_dim)
                    item_embeddings: torch.Tensor
                        Updated item embeddings (num_items x embedding_dim)
                """
                # Get initial embeddings
                user_emb = self.user_embedding.weight
                item_emb = self.item_embedding.weight
                
                # Concatenate embeddings
                all_emb = torch.cat([user_emb, item_emb], dim=0)
                
                # Message propagation through layers
                embs = [all_emb]
                
                # Apply node dropout to adjacency matrix if needed
                if self.node_dropout > 0 and self.training:
                    adj_matrix = self._drop_nodes(adj_matrix, self.node_dropout)
                
                # Layer propagation
                for layer in self.layers:
                    if self.message_dropout > 0 and self.training:
                        all_emb = F.dropout(all_emb, p=self.message_dropout)
                    
                    all_emb = layer(all_emb, adj_matrix)
                    embs.append(all_emb)
                
                # Aggregate embeddings from all layers (mean)
                all_emb = torch.stack(embs, dim=1).mean(dim=1)
                
                # Split embeddings back to users and items
                user_final = all_emb[:self.num_users]
                item_final = all_emb[self.num_users:]
                
                return user_final, item_final
            
            def _drop_nodes(self, adj_matrix, dropout_rate):
                """Apply node dropout to the adjacency matrix"""
                # Create node dropout mask
                node_mask = torch.FloatTensor(adj_matrix.size(0)).uniform_() >= dropout_rate
                
                # Convert to sparse format for efficiency
                indices = adj_matrix._indices()
                values = adj_matrix._values()
                
                # Apply mask to nodes
                mask = node_mask[indices[0]] & node_mask[indices[1]]
                indices = indices[:, mask]
                values = values[mask]
                
                # Create new sparse tensor
                shape = adj_matrix.size()
                new_adj = torch.sparse.FloatTensor(indices, values, shape)
                
                return new_adj
                
            def predict(self, user_indices, item_indices, user_emb, item_emb):
                """
                Make predictions for user-item pairs
                
                Parameters:
                -----------
                user_indices: torch.Tensor
                    User indices
                item_indices: torch.Tensor
                    Item indices
                user_emb: torch.Tensor
                    User embeddings
                item_emb: torch.Tensor
                    Item embeddings
                    
                Returns:
                --------
                torch.Tensor
                    Prediction scores
                """
                # Get user and item embeddings
                users = user_emb[user_indices]
                items = item_emb[item_indices]
                
                # Apply element-wise product + concatenation
                element_product = users * items
                concat = torch.cat([users, items], dim=-1)
                
                # Make predictions
                scores = self.prediction(concat)
                
                return scores.squeeze()
        
        # Create model
        model = GNNModel(
            self.num_users,
            self.num_items,
            self.embedding_dim,
            self.num_layers,
            self.dropout,
            self.node_dropout,
            self.message_dropout,
            self.aggregation
        ).to(self.device)
        
        self.model = model
        self.logger.info(f"Built GNN model with {self.num_layers} layers and {self.embedding_dim} dimensions")
        
    def _normalize_adj_matrix(self, adj_matrix):
        """
        Normalize adjacency matrix for graph convolution
        
        Parameters:
        -----------
        adj_matrix: torch.Tensor
            Adjacency matrix (N x N)
            
        Returns:
        --------
        torch.sparse.FloatTensor
            Normalized adjacency matrix
        """
        if self.add_self_loops:
            # Add self loops: A' = A + I
            adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0), device=self.device)
        
        if self.normalization == 'symmetric':
            # Symmetric normalization: D^(-1/2) A D^(-1/2)
            deg = torch.sum(adj_matrix, dim=1)
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
            
            deg_inv_sqrt = torch.diag(deg_inv_sqrt)
            normalized_adj = torch.mm(torch.mm(deg_inv_sqrt, adj_matrix), deg_inv_sqrt)
            
        elif self.normalization == 'random_walk':
            # Random walk normalization: D^(-1) A
            deg = torch.sum(adj_matrix, dim=1)
            deg_inv = 1.0 / deg
            deg_inv[torch.isinf(deg_inv)] = 0
            
            deg_inv = torch.diag(deg_inv)
            normalized_adj = torch.mm(deg_inv, adj_matrix)
            
        else:  # No normalization
            normalized_adj = adj_matrix
        
        # Convert to sparse tensor for efficiency
        indices = torch.nonzero(normalized_adj).t()
        values = normalized_adj[indices[0], indices[1]]
        
        shape = normalized_adj.size()
        sparse_adj = torch.sparse.FloatTensor(indices, values, shape)
        
        return sparse_adj
        
    def fit(self, interactions, validation=None):
        """
        Train the GNN recommendation model
        
        Parameters:
        -----------
        interactions: DataFrame or list of tuples
            User-item interactions (user_id, item_id, [rating])
        validation: DataFrame or list of tuples, optional
            Validation interactions
            
        Returns:
        --------
        self
        """
        # Validate inputs
        validate_fit_inputs(user_ids, item_ids, ratings)
        
        self.logger.info(f"Starting training of {self.name}")
        
        # Build interaction graph
        self.nx_graph = self._build_graph(interactions)
        
        # Create adjacency matrix
        adj_matrix = nx.to_numpy_array(self.nx_graph)
        adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)
        
        # Normalize adjacency matrix
        self.adj_matrix = self._normalize_adj_matrix(adj_matrix)
        
        # Build model
        self._build_model()
        
        # Prepare optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Prepare training data
        train_data = []
        for user, item, *rest in interactions:
            user_idx = self.user_mapping[user]
            item_idx = self.item_mapping[item] - self.num_users  # Adjust item index
            
            train_data.append((user_idx, item_idx))
        
        train_data = np.array(train_data)
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Shuffle training data
            np.random.shuffle(train_data)
            
            # Mini-batch training
            epoch_loss = 0
            
            for i in range(0, len(train_data), self.batch_size):
                batch = train_data[i:i+self.batch_size]
                user_indices = torch.LongTensor(batch[:, 0]).to(self.device)
                pos_item_indices = torch.LongTensor(batch[:, 1]).to(self.device)
                
                # Sample negative items
                neg_item_indices = torch.randint(
                    0, self.num_items, 
                    size=pos_item_indices.size()
                ).to(self.device)
                
                # Get embeddings from GNN
                user_emb, item_emb = self.model(self.adj_matrix)
                
                # Compute predictions
                pos_scores = self.model.predict(
                    user_indices, pos_item_indices, user_emb, item_emb
                )
                neg_scores = self.model.predict(
                    user_indices, neg_item_indices, user_emb, item_emb
                )
                
                # Compute loss
                if self.loss_type == 'bpr':
                    loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
                elif self.loss_type == 'bce':
                    loss = F.binary_cross_entropy_with_logits(
                        pos_scores, 
                        torch.ones_like(pos_scores)
                    ) + F.binary_cross_entropy_with_logits(
                        neg_scores, 
                        torch.zeros_like(neg_scores)
                    )
                else:  # ce
                    pred_scores = torch.cat([pos_scores, neg_scores])
                    labels = torch.cat([
                        torch.ones_like(pos_scores), 
                        torch.zeros_like(neg_scores)
                    ])
                    loss = F.binary_cross_entropy_with_logits(pred_scores, labels)
                
                # Update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= (len(train_data) / self.batch_size)
            
            # Logging
            if self.verbose:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}")
                
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Restore best model
        self.model.load_state_dict(best_model)
        self.is_fitted = True
        
        self.logger.info("Training completed")
        return self
        
    def recommend(self, user_id, top_n=10, exclude_seen=True, include_scores=False):
        """
        Generate top-N recommendations for a user
        
        Parameters:
        -----------
        user_id: any
            User ID
        top_n: int
            Number of recommendations
        exclude_seen: bool
            Whether to exclude items the user has already interacted with
        include_scores: bool
            Whether to include prediction scores in the output
            
        Returns:
        --------
        list
            List of recommended item IDs (or tuples of (item_id, score) if include_scores=True)
        """
        # Validate inputs
        validate_model_fitted(self.is_fitted, self.name)
        validate_user_id(user_id, self.user_map if hasattr(self, 'user_map') else {})
        validate_top_k(top_k if 'top_k' in locals() else 10)
        
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
            
        if user_id not in self.user_mapping:
            raise ValueError(f"Unknown user: {user_id}")
            
        user_idx = self.user_mapping[user_id]
        
        # Get seen items for this user
        seen_items = set()
        if exclude_seen:
            for neighbor in self.nx_graph.neighbors(user_idx):
                if neighbor >= self.num_users:  # Ensure it's an item node
                    seen_items.add(neighbor - self.num_users)
        
        # Get model in evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Get embeddings from GNN
            user_emb, item_emb = self.model(self.adj_matrix)
            
            # Get user embedding
            user_embedding = user_emb[user_idx]
            
            # Compute scores for all items
            scores = []
            user_embedding = user_embedding.expand(self.num_items, -1)
            item_embeddings = item_emb
            
            # Compute similarity scores
            similarity = self.model.compute_similarity(user_embedding, item_embeddings)
            scores = similarity.cpu().numpy()
            
            # Create list of (item_id, score) pairs
            item_scores = []
            for i in range(self.num_items):
                if exclude_seen and i in seen_items:
                    continue
                
                original_item_id = self.reverse_item_mapping[i + self.num_users]
                item_scores.append((original_item_id, scores[i]))
            
            # Sort by score in descending order
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-N recommendations
            if include_scores:
                return item_scores[:top_n]
            else:
                return [item for item, _ in item_scores[:top_n]]
    
    def visualize_graph(self, user_id=None, top_n=5, recommended_nodes=None, node_labels=None):
        """
        Visualize the recommendation graph
        
        Parameters:
        -----------
        user_id: any, optional
            If provided, highlight this user and their connections
        top_n: int
            Number of recommendations to highlight
        recommended_nodes: list, optional
            List of recommended node IDs to highlight
        node_labels: dict, optional
            Dictionary mapping node IDs to labels
        """
        if not hasattr(self, 'nx_graph'):
            raise ValueError("Model has not been fitted or graph is not available")
        
        # If user_id is provided, get recommendations
        if user_id is not None and recommended_nodes is None:
            user_idx = self.user_mapping.get(user_id)
            if user_idx is None:
                raise ValueError(f"Unknown user: {user_id}")
                
            # Get recommendations
            recs = self.recommend(user_id, top_n=top_n)
            
            # Convert to node IDs in the graph
            recommended_nodes = [self.item_mapping[item] for item in recs]
        
        # Create node labels if not provided
        if node_labels is None:
            node_labels = {}
            for original_id, idx in self.user_mapping.items():
                node_labels[idx] = f"U: {original_id}"
            for original_id, idx in self.item_mapping.items():
                node_labels[idx] = f"I: {original_id}"
        
        # Highlight the user if provided
        top_nodes = []
        if user_id is not None:
            top_nodes = [self.user_mapping[user_id]]
            
        # Use spring layout for visualization
        pos = nx.spring_layout(self.nx_graph)
        
        # Draw the graph
        draw_graph(
            self.nx_graph, 
            pos=pos, 
            top_nodes=top_nodes, 
            recommended_nodes=recommended_nodes, 
            node_labels=node_labels
        )
    
    def save(self, filepath):
        """
        Save model to file
        
        Parameters:
        -----------
        filepath: str
            Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Prepare data to save
        save_data = {
            'model_config': {
                'embedding_dim': self.embedding_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'node_dropout': self.node_dropout,
                'message_dropout': self.message_dropout,
                'aggregation': self.aggregation,
                'add_self_loops': self.add_self_loops,
                'normalization': self.normalization,
                'loss_type': self.loss_type,
            },
            'training_config': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'weight_decay': self.weight_decay,
            },
            'graph_data': {
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'reverse_user_mapping': self.reverse_user_mapping,
                'reverse_item_mapping': self.reverse_item_mapping,
                'num_users': self.num_users,
                'num_items': self.num_items,
            },
            'model_state': self.model.state_dict(),
            'adj_matrix': self.adj_matrix
        }
        
        # Save to file
        torch.save(save_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, device=None):
        """
        Load model from file
        
        Parameters:
        -----------
        filepath: str
            Path to the saved model
        device: str, optional
            Device to load the model on
            
        Returns:
        --------
        GNNRecommender
            Loaded model
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Load saved data
        save_data = torch.load(filepath, map_location=device)
        
        # Create a new instance with saved configuration
        instance = cls(
            name=os.path.basename(filepath).split('.')[0],
            embedding_dim=save_data['model_config']['embedding_dim'],
            num_layers=save_data['model_config']['num_layers'],
            dropout=save_data['model_config']['dropout'],
            node_dropout=save_data['model_config']['node_dropout'],
            message_dropout=save_data['model_config']['message_dropout'],
            aggregation=save_data['model_config']['aggregation'],
            add_self_loops=save_data['model_config']['add_self_loops'],
            normalization=save_data['model_config']['normalization'],
            loss_type=save_data['model_config']['loss_type'],
            learning_rate=save_data['training_config']['learning_rate'],
            batch_size=save_data['training_config']['batch_size'],
            epochs=save_data['training_config']['epochs'],
            weight_decay=save_data['training_config']['weight_decay'],
            device=device
        )
        
        # Restore graph data
        instance.user_mapping = save_data['graph_data']['user_mapping']
        instance.item_mapping = save_data['graph_data']['item_mapping']
        instance.reverse_user_mapping = save_data['graph_data']['reverse_user_mapping']
        instance.reverse_item_mapping = save_data['graph_data']['reverse_item_mapping']
        instance.num_users = save_data['graph_data']['num_users']
        instance.num_items = save_data['graph_data']['num_items']
        
        # Restore adjacency matrix
        instance.adj_matrix = save_data['adj_matrix']
        
        # Create NetworkX graph
        instance.nx_graph = nx.Graph()
        
        # Build model and load state
        instance._build_model()
        instance.model.load_state_dict(save_data['model_state'])
        instance.model.eval()
        
        instance.is_fitted = True
        return instance