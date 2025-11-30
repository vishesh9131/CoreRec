import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.sparse import csr_matrix
from corerec.api.base_recommender import BaseRecommender
from corerec.api.exceptions import ModelNotFittedError, InvalidParameterError
import pickle
from pathlib import Path
from typing import Union
import logging

logger = logging.getLogger(__name__)


class GNNRec(BaseRecommender):
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
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

    def _build_model(self, num_users: int, num_items: int, laplacian_matrix: torch.Tensor):
        class EmbeddingPropagationLayer(nn.Module):
            def __init__(self, in_dim: int, out_dim: int):
                super().__init__()
                self.W1 = nn.Linear(in_dim, out_dim)
                self.W2 = nn.Linear(in_dim, out_dim)

            def forward(self, E, L):
                # Message construction and aggregation
                # E: (N+M, d) embeddings
                # L: (N+M, N+M) Laplacian matrix
                # Equation (7) from NGCF paper: E^(l) = LeakyReLU((L+I)E^(l-1)W1 + L(E^(l-1) ⊙ E^(l-1))W2)
                L_plus_I = L + torch.eye(L.size(0), device=L.device)
                # (L+I) @ E @ W1
                term1 = (L_plus_I @ E) @ self.W1.weight.t() + (self.W1.bias if self.W1.bias is not None else 0)
                # L @ (E ⊙ E) @ W2
                E_interaction = E * E  # Element-wise product
                term2 = (L @ E_interaction) @ self.W2.weight.t() + (self.W2.bias if self.W2.bias is not None else 0)
                E_new = F.leaky_relu(term1 + term2)
                return E_new

        class GNNModel(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim, num_layers, dropout, laplacian):
                super().__init__()
                self.num_users = num_users
                self.num_items = num_items
                self.embedding_dim = embedding_dim
                self.num_layers = num_layers
                self.dropout = dropout
                self.laplacian = laplacian

                # Initial embeddings (E^(0))
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)

                # Embedding propagation layers
                self.prop_layers = nn.ModuleList()
                for i in range(num_layers):
                    self.prop_layers.append(EmbeddingPropagationLayer(embedding_dim, embedding_dim))

                # Initialize weights using Xavier
                self._init_weights()

            def _init_weights(self):
                nn.init.xavier_uniform_(self.user_embedding.weight)
                nn.init.xavier_uniform_(self.item_embedding.weight)
                for layer in self.prop_layers:
                    nn.init.xavier_uniform_(layer.W1.weight)
                    nn.init.xavier_uniform_(layer.W2.weight)
                    if layer.W1.bias is not None:
                        nn.init.zeros_(layer.W1.bias)
                    if layer.W2.bias is not None:
                        nn.init.zeros_(layer.W2.bias)

            def forward(self, user_indices, item_indices):
                # Get initial embeddings E^(0)
                user_emb = self.user_embedding.weight
                item_emb = self.item_embedding.weight
                E = torch.cat([user_emb, item_emb], dim=0)  # (N+M, d)

                # Store embeddings from all layers
                embeddings_list = [E]

                # Propagate through layers
                for layer in self.prop_layers:
                    E = layer(E, self.laplacian)
                    E = F.dropout(E, p=self.dropout, training=self.training)
                    embeddings_list.append(E)

                # Concatenate embeddings from all layers (Equation 9)
                E_final = torch.cat(embeddings_list, dim=1)  # (N+M, d*(L+1))

                # Split into user and item embeddings
                user_embeddings = E_final[:self.num_users]  # (N, d*(L+1))
                item_embeddings = E_final[self.num_users:]  # (M, d*(L+1))

                # Get embeddings for specific users and items
                users_emb = user_embeddings[user_indices]  # (batch, d*(L+1))
                items_emb = item_embeddings[item_indices]  # (batch, d*(L+1))

                # Inner product for prediction (Equation 10)
                scores = (users_emb * items_emb).sum(dim=1)

                return torch.sigmoid(scores)

        return GNNModel(num_users, num_items, self.embedding_dim, self.num_gnn_layers, self.dropout, laplacian_matrix).to(
            self.device
        )

    def fit(
        self, user_ids: List[int], item_ids: List[int], ratings: List[float], **kwargs
    ) -> "GNNRec":
        """Train the GNNRec model."""
        from corerec.utils.validation import validate_fit_inputs

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

        self.user_item_matrix = csr_matrix(
            (ratings, (user_indices, item_indices)), shape=(num_users, num_items)
        )

        # Create adjacency matrix for user-item graph (bipartite)
        # A = [[0, R], [R^T, 0]] where R is user-item interaction matrix
        adj_size = num_users + num_items
        adj_matrix = torch.zeros(adj_size, adj_size, device=self.device)
        
        # User -> Item connections
        user_idx_tensor = torch.tensor(user_indices, dtype=torch.long, device=self.device)
        item_idx_tensor = torch.tensor(item_indices, dtype=torch.long, device=self.device)
        adj_matrix[user_idx_tensor, item_idx_tensor + num_users] = 1.0
        # Item -> User connections
        adj_matrix[item_idx_tensor + num_users, user_idx_tensor] = 1.0

        # Compute Laplacian matrix L = D^(-1/2) A D^(-1/2)
        # D is diagonal degree matrix
        rowsum = adj_matrix.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        laplacian_matrix = d_mat_inv_sqrt @ adj_matrix @ d_mat_inv_sqrt

        # Build model
        self.model = self._build_model(num_users, num_items, laplacian_matrix)
        self.laplacian_matrix = laplacian_matrix

        # Create training data
        train_user_indices = torch.tensor(user_indices, dtype=torch.long).to(self.device)
        train_item_indices = torch.tensor(item_indices, dtype=torch.long).to(self.device)
        train_labels = torch.tensor(ratings, dtype=torch.float).to(self.device)

        # Define optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        # Train the model
        self.model.train()
        n_batches = len(train_user_indices) // self.batch_size + (
            1 if len(train_user_indices) % self.batch_size != 0 else 0
        )

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
                outputs = self.model(users, items)

                # Compute loss
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if self.verbose:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/n_batches:.4f}")

        self.is_fitted = True
        return self

    def predict(self, user_id: int, item_id: int, **kwargs) -> float:
        """Predict rating for a user-item pair."""
        from corerec.utils.validation import validate_model_fitted

        # Validate inputs
        validate_model_fitted(self.is_fitted, self.name)

        if self.model is None:
            raise ValueError("Model has not been trained yet")

        if user_id not in self.user_map or item_id not in self.item_map:
            return 0.0

        # Get indices
        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]

        # Generate prediction
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
            item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
            prediction = self.model(user_tensor, item_tensor)
            return float(prediction.item())

    def recommend(self, user_id: int, top_k: int = 10, **kwargs) -> List[int]:
        """Generate top-K recommendations for a user."""
        from corerec.utils.validation import validate_model_fitted, validate_user_id, validate_top_k

        # Validate inputs
        validate_model_fitted(self.is_fitted, self.name)
        validate_user_id(user_id, self.user_map if hasattr(self, "user_map") else {})
        validate_top_k(top_k)

        if user_id not in self.user_map:
            return []

        # Get user index
        user_idx = self.user_map[user_id]

        # Get items the user has already interacted with
        exclude_seen = kwargs.get("exclude_seen", True)
        seen_items = set()
        if exclude_seen:
            seen_items = set(self.user_item_matrix[user_idx].indices)

        num_items = len(self.item_map)

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
            batch_users_tensor = torch.tensor(batch_users, dtype=torch.long).to(self.device)
            batch_items_tensor = torch.tensor(batch_items, dtype=torch.long).to(self.device)

            # Get predictions
            with torch.no_grad():
                batch_preds = self.model(batch_users_tensor, batch_items_tensor).cpu().detach().tolist()

            # Add to predictions
            for item_idx, pred in zip(batch_items, batch_preds):
                if exclude_seen and item_idx in seen_items:
                    continue
                predictions.append((item_idx, pred))

        # Sort predictions and get top-K
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_item_indices = [item_idx for item_idx, _ in predictions[:top_k]]

        # Map indices back to original item IDs
        top_items = [
            self.reverse_item_map[idx] for idx in top_item_indices if idx in self.reverse_item_map
        ]

        return top_items

    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save model to disk."""
        import pickle

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.verbose:
            logger.info(f"{self.name} model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "GNNRec":
        """Load model from disk."""
        import pickle

        with open(path, "rb") as f:
            model = pickle.load(f)

        if hasattr(model, "verbose") and model.verbose:
            logger.info(f"Model loaded from {path}")

        return model
