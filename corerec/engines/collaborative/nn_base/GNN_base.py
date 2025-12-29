import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Optional, Union, Tuple, Any
import logging
from pathlib import Path

from corerec.api.base_recommender import BaseRecommender


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer.

    Implements message passing on a graph for representation learning.

    Architecture:
    ┌─────────────────┐
    │  Node Features  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Message Passing │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │Updated Embeddings│
    └─────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, in_dim: int, out_dim: int, activation=F.relu, dropout: float = 0.1):
        """
        Initialize graph convolutional layer.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            activation: Activation function
            dropout: Dropout rate
        """
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for graph convolution.

        Args:
            features: Node feature tensor of shape (num_nodes, in_dim)
            adj_matrix: Adjacency matrix of shape (num_nodes, num_nodes)

        Returns:
            Updated node embeddings of shape (num_nodes, out_dim)
        """
        # Linear transformation
        support = torch.mm(features, self.weight)

        # Message passing (graph convolution)
        output = torch.spmm(adj_matrix, support)

        # Add bias and apply activation
        output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        # Apply dropout
        output = self.dropout(output)

        return output


class GNNModel(nn.Module):
    """
    Graph Neural Network Model for recommendation.

    This model builds a user-item interaction graph and uses graph convolutional
    networks to learn node embeddings for recommendation.

    Architecture:
    ┌─────────────────┐
    │  User-Item Graph│
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Graph Conv 1   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Graph Conv 2   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Prediction    │
    └─────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
    ):
        """
        Initialize GNN model.

        Args:
            num_users: Number of users
            num_items: Number of items
            embed_dim: Initial embedding dimension
            hidden_dims: Hidden dimensions for graph convolution layers
            dropout: Dropout rate
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_users + num_items
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims

        # Initial node embeddings
        self.node_embedding = nn.Embedding(self.num_items, embed_dim)

        # Graph convolution layers
        self.layers = nn.ModuleList()
        in_dim = embed_dim

        for hidden_dim in hidden_dims:
            self.layers.append(GraphConvLayer(in_dim, hidden_dim, dropout=dropout))
            in_dim = hidden_dim

        # Prediction layer
        self.predict = nn.Linear(in_dim, 1)

        # Initialize embeddings
        nn.init.normal_(self.node_embedding.weight, std=0.1)

    def forward(
        self, adj_matrix: torch.Tensor, user_idx: torch.Tensor = None, item_idx: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the GNN model.

        Args:
            adj_matrix: Sparse adjacency matrix
            user_idx: User indices for prediction (optional)
            item_idx: Item indices for prediction (optional)

        Returns:
            Node embeddings or prediction scores
        """
        # Get initial node embeddings
        x = self.node_embedding.weight

        # Apply graph convolution layers
        for layer in self.layers:
            x = layer(x, adj_matrix)

        # Make predictions if user_idx and item_idx are provided
        if user_idx is not None and item_idx is not None:
            user_embeds = x[user_idx]
            item_embeds = x[item_idx + self.num_users]  # Offset for item indices

            # Dot product for predictions
            scores = torch.sum(user_embeds * item_embeds, dim=1)

            return torch.sigmoid(scores.view(-1, 1))

        return x


class GNN_base(BaseRecommender):
    """
    Graph Neural Network for recommendation.

    This model builds a user-item interaction graph and uses graph convolutional
    networks to learn node embeddings for recommendation.

    Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │                       GNN_base                            │
    │                                                           │
    │  ┌─────────────┐    ┌───────────────┐    ┌────────────┐  │
    │  │ User & Item │    │  Graph Build  │    │Data Process│  │
    │  └──────┬──────┘    └───────┬───────┘    └──────┬─────┘  │
    │         │                   │                   │         │
    │         └───────────┬───────┴─────────┬─────────┘         │
    │                     │                 │                    │
    │                     ▼                 ▼                    │
    │            ┌────────────────┐  ┌─────────────┐            │
    │            │   GNN Model    │  │Training Loop│            │
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

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        name: str = "GNN",
        embed_dim: int = 64,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 1024,
        num_epochs: int = 30,
        neg_samples: int = 5,
        device: str = None,
        seed: int = 42,
        verbose: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the GNN model.

        Args:
            name: Model name
            embed_dim: Initial embedding dimension
            hidden_dims: Hidden dimensions for graph convolution layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Number of samples per batch
            num_epochs: Maximum number of training epochs
            neg_samples: Number of negative samples per positive interaction
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
            self.hidden_dims = config.get("hidden_dims", hidden_dims)
            self.dropout = config.get("dropout", dropout)
            self.learning_rate = config.get("learning_rate", learning_rate)
            self.batch_size = config.get("batch_size", batch_size)
            self.num_epochs = config.get("num_epochs", num_epochs)
            self.neg_samples = config.get("neg_samples", neg_samples)
        else:
            self.embed_dim = embed_dim
            self.hidden_dims = hidden_dims
            self.dropout = dropout
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.neg_samples = neg_samples

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Setup logger
        self._setup_logger()

        # Initialize model
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.BCELoss()

        # Initialize data structures
        self.user_encoding = {}
        self.item_encoding = {}
        self.adj_matrix = None
        self.interactions = None
        self.num_users = 0
        self.num_items = 0

        if self.verbose:
            self.logger.info(
                f"Initialized {self.name} model with {self.embed_dim} embedding dimensions"
            )

    def _setup_logger(self):
        """Setup logger for the model."""
        self.logger = logging.getLogger(f"{self.name}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False

    def _preprocess_data(self, data: List[Tuple[Any, Any, float]]):
        """
        Preprocess data for training.

        Args:
            data: List of (user_id, item_id, rating) tuples
        """
        # Create user and item encodings
        unique_users = set()
        unique_items = set()

        for user_id, item_id, _ in data:
            unique_users.add(user_id)
            unique_items.add(item_id)

        self.user_encoding = {user_id: idx for idx, user_id in enumerate(sorted(unique_users))}
        self.item_encoding = {item_id: idx for idx, item_id in enumerate(sorted(unique_items))}
        self.user_map = self.user_encoding  # Alias for test compatibility
        self.item_map = self.item_encoding  # Alias for test compatibility

        self.num_users = len(self.user_encoding)
        self.num_items = len(self.item_encoding)

        # Store interactions
        self.interactions = []
        for user_id, item_id, rating in data:
            user_idx = self.user_encoding[user_id]
            item_idx = self.item_encoding[item_id]
            self.interactions.append((user_idx, item_idx, float(rating > 0)))

        # Build adjacency matrix for user-item graph
        self._build_adj_matrix()

        if self.verbose:
            self.logger.info(f"Processed data: {self.num_users} users, {self.num_items} items")
            self.logger.info(f"Total interactions: {len(self.interactions)}")

    def _build_adj_matrix(self):
        """Build normalized adjacency matrix for the user-item interaction graph."""
        total_nodes = self.num_users + self.num_items

        # Create sparse adjacency matrix
        adj_indices = []
        for user_idx, item_idx, _ in self.interactions:
            # User -> Item edge
            adj_indices.append([user_idx, self.num_users + item_idx])
            # Item -> User edge (symmetric)
            adj_indices.append([self.num_users + item_idx, user_idx])

        # Add self-loops
        for i in range(total_nodes):
            adj_indices.append([i, i])

        adj_indices = np.array(adj_indices).T
        adj_values = np.ones(len(adj_indices[0]))

        # Create sparse tensor
        adj_shape = (total_nodes, total_nodes)
        adj_matrix = sp.coo_matrix((adj_values, (adj_indices[0], adj_indices[1])), shape=adj_shape)

        # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
        rowsum = np.array(adj_matrix.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        normalized_adj = adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

        # Convert to torch sparse tensor
        normalized_adj = normalized_adj.tocoo()
        indices = torch.LongTensor([normalized_adj.row, normalized_adj.col])
        values = torch.FloatTensor(normalized_adj.data)
        self.adj_matrix = torch.sparse.FloatTensor(
            indices, values, torch.Size(normalized_adj.shape)
        ).to(self.device)

    def _build_model(self):
        """Build the GNN model."""
        self.model = GNNModel(
            num_users=self.num_users,
            num_items=self.num_items,
            embed_dim=self.embed_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.verbose:
            self.logger.info(f"Built GNN model")
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Total parameters: {total_params}")

    def _generate_training_samples(self):
        """
        Generate training samples with negative sampling.

        Returns:
            user_indices, item_indices, labels
        """
        # Positive samples
        user_indices = [user_idx for user_idx, _, _ in self.interactions]
        item_indices = [item_idx for _, item_idx, _ in self.interactions]
        labels = [1.0] * len(self.interactions)

        # Negative sampling
        for user_idx, _, _ in self.interactions:
            user_interacted_items = set(
                item_idx for u_idx, item_idx, _ in self.interactions if u_idx == user_idx
            )

            # Sample negative items
            for _ in range(self.neg_samples):
                neg_item_idx = np.random.randint(0, self.num_items)

                # Ensure negative item is truly negative
                while neg_item_idx in user_interacted_items:
                    neg_item_idx = np.random.randint(0, self.num_items)

                user_indices.append(user_idx)
                item_indices.append(neg_item_idx)
                labels.append(0.0)

        return (
            torch.LongTensor(user_indices),
            torch.LongTensor(item_indices),
            torch.FloatTensor(labels).view(-1, 1),
        )

    def fit(self, data: List[Tuple[Any, Any, float]]) -> "GNN_base":
        """
        Fit the GNN model.

        Args:
            data: List of (user_id, item_id, rating) tuples

        Returns:
            Fitted model
        """
        if self.verbose:
            self.logger.info(f"Fitting {self.name} model on {len(data)} interactions")

        # Preprocess data
        self._preprocess_data(data)

        # Build model
        self._build_model()

        # Generate all training samples
        user_indices, item_indices, labels = self._generate_training_samples()

        # Convert to tensors and send to device
        user_indices = user_indices.to(self.device)
        item_indices = item_indices.to(self.device)
        labels = labels.to(self.device)

        dataset_size = len(labels)
        num_batches = (dataset_size + self.batch_size - 1) // self.batch_size

        # Training loop
        self.loss_history = []

        for epoch in range(self.num_epochs):
            # Shuffle data
            indices = torch.randperm(dataset_size)
            user_indices_shuffled = user_indices[indices]
            item_indices_shuffled = item_indices[indices]
            labels_shuffled = labels[indices]

            epoch_loss = 0

            for i in range(0, dataset_size, self.batch_size):
                # Get batch
                batch_user_indices = user_indices_shuffled[i : i + self.batch_size]
                batch_item_indices = item_indices_shuffled[i : i + self.batch_size]
                batch_labels = labels_shuffled[i : i + self.batch_size]

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(self.adj_matrix, batch_user_indices, batch_item_indices)

                # Compute loss
                loss = self.loss_fn(outputs, batch_labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / num_batches
            self.loss_history.append(avg_epoch_loss)

            if self.verbose and (epoch + 1) % 5 == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_epoch_loss:.4f}")

        self.is_fitted = True
        return self

    def predict(self, user_id: Any, item_id: Any) -> float:
        """
        Predict rating for a user-item pair.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Predicted rating
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        # Check if user and item exist in training data
        if user_id not in self.user_encoding:
            raise ValueError(f"User ID {user_id} not found in training data")

        if item_id not in self.item_encoding:
            raise ValueError(f"Item ID {item_id} not found in training data")

        # Get indices
        user_idx = self.user_encoding[user_id]
        item_idx = self.item_encoding[item_id]

        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        item_tensor = torch.LongTensor([item_idx]).to(self.device)

        # Get prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(self.adj_matrix, user_tensor, item_tensor).item()

        return prediction

    def recommend(
        self, user_id: Any, top_n: int = 10, exclude_seen: bool = True
    ) -> List[Tuple[Any, float]]:
        """
        Generate top-N recommendations for a user.

        Args:
            user_id: User ID
            top_n: Number of recommendations to generate
            exclude_seen: Whether to exclude items the user has already interacted with

        Returns:
            List of (item_id, score) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making recommendations")

        # Check if user exists in training data
        if user_id not in self.user_encoding:
            raise ValueError(f"User ID {user_id} not found in training data")

        user_idx = self.user_encoding[user_id]

        # Get user's interacted items
        if exclude_seen:
            interacted_items = set(
                item_idx for u_idx, item_idx, _ in self.interactions if u_idx == user_idx
            )
        else:
            interacted_items = set()

        # Get predictions for all items
        candidates = []
        for item_id, item_idx in self.item_encoding.items():
            if item_idx not in interacted_items:
                score = self.predict(user_id, item_id)
                candidates.append((item_id, score))

        # Sort by score in descending order
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top-n items
        return candidates[:top_n]

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

        # Convert adjacency matrix to indices and values for storage
        adj_indices = self.adj_matrix._indices().cpu()
        adj_values = self.adj_matrix._values().cpu()
        adj_size = self.adj_matrix.size()

        # Prepare data to save
        model_data = {
            "model_config": {
                "embed_dim": self.embed_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "name": self.name,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "neg_samples": self.neg_samples,
                "seed": self.seed,
                "verbose": self.verbose,
            },
            "data": {
                "user_encoding": self.user_encoding,
                "item_encoding": self.item_encoding,
                "interactions": self.interactions,
                "num_users": self.num_users,
                "num_items": self.num_items,
                "adj_indices": adj_indices,
                "adj_values": adj_values,
                "adj_size": adj_size,
            },
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss_history": self.loss_history if hasattr(self, "loss_history") else [],
        }

        # Save to file
        torch.save(model_data, filepath)

        if self.verbose:
            self.logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "GNN_base":
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        model_data = torch.load(filepath, map_location=device)

        # Create model instance with saved config
        instance = cls(
            name=model_data["model_config"]["name"],
            embed_dim=model_data["model_config"]["embed_dim"],
            hidden_dims=model_data["model_config"]["hidden_dims"],
            dropout=model_data["model_config"]["dropout"],
            learning_rate=model_data["model_config"]["learning_rate"],
            batch_size=model_data["model_config"]["batch_size"],
            num_epochs=model_data["model_config"]["num_epochs"],
            neg_samples=model_data["model_config"]["neg_samples"],
            seed=model_data["model_config"]["seed"],
            verbose=model_data["model_config"]["verbose"],
            device=device,
        )

        # Restore data
        data = model_data["data"]
        instance.user_encoding = data["user_encoding"]
        instance.item_encoding = data["item_encoding"]
        instance.interactions = data["interactions"]
        instance.num_users = data["num_users"]
        instance.num_items = data["num_items"]

        # Restore adjacency matrix
        adj_indices = data["adj_indices"]
        adj_values = data["adj_values"]
        adj_size = data["adj_size"]
        instance.adj_matrix = torch.sparse.FloatTensor(adj_indices, adj_values, adj_size).to(device)

        # Build and load model
        instance._build_model()
        instance.model.load_state_dict(model_data["model_state"])
        instance.optimizer.load_state_dict(model_data["optimizer_state"])
        instance.loss_history = model_data.get("loss_history", [])

        instance.is_fitted = True
        return instance

    def train(self):
        """Required by base class but implemented as fit."""
        pass
