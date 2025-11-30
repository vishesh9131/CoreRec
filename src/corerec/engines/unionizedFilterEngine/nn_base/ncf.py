import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import pickle
from scipy.sparse import csr_matrix
import logging
from collections import defaultdict
import pandas as pd

from corerec.base_recommender import BaseCorerec


class GMFLayer(nn.Module):
    """
    Generalized Matrix Factorization Layer for NCF.

    Performs element-wise product of user and item embeddings.
    """

    def __init__(self, num_users, num_items, embedding_dim=64):
        super(GMFLayer, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        """
        Forward pass of the GMF layer.

        Parameters:
        -----------
        user_indices: torch.Tensor
            Tensor of user indices
        item_indices: torch.Tensor
            Tensor of item indices

        Returns:
        --------
        torch.Tensor
            Element-wise product of user and item embeddings
        """
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)

        # Element-wise product
        return user_embedding * item_embedding


class MLPLayer(nn.Module):
    """
    Multi-Layer Perceptron Layer for NCF.

    Processes concatenated user and item embeddings through hidden layers.
    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
        hidden_layers=(128, 64, 32),
        dropout=0.2,
        batch_norm=True,
        activation="relu",
    ):
        super(MLPLayer, self).__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        # Choose activation function
        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "leaky_relu":
            activation_fn = nn.LeakyReLU(0.2)
        elif activation == "tanh":
            activation_fn = nn.Tanh()
        elif activation == "sigmoid":
            activation_fn = nn.Sigmoid()
        else:  # Default to GELU
            activation_fn = nn.GELU()

        # Build MLP layers
        layers = []
        input_dim = embedding_dim * 2  # Concatenated user and item embeddings

        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(activation_fn)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_layers[-1] if hidden_layers else input_dim

    def forward(self, user_indices, item_indices):
        """
        Forward pass of the MLP layer.

        Parameters:
        -----------
        user_indices: torch.Tensor
            Tensor of user indices
        item_indices: torch.Tensor
            Tensor of item indices

        Returns:
        --------
        torch.Tensor
            Output of the MLP layers
        """
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)

        # Concatenate embeddings
        concat = torch.cat([user_embedding, item_embedding], dim=1)

        # Pass through MLP
        return self.mlp(concat)


class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering Model.

    Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
    for collaborative filtering.
    """

    def __init__(
        self,
        num_users,
        num_items,
        model_type="NeuMF",
        gmf_embedding_dim=64,
        mlp_embedding_dim=64,
        mlp_hidden_layers=(128, 64, 32),
        dropout=0.2,
        batch_norm=True,
        activation="relu",
        pretrained_user_embeddings=None,
        pretrained_item_embeddings=None,
        trainable_embeddings=True,
    ):
        """
        Initialize the NCF model.

        Parameters:
        -----------
        num_users: int
            Number of users
        num_items: int
            Number of items
        model_type: str
            Model type: 'GMF', 'MLP', or 'NeuMF' (fusion of both)
        gmf_embedding_dim: int
            Embedding dimension for GMF component
        mlp_embedding_dim: int
            Embedding dimension for MLP component
        mlp_hidden_layers: tuple
            Hidden layer dimensions for MLP
        dropout: float
            Dropout rate for MLP
        batch_norm: bool
            Whether to use batch normalization
        activation: str
            Activation function: 'relu', 'leaky_relu', 'tanh', 'sigmoid', or 'gelu'
        pretrained_user_embeddings: np.ndarray, optional
            Pretrained user embeddings
        pretrained_item_embeddings: np.ndarray, optional
            Pretrained item embeddings
        trainable_embeddings: bool
            Whether embeddings are trainable
        """
        super(NCFModel, self).__init__()

        self.model_type = model_type
        self.num_users = num_users
        self.num_items = num_items

        # GMF component
        if model_type in ["GMF", "NeuMF"]:
            self.gmf = GMFLayer(num_users, num_items, embedding_dim=gmf_embedding_dim)

            # Load pretrained embeddings if provided
            if pretrained_user_embeddings is not None:
                self.gmf.user_embedding.weight.data.copy_(
                    torch.from_numpy(pretrained_user_embeddings)
                )
            if pretrained_item_embeddings is not None:
                self.gmf.item_embedding.weight.data.copy_(
                    torch.from_numpy(pretrained_item_embeddings)
                )

            # Set trainable status
            self.gmf.user_embedding.weight.requires_grad = trainable_embeddings
            self.gmf.item_embedding.weight.requires_grad = trainable_embeddings

        # MLP component
        if model_type in ["MLP", "NeuMF"]:
            self.mlp = MLPLayer(
                num_users,
                num_items,
                embedding_dim=mlp_embedding_dim,
                hidden_layers=mlp_hidden_layers,
                dropout=dropout,
                batch_norm=batch_norm,
                activation=activation,
            )

            # Load pretrained embeddings if provided
            if pretrained_user_embeddings is not None:
                self.mlp.user_embedding.weight.data.copy_(
                    torch.from_numpy(pretrained_user_embeddings)
                )
            if pretrained_item_embeddings is not None:
                self.mlp.item_embedding.weight.data.copy_(
                    torch.from_numpy(pretrained_item_embeddings)
                )

            # Set trainable status
            self.mlp.user_embedding.weight.requires_grad = trainable_embeddings
            self.mlp.item_embedding.weight.requires_grad = trainable_embeddings

        # Final prediction layer
        if model_type == "GMF":
            self.prediction = nn.Linear(gmf_embedding_dim, 1)
        elif model_type == "MLP":
            self.prediction = nn.Linear(self.mlp.output_dim, 1)
        else:  # NeuMF
            self.prediction = nn.Linear(gmf_embedding_dim + self.mlp.output_dim, 1)

    def forward(self, user_indices, item_indices):
        """
        Forward pass of the NCF model.

        Parameters:
        -----------
        user_indices: torch.Tensor
            Tensor of user indices
        item_indices: torch.Tensor
            Tensor of item indices

        Returns:
        --------
        torch.Tensor
            Predicted ratings or scores
        """
        if self.model_type == "GMF":
            gmf_output = self.gmf(user_indices, item_indices)
            prediction = self.prediction(gmf_output)
        elif self.model_type == "MLP":
            mlp_output = self.mlp(user_indices, item_indices)
            prediction = self.prediction(mlp_output)
        else:  # NeuMF
            gmf_output = self.gmf(user_indices, item_indices)
            mlp_output = self.mlp(user_indices, item_indices)

            # Concatenate GMF and MLP outputs
            concat = torch.cat([gmf_output, mlp_output], dim=1)
            prediction = self.prediction(concat)

        return prediction


class NCF(BaseCorerec):
    """
    Neural Collaborative Filtering (NCF)

    A neural network-based approach to collaborative filtering that models user-item
    interactions with multi-layer perceptrons. It can implement three variants:
    Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and
    Neural Matrix Factorization (NeuMF) which fuses both.

    Features:
    - Flexible architecture with multiple model variants
    - Supports implicit and explicit feedback
    - Handles cold-start scenarios with pretrained embeddings
    - Configurable neural network architecture
    - Multiple training objectives (point-wise, pair-wise, list-wise)

    Reference:
    He et al. "Neural Collaborative Filtering" (WWW 2017)
    """

    def __init__(
        self,
        name: str = "NCF",
        model_type: str = "NeuMF",  # Options: GMF, MLP, NeuMF
        gmf_embedding_dim: int = 64,
        mlp_embedding_dim: int = 64,
        mlp_hidden_layers: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.2,
        batch_norm: bool = True,
        activation: str = "relu",
        learning_rate: float = 0.001,
        batch_size: int = 256,
        num_epochs: int = 20,
        negative_samples: int = 4,
        loss_type: str = "bce",  # Options: bce, bpr, hinge
        l2_regularization: float = 0.00001,
        early_stopping_patience: int = 5,
        pretrained_user_embeddings: Optional[np.ndarray] = None,
        pretrained_item_embeddings: Optional[np.ndarray] = None,
        trainable_embeddings: bool = True,
        sample_strategy: str = "uniform",  # Options: uniform, popularity
        verbose: bool = False,
        seed: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(name=name, trainable=True, verbose=verbose)

        # Model hyperparameters
        self.model_type = model_type
        self.gmf_embedding_dim = gmf_embedding_dim
        self.mlp_embedding_dim = mlp_embedding_dim
        self.mlp_hidden_layers = mlp_hidden_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.negative_samples = negative_samples
        self.loss_type = loss_type
        self.l2_regularization = l2_regularization
        self.early_stopping_patience = early_stopping_patience

        # Embedding settings
        self.pretrained_user_embeddings = pretrained_user_embeddings
        self.pretrained_item_embeddings = pretrained_item_embeddings
        self.trainable_embeddings = trainable_embeddings

        # Sampling strategy
        self.sample_strategy = sample_strategy

        # Other settings
        self.seed = seed
        self.device = device

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Initialize logging
        self.logger = self._setup_logger()

        # These will be set during fit
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.num_users = 0
        self.num_items = 0
        self.item_popularity = None
        self.model = None
        self.user_history = None
        self.is_fitted = False

    def _setup_logger(self):
        """Set up a logger for the model"""
        logger = logging.getLogger(f"{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger

    def _preprocess_data(self, data):
        """
        Preprocess input data for training

        Parameters:
        -----------
        data: DataFrame
            Input data containing user_id, item_id, and rating columns

        Returns:
        --------
        processed_data: Dict
            Dictionary containing processed data information
        """
        # Create user and item mappings
        unique_users = sorted(data["user_id"].unique())
        unique_items = sorted(data["item_id"].unique())

        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}

        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}

        self.num_users = len(unique_users)
        self.num_items = len(unique_items)

        # Create user-item interaction matrix for negative sampling
        self.user_history = defaultdict(set)
        for _, row in data.iterrows():
            user_idx = self.user_mapping[row["user_id"]]
            item_idx = self.item_mapping[row["item_id"]]
            self.user_history[user_idx].add(item_idx)

        # Calculate item popularity for popularity-based sampling
        if self.sample_strategy == "popularity":
            item_counts = data["item_id"].value_counts()
            self.item_popularity = np.zeros(self.num_items)
            for item, count in item_counts.items():
                self.item_popularity[self.item_mapping[item]] = count

            # Convert to probability distribution
            self.item_popularity = self.item_popularity / self.item_popularity.sum()

        self.logger.info(f"Processed data with {self.num_users} users and {self.num_items} items")

        return {
            "user_mapping": self.user_mapping,
            "item_mapping": self.item_mapping,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "user_history": self.user_history,
        }

    def _sample_negatives(self, user_idx, n_samples=1):
        """
        Sample negative items for a user

        Parameters:
        -----------
        user_idx: int
            User index
        n_samples: int
            Number of negative samples to generate

        Returns:
        --------
        list
            List of negative item indices
        """
        positive_items = self.user_history[user_idx]

        # Different sampling strategies
        if self.sample_strategy == "uniform":
            # Uniform sampling
            negative_items = []
            while len(negative_items) < n_samples:
                item = np.random.randint(0, self.num_items)
                if item not in positive_items and item not in negative_items:
                    negative_items.append(item)
        else:  # popularity-based sampling
            # Popularity-based sampling
            negative_items = []
            while len(negative_items) < n_samples:
                item = np.random.choice(self.num_items, p=self.item_popularity)
                if item not in positive_items and item not in negative_items:
                    negative_items.append(item)

        return negative_items

    def _build_model(self):
        """Build the NCF model"""
        self.model = NCFModel(
            num_users=self.num_users,
            num_items=self.num_items,
            model_type=self.model_type,
            gmf_embedding_dim=self.gmf_embedding_dim,
            mlp_embedding_dim=self.mlp_embedding_dim,
            mlp_hidden_layers=self.mlp_hidden_layers,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            activation=self.activation,
            pretrained_user_embeddings=self.pretrained_user_embeddings,
            pretrained_item_embeddings=self.pretrained_item_embeddings,
            trainable_embeddings=self.trainable_embeddings,
        ).to(self.device)

        self.logger.info(f"Built {self.model_type} model")

    def fit(self, data, validation_data=None):
        """
        Train the NCF model

        Parameters:
        -----------
        data: DataFrame
            Training data with user_id, item_id, and rating columns
        validation_data: DataFrame, optional
            Validation data with the same format as training data

        Returns:
        --------
        self
            Trained model instance
        """
        self.logger.info("Started model training")

        # Preprocess data
        self._preprocess_data(data)

        # Build model if not already built
        if self.model is None:
            self._build_model()

        # Prepare data for training
        train_users = []
        train_items = []
        train_ratings = []

        for _, row in data.iterrows():
            user_idx = self.user_mapping[row["user_id"]]
            item_idx = self.item_mapping[row["item_id"]]
            # Convert rating to binary for implicit feedback
            rating = 1.0 if "rating" not in row or row["rating"] > 0 else 0.0

            train_users.append(user_idx)
            train_items.append(item_idx)
            train_ratings.append(rating)

            # Add negative samples
            if self.negative_samples > 0:
                neg_items = self._sample_negatives(user_idx, self.negative_samples)
                for neg_item in neg_items:
                    train_users.append(user_idx)
                    train_items.append(neg_item)
                    train_ratings.append(0.0)  # Negative sample

        train_users = torch.LongTensor(train_users).to(self.device)
        train_items = torch.LongTensor(train_items).to(self.device)
        train_ratings = torch.FloatTensor(train_ratings).unsqueeze(1).to(self.device)

        # Prepare validation data if provided
        if validation_data is not None:
            val_users = []
            val_items = []
            val_ratings = []

            for _, row in validation_data.iterrows():
                if row["user_id"] in self.user_mapping and row["item_id"] in self.item_mapping:
                    user_idx = self.user_mapping[row["user_id"]]
                    item_idx = self.item_mapping[row["item_id"]]
                    # Convert rating to binary for implicit feedback
                    rating = 1.0 if "rating" not in row or row["rating"] > 0 else 0.0

                    val_users.append(user_idx)
                    val_items.append(item_idx)
                    val_ratings.append(rating)

            val_users = torch.LongTensor(val_users).to(self.device)
            val_items = torch.LongTensor(val_items).to(self.device)
            val_ratings = torch.FloatTensor(val_ratings).unsqueeze(1).to(self.device)

        # Set up optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization
        )

        # Set up loss function
        if self.loss_type == "bce":
            criterion = nn.BCEWithLogitsLoss()
        elif self.loss_type == "bpr":
            import torch.nn.functional as F

            criterion = lambda pos, neg: -F.logsigmoid(pos - neg).mean()
        elif self.loss_type == "hinge":
            import torch.nn.functional as F

            criterion = lambda pos, neg: F.relu(1.0 - (pos - neg)).mean()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Training mode
            self.model.train()

            # Shuffle training data
            indices = np.arange(len(train_users))
            np.random.shuffle(indices)

            # Mini-batch training
            epoch_loss = 0
            n_batches = 0

            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i : min(i + self.batch_size, len(indices))]

                # Get batch data
                batch_users = train_users[batch_indices]
                batch_items = train_items[batch_indices]
                batch_ratings = train_ratings[batch_indices]

                # Forward pass
                predictions = self.model(batch_users, batch_items)

                # Compute loss based on loss type
                if self.loss_type == "bce":
                    loss = criterion(predictions, batch_ratings)
                else:
                    import torch.nn.functional as F

                    # For BPR and hinge loss, reshape data to have positive and negative pairs
                    loss = 0  # Initialize loss

                    for j in range(0, len(batch_indices), self.negative_samples + 1):
                        if j + self.negative_samples < len(batch_indices):
                            user_idx = batch_users[j]
                            pos_item_idx = batch_items[j]
                            pos_score = self.model(user_idx.unsqueeze(0), pos_item_idx.unsqueeze(0))

                            neg_losses = []
                            for k in range(1, self.negative_samples + 1):
                                neg_item_idx = batch_items[j + k]
                                neg_score = self.model(
                                    user_idx.unsqueeze(0), neg_item_idx.unsqueeze(0)
                                )
                                neg_losses.append(criterion(pos_score, neg_score))

                            # Average loss over negative samples
                            loss += sum(neg_losses) / len(neg_losses)

                    loss /= len(batch_indices) / (self.negative_samples + 1)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= n_batches

            # Validation
            if validation_data is not None:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(val_users, val_items)

                    if self.loss_type == "bce":
                        val_loss = criterion(val_predictions, val_ratings).item()
                    else:
                        import torch.nn.functional as F

                        # For BPR and hinge loss, we need to compute it differently
                        # This is simplified for validation
                        val_loss = F.binary_cross_entropy_with_logits(
                            val_predictions, val_ratings
                        ).item()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_model = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break

                self.logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            else:
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")

        # Load best model if validation was used
        if validation_data is not None and "best_model" in locals():
            self.model.load_state_dict(best_model)

        self.is_fitted = True
        self.logger.info("Training completed")

        return self

    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair

        Parameters:
        -----------
        user_id: any
            User ID
        item_id: any
            Item ID

        Returns:
        --------
        float
            Predicted rating or score
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        # Check if user and item exist in training data
        if user_id not in self.user_mapping:
            raise ValueError(f"Unknown user: {user_id}")
        if item_id not in self.item_mapping:
            raise ValueError(f"Unknown item: {item_id}")

        # Get indices
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]

        # Predict
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            item_tensor = torch.LongTensor([item_idx]).to(self.device)
            prediction = self.model(user_tensor, item_tensor)

            # Apply sigmoid for binary classes
            if self.loss_type == "bce":
                prediction = torch.sigmoid(prediction)

        return prediction.item()

    def recommend(self, user_id, top_n=10, items_to_ignore=None):
        """
        Generate top-N recommendations for a user

        Parameters:
        -----------
        user_id: any
            User ID
        top_n: int
            Number of recommendations
        items_to_ignore: list, optional
            List of items to exclude from recommendations

        Returns:
        --------
        list
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        if user_id not in self.user_mapping:
            raise ValueError(f"Unknown user: {user_id}")

        user_idx = self.user_mapping[user_id]

        # Get items to ignore
        ignore_indices = set()
        if items_to_ignore:
            for item in items_to_ignore:
                if item in self.item_mapping:
                    ignore_indices.add(self.item_mapping[item])

        # Add items from user history to ignore list
        if user_idx in self.user_history:
            ignore_indices.update(self.user_history[user_idx])

        # Predict scores for all items
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx] * self.num_items).to(self.device)
            item_tensor = torch.LongTensor(range(self.num_items)).to(self.device)
            predictions = self.model(user_tensor, item_tensor)

            # Apply sigmoid for binary classes
            if self.loss_type == "bce":
                predictions = torch.sigmoid(predictions)

            # Convert to numpy array
            predictions = predictions.cpu().numpy().flatten()

        # Create list of (item_id, score) pairs, excluding ignored items
        item_scores = []
        for i in range(self.num_items):
            if i not in ignore_indices:
                item_scores.append((self.reverse_item_mapping[i], predictions[i]))

        # Sort by score in descending order
        item_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-N recommendations
        return [item for item, _ in item_scores[:top_n]]

    def save(self, filepath):
        """
        Save model to file

        Parameters:
        -----------
        filepath: str
            Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        # Prepare data to save
        model_data = {
            "model_config": {
                "model_type": self.model_type,
                "gmf_embedding_dim": self.gmf_embedding_dim,
                "mlp_embedding_dim": self.mlp_embedding_dim,
                "mlp_hidden_layers": self.mlp_hidden_layers,
                "dropout": self.dropout,
                "batch_norm": self.batch_norm,
                "activation": self.activation,
            },
            "training_config": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "negative_samples": self.negative_samples,
                "loss_type": self.loss_type,
                "l2_regularization": self.l2_regularization,
                "sample_strategy": self.sample_strategy,
            },
            "user_item_data": {
                "user_mapping": self.user_mapping,
                "item_mapping": self.item_mapping,
                "reverse_user_mapping": self.reverse_user_mapping,
                "reverse_item_mapping": self.reverse_item_mapping,
                "num_users": self.num_users,
                "num_items": self.num_items,
                "user_history": {k: list(v) for k, v in self.user_history.items()},
                "item_popularity": self.item_popularity,
            },
            "model_state": self.model.state_dict(),
        }

        # Save to file
        torch.save(model_data, filepath)
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
        NCF
            Loaded model
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load saved data
        model_data = torch.load(filepath, map_location=device)

        # Create a new instance with saved configuration
        instance = cls(
            name=os.path.basename(filepath).split(".")[0],
            model_type=model_data["model_config"]["model_type"],
            gmf_embedding_dim=model_data["model_config"]["gmf_embedding_dim"],
            mlp_embedding_dim=model_data["model_config"]["mlp_embedding_dim"],
            mlp_hidden_layers=model_data["model_config"]["mlp_hidden_layers"],
            dropout=model_data["model_config"]["dropout"],
            batch_norm=model_data["model_config"]["batch_norm"],
            activation=model_data["model_config"]["activation"],
            learning_rate=model_data["training_config"]["learning_rate"],
            batch_size=model_data["training_config"]["batch_size"],
            num_epochs=model_data["training_config"]["num_epochs"],
            negative_samples=model_data["training_config"]["negative_samples"],
            loss_type=model_data["training_config"]["loss_type"],
            l2_regularization=model_data["training_config"]["l2_regularization"],
            sample_strategy=model_data["training_config"]["sample_strategy"],
            device=device,
        )

        # Restore user-item data
        instance.user_mapping = model_data["user_item_data"]["user_mapping"]
        instance.item_mapping = model_data["user_item_data"]["item_mapping"]
        instance.reverse_user_mapping = model_data["user_item_data"]["reverse_user_mapping"]
        instance.reverse_item_mapping = model_data["user_item_data"]["reverse_item_mapping"]
        instance.num_users = model_data["user_item_data"]["num_users"]
        instance.num_items = model_data["user_item_data"]["num_items"]
        instance.user_history = {
            k: set(v) for k, v in model_data["user_item_data"]["user_history"].items()
        }
        instance.item_popularity = model_data["user_item_data"]["item_popularity"]

        # Build model and load state
        instance._build_model()
        instance.model.load_state_dict(model_data["model_state"])
        instance.model.eval()

        instance.is_fitted = True
        return instance
