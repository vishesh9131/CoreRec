import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
import logging
from pathlib import Path

from corerec.base_recommender import BaseCorerec


class FieldGate(nn.Module):
    """
    Field Gating mechanism for GateNet.

    This module implements field gating that controls information flow
    from different feature fields based on their relevance.

    Architecture:
    ┌─────────────────┐
    │  Field Features │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Field Attention │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Gated Features │
    └─────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, num_fields: int, embed_dim: int):
        """
        Initialize field gate.

        Args:
            num_fields: Number of feature fields
            embed_dim: Embedding dimension
        """
        super().__init__()

        # Field attention network
        self.field_attention = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply field gating to embeddings.

        Args:
            embeddings: Tensor of shape (batch_size, num_fields, embed_dim)

        Returns:
            Gated features of shape (batch_size, num_fields, embed_dim)
        """
        batch_size, num_fields, embed_dim = embeddings.size()

        # Reshape for field attention
        reshaped = embeddings.reshape(-1, embed_dim)  # (batch_size * num_fields, embed_dim)

        # Calculate attention weights
        attention = self.field_attention(reshaped)  # (batch_size * num_fields, 1)
        attention = attention.reshape(batch_size, num_fields, 1)  # (batch_size, num_fields, 1)

        # Apply gate
        gated_embeddings = embeddings * attention

        return gated_embeddings


class FeatureInteraction(nn.Module):
    """
    Feature Interaction module for GateNet.

    This module models interactions between gated feature fields.

    Architecture:
    ┌─────────────────┐
    │  Gated Features │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Cross Features  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Pooled Output  │
    └─────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, num_fields: int, embed_dim: int):
        """
        Initialize feature interaction module.

        Args:
            num_fields: Number of feature fields
            embed_dim: Embedding dimension
        """
        super().__init__()

        # Cross feature transformation
        self.cross_transform = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

    def forward(self, gated_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply feature interaction to gated embeddings.

        Args:
            gated_embeddings: Tensor of shape (batch_size, num_fields, embed_dim)

        Returns:
            Interaction features of shape (batch_size, embed_dim)
        """
        # Sum pooling across fields
        pooled = torch.sum(gated_embeddings, dim=1)  # (batch_size, embed_dim)

        # Transform for cross features
        transformed = self.cross_transform(pooled)  # (batch_size, embed_dim)

        return transformed


class MLPModule(nn.Module):
    """
    Multi-Layer Perceptron module for GateNet.

    Simple MLP with configurable layers and activation functions.

    Architecture:
    ┌─────────────────┐
    │     Input       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Hidden Layers  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │    Output Layer │
    └─────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self, input_dim: int, hidden_dims: List[int], output_dim: int = 1, dropout: float = 0.1
    ):
        """
        Initialize MLP module.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.mlp(x)


class GateNetModel(nn.Module):
    """
    GateNet Model for recommendation.

    This model uses field gating to control information flow from feature fields
    and model their interactions effectively.

    Architecture:
    ┌─────────────────┐
    │  Input Features │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Embeddings    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Field Gates   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │Feature Interaction│
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ MLP + Prediction│
    └─────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        field_dims: List[int],
        embed_dim: int = 16,
        mlp_dims: List[int] = [128, 64],
        dropout: float = 0.1,
    ):
        """
        Initialize GateNet model.

        Args:
            field_dims: List of feature field dimensions
            embed_dim: Embedding dimension
            mlp_dims: List of hidden layer dimensions for MLP
            dropout: Dropout rate
        """
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)

        # Embedding layers
        self.embedding = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in field_dims])

        # Initialize embeddings
        for embed in self.embedding:
            nn.init.xavier_uniform_(embed.weight)

        # Field gating module
        self.field_gate = FieldGate(self.num_fields, embed_dim)

        # Feature interaction module
        self.feature_interaction = FeatureInteraction(self.num_fields, embed_dim)

        # MLP for prediction
        self.mlp = MLPModule(
            input_dim=embed_dim, hidden_dims=mlp_dims, output_dim=1, dropout=dropout
        )

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass of the GateNet model.

        Args:
            x: Input tensor of shape (batch_size, num_fields)

        Returns:
            Prediction tensor of shape (batch_size, 1)
        """
        # Get field embeddings
        field_embeds = [embedding(x[:, i]) for i, embedding in enumerate(self.embedding)]
        embeddings = torch.stack(field_embeds, dim=1)  # (batch_size, num_fields, embed_dim)

        # Apply field gating
        gated_embeddings = self.field_gate(embeddings)

        # Apply feature interaction
        interaction_features = self.feature_interaction(gated_embeddings)

        # Apply MLP for prediction
        output = self.mlp(interaction_features)

        # Apply sigmoid for classification/recommendation score
        return torch.sigmoid(output)


class GateNet_base(BaseCorerec):
    """
    GateNet for recommendation.

    This model enhances feature representation by using gating mechanisms
    to control information flow from different feature fields and model
    their interactions effectively.

    Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │                      GateNet_base                         │
    │                                                           │
    │  ┌─────────────┐    ┌───────────────┐    ┌────────────┐  │
    │  │ User & Item │    │Feature Mapping│    │Data Process│  │
    │  └──────┬──────┘    └───────┬───────┘    └──────┬─────┘  │
    │         │                   │                   │         │
    │         └───────────┬───────┴─────────┬─────────┘         │
    │                     │                 │                    │
    │                     ▼                 ▼                    │
    │            ┌────────────────┐  ┌─────────────┐            │
    │            │ GateNet Model  │  │Training Loop│            │
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
        name: str = "GateNet",
        embed_dim: int = 16,
        mlp_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        num_epochs: int = 20,
        patience: int = 5,
        shuffle: bool = True,
        device: str = None,
        seed: int = 42,
        verbose: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the GateNet model.

        Args:
            name: Model name
            embed_dim: Embedding dimension
            mlp_dims: Hidden layer dimensions for MLP
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Number of samples per batch
            num_epochs: Maximum number of training epochs
            patience: Early stopping patience
            shuffle: Whether to shuffle data during training
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
            self.mlp_dims = config.get("mlp_dims", mlp_dims)
            self.dropout = config.get("dropout", dropout)
            self.learning_rate = config.get("learning_rate", learning_rate)
            self.batch_size = config.get("batch_size", batch_size)
            self.num_epochs = config.get("num_epochs", num_epochs)
            self.patience = config.get("patience", patience)
            self.shuffle = config.get("shuffle", shuffle)
        else:
            self.embed_dim = embed_dim
            self.mlp_dims = mlp_dims
            self.dropout = dropout
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.patience = patience
            self.shuffle = shuffle

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
        self.field_names = []
        self.field_dims = []
        self.field_mapping = {}

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

    def _preprocess_data(self, data: List[Dict[str, Any]]):
        """
        Preprocess data for training.

        Args:
            data: List of dictionaries with features and label
        """
        # Extract field names
        all_fields = set()
        for sample in data:
            for field in sample.keys():
                if field != "label":
                    all_fields.add(field)

        self.field_names = sorted(list(all_fields))
        if self.verbose:
            self.logger.info(f"Identified {len(self.field_names)} fields: {self.field_names}")

        # Create field mappings
        for field in self.field_names:
            self.field_mapping[field] = {}
            values = set()

            # Collect all values for this field
            for sample in data:
                if field in sample:
                    values.add(sample[field])

            # Map values to indices
            for i, value in enumerate(sorted(list(values))):
                self.field_mapping[field][value] = i + 1  # Reserve 0 for unknown/missing

            # Set field dimension
            self.field_dims.append(len(self.field_mapping[field]) + 1)  # +1 for unknown/missing

        if self.verbose:
            self.logger.info(f"Field dimensions: {self.field_dims}")

    def _build_model(self):
        """Build the GateNet model."""
        self.model = GateNetModel(
            field_dims=self.field_dims,
            embed_dim=self.embed_dim,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.verbose:
            self.logger.info(f"Built GateNet model with {len(self.field_dims)} fields")
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Total parameters: {total_params}")

    def _prepare_batch(self, batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for training.

        Args:
            batch: List of dictionaries with features and label

        Returns:
            Tuple of (features, labels)
        """
        batch_size = len(batch)

        # Initialize tensors
        features = torch.zeros((batch_size, len(self.field_names)), dtype=torch.long)
        labels = torch.zeros((batch_size, 1), dtype=torch.float)

        # Fill tensors with data
        for i, sample in enumerate(batch):
            # Features
            for j, field in enumerate(self.field_names):
                if field in sample:
                    value = sample[field]
                    field_idx = self.field_mapping[field].get(value, 0)  # Use 0 for unknown values
                    features[i, j] = field_idx

            # Label
            if "label" in sample:
                labels[i, 0] = float(sample["label"])

        return features.to(self.device), labels.to(self.device)

    def fit(self, data: List[Dict[str, Any]]) -> "GateNet_base":
        """
        Fit the GateNet model.

        Args:
            data: List of dictionaries with features and label

        Returns:
            Fitted model
        """
        if self.verbose:
            self.logger.info(f"Fitting {self.name} model on {len(data)} samples")

        # Preprocess data
        self._preprocess_data(data)

        # Build model
        self._build_model()

        # Training loop
        num_batches = (len(data) + self.batch_size - 1) // self.batch_size
        best_loss = float("inf")
        patience_counter = 0
        self.loss_history = []

        for epoch in range(self.num_epochs):
            if self.shuffle:
                np.random.shuffle(data)

            epoch_loss = 0

            for i in range(0, len(data), self.batch_size):
                batch = data[i : i + self.batch_size]

                # Prepare data
                features, labels = self._prepare_batch(batch)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(features)

                # Compute loss
                loss = self.loss_fn(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / num_batches
            self.loss_history.append(avg_epoch_loss)

            if self.verbose:
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_epoch_loss:.4f}")

            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        self.is_fitted = True
        return self

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Predict probability for a single sample.

        Args:
            features: Dictionary with feature values

        Returns:
            Predicted probability
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        # Convert features to tensor
        feature_tensor = torch.zeros(1, len(self.field_names), dtype=torch.long)

        for i, field in enumerate(self.field_names):
            if field in features:
                value = features[field]
                field_idx = self.field_mapping.get(field, {}).get(value, 0)
                feature_tensor[0, i] = field_idx

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            feature_tensor = feature_tensor.to(self.device)
            prediction = self.model(feature_tensor).item()

        return prediction

    def recommend(
        self, user_features: Dict[str, Any], item_pool: List[Dict[str, Any]], top_n: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Generate recommendations for a user.

        Args:
            user_features: Dictionary with user features
            item_pool: List of dictionaries with item features
            top_n: Number of recommendations to generate

        Returns:
            List of (item, score) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making recommendations")

        # Score each item in the pool
        scored_items = []
        for item in item_pool:
            # Merge user and item features
            features = {**user_features, **item}

            # Make prediction
            score = self.predict(features)
            scored_items.append((item, score))

        # Sort by score in descending order
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # Return top-n items
        return scored_items[:top_n]

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

        # Prepare data to save
        model_data = {
            "model_config": {
                "embed_dim": self.embed_dim,
                "mlp_dims": self.mlp_dims,
                "dropout": self.dropout,
                "name": self.name,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "patience": self.patience,
                "shuffle": self.shuffle,
                "seed": self.seed,
                "verbose": self.verbose,
            },
            "field_data": {
                "field_names": self.field_names,
                "field_dims": self.field_dims,
                "field_mapping": self.field_mapping,
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
    def load(cls, filepath: str, device: Optional[str] = None) -> "GateNet_base":
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
            mlp_dims=model_data["model_config"]["mlp_dims"],
            dropout=model_data["model_config"]["dropout"],
            learning_rate=model_data["model_config"]["learning_rate"],
            batch_size=model_data["model_config"]["batch_size"],
            num_epochs=model_data["model_config"]["num_epochs"],
            patience=model_data["model_config"]["patience"],
            shuffle=model_data["model_config"]["shuffle"],
            seed=model_data["model_config"]["seed"],
            verbose=model_data["model_config"]["verbose"],
            device=device,
        )

        # Restore field data
        instance.field_names = model_data["field_data"]["field_names"]
        instance.field_dims = model_data["field_data"]["field_dims"]
        instance.field_mapping = model_data["field_data"]["field_mapping"]
        instance.loss_history = model_data.get("loss_history", [])

        # Build and load model
        instance._build_model()
        instance.model.load_state_dict(model_data["model_state"])
        instance.optimizer.load_state_dict(model_data["optimizer_state"])

        instance.is_fitted = True
        return instance

    def train(self):
        """Required by base class but implemented as fit."""
        pass
