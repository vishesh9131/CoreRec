import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
import os
import logging
import pickle
from pathlib import Path
from collections import defaultdict

from corerec.api.base_recommender import BaseRecommender


class FeatureEmbedding(nn.Module):
    """
    Feature embedding module for ESCMM.

    Maps categorical features to dense vectors.

    Architecture:
    ┌─────────────────┐
    │  Sparse Features│
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Embeddings    │
    └─────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, field_dims: List[int], embed_dim: int):
        """
        Initialize feature embedding layer.

        Args:
            field_dims: List of feature field dimensions
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in field_dims])

        # Initialize embeddings with Xavier uniform
        for embed in self.embedding:
            nn.init.xavier_uniform_(embed.weight)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass for feature embedding.

        Args:
            x: Sparse feature tensor of shape (batch_size, num_fields)

        Returns:
            Embedded features of shape (batch_size, num_fields, embed_dim)
        """
        return torch.stack(
            [embedding(x[:, i]) for i, embedding in enumerate(self.embedding)], dim=1
        )


class FeatureConvolution(nn.Module):
    """
    Feature convolution module for ESCMM.

    Applies 1D convolutions to learn local feature patterns.

    Architecture:
    ┌─────────────────┐
    │  Embeddings     │
    └────────┬────────┘
             │
             ▼
    ┌────────────────┐
    │    Conv1D      │───► ReLU
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │    MaxPool     │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │ Flattened Output│
    └────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        num_fields: int,
        embed_dim: int,
        num_filters: int = 64,
        kernel_sizes: List[int] = [2, 3, 4],
    ):
        """
        Initialize feature convolution layer.

        Args:
            num_fields: Number of feature fields
            embed_dim: Embedding dimension
            num_filters: Number of convolution filters
            kernel_sizes: List of kernel sizes for convolution
        """
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=(k - 1) // 2,
                )
                for k in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature convolution.

        Args:
            x: Embedded features of shape (batch_size, num_fields, embed_dim)

        Returns:
            Convoluted features of shape (batch_size, num_fields, num_filters * len(kernel_sizes))
        """
        # Reshape for convolution: (batch_size, num_fields, embed_dim) -> (batch_size * num_fields, embed_dim, 1)
        batch_size, num_fields, embed_dim = x.size()
        x = x.view(-1, embed_dim, 1)

        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            # Apply 1D convolution and ReLU activation
            conv_out = F.relu(conv(x))

            # Apply max pooling over time dimension
            pooled = F.max_pool1d(conv_out, conv_out.size(2))

            # Flatten pooled output
            pooled = pooled.squeeze(2)

            # Add to list of outputs
            conv_outputs.append(pooled)

        # Concatenate outputs from different kernel sizes
        out = torch.cat(conv_outputs, dim=1)

        # Reshape back to (batch_size, num_fields, num_filters * len(kernel_sizes))
        out = out.view(batch_size, num_fields, -1)

        return out


class DNN(nn.Module):
    """
    Deep Neural Network module for ESCMM.

    Used separately for CTR and CVR towers.

    Architecture:
    ┌─────────────┐
    │Input Features│
    └──────┬──────┘
           │
           ▼
    ┌─────────────────┐
    │ Linear Layer 1  │───► BatchNorm ───► ReLU ───► Dropout
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Linear Layer 2  │───► BatchNorm ───► ReLU ───► Dropout
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Linear Layer N  │
    └─────────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1, batchnorm: bool = True
    ):
        """
        Initialize the DNN module.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            batchnorm: Whether to use batch normalization
        """
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DNN.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.mlp(x)


class ESCMMModel(nn.Module):
    """
    Entire Space Convolutional Multi-task Model (ESCMM) for CVR prediction.

    ESCMM extends ESMM by integrating convolutional networks to better
    capture local patterns in feature interactions.

    Architecture:
    ┌─────────────────┐
    │  Input Features │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Embeddings    │
    └─────────┬───────┘
              │
              ▼
    ┌────────────────────┐
    │ Feature Convolution │
    └────────┬───────────┘
             │
      ┌──────┴───────┐
      │              │
      ▼              ▼
    ┌───────┐      ┌───────┐
    │CTR Net│      │CVR Net│
    └───┬───┘      └───┬───┘
        │              │
        ▼              ▼
      ┌───┐          ┌───┐
      │pCTR│          │pCVR│
      └─┬─┘          └─┬─┘
        │              │
        └──────┬───────┘
               │
               ▼
             ┌───┐
             │pCTCVR│
             └───┘

    References:
        - Inspired by ESMM (Ma, X., et al. 2018) but with added convolutional layers
          for better feature learning.

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        field_dims: List[int],
        embed_dim: int = 16,
        num_filters: int = 64,
        kernel_sizes: List[int] = [2, 3, 4],
        ctr_hidden_dims: List[int] = [128, 64],
        cvr_hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        batchnorm: bool = True,
    ):
        """
        Initialize the ESCMM model.

        Args:
            field_dims: Dimensions of feature fields
            embed_dim: Embedding dimension
            num_filters: Number of filters in convolutional layer
            kernel_sizes: List of kernel sizes for convolution
            ctr_hidden_dims: Hidden layer dimensions for CTR tower
            cvr_hidden_dims: Hidden layer dimensions for CVR tower
            dropout: Dropout rate
            batchnorm: Whether to use batch normalization
        """
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)

        # Feature embedding layer
        self.embedding = FeatureEmbedding(field_dims, embed_dim)

        # Feature convolution layer
        self.feature_conv = FeatureConvolution(
            num_fields=self.num_fields,
            embed_dim=embed_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
        )

        # Calculate input dimension for DNN after convolution
        # Each field has num_filters * len(kernel_sizes) features after convolution
        conv_output_dim = self.num_fields * num_filters * len(kernel_sizes)

        # CTR tower
        self.ctr_net = DNN(
            input_dim=conv_output_dim,
            hidden_dims=ctr_hidden_dims,
            dropout=dropout,
            batchnorm=batchnorm,
        )

        # CVR tower
        self.cvr_net = DNN(
            input_dim=conv_output_dim,
            hidden_dims=cvr_hidden_dims,
            dropout=dropout,
            batchnorm=batchnorm,
        )

        # CTR prediction layer
        self.ctr_pred = nn.Linear(ctr_hidden_dims[-1], 1)

        # CVR prediction layer
        self.cvr_pred = nn.Linear(cvr_hidden_dims[-1], 1)

    def forward(self, x: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ESCMM model.

        Args:
            x: Input tensor of shape (batch_size, num_fields)

        Returns:
            Tuple of (pCTR, pCVR, pCTCVR) tensors, each of shape (batch_size, 1)
        """
        # Get embeddings
        embed_x = self.embedding(x)
        batch_size = embed_x.size(0)

        # Apply feature convolution
        conv_x = self.feature_conv(embed_x)

        # Flatten convoluted features
        flat_x = conv_x.view(batch_size, -1)

        # CTR tower
        ctr_hidden = self.ctr_net(flat_x)
        ctr_output = self.ctr_pred(ctr_hidden)
        p_ctr = torch.sigmoid(ctr_output)

        # CVR tower
        cvr_hidden = self.cvr_net(flat_x)
        cvr_output = self.cvr_pred(cvr_hidden)
        p_cvr = torch.sigmoid(cvr_output)

        # CTCVR = CTR * CVR
        p_ctcvr = p_ctr * p_cvr

        return p_ctr, p_cvr, p_ctcvr


class ESCMM_base(BaseRecommender):
    """
    Entire Space Convolutional Multi-task Model (ESCMM) implementation.

    ESCMM extends ESMM by integrating convolutional networks to better
    capture local patterns in feature interactions for CTR and CVR prediction.

    Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │                       ESCMM_base                          │
    │                                                           │
    │  ┌─────────────┐    ┌───────────────┐    ┌────────────┐  │
    │  │ User & Item │    │Feature Mapping│    │Data Process│  │
    │  └──────┬──────┘    └───────┬───────┘    └──────┬─────┘  │
    │         │                   │                   │         │
    │         └───────────┬───────┴─────────┬─────────┘         │
    │                     │                 │                    │
    │                     ▼                 ▼                    │
    │            ┌────────────────┐  ┌─────────────┐            │
    │            │ ESCMM Model    │  │Training Loop│            │
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

    References:
        - Inspired by ESMM (Ma, X., et al. 2018) but with added convolutional layers
          for better feature learning.

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        name: str = "ESCMM",
        embed_dim: int = 16,
        num_filters: int = 64,
        kernel_sizes: List[int] = [2, 3, 4],
        ctr_hidden_dims: List[int] = [128, 64],
        cvr_hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        batchnorm: bool = True,
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
        Initialize the ESCMM model.

        Args:
            name: Model name
            embed_dim: Embedding dimension
            num_filters: Number of filters in convolutional layer
            kernel_sizes: List of kernel sizes for convolution
            ctr_hidden_dims: Hidden layer dimensions for CTR tower
            cvr_hidden_dims: Hidden layer dimensions for CVR tower
            dropout: Dropout rate
            batchnorm: Whether to use batch normalization
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
        super().__init__()
        self.name = name
        self.seed = seed
        self.verbose = verbose
        self.is_fitted = False

        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Process config if provided
        if config is not None:
            self.embed_dim = config.get("embed_dim", embed_dim)
            self.num_filters = config.get("num_filters", num_filters)
            self.kernel_sizes = config.get("kernel_sizes", kernel_sizes)
            self.ctr_hidden_dims = config.get("ctr_hidden_dims", ctr_hidden_dims)
            self.cvr_hidden_dims = config.get("cvr_hidden_dims", cvr_hidden_dims)
            self.dropout = config.get("dropout", dropout)
            self.batchnorm = config.get("batchnorm", batchnorm)
            self.learning_rate = config.get("learning_rate", learning_rate)
            self.batch_size = config.get("batch_size", batch_size)
            self.num_epochs = config.get("num_epochs", num_epochs)
            self.patience = config.get("patience", patience)
            self.shuffle = config.get("shuffle", shuffle)
        else:
            self.embed_dim = embed_dim
            self.num_filters = num_filters
            self.kernel_sizes = kernel_sizes
            self.ctr_hidden_dims = ctr_hidden_dims
            self.cvr_hidden_dims = cvr_hidden_dims
            self.dropout = dropout
            self.batchnorm = batchnorm
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
            data: List of dictionaries with features, click and conversion labels
        """
        # Extract field names
        all_fields = set()
        for sample in data:
            for field in sample.keys():
                if field not in ["click_label", "conversion_label"]:
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
        """Build the ESCMM model."""
        self.model = ESCMMModel(
            field_dims=self.field_dims,
            embed_dim=self.embed_dim,
            num_filters=self.num_filters,
            kernel_sizes=self.kernel_sizes,
            ctr_hidden_dims=self.ctr_hidden_dims,
            cvr_hidden_dims=self.cvr_hidden_dims,
            dropout=self.dropout,
            batchnorm=self.batchnorm,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Binary cross entropy loss for both tasks
        self.ctr_loss_fn = nn.BCELoss()
        self.ctcvr_loss_fn = nn.BCELoss()

        if self.verbose:
            self.logger.info(f"Built ESCMM model with {len(self.field_dims)} fields")
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Total parameters: {total_params}")

    def _prepare_batch(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for training.

        Args:
            batch: List of dictionaries with features, click and conversion labels

        Returns:
            Tuple of (features, click_labels, conversion_labels)
        """
        batch_size = len(batch)

        # Initialize tensors
        features = torch.zeros((batch_size, len(self.field_names)), dtype=torch.long)
        click_labels = torch.zeros((batch_size, 1), dtype=torch.float)
        conversion_labels = torch.zeros((batch_size, 1), dtype=torch.float)

        # Fill tensors with data
        for i, sample in enumerate(batch):
            # Features
            for j, field in enumerate(self.field_names):
                if field in sample:
                    value = sample[field]
                    field_idx = self.field_mapping[field].get(value, 0)  # Use 0 for unknown values
                    features[i, j] = field_idx

            # Click label (CTR)
            if "click_label" in sample:
                click_labels[i, 0] = float(sample["click_label"])

            # Conversion label (CTCVR)
            # Note: In ESCMM, conversion_label is only valid when click_label is 1
            # If click_label is 0, conversion_label must be 0
            if "conversion_label" in sample and click_labels[i, 0] == 1:
                conversion_labels[i, 0] = float(sample["conversion_label"])

        return (
            features.to(self.device),
            click_labels.to(self.device),
            conversion_labels.to(self.device),
        )

    def fit(self, data: List[Dict[str, Any]]) -> "ESCMM_base":
        """
        Fit the ESCMM model.

        Args:
            data: List of dictionaries with features, click and conversion labels

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
                features, click_labels, conversion_labels = self._prepare_batch(batch)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass: p_ctr, p_cvr, p_ctcvr
                p_ctr, _, p_ctcvr = self.model(features)

                # Compute CTR loss
                ctr_loss = self.ctr_loss_fn(p_ctr, click_labels)

                # Compute CTCVR loss
                ctcvr_loss = self.ctcvr_loss_fn(p_ctcvr, conversion_labels)

                # Total loss (equal weighting of tasks)
                total_loss = ctr_loss + ctcvr_loss

                # Backward pass and optimize
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()

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

    def predict(self, features: Dict[str, Any], task: str = "ctr") -> float:
        """
        Predict probability for a single sample.

        Args:
            features: Dictionary with feature values
            task: Prediction task - 'ctr', 'cvr', or 'ctcvr'

        Returns:
            Predicted probability
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        if task not in ["ctr", "cvr", "ctcvr"]:
            raise ValueError("Task must be one of 'ctr', 'cvr', or 'ctcvr'")

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
            p_ctr, p_cvr, p_ctcvr = self.model(feature_tensor)

            if task == "ctr":
                return p_ctr.item()
            elif task == "cvr":
                return p_cvr.item()
            else:  # ctcvr
                return p_ctcvr.item()

    def recommend(
        self,
        user_features: Dict[str, Any],
        item_pool: List[Dict[str, Any]],
        top_n: int = 10,
        task: str = "ctcvr",
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Generate recommendations for a user.

        Args:
            user_features: Dictionary with user features
            item_pool: List of dictionaries with item features
            top_n: Number of recommendations to generate
            task: Recommendation task - 'ctr', 'cvr', or 'ctcvr'

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

            # Make prediction based on task
            score = self.predict(features, task=task)
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
                "num_filters": self.num_filters,
                "kernel_sizes": self.kernel_sizes,
                "ctr_hidden_dims": self.ctr_hidden_dims,
                "cvr_hidden_dims": self.cvr_hidden_dims,
                "dropout": self.dropout,
                "batchnorm": self.batchnorm,
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
    def load(cls, filepath: str, device: Optional[str] = None) -> "ESCMM_base":
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
            num_filters=model_data["model_config"]["num_filters"],
            kernel_sizes=model_data["model_config"]["kernel_sizes"],
            ctr_hidden_dims=model_data["model_config"]["ctr_hidden_dims"],
            cvr_hidden_dims=model_data["model_config"]["cvr_hidden_dims"],
            dropout=model_data["model_config"]["dropout"],
            batchnorm=model_data["model_config"]["batchnorm"],
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
