import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
import os
import pickle
import yaml
import logging
from pathlib import Path
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from corerec.api.base_recommender import BaseRecommender


class HookManager:
    """
    Manager for model hooks to inspect internal states.

    This class provides functionality to register hooks on specific layers
    of PyTorch models to capture and analyze their activations during forward passes.

    Architecture:
    ┌───────────────┐
    │  HookManager  │
    ├───────────────┤
    │  hooks        │◄───── Stores hook handles
    │  activations  │◄───── Stores layer outputs
    └───────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self):
        """Initialize the hook manager."""
        self.hooks = {}
        self.activations = {}

    def _get_activation(self, name):
        """
        Get activation for a specific layer.

        Args:
            name: Name of the layer to capture activations from.

        Returns:
            A hook function that stores layer outputs.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """

        def hook(model, input, output):
            self.activations[name] = output.detach()

        return hook

    def register_hook(self, model, layer_name, callback=None):
        """
        Register a hook for a specific layer.

        Args:
            model: PyTorch model to register hook on.
            layer_name: Name of the layer to hook.
            callback: Custom hook function (optional). If None, will use default.

        Returns:
            bool: Whether the hook was successfully registered.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            if callback is None:
                callback = self._get_activation(layer_name)
            handle = layer.register_forward_hook(callback)
            self.hooks[layer_name] = handle
            return True

        # Try to find the layer in submodules
        for name, module in model.named_modules():
            if name == layer_name:
                if callback is None:
                    callback = self._get_activation(layer_name)
                handle = module.register_forward_hook(callback)
                self.hooks[layer_name] = handle
                return True

        return False

    def remove_hook(self, layer_name):
        """
        Remove a hook for a specific layer.

        Args:
            layer_name: Name of the layer to remove hook from.

        Returns:
            bool: Whether the hook was successfully removed.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if layer_name in self.hooks:
            self.hooks[layer_name].remove()
            del self.hooks[layer_name]
            return True
        return False

    def get_activation(self, layer_name):
        """
        Get the activation for a specific layer.

        Args:
            layer_name: Name of the layer to get activations for.

        Returns:
            Tensor of activations or None if not found.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return self.activations.get(layer_name, None)

    def clear_activations(self):
        """
        Clear all stored activations.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.activations = {}


class CrossLayer(nn.Module):
    """
    Cross Layer for the DCN model.

    Performs explicit feature interactions by computing x0·x_l^T·w + b + x_l

    Architecture:
    Input x0 (Original) ───┐      Input xl (Previous Layer) ───┐
                           │                                   │
                           ▼                                   ▼
    ┌───────────────────────────────────────────────────────────┐
    │                      Cross Operation                       │
    │                                                           │
    │                  x0·x_l^T·weight + bias                   │
    └───────────────────────────────────────────────────────────┘
                           │                                   │
                           └───────────────────────────► + ◄───┘
                                                       │
                                                       ▼
                                                     Output

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, input_dim: int):
        """
        Initialize the Cross Layer.

        Args:
            input_dim: Dimension of input features.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super(CrossLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim, 1))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the cross layer.

        Args:
            x0: Original input (batch_size, input_dim)
            xl: Output from previous cross layer (batch_size, input_dim)

        Returns:
            Tensor after cross computation with residual connection.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Convert to float if needed, ensure correct dtype for matmul
        x0 = x0.float()
        xl = xl.float()

        # Compute cross term using correct DCN formula: x0 * (xl^T * w) + b + xl
        # xl^T * w: (batch, input_dim) @ (input_dim, 1) -> (batch, 1)
        xl_w = torch.matmul(xl, self.weight)  # (batch, 1)

        # x0 * (xl^T * w): element-wise multiply broadcasts correctly
        cross_term = x0 * xl_w  # (batch, input_dim) * (batch, 1) -> (batch, input_dim)

        # Add bias and residual connection
        output = cross_term + self.bias.squeeze() + xl  # all (batch, input_dim)

        return output


class DNN(nn.Module):
    """
    Deep Neural Network component for the DCN model.

    Computes implicit feature interactions through a series of fully connected layers.

    Architecture:
    Input ──►  FC Layer 1 ──► Activation ──► Dropout
              │
              ▼
         FC Layer 2 ──► Activation ──► Dropout
              │
              ▼
             ...
              │
              ▼
         FC Layer N ──► Output

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        dropout_rate: float = 0.0,
        activation: str = "relu",
    ):
        """
        Initialize the Deep Neural Network.

        Args:
            input_dim: Dimension of input features.
            hidden_layers: List of hidden layer dimensions.
            dropout_rate: Dropout probability for regularization.
            activation: Activation function to use ('relu', 'tanh', etc.)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super(DNN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        # Map activation string to function
        self.activation_map = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "leaky_relu": F.leaky_relu,
        }
        self.activation_fn = self.activation_map.get(activation.lower(), F.relu)

        # Create layers
        layer_dims = [input_dim] + hidden_layers
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DNN.

        Args:
            x: Input tensor (batch_size, input_dim).

        Returns:
            Output tensor after passing through all layers.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation and dropout to all but the last layer
            if i < len(self.layers) - 1:
                x = self.activation_fn(x)
                x = self.dropout(x)
        return x


class DCNModel(nn.Module):
    """
    Deep & Cross Network (DCN) model architecture.

    Combines a cross network for explicit feature interactions with a
    deep neural network for implicit feature interactions.

    Architecture:
                   Input
                     │
                     ▼
               ┌─────────────┐
           ┌───► Cross Net   │───┐
           │   └─────────────┘   │
           │                    │
    Input ─┤                     ├─► Concatenate ──► Output Layer ──► Prediction
           │                    │
           │   ┌─────────────┐   │
           └───►    DNN     │───┘
               └─────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        num_features: int,
        embedding_dim: int,
        num_cross_layers: int = 3,
        hidden_layers: List[int] = [128, 64],
        dropout_rate: float = 0.0,
        activation: str = "relu",
    ):
        """
        Initialize the DCN model.

        Args:
            num_features: Number of features (fields).
            embedding_dim: Dimension of the embeddings.
            num_cross_layers: Number of cross layers.
            hidden_layers: List of hidden layer dimensions for DNN.
            dropout_rate: Dropout probability for DNN.
            activation: Activation function for DNN.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super(DCNModel, self).__init__()

        # Input dimension after field embeddings
        self.input_dim = num_features

        # Cross network
        self.cross_layers = nn.ModuleList()
        for _ in range(num_cross_layers):
            self.cross_layers.append(CrossLayer(self.input_dim))

        # Deep network
        self.deep_network = DNN(
            input_dim=self.input_dim,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        # Final output layer
        # Combine outputs from cross network and deep network
        cross_output_dim = self.input_dim
        deep_output_dim = hidden_layers[-1] if hidden_layers else self.input_dim
        self.final_layer = nn.Linear(cross_output_dim + deep_output_dim, 1)

        # Sigmoid for final output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DCN model.

        Args:
            x: Input tensor (batch_size, num_features).

        Returns:
            Predicted scores (batch_size, 1).

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Convert to float to ensure compatibility
        x = x.float()

        # Cross network
        cross_input = x
        cross_output = x
        for cross_layer in self.cross_layers:
            cross_output = cross_layer(cross_input, cross_output)

        # Deep network
        deep_output = self.deep_network(x)

        # Combine outputs
        combined = torch.cat([cross_output, deep_output], dim=1)

        # Final prediction
        output = self.final_layer(combined)
        return self.sigmoid(output)


class DCN_base(BaseRecommender):
    """
    Deep & Cross Network for Recommendation

    A neural network architecture that combines a cross network for explicit
    feature interactions with a deep neural network for implicit feature interactions.
    It's particularly effective for CTR prediction and feature-rich recommendation tasks.

    Features:
    - Explicit feature crossing via Cross Network
    - Implicit feature interactions via Deep Network
    - Handles categorical and continuous features
    - Configurable architecture for both components
    - Efficient implementation for large-scale data

    Architecture:
                   Features
                     │
       ┌─────────────┴─────────────┐
       │                         │
       ▼                         ▼
    ┌──────┐                  ┌──────┐
    │ Cross│                  │ Deep │
    │ Net  │                  │ Net  │
    └──────┘                  └──────┘
       │                         │
       └─────────────┬─────────────┘
                     │
                     ▼
                ┌─────────┐
                │ Combine │
                └─────────┘
                     │
                     ▼
                ┌─────────┐
                │ Predict │
                └─────────┘

    Reference:
    Wang et al. "Deep & Cross Network for Ad Click Predictions" (2017)

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        name: str = "DCN",
        config: Optional[Dict[str, Any]] = None,
        trainable: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize the DCN model.

        Args:
            name: Name of the model.
            config: Configuration dictionary, can contain:
                - embedding_dim: Dimension of embeddings (default: 16)
                - num_cross_layers: Number of cross layers (default: 3)
                - deep_layers: List of hidden layer dimensions for DNN (default: [128, 64, 32])
                - dropout: Dropout rate for regularization (default: 0.2)
                - activation: Activation function for DNN (default: 'relu')
                - learning_rate: Learning rate for optimizer (default: 0.001)
                - batch_size: Training batch size (default: 256)
                - num_epochs: Number of training epochs (default: 20)
                - l2_reg: L2 regularization coefficient (default: 0.00001)
                - early_stopping_patience: Patience for early stopping (default: 5)
                - feature_types: Dictionary mapping feature names to types
                - device: Device to run model on ('cuda' or 'cpu')
            trainable: Whether the model should be trainable.
            verbose: Whether to show verbose output.
            seed: Random seed for reproducibility.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        # Set default configuration
        default_config = {
            "embedding_dim": 16,
            "num_cross_layers": 3,
            "deep_layers": [128, 64, 32],
            "dropout": 0.2,
            "activation": "relu",
            "learning_rate": 0.001,
            "batch_size": 256,
            "num_epochs": 20,
            "l2_reg": 0.00001,
            "early_stopping_patience": 5,
            "feature_types": {},
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        # Update with provided config
        self.config = default_config
        if config:
            self.config.update(config)

        # Set attributes from config
        self.embedding_dim = self.config["embedding_dim"]
        self.num_cross_layers = self.config["num_cross_layers"]
        self.deep_layers = self.config["deep_layers"]
        self.dropout = self.config["dropout"]
        self.activation = self.config["activation"]
        self.learning_rate = self.config["learning_rate"]
        self.batch_size = self.config["batch_size"]
        self.num_epochs = self.config["num_epochs"]
        self.l2_reg = self.config["l2_reg"]
        self.early_stopping_patience = self.config["early_stopping_patience"]
        self.feature_types = self.config["feature_types"]
        self.device = torch.device(self.config["device"])

        # Set seed for reproducibility
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Initialize hook manager
        self.hook_manager = HookManager()

        # Initialize logging
        self.logger = self._setup_logger()

        # These will be set during fit
        self.feature_names = []
        self.feature_mappings = {}
        self.feature_dims = {}
        self.model = None
        self.is_fitted = False
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}

    def _setup_logger(self):
        """
        Setup logger for the model.

        Returns:
            Logger instance.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        logger = logging.getLogger(f"DCN_{self.name}")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _preprocess_features(self, data: pd.DataFrame) -> None:
        """
        Preprocess features for the DCN model.

        Args:
            data: DataFrame containing training data.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.logger.info("Preprocessing features...")

        # Process users and items
        if "user_id" in data.columns:
            unique_users = sorted(data["user_id"].unique())
            self.user_map = {user: idx for idx, user in enumerate(unique_users)}
            self.reverse_user_map = {idx: user for user, idx in self.user_map.items()}

        if "item_id" in data.columns:
            unique_items = sorted(data["item_id"].unique())
            self.item_map = {item: idx for idx, item in enumerate(unique_items)}
            self.reverse_item_map = {idx: item for item, idx in self.item_map.items()}

        # Process other features
        for col in data.columns:
            if col in ["user_id", "item_id", "rating", "timestamp"]:
                continue

            # Determine feature type
            if col in self.feature_types:
                feature_type = self.feature_types[col]
            else:
                # Auto-detect feature type
                if data[col].dtype == object or data[col].nunique() < 10:
                    feature_type = "categorical"
                else:
                    feature_type = "numerical"

            # Process based on feature type
            if feature_type == "categorical":
                unique_values = sorted(data[col].unique())
                mapping = {val: idx for idx, val in enumerate(unique_values)}

                self.feature_names.append(col)
                self.feature_mappings[col] = mapping
                self.feature_dims[col] = len(unique_values)
            else:  # numerical
                # Store mean and std for normalization
                mean = data[col].mean()
                std = data[col].std() or 1.0  # Avoid division by zero

                self.feature_names.append(col)
                self.feature_mappings[col] = {"mean": mean, "std": std}
                self.feature_dims[col] = 1  # Continuous feature

        self.logger.info(f"Processed {len(self.feature_names)} features")

    def _build_model(self) -> None:
        """
        Build the DCN model.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        num_features = len(self.feature_names) + 2  # +2 for user and item features

        self.model = DCNModel(
            num_features=num_features,
            embedding_dim=self.embedding_dim,
            num_cross_layers=self.num_cross_layers,
            hidden_layers=self.deep_layers,
            dropout_rate=self.dropout,
            activation=self.activation,
        ).to(self.device)

        self.logger.info(f"Built DCN model with {num_features} features")

    def _prepare_batch(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of data for training.

        Args:
            data: DataFrame containing a batch of data.

        Returns:
            Tuple of (features, labels).

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Extract labels (usually ratings or binary feedback)
        if "rating" in data.columns:
            labels = data["rating"].values
            # Normalize ratings to [0, 1] if they're not already
            if labels.max() > 1:
                labels = labels / labels.max()
        else:
            # Assume binary feedback (e.g., clicks, purchases)
            labels = np.ones(len(data))

        # Prepare features
        features = np.zeros((len(data), len(self.feature_names) + 2))  # +2 for user and item

        # Add user and item features
        if "user_id" in data.columns:
            features[:, 0] = data["user_id"].map(self.user_map).values

        if "item_id" in data.columns:
            features[:, 1] = data["item_id"].map(self.item_map).values

        # Add other features
        for i, feat_name in enumerate(self.feature_names):
            if feat_name in data.columns:
                if (
                    isinstance(self.feature_mappings[feat_name], dict)
                    and "mean" in self.feature_mappings[feat_name]
                ):
                    # Numerical feature - normalize
                    mean = self.feature_mappings[feat_name]["mean"]
                    std = self.feature_mappings[feat_name]["std"]
                    features[:, i + 2] = (data[feat_name].values - mean) / std
                else:
                    # Categorical feature - map to index
                    features[:, i + 2] = (
                        data[feat_name].map(self.feature_mappings[feat_name]).fillna(0).values
                    )

        # Convert to tensors
        features_tensor = torch.FloatTensor(features).to(self.device)
        labels_tensor = torch.FloatTensor(labels).unsqueeze(1).to(self.device)

        return features_tensor, labels_tensor

    def _prepare_data(self, data: List[Tuple] or pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for the DCN model.

        Args:
            data: Input data either as a list of tuples (user_id, item_id, rating, [timestamp])
                  or as a DataFrame.

        Returns:
            DataFrame with processed data.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Convert to DataFrame if necessary
        if isinstance(data, list):
            if len(data[0]) == 3:  # (user_id, item_id, rating)
                df = pd.DataFrame(data, columns=["user_id", "item_id", "rating"])
            elif len(data[0]) == 4:  # (user_id, item_id, rating, timestamp)
                df = pd.DataFrame(data, columns=["user_id", "item_id", "rating", "timestamp"])
            else:
                raise ValueError(
                    "Data format not recognized. Expected (user_id, item_id, rating[, timestamp])"
                )
        else:
            df = data.copy()

        return df

    def fit(
        self, data: List[Tuple] or pd.DataFrame, validation_data: List[Tuple] or pd.DataFrame = None
    ) -> "DCN_base":
        """
        Train the DCN model.

        Args:
            data: Training data as list of tuples (user_id, item_id, rating, [timestamp])
                  or as a DataFrame.
            validation_data: Optional validation data in the same format as training data.

        Returns:
            Trained model instance.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.trainable:
            self.logger.warning("Model is not trainable. Skipping training.")
            return self

        # Prepare data
        train_df = self._prepare_data(data)

        # Preprocess features
        self._preprocess_features(train_df)

        # Build model
        self._build_model()

        # Prepare validation data if provided
        val_df = None
        if validation_data is not None:
            val_df = self._prepare_data(validation_data)

        # Training loop
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg
        )
        criterion = nn.BCELoss()

        # Early stopping setup
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        # Training history
        self.train_loss_history = []
        self.val_loss_history = []

        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            # Create batches
            indices = np.random.permutation(len(train_df))
            batch_size = self.batch_size

            with tqdm(
                total=len(indices),
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                disable=not self.verbose,
            ) as pbar:
                for start_idx in range(0, len(indices), batch_size):
                    end_idx = min(start_idx + batch_size, len(indices))
                    batch_indices = indices[start_idx:end_idx]
                    batch_df = train_df.iloc[batch_indices]

                    # Prepare batch
                    X_batch, y_batch = self._prepare_batch(batch_df)

                    # Forward pass
                    optimizer.zero_grad()
                    y_pred = self.model(X_batch)
                    loss = criterion(y_pred, y_batch)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    # Update metrics
                    train_loss += loss.item() * len(X_batch)
                    pbar.update(len(batch_indices))

            train_loss /= len(train_df)
            self.train_loss_history.append(train_loss)

            # Validation
            val_loss = 0.0
            if val_df is not None:
                self.model.eval()
                with torch.no_grad():
                    # Create batches for validation
                    val_indices = np.arange(len(val_df))
                    for start_idx in range(0, len(val_indices), batch_size):
                        end_idx = min(start_idx + batch_size, len(val_indices))
                        batch_indices = val_indices[start_idx:end_idx]
                        batch_df = val_df.iloc[batch_indices]

                        # Prepare batch
                        X_batch, y_batch = self._prepare_batch(batch_df)

                        # Forward pass
                        y_pred = self.model(X_batch)
                        loss = criterion(y_pred, y_batch)

                        val_loss += loss.item() * len(X_batch)

                val_loss /= len(val_df)
                self.val_loss_history.append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        # Restore best model
                        self.model.load_state_dict(best_model_state)
                        break

            # Logging
            if self.verbose:
                if val_df is not None:
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )
                else:
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}"
                    )

        self.is_fitted = True
        self.logger.info("Training completed")
        return self

    def predict(
        self, user_id: Any, item_id: Any, additional_features: Dict[str, Any] = None
    ) -> float:
        """
        Predict rating for a user-item pair.

        Args:
            user_id: User ID.
            item_id: Item ID.
            additional_features: Additional features for prediction.

        Returns:
            Predicted rating or score.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Check if user and item exist in the mapping
        if user_id not in self.user_map or item_id not in self.item_map:
            # Return default score for unknown user or item
            return 0.5

        # Create features array
        features = np.zeros(len(self.feature_names) + 2)  # +2 for user and item

        # Add user and item features
        features[0] = self.user_map[user_id]
        features[1] = self.item_map[item_id]

        # Add additional features if provided
        if additional_features:
            for i, feat_name in enumerate(self.feature_names):
                if feat_name in additional_features:
                    if (
                        isinstance(self.feature_mappings[feat_name], dict)
                        and "mean" in self.feature_mappings[feat_name]
                    ):
                        # Numerical feature - normalize
                        mean = self.feature_mappings[feat_name]["mean"]
                        std = self.feature_mappings[feat_name]["std"]
                        features[i + 2] = (additional_features[feat_name] - mean) / std
                    else:
                        # Categorical feature - map to index
                        value = additional_features[feat_name]
                        if value in self.feature_mappings[feat_name]:
                            features[i + 2] = self.feature_mappings[feat_name][value]

        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features_tensor).item()

        return prediction

    def recommend(
        self,
        user_id: Any,
        top_n: int = 10,
        additional_features: Dict[str, Any] = None,
        items_to_ignore: List[Any] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Recommend items for a user.

        Args:
            user_id: User ID.
            top_n: Number of recommendations to return.
            additional_features: Additional features for prediction.
            items_to_ignore: List of items to exclude from recommendations.

        Returns:
            List of (item_id, score) tuples sorted by score in descending order.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Check if user exists in the mapping
        if user_id not in self.user_map:
            self.logger.warning(f"User {user_id} not in training data.")
            return []

        items_to_ignore = items_to_ignore or []

        # Get all items to score
        items_to_score = [item for item in self.item_map.keys() if item not in items_to_ignore]

        # Score all items
        scores = []
        for item_id in items_to_score:
            score = self.predict(user_id, item_id, additional_features)
            scores.append((item_id, score))

        # Sort by score and return top_n
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def register_hook(self, layer_name: str, callback: callable = None) -> bool:
        """
        Register a hook for a specific layer.

        Args:
            layer_name: Name of the layer to hook.
            callback: Custom hook function.

        Returns:
            Whether the hook was successfully registered.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not hasattr(self, "model") or self.model is None:
            self.logger.error("Model not initialized. Call fit() first.")
            return False

        return self.hook_manager.register_hook(self.model, layer_name, callback)

    def get_activation(self, layer_name: str) -> torch.Tensor:
        """
        Get activation for a specific layer.

        Args:
            layer_name: Name of the layer.

        Returns:
            Activation tensor.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return self.hook_manager.get_activation(layer_name)

    def save(self, filepath: str) -> None:
        """
        Save model to file.

        Args:
            filepath: Path to save the model to.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            self.logger.error("Model is not fitted yet. Call fit() first.")
            return

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        # Prepare model data
        model_data = {
            "model_config": {
                "embedding_dim": self.embedding_dim,
                "num_cross_layers": self.num_cross_layers,
                "deep_layers": self.deep_layers,
                "dropout": self.dropout,
                "activation": self.activation,
            },
            "feature_info": {
                "feature_names": self.feature_names,
                "feature_mappings": self.feature_mappings,
                "feature_dims": self.feature_dims,
            },
            "user_item_mappings": {
                "user_map": self.user_map,
                "item_map": self.item_map,
                "reverse_user_map": self.reverse_user_map,
                "reverse_item_map": self.reverse_item_map,
            },
            "model_state": self.model.state_dict(),
            "train_loss_history": self.train_loss_history,
            "val_loss_history": getattr(self, "val_loss_history", []),
            "seed": self.seed,
            "version": "1.0",
        }

        # Save model
        torch.save(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

        # Save config separately for human readability
        config_path = f"{os.path.splitext(filepath)[0]}_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(
                {
                    "name": self.name,
                    "embedding_dim": self.embedding_dim,
                    "num_cross_layers": self.num_cross_layers,
                    "deep_layers": self.deep_layers,
                    "dropout": self.dropout,
                    "activation": self.activation,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "seed": self.seed,
                    "version": "1.0",
                    "saved_at": str(datetime.now()),
                },
                f,
            )

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "DCN_base":
        """
        Load model from file.

        Args:
            filepath: Path to load the model from.
            device: Device to load the model on. If None, will use cuda if available.

        Returns:
            Loaded model.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model data
        model_data = torch.load(filepath, map_location=device)

        # Create new instance
        instance = cls(
            name=os.path.basename(filepath).split(".")[0],
            embedding_dim=model_data["model_config"]["embedding_dim"],
            num_cross_layers=model_data["model_config"]["num_cross_layers"],
            deep_layers=model_data["model_config"]["deep_layers"],
            dropout=model_data["model_config"]["dropout"],
            activation=model_data["model_config"]["activation"],
            device=device,
        )

        # Restore feature information
        instance.feature_names = model_data["feature_info"]["feature_names"]
        instance.feature_mappings = model_data["feature_info"]["feature_mappings"]
        instance.feature_dims = model_data["feature_info"]["feature_dims"]

        # Restore user-item mappings
        instance.user_map = model_data["user_item_mappings"]["user_map"]
        instance.item_map = model_data["user_item_mappings"]["item_map"]
        instance.reverse_user_map = model_data["user_item_mappings"]["reverse_user_map"]
        instance.reverse_item_map = model_data["user_item_mappings"]["reverse_item_map"]

        # Restore loss history
        instance.train_loss_history = model_data.get("train_loss_history", [])
        if "val_loss_history" in model_data:
            instance.val_loss_history = model_data["val_loss_history"]

        # Build and restore model
        instance._build_model()
        instance.model.load_state_dict(model_data["model_state"])
        instance.is_fitted = True

        return instance

    def export_feature_importance(self) -> Dict[str, float]:
        """
        Export feature importance.

        Feature importance is calculated by analyzing the weights of the cross layers.
        Higher absolute weight values indicate greater importance.

        Returns:
            Dictionary mapping feature names to importance scores.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Get cross layer weights
        feature_weights = []
        for i in range(self.num_cross_layers):
            layer_name = f"cross_net.cross_layers.{i}"
            for name, module in self.model.named_modules():
                if name == layer_name:
                    # Get absolute weights
                    weights = module.weight.abs().cpu().detach().numpy()
                    feature_weights.append(weights)

        # Average weights across layers
        avg_weights = np.mean(np.concatenate(feature_weights, axis=1), axis=1)

        # Map to feature names
        feature_importance = {}
        for i, name in enumerate(self.feature_names):
            feature_importance[name] = float(avg_weights[i])

        # Normalize to sum to 1
        total = sum(feature_importance.values())
        for name in feature_importance:
            feature_importance[name] /= total

        return feature_importance

    def set_device(self, device: str) -> None:
        """
        Set the device to run the model on.

        Args:
            device: Device to run the model on ('cpu' or 'cuda').

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.device = torch.device(device)
        if hasattr(self, "model") and self.model is not None:
            self.model.to(self.device)
