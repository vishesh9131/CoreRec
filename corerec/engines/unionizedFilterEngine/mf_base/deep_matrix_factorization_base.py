import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import os
from collections import defaultdict

from corerec.api.base_recommender import BaseRecommender
from corerec.utils.validation import (
    validate_fit_inputs,
    validate_user_id,
    validate_top_k,
    validate_model_fitted,
)


class FMLayer(nn.Module):
    """
    Factorization Machine Layer for DeepFM

    Implements the core FM component that models pairwise feature interactions.
    """

    def __init__(self, field_dims, embedding_dim=16):
        super(FMLayer, self).__init__()
        self.field_dims = field_dims
        self.embedding_dim = embedding_dim
        self.num_fields = len(field_dims)

        # First-order weights
        self.first_order_weights = nn.Parameter(torch.Tensor(sum(field_dims), 1))
        nn.init.xavier_normal_(self.first_order_weights)

        # Embedding weights for second-order interactions
        self.embedding = nn.Embedding(sum(field_dims), embedding_dim)
        nn.init.xavier_normal_(self.embedding.weight)

        # Offset indices for features
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)

    def forward(self, x):
        """
        Forward pass of the FM layer

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor with shape (batch_size, num_fields)

        Returns:
        --------
        torch.Tensor
            Output tensor with shape (batch_size, 1)
        """
        # Add offset to input indices
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        # First-order term
        first_order = torch.sum(self.first_order_weights.squeeze(1)[x], dim=1, keepdim=True)

        # Second-order term
        embedding_x = self.embedding(x)  # (batch, fields, embed_dim)
        square_of_sum = torch.sum(embedding_x, dim=1) ** 2  # (batch, embed_dim)
        sum_of_square = torch.sum(embedding_x**2, dim=1)  # (batch, embed_dim)
        second_order = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        return first_order + second_order


class MLPLayer(nn.Module):
    """
    Multi-Layer Perceptron Layer for DeepFM

    Implements the DNN component that captures high-order feature interactions.
    """

    def __init__(
        self,
        input_dim,
        hidden_layers=(400, 400, 400),
        dropout=0.3,
        batch_norm=True,
        activation="relu",
    ):
        super(MLPLayer, self).__init__()

        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:  # Default to GELU
            self.activation = nn.GELU()

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(self.activation)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x):
        """
        Forward pass of the MLP layer

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor

        Returns:
        --------
        torch.Tensor
            Output tensor
        """
        return self.mlp(x)


class DeepFMModel(nn.Module):
    """
    DeepFM Model

    Combines factorization machines for low-order feature interactions
    with deep neural networks for high-order interactions.
    """

    def __init__(
        self,
        field_dims,
        embedding_dim=16,
        hidden_layers=(400, 400, 400),
        dropout=0.3,
        batch_norm=True,
        activation="relu",
    ):
        super(DeepFMModel, self).__init__()

        self.field_dims = field_dims
        self.embedding_dim = embedding_dim
        self.num_fields = len(field_dims)

        # FM component
        self.fm = FMLayer(field_dims, embedding_dim)

        # Feature embeddings for deep component
        self.feature_embeddings = nn.Embedding(sum(field_dims), embedding_dim)
        nn.init.xavier_normal_(self.feature_embeddings.weight)

        # Offset indices for features
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)

        # Deep component
        self.mlp = MLPLayer(
            self.num_fields * embedding_dim, hidden_layers, dropout, batch_norm, activation
        )

        # Final output layer
        self.output_layer = nn.Linear(self.mlp.output_dim + 1, 1)

    def forward(self, x):
        """
        Forward pass of the DeepFM model

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor with shape (batch_size, num_fields)

        Returns:
        --------
        torch.Tensor
            Output prediction tensor with shape (batch_size, 1)
        """
        # Add offset to input indices
        x_with_offset = x + x.new_tensor(self.offsets).unsqueeze(0)

        # FM part
        fm_output = self.fm(x)

        # Deep part
        embedding_x = self.feature_embeddings(x_with_offset)  # (batch, fields, embed_dim)
        embedding_x = embedding_x.view(-1, self.num_fields * self.embedding_dim)
        deep_output = self.mlp(embedding_x)

        # Concatenate FM and Deep outputs
        concat_output = torch.cat([fm_output, deep_output], dim=1)

        # Final prediction
        prediction = self.output_layer(concat_output)

        return prediction


class DeepFM(BaseRecommender):
    """
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

    Combines factorization machines with deep neural networks to model both low-order and high-order
    feature interactions simultaneously. It integrates the power of factorization machines for
    recommendation with deep learning.

    Features:
    - Joint training of FM and DNN components
    - Shared feature embeddings between the two components
    - Explicit modeling of low-order feature interactions
    - Implicit modeling of high-order feature interactions
    - Handles both categorical and numerical features
    - Effective for CTR prediction and recommendation tasks

    Reference:
    Guo et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction" (IJCAI 2017)
    """

    def __init__(
        self,
        name: str = "DeepFM",
        embedding_dim: int = 16,
        hidden_layers: Tuple[int, ...] = (400, 400, 400),
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
        learning_rate: float = 0.001,
        batch_size: int = 2048,
        epochs: int = 20,
        l2_reg: float = 0.00001,
        early_stopping_patience: int = 5,
        trainable: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        # Model hyperparameters
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.early_stopping_patience = early_stopping_patience

        # Other settings
        self.seed = seed
        self.device = device

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Initialize logging
        self.logger = self._setup_logger()

        # These will be set during fit
        self.field_dims = None
        self.feature_mappings = {}
        self.model = None
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

    def _preprocess_features(self, data):
        """
        Preprocess input features and create feature mappings

        Parameters:
        -----------
        data: DataFrame
            Input data containing user_id, item_id, and other features

        Returns:
        --------
        processed_features: dict
            Dictionary of processed feature information
        """
        # Extract feature columns
        feature_cols = [col for col in data.columns if col not in ["rating", "timestamp"]]
        self.feature_mappings = {}
        field_dims = []

        for col in feature_cols:
            # Process categorical features
            unique_values = sorted(data[col].unique())
            mapping = {val: idx for idx, val in enumerate(unique_values)}

            self.feature_mappings[col] = mapping
            field_dims.append(len(unique_values))

        self.field_dims = field_dims
        self.logger.info(f"Processed {len(feature_cols)} features with dimensions: {field_dims}")

        return {"field_dims": field_dims, "feature_mappings": self.feature_mappings}

    def _encode_features(self, data):
        """
        Encode input features using mappings

        Parameters:
        -----------
        data: DataFrame
            Input data containing features

        Returns:
        --------
        torch.LongTensor
            Tensor of encoded features
        """
        if not self.feature_mappings:
            raise ValueError("Feature mappings not created. Call _preprocess_features first.")

        feature_cols = list(self.feature_mappings.keys())
        encoded_data = np.zeros((len(data), len(feature_cols)), dtype=np.int64)

        for i, col in enumerate(feature_cols):
            mapping = self.feature_mappings[col]
            encoded_data[:, i] = np.array([mapping[val] for val in data[col]])

        return torch.LongTensor(encoded_data)

    def _build_model(self):
        """Build the DeepFM model"""
        self.model = DeepFMModel(
            field_dims=self.field_dims,
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            activation=self.activation,
        ).to(self.device)

        self.logger.info(f"Built DeepFM model with {len(self.field_dims)} fields")

    def fit(self, data, validation_data=None):
        """
        Train the DeepFM model

        Parameters:
        -----------
        data: DataFrame
            Training data with user_id, item_id, features, and ratings
        validation_data: DataFrame, optional
            Validation data with the same format as training data

        Returns:
        --------
        self
            Trained model instance
        """
        # Validate inputs
        validate_fit_inputs(user_ids, item_ids, ratings)

        self.logger.info("Started model training")

        # Preprocess features
        self._preprocess_features(data)

        # Build model if not already built
        if self.model is None:
            self._build_model()

        # Encode features
        X = self._encode_features(data)
        y = torch.FloatTensor(data["rating"].values).unsqueeze(1)

        # Prepare validation data if provided
        if validation_data is not None:
            X_val = self._encode_features(validation_data)
            y_val = torch.FloatTensor(validation_data["rating"].values).unsqueeze(1)

        # Set up optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg
        )

        # Set up loss function
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training mode
            self.model.train()

            # Shuffle indices
            indices = torch.randperm(len(X))

            # Mini-batch training
            total_loss = 0

            for i in range(0, len(X), self.batch_size):
                # Get batch indices
                batch_indices = indices[i : i + self.batch_size]

                # Get batch data
                X_batch = X[batch_indices].to(self.device)
                y_batch = y[batch_indices].to(self.device)

                # Forward pass
                y_pred = self.model(X_batch)

                # Compute loss
                loss = criterion(y_pred, y_batch)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_indices)

            avg_loss = total_loss / len(X)

            # Validation if provided
            if validation_data is not None:
                # Evaluation mode
                self.model.eval()

                with torch.no_grad():
                    # Forward pass on validation data
                    y_val_pred = []
                    for i in range(0, len(X_val), self.batch_size):
                        X_val_batch = X_val[i : i + self.batch_size].to(self.device)
                        batch_pred = self.model(X_val_batch)
                        y_val_pred.append(batch_pred)

                    # Combine predictions
                    y_val_pred = torch.cat(y_val_pred)

                    # Compute validation loss
                    val_loss = criterion(y_val_pred.cpu(), y_val).item()

                    # Log progress
                    if self.verbose:
                        self.logger.info(
                            f"Epoch {epoch+1}/{self.epochs}, "
                            f"Train Loss: {avg_loss:.4f}, "
                            f"Val Loss: {val_loss:.4f}"
                        )

                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        # Load best model
                        self.model.load_state_dict(best_state_dict)
                        break
            else:
                # Log progress without validation
                if self.verbose:
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.epochs}, " f"Train Loss: {avg_loss:.4f}"
                    )

        self.is_fitted = True
        self.logger.info("Training completed")

        return self

    def predict(self, user_id, item_id, additional_features=None):
        """
        Predict rating for a user-item pair

        Parameters:
        -----------
        user_id: any
            User ID
        item_id: any
            Item ID
        additional_features: dict, optional
            Additional features for the prediction

        Returns:
        --------
        float
            Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        # Create feature dictionary
        features = {"user_id": user_id, "item_id": item_id}

        # Add additional features if provided
        if additional_features:
            features.update(additional_features)

        # Create DataFrame with a single row
        import pandas as pd

        data = pd.DataFrame([features])

        # Encode features
        X = self._encode_features(data)

        # Predict
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            prediction = self.model(X).item()

        return prediction

    def recommend(self, user_id, top_n=10, items_to_ignore=None, additional_features=None):
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
        additional_features: dict, optional
            Additional features for prediction

        Returns:
        --------
        list
            List of recommended item IDs
        """
        # Validate inputs
        validate_model_fitted(self.is_fitted, self.name)
        validate_user_id(user_id, self.user_map if hasattr(self, "user_map") else {})
        validate_top_k(top_k if "top_k" in locals() else 10)

        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        # Get all unique items
        all_items = list(self.feature_mappings["item_id"].keys())

        # Filter out items to ignore
        if items_to_ignore:
            candidate_items = [item for item in all_items if item not in items_to_ignore]
        else:
            candidate_items = all_items

        # Get predictions for all candidate items
        scores = []
        for item_id in candidate_items:
            prediction = self.predict(user_id, item_id, additional_features)
            scores.append((item_id, prediction))

        # Sort by predicted rating in descending order
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-N recommendations
        return [item_id for item_id, _ in scores[:top_n]]

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
                "embedding_dim": self.embedding_dim,
                "hidden_layers": self.hidden_layers,
                "dropout": self.dropout,
                "batch_norm": self.batch_norm,
                "activation": self.activation,
            },
            "training_config": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "l2_reg": self.l2_reg,
            },
            "feature_info": {
                "field_dims": self.field_dims,
                "feature_mappings": self.feature_mappings,
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
        DeepFM
            Loaded model
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load saved data
        model_data = torch.load(filepath, map_location=device)

        # Create a new instance with saved configuration
        instance = cls(
            name=os.path.basename(filepath).split(".")[0],
            embedding_dim=model_data["model_config"]["embedding_dim"],
            hidden_layers=model_data["model_config"]["hidden_layers"],
            dropout=model_data["model_config"]["dropout"],
            batch_norm=model_data["model_config"]["batch_norm"],
            activation=model_data["model_config"]["activation"],
            learning_rate=model_data["training_config"]["learning_rate"],
            batch_size=model_data["training_config"]["batch_size"],
            epochs=model_data["training_config"]["epochs"],
            l2_reg=model_data["training_config"]["l2_reg"],
            device=device,
        )

        # Restore feature information
        instance.field_dims = model_data["feature_info"]["field_dims"]
        instance.feature_mappings = model_data["feature_info"]["feature_mappings"]

        # Build model and load state
        instance._build_model()
        instance.model.load_state_dict(model_data["model_state"])
        instance.is_fitted = True

        return instance
