from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import logging
import pickle
from pathlib import Path
import scipy.sparse as sp
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from corerec.base_recommender import BaseCorerec


class ResidualUnit(nn.Module):
    """
    Residual Unit used in Deep Crossing.
    
    Architecture:
    
    Input
     │
     ├─────────────┐
     │             │
     ▼             │
    [Linear]       │
     │             │
     ▼             │
    [BatchNorm]    │
     │             │
     ▼             │
    [Activation]   │
     │             │
     ▼             │
    [Linear]       │
     │             │
     ▼             │
    [BatchNorm]    │
     │             │
     ▼             │
     + ◄───────────┘
     │
     ▼
    Output
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, input_dim: int, activation: str = 'relu'):
        """
        Initialize a residual unit.
        
        Args:
            input_dim: Dimension of input features.
            activation: Activation function to use ('relu', 'tanh', etc).
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual unit.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        identity = x
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.activation(out)
        
        return out


class DeepCrossingModel(nn.Module):
    """
    Deep Crossing model architecture.
    
    Architecture:
    
    ┌─────────────────┐
    │ Feature Inputs  │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │   Embeddings    │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Stacking Layer  │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Residual Units  │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Output Layer   │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │    Prediction   │
    └─────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self,
        num_features: int,
        embedding_dim: int,
        hidden_units: List[int],
        num_residual_units: int,
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        """
        Initialize the Deep Crossing model.
        
        Args:
            num_features: Number of input features.
            embedding_dim: Dimension of feature embeddings.
            hidden_units: List of hidden unit dimensions.
            num_residual_units: Number of residual units.
            activation: Activation function to use.
            dropout: Dropout rate.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        
        # Embedding layer for each feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(1000, embedding_dim, padding_idx=0)  # Default max size, will be resized later
            for _ in range(num_features)
        ])
        
        # Stacking layer (concatenation happens in forward)
        total_emb_dim = num_features * embedding_dim
        
        # Residual units
        self.residual_units = nn.ModuleList()
        input_dim = total_emb_dim
        
        for hidden_dim in hidden_units:
            # Dimension matching layer
            self.residual_units.append(nn.Linear(input_dim, hidden_dim))
            self.residual_units.append(nn.BatchNorm1d(hidden_dim))
            self.residual_units.append(getattr(nn, activation)())
            
            # Residual units with the same dimension
            for _ in range(num_residual_units):
                self.residual_units.append(ResidualUnit(hidden_dim, activation))
                
            input_dim = hidden_dim
        
        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, feature_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            feature_indices: Tensor of feature indices.
                Shape: (batch_size, num_features)
            
        Returns:
            Predicted scores.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Get embeddings for each feature
        embedded_features = []
        for i, embedding_layer in enumerate(self.embeddings):
            # Get indices for the current feature
            feature_idx = feature_indices[:, i]
            embedded = embedding_layer(feature_idx)
            embedded_features.append(embedded)
        
        # Concatenate all embeddings
        x = torch.cat(embedded_features, dim=1)
        
        # Pass through residual units
        for layer in self.residual_units:
            x = layer(x)
        
        # Final output
        x = self.dropout(x)
        logits = self.output(x)
        
        return self.sigmoid(logits)
    
    def reset_parameters(self):
        """
        Reset parameters using Xavier/Glorot initialization.
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)


class HookManager:
    """
    Manager for model hooks to inspect internal states.
    
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
            if layer_name in self.activations:
                del self.activations[layer_name]
            return True
        return False
    
    def clear(self):
        """Clear all hooks and activations."""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
        self.activations.clear()
    
    def get_activation(self, layer_name):
        """
        Get the activation for a layer.
        
        Args:
            layer_name: Name of the layer.
            
        Returns:
            Activation tensor, or None if not available.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return self.activations.get(layer_name, None)


class DeepCrossing_base(BaseCorerec):
    """
    Deep Crossing model base class.
    
    DeepCrossing is a deep learning model for recommendation that uses residual units
    to learn feature interactions. It processes categorical features with embeddings
    and then applies multiple residual units to capture complex interactions.
    
    Architecture:
    
    ┌──────────────────────┐
    │ DeepCrossing_base    │
    ├──────────────────────┤
    │ 1. Preprocess Data   │
    │ 2. Create Embeddings │
    │ 3. Stack Features    │
    │ 4. Residual Units    │
    │ 5. Output Layer      │
    └──────────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self,
        name: str = "DeepCrossing",
        config: Optional[Dict[str, Any]] = None,
        trainable: bool = True,
        verbose: bool = True,
        seed: int = 42
    ):
        """
        Initialize the DeepCrossing model.
        
        Args:
            name: Model name.
            config: Configuration dictionary.
            trainable: Whether the model is trainable.
            verbose: Whether to print verbose output.
            seed: Random seed for reproducibility.
        """
        super().__init__(name, trainable, verbose)
        
        # Set default config if none is provided
        if config is None:
            config = {}
            
        self.embedding_dim = config.get('embedding_dim', 16)
        self.hidden_units = config.get('hidden_units', [64, 32, 16])
        self.num_residual_units = config.get('num_residual_units', 2)
        self.dropout = config.get('dropout', 0.1)
        self.activation = config.get('activation', 'ReLU')
        self.batch_size = config.get('batch_size', 256)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.num_epochs = config.get('num_epochs', 10)
        self.seed = seed or np.random.randint(1000000)
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # For tracking training progress
        self.train_loss_history = []
        self.val_loss_history = []
        
        # Set up logging
        self.logger = logging.getLogger(f"{self.name}_logger")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            
            if not verbose:
                self.logger.setLevel(logging.WARNING)
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        
        # Initialize hook manager for model introspection
        self.hook_manager = HookManager()
        
        # Model will be initialized during fit
        self.model = None
        self.is_fitted = False
    
    def _preprocess_data(self, interactions):
        """
        Preprocess interaction data.
        
        Args:
            interactions: List of (user_id, item_id, features) tuples.
            
        Returns:
            Processed feature data ready for model.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Extract users, items, and features
        self.user_ids = set()
        self.item_ids = set()
        self.feature_names = set()
        
        # Analyze interactions to determine users, items, and features
        for user_id, item_id, features in interactions:
            self.user_ids.add(user_id)
            self.item_ids.add(item_id)
            if isinstance(features, dict):
                for key in features:
                    self.feature_names.add(key)
        
        # Convert to sorted lists for deterministic ordering
        self.user_ids = sorted(list(self.user_ids))
        self.item_ids = sorted(list(self.item_ids))
        self.feature_names = sorted(list(self.feature_names))
        
        # Create mappings for users and items
        self.user_map = {user_id: idx + 1 for idx, user_id in enumerate(self.user_ids)}  # 0 reserved for padding
        self.item_map = {item_id: idx + 1 for idx, item_id in enumerate(self.item_ids)}  # 0 reserved for padding
        
        # Create reverse mappings
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_map.items()}
        
        # Handle categorical features
        self.feature_mappings = {}
        feature_values = {feature: set() for feature in self.feature_names}
        
        for _, _, features in interactions:
            if isinstance(features, dict):
                for key, value in features.items():
                    if key in self.feature_names:
                        feature_values[key].add(value)
        
        # Create mappings for categorical features
        for feature in self.feature_names:
            values = sorted(list(feature_values[feature]))
            self.feature_mappings[feature] = {val: idx + 1 for idx, val in enumerate(values)}  # 0 reserved for padding
        
        # Determine feature dimensions for embeddings
        self.feature_dims = {}
        self.feature_dims['user'] = len(self.user_ids) + 1  # +1 for padding
        self.feature_dims['item'] = len(self.item_ids) + 1  # +1 for padding
        
        for feature in self.feature_names:
            self.feature_dims[feature] = len(self.feature_mappings[feature]) + 1  # +1 for padding
        
        # Process interactions into training data
        X = []
        y = []
        
        for user_id, item_id, features in interactions:
            user_idx = self.user_map[user_id]
            item_idx = self.item_map[item_id]
            
            # Process feature indices
            feature_indices = [user_idx, item_idx]
            
            if isinstance(features, dict):
                for feature in self.feature_names:
                    if feature in features:
                        feature_idx = self.feature_mappings[feature].get(features[feature], 0)
                    else:
                        feature_idx = 0  # Use padding index for missing features
                    feature_indices.append(feature_idx)
            
            # For simplicity, we'll use binary labels (1 for positive interactions)
            label = 1
            
            X.append(feature_indices)
            y.append(label)
        
        return X, y
    
    def _build_model(self):
        """
        Build the Deep Crossing model.
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        num_features = 2 + len(self.feature_names)  # user, item, and other features
        
        self.model = DeepCrossingModel(
            num_features=num_features,
            embedding_dim=self.embedding_dim,
            hidden_units=self.hidden_units,
            num_residual_units=self.num_residual_units,
            activation=self.activation.lower(),
            dropout=self.dropout
        ).to(self.device)
        
        # Resize embeddings to actual dimensions
        self.model.embeddings[0] = nn.Embedding(
            self.feature_dims['user'], self.embedding_dim, padding_idx=0
        ).to(self.device)
        
        self.model.embeddings[1] = nn.Embedding(
            self.feature_dims['item'], self.embedding_dim, padding_idx=0
        ).to(self.device)
        
        for i, feature in enumerate(self.feature_names, start=2):
            self.model.embeddings[i] = nn.Embedding(
                self.feature_dims[feature], self.embedding_dim, padding_idx=0
            ).to(self.device)
        
        # Initialize weights
        self.model.reset_parameters()
    
    def _generate_negative_samples(self, X, num_negatives=4):
        """
        Generate negative samples for training.
        
        Args:
            X: List of feature indices for positive interactions.
            num_negatives: Number of negative samples per positive interaction.
            
        Returns:
            Extended X and y with negative samples.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        neg_X = []
        neg_y = []
        
        # Get all possible items
        all_items = list(range(1, len(self.item_ids) + 1))
        
        for indices in X:
            user_idx = indices[0]
            pos_item_idx = indices[1]
            
            for _ in range(num_negatives):
                # Sample negative item
                while True:
                    neg_item_idx = np.random.choice(all_items)
                    if neg_item_idx != pos_item_idx:
                        break
                
                # Copy all indices but replace item index
                neg_indices = indices.copy()
                neg_indices[1] = neg_item_idx
                
                neg_X.append(neg_indices)
                neg_y.append(0)  # 0 for negative interaction
        
        # Combine positive and negative samples
        extended_X = X + neg_X
        extended_y = [1] * len(X) + neg_y
        
        return extended_X, extended_y
    
    def _train(self, X, y, X_val=None, y_val=None):
        """
        Train the model.
        
        Args:
            X: List of feature indices.
            y: List of target values.
            X_val: Validation feature indices.
            y_val: Validation target values.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Prepare data
        X = np.array(X)
        y = np.array(y)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(X).to(self.device),
            torch.FloatTensor(y).unsqueeze(1).to(self.device)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Validation data
        if X_val is not None and y_val is not None:
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            val_dataset = torch.utils.data.TensorDataset(
                torch.LongTensor(X_val).to(self.device),
                torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            self.train_loss_history.append(epoch_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_dataloader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_dataloader)
                self.val_loss_history.append(val_loss)
                
                if self.verbose:
                    self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, "
                                    f"Train Loss: {epoch_loss:.4f}, "
                                    f"Val Loss: {val_loss:.4f}")
                self.model.train()
            elif self.verbose:
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {epoch_loss:.4f}")
    
    def fit(self, interactions, validation_interactions=None):
        """
        Fit the model to the interactions.
        
        Args:
            interactions: List of (user_id, item_id, features) tuples.
            validation_interactions: Optional validation interactions.
            
        Returns:
            self
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.logger.info(f"Fitting {self.name} model to {len(interactions)} interactions")
        
        # Preprocess data
        X, y = self._preprocess_data(interactions)
        
        # Generate negative samples
        X, y = self._generate_negative_samples(X)
        
        # Preprocess validation data if provided
        X_val, y_val = None, None
        if validation_interactions:
            X_val, y_val = self._preprocess_data(validation_interactions)
            X_val, y_val = self._generate_negative_samples(X_val)
        
        # Build model
        self._build_model()
        
        # Train model
        self._train(X, y, X_val, y_val)
        
        self.is_fitted = True
        return self
    
    def predict(self, user_id, item_id, features=None):
        """
        Predict the score for a user-item pair.
        
        Args:
            user_id: ID of the user.
            item_id: ID of the item.
            features: Optional dictionary of additional features.
            
        Returns:
            Predicted score.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        # Check if user and item exist in the training data
        if user_id not in self.user_map:
            self.logger.warning(f"User {user_id} not in training data.")
            return 0.0
        
        if item_id not in self.item_map:
            self.logger.warning(f"Item {item_id} not in training data.")
            return 0.0
        
        # Get indices
        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]
        
        # Prepare feature indices
        feature_indices = [user_idx, item_idx]
        
        # Add additional features if provided
        if features:
            for feature in self.feature_names:
                if feature in features and features[feature] in self.feature_mappings[feature]:
                    feature_idx = self.feature_mappings[feature][features[feature]]
                else:
                    feature_idx = 0  # Padding index for unknown features
                feature_indices.append(feature_idx)
        else:
            # Use padding indices for missing features
            feature_indices.extend([0] * len(self.feature_names))
        
        # Prepare input tensor
        input_tensor = torch.LongTensor([feature_indices]).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
        
        return output.item()
    
    def recommend(self, user_id, top_n=10, features=None, exclude_seen=True):
        """
        Recommend items for a user.
        
        Args:
            user_id: ID of the user.
            top_n: Number of recommendations to return.
            features: Optional dictionary of additional features.
            exclude_seen: Whether to exclude items the user has already interacted with.
            
        Returns:
            List of (item_id, score) tuples.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        if user_id not in self.user_map:
            self.logger.warning(f"User {user_id} not in training data.")
            return []
        
        # Get user index
        user_idx = self.user_map[user_id]
        
        # Get seen items for this user if excluding them
        seen_items = set()
        if exclude_seen:
            for i, (uid, iid, _) in enumerate(zip(self.user_ids, self.item_ids, self.feature_names)):
                if uid == user_id:
                    seen_items.add(iid)
        
        # Generate predictions for all items
        predictions = []
        self.model.eval()
        
        for item_id in self.item_ids:
            if exclude_seen and item_id in seen_items:
                continue
                
            score = self.predict(user_id, item_id, features)
            predictions.append((item_id, score))
        
        # Sort by predicted score and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model to.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            self.logger.error("Model is not fitted yet. Call fit() first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model_config': {
                'embedding_dim': self.embedding_dim,
                'hidden_units': self.hidden_units,
                'num_residual_units': self.num_residual_units,
                'dropout': self.dropout,
                'activation': self.activation
            },
            'feature_info': {
                'user_ids': self.user_ids,
                'item_ids': self.item_ids,
                'feature_names': self.feature_names,
                'feature_mappings': self.feature_mappings,
                'feature_dims': self.feature_dims
            },
            'mappings': {
                'user_map': self.user_map,
                'item_map': self.item_map,
                'reverse_user_map': self.reverse_user_map,
                'reverse_item_map': self.reverse_item_map
            },
            'model_state': self.model.state_dict(),
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'seed': self.seed,
            'version': '1.0'
        }
        
        # Save model
        torch.save(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, device=None):
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from.
            device: Device to load the model on.
            
        Returns:
            Loaded model.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Load model data
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model_data = torch.load(filepath, map_location=device)
        
        # Create instance
        instance = cls(
            name=os.path.basename(filepath).split('.')[0],
            config={
                'embedding_dim': model_data['model_config']['embedding_dim'],
                'hidden_units': model_data['model_config']['hidden_units'],
                'num_residual_units': model_data['model_config']['num_residual_units'],
                'dropout': model_data['model_config']['dropout'],
                'activation': model_data['model_config']['activation'],
                'batch_size': model_data['model_config']['batch_size'],
                'learning_rate': model_data['model_config']['learning_rate'],
                'num_epochs': model_data['model_config']['num_epochs'],
                'seed': model_data['seed'],
                'device': device
            },
            verbose=True
        )
        
        # Restore feature information
        instance.user_ids = model_data['feature_info']['user_ids']
        instance.item_ids = model_data['feature_info']['item_ids']
        instance.feature_names = model_data['feature_info']['feature_names']
        instance.feature_mappings = model_data['feature_info']['feature_mappings']
        instance.feature_dims = model_data['feature_info']['feature_dims']
        
        # Restore mappings
        instance.user_map = model_data['mappings']['user_map']
        instance.item_map = model_data['mappings']['item_map']
        instance.reverse_user_map = model_data['mappings']['reverse_user_map']
        instance.reverse_item_map = model_data['mappings']['reverse_item_map']
        
        # Restore loss history
        instance.train_loss_history = model_data['train_loss_history']
        instance.val_loss_history = model_data['val_loss_history']
        
        # Build and restore model
        instance._build_model()
        instance.model.load_state_dict(model_data['model_state'])
        instance.is_fitted = True
        
        return instance
    
    def export_feature_importance(self):
        """
        Export feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        # Get weights from residual units
        feature_weights = []
        for i, layer in enumerate(self.model.residual_units):
            if isinstance(layer, nn.Linear):
                # Get absolute weights for this layer
                weights = layer.weight.abs().cpu().detach().numpy()
                feature_weights.append(weights.mean(axis=0))
        
        # If we have multiple layers, average across layers
        if feature_weights:
            avg_weights = np.mean(feature_weights, axis=0)
            
            # Map to feature names (user, item, and other features)
            all_features = ['user', 'item'] + self.feature_names
            importance = {}
            
            # For embedding layers, we need to sum across embedding dimensions
            start_idx = 0
            for feature in all_features:
                end_idx = start_idx + self.embedding_dim
                importance[feature] = float(avg_weights[start_idx:end_idx].sum())
                start_idx = end_idx
            
            # Normalize to sum to 1
            total = sum(importance.values())
            if total > 0:
                for feature in importance:
                    importance[feature] /= total
            
            return importance
        
        return {feature: 1.0 / (2 + len(self.feature_names)) for feature in ['user', 'item'] + self.feature_names}
    
    def register_hook(self, layer_name, callback=None):
        """
        Register a hook for a specific layer.
        
        Args:
            layer_name: Name of the layer to hook.
            callback: Custom hook function (optional). If None, will use default.
            
        Returns:
            bool: Whether the hook was successfully registered.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not hasattr(self, 'model') or self.model is None:
            self.logger.error("Model not initialized. Call fit() first.")
            return False
        
        return self.hook_manager.register_hook(self.model, layer_name, callback)
    
    def get_activation(self, layer_name):
        """
        Get activation for a specific layer.
        
        Args:
            layer_name: Name of the layer.
            
        Returns:
            Activation tensor.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return self.hook_manager.get_activation(layer_name)
    
    def set_device(self, device: str) -> None:
        """
        Set the device to run the model on.
        
        Args:
            device: Device to run the model on ('cpu' or 'cuda').
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.device = torch.device(device)
        if hasattr(self, 'model') and self.model is not None:
            self.model.to(self.device)
    
    def update_incremental(self, new_interactions: List[Tuple], new_user_ids: List = None, new_item_ids: List = None):
        """
        Update the model with new interactions.
        
        Args:
            new_interactions: List of (user_id, item_id, features) tuples.
            new_user_ids: List of new user IDs.
            new_item_ids: List of new item IDs.
            
        Returns:
            self: Updated model.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            return self.fit(new_interactions)
        
        # Add new users and items to mappings
        if new_user_ids:
            for user_id in new_user_ids:
                if user_id not in self.user_map:
                    user_idx = len(self.user_map) + 1
                    self.user_map[user_id] = user_idx
                    self.reverse_user_map[user_idx] = user_id
                    self.user_ids.append(user_id)
        
        if new_item_ids:
            for item_id in new_item_ids:
                if item_id not in self.item_map:
                    item_idx = len(self.item_map) + 1
                    self.item_map[item_id] = item_idx
                    self.reverse_item_map[item_idx] = item_id
                    self.item_ids.append(item_id)
        
        # Extract and process new features
        self._extract_features(new_interactions)
        
        # Rebuild model with updated feature dimensions
        old_state_dict = None
        if hasattr(self, 'model') and self.model is not None:
            old_state_dict = self.model.state_dict()
        
        # Rebuild model
        self._build_model()
        
        # Restore weights for existing parameters
        if old_state_dict is not None:
            # Get new state dict
            new_state_dict = self.model.state_dict()
            
            # Only copy params that exist in both state dicts and have the same shape
            for name, param in old_state_dict.items():
                if name in new_state_dict and new_state_dict[name].shape == param.shape:
                    new_state_dict[name].copy_(param)
            
            # Load updated state dict
            self.model.load_state_dict(new_state_dict)
        
        # Fine-tune with new interactions
        self._train(new_interactions, epochs=min(5, self.num_epochs))
        
        return self