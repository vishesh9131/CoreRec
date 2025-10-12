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


class FM(nn.Module):
    """
    Factorization Machine component for DeepFM.
    
    Architecture:
    
    ┌───────────┐
    │ Features  │
    └─────┬─────┘
          │
          ▼
    ┌─────────────────┐    ┌───────────────────┐
    │ First-order Term│    │ Second-order Term │
    │ (Linear)        │    │ (Interaction)     │
    └────────┬────────┘    └─────────┬─────────┘
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                     ┌───────┐
                     │  Sum  │
                     └───────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, field_dims: List[int], embed_dim: int):
        """
        Initialize FM component.
        
        Args:
            field_dims: Dimensions of each field.
            embed_dim: Embedding dimension.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FM component.
        
        Args:
            x: Input tensor of shape (batch_size, num_fields).
            
        Returns:
            Output tensor of shape (batch_size, 1).
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # First order term
        first_order = self.linear(x)
        
        # Second order term
        embeddings = self.embedding(x)
        square_of_sum = torch.sum(embeddings, dim=1) ** 2
        sum_of_square = torch.sum(embeddings ** 2, dim=1)
        second_order = 0.5 * (square_of_sum - sum_of_square).sum(1, keepdim=True)
        
        return first_order + second_order


class FeaturesLinear(nn.Module):
    """
    Linear part of factorization machine for modeling first-order interactions.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, field_dims: List[int]):
        """
        Initialize features linear component.
        
        Args:
            field_dims: Dimensions of each field.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), 1)
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.fc.weight.data)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear component.
        
        Args:
            x: Input tensor of shape (batch_size, num_fields).
            
        Returns:
            Output tensor of shape (batch_size, 1).
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.fc(x).sum(dim=1) + self.bias


class FeaturesEmbedding(nn.Module):
    """
    Embedding layer for sparse features in the DeepFM model.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, field_dims: List[int], embed_dim: int):
        """
        Initialize features embedding component.
        
        Args:
            field_dims: Dimensions of each field.
            embed_dim: Embedding dimension.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.embedding.weight.data)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedding component.
        
        Args:
            x: Input tensor of shape (batch_size, num_fields).
            
        Returns:
            Output tensor of shape (batch_size, num_fields, embed_dim).
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer perceptron for the deep part of DeepFM.
    
    Architecture:
    
    ┌───────────┐
    │   Input   │
    └─────┬─────┘
          │
          ▼
    ┌─────────────┐
    │ Hidden Layer│
    │    (ReLU)   │
    └─────┬───────┘
          │
          ▼
    ┌─────────────┐
    │ Hidden Layer│
    │    (ReLU)   │
    └─────┬───────┘
          │
          ▼
    ┌─────────────┐
    │ Output Layer│
    └─────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.0, output_layer: bool = True):
        """
        Initialize MLP component.
        
        Args:
            input_dim: Input dimension.
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout probability.
            output_layer: Whether to include output layer.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        layers = []
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        if output_layer:
            if hidden_dims:
                layers.append(nn.Linear(hidden_dims[-1], 1))
            else:
                layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP component.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return self.mlp(x)


class DeepFMModel(nn.Module):
    """
    DeepFM Model combining factorization machines and deep neural networks.
    
    Architecture:
    
    ┌───────────┐
    │   Input   │
    └─────┬─────┘
          │
          ├────────────┬─────────────┐
          │            │             │
          ▼            ▼             ▼
    ┌───────────┐ ┌─────────┐ ┌─────────────┐
    │   Linear  │ │    FM   │ │ Feature Emb │
    └─────┬─────┘ └────┬────┘ └──────┬──────┘
          │            │             │
          │            │             ▼
          │            │      ┌─────────────┐
          │            │      │     MLP     │
          │            │      └──────┬──────┘
          │            │             │
          └────────────┼─────────────┘
                       │
                       ▼
                  ┌─────────┐
                  │   Sum   │
                  └─────────┘
                       │
                       ▼
                  ┌─────────┐
                  │ Sigmoid │
                  └─────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, field_dims: List[int], embed_dim: int, mlp_dims: List[int], dropout: float = 0.0):
        """
        Initialize DeepFM model.
        
        Args:
            field_dims: Dimensions of each field.
            embed_dim: Embedding dimension.
            mlp_dims: Dimensions of MLP layers.
            dropout: Dropout probability.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        self.fm = FM(field_dims, embed_dim)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.mlp = MultiLayerPerceptron(len(field_dims) * embed_dim, mlp_dims, dropout, True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DeepFM model.
        
        Args:
            x: Input tensor of shape (batch_size, num_fields).
            
        Returns:
            Output tensor of shape (batch_size, 1).
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # FM component
        fm_output = self.fm(x)
        
        # Deep component
        embeddings = self.embedding(x)
        mlp_input = embeddings.view(embeddings.size(0), -1)
        mlp_output = self.mlp(mlp_input)
        
        # Combine outputs
        output = fm_output + mlp_output
        
        return torch.sigmoid(output)


class HookManager:
    """
    Manager for model hooks to inspect internal model states.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self):
        """
        Initialize hook manager.
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.hooks = {}
        self.activations = {}
        
    def register_hook(self, model: nn.Module, layer_name: str, hook_fn=None) -> bool:
        """
        Register a hook for a specific layer.
        
        Args:
            model: The model to register the hook on.
            layer_name: Name of the layer to hook.
            hook_fn: Hook function (optional). If None, will use default.
            
        Returns:
            bool: Whether the hook was successfully registered.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        for name, module in model.named_modules():
            if name == layer_name:
                def hook(module, input, output):
                    self.activations[layer_name] = output.detach()
                    if hook_fn:
                        return hook_fn(module, input, output)
                    return output
                
                handle = module.register_forward_hook(hook)
                self.hooks[layer_name] = handle
                return True
        return False
    
    def remove_hook(self, layer_name: str) -> bool:
        """
        Remove a hook.
        
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
    
    def clear_hooks(self) -> None:
        """
        Remove all hooks.
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        for layer_name in list(self.hooks.keys()):
            self.remove_hook(layer_name)
            
    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        Get activation for a specific layer.
        
        Args:
            layer_name: Name of the layer.
            
        Returns:
            Activation tensor if exists, None otherwise.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return self.activations.get(layer_name)


class DeepFM_base(BaseCorerec):
    """
    DeepFM base class for recommendation.
    
    DeepFM is a factorization-machine based neural network for recommendation.
    It combines the power of factorization machines for recommendation and deep learning
    for feature learning in a new neural network architecture.
    
    Architecture Overview:
    
      ┌────────────┐     ┌───────────┐     ┌───────────┐
      │ User Input │     │Item Input │     │ Features  │
      └──────┬─────┘     └─────┬─────┘     └─────┬─────┘
             │                 │                 │
             └────────────────┼─────────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │ Input Layer   │
                      └───────┬───────┘
                              │
                 ┌────────────┴───────────┐
                 │                        │
                 ▼                        ▼
         ┌───────────────┐       ┌───────────────┐
         │ FM Component  │       │ Deep Component│
         └───────┬───────┘       └───────┬───────┘
                 │                       │
                 └───────────┬───────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │ Output Layer│
                      └─────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self,
        name: str = "DeepFM",
        embed_dim: int = 16,
        mlp_dims: List[int] = [64, 32, 16],
        dropout: float = 0.1,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        weight_decay: float = 1e-6,
        device: Optional[str] = None,
        seed: int = 42,
        verbose: bool = True
    ):
        """
        Initialize DeepFM base model.
        
        Args:
            name: Model name.
            embed_dim: Embedding dimension.
            mlp_dims: List of MLP layer dimensions.
            dropout: Dropout probability.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            num_epochs: Number of training epochs.
            weight_decay: L2 regularization.
            device: Device to run the model on ('cpu' or 'cuda').
            seed: Random seed.
            verbose: Whether to print progress.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.name = name
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.verbose = verbose
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize other attributes
        self.model = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.feature_names = []
        self.categorical_features = {}
        self.numerical_features = {}
        self.feature_encoders = {}
        self.numerical_means = {}
        self.numerical_stds = {}
        self.field_dims = []
        self.loss_history = []
        self.is_fitted = False
        
        # Logger
        self.logger = logging.getLogger(f"{self.name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        
        # Hook manager
        self.hook_manager = HookManager()
        
        self.config = {
            'name': name,
            'embed_dim': embed_dim,
            'mlp_dims': mlp_dims,
            'dropout': dropout,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'weight_decay': weight_decay,
            'seed': seed,
            'verbose': verbose
        }
    
    def _extract_features(self, interactions: List[Tuple]) -> None:
        """
        Extract and preprocess features from interactions.
        
        Args:
            interactions: List of (user_id, item_id, features) tuples.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Detect feature names
        if not self.feature_names:
            if len(interactions) > 0 and len(interactions[0]) > 2 and isinstance(interactions[0][2], dict):
                # Extract feature names from the first interaction that has features
                for i, (_, _, features) in enumerate(interactions):
                    if isinstance(features, dict) and features:
                        self.feature_names = list(features.keys())
                        break
            else:
                self.feature_names = []
        
        # Skip if no features
        if not self.feature_names:
            return
        
        # Collect feature values
        all_values = {feature: [] for feature in self.feature_names}
        for _, _, features in interactions:
            if isinstance(features, dict):
                for feature in self.feature_names:
                    if feature in features:
                        all_values[feature].append(features[feature])
                    else:
                        all_values[feature].append(None)
        
        # Determine feature types and create encoders
        for feature in self.feature_names:
            values = [v for v in all_values[feature] if v is not None]
            if not values:
                continue
            
            # Check if feature is categorical or numerical
            if all(isinstance(v, (str, bool)) for v in values) or all(isinstance(v, (int, float)) and v == int(v) for v in values):
                # Categorical feature
                unique_values = list(set(values))
                self.categorical_features[feature] = unique_values
                self.feature_encoders[feature] = {val: i+1 for i, val in enumerate(unique_values)}
                # +1 for padding (0)
                if feature not in self.field_dims:
                    self.field_dims.append(len(unique_values) + 1)
            else:
                # Numerical feature
                numerical_values = [float(v) for v in values if v is not None]
                if numerical_values:
                    self.numerical_features[feature] = True
                    self.numerical_means[feature] = np.mean(numerical_values)
                    self.numerical_stds[feature] = np.std(numerical_values) + 1e-8  # Avoid division by zero
                    # Categorical bins for numerical features (10 bins)
                    if feature not in self.field_dims:
                        self.field_dims.append(11)  # 10 bins + padding
    
    def _build_model(self) -> None:
        """
        Build the DeepFM model.
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        field_dims = [len(self.user_map) + 1, len(self.item_map) + 1] + self.field_dims
        self.model = DeepFMModel(field_dims, self.embed_dim, self.mlp_dims, self.dropout).to(self.device)
    
    def _encode_features(self, user_id, item_id, features=None) -> torch.Tensor:
        """
        Encode user, item, and features into model input.
        
        Args:
            user_id: User ID.
            item_id: Item ID.
            features: Optional features dictionary.
            
        Returns:
            Tensor of encoded features.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if user_id in self.user_map:
            user_idx = self.user_map[user_id]
        else:
            raise ValueError(f"User {user_id} not found in user map")
            
        if item_id in self.item_map:
            item_idx = self.item_map[item_id]
        else:
            raise ValueError(f"Item {item_id} not found in item map")
        
        # Encode user and item
        encoded = [user_idx, item_idx]
        
        # Encode additional features
        if features and self.feature_names:
            for feature in self.feature_names:
                if feature in self.categorical_features:
                    # Categorical feature
                    if feature in features and features[feature] in self.feature_encoders[feature]:
                        encoded.append(self.feature_encoders[feature][features[feature]])
                    else:
                        encoded.append(0)  # Padding value for unknown
                elif feature in self.numerical_features:
                    # Numerical feature
                    if feature in features and features[feature] is not None:
                        # Normalize and bin the value
                        normalized = (float(features[feature]) - self.numerical_means[feature]) / self.numerical_stds[feature]
                        bin_idx = min(int(normalized * 2) + 5, 10)  # Map to 0-10 range
                        encoded.append(bin_idx)
                    else:
                        encoded.append(0)  # Padding value for unknown
        
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def _prepare_batch(self, interactions: List[Tuple], add_negative_samples: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of data for training.
        
        Args:
            interactions: List of (user_id, item_id, features) tuples.
            add_negative_samples: Whether to add negative samples.
            
        Returns:
            Tuple of (X, y) tensors.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        batch_X = []
        batch_y = []
        
        # Process positive interactions
        for user_id, item_id, features in interactions:
            if user_id in self.user_map and item_id in self.item_map:
                x = self._encode_features(user_id, item_id, features)
                batch_X.append(x)
                batch_y.append(1.0)
        
        # Add negative samples
        if add_negative_samples:
            for user_id, _, features in interactions:
                if user_id in self.user_map:
                    # Sample a random item as negative
                    for _ in range(1):  # One negative per positive
                        neg_item_idx = np.random.randint(1, len(self.item_map) + 1)
                        neg_item_id = self.reverse_item_map[neg_item_idx]
                        
                        # Skip if this is a positive interaction
                        user_idx = self.user_map[user_id]
                        if any(u_id == user_id and i_id == neg_item_id for u_id, i_id, _ in interactions):
                            continue
                        
                        x = self._encode_features(user_id, neg_item_id, features)
                        batch_X.append(x)
                        batch_y.append(0.0)
        
        if not batch_X:
            return None, None
        
        X = torch.cat(batch_X, dim=0)
        y = torch.tensor(batch_y, dtype=torch.float32).to(self.device)
        
        return X, y
    
    def _train(self, interactions: List[Tuple], num_epochs: Optional[int] = None) -> None:
        """
        Train the model.
        
        Args:
            interactions: List of (user_id, item_id, features) tuples.
            num_epochs: Number of training epochs (optional).
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if num_epochs is None:
            num_epochs = self.num_epochs
        
        # Get total number of interactions
        n_interactions = len(interactions)
        if n_interactions == 0:
            self.logger.warning("No interactions to train on")
            return
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Set model to training mode
        self.model.train()
        
        # Training loop
        epoch_iterator = range(num_epochs)
        if self.verbose:
            epoch_iterator = tqdm(epoch_iterator, desc="Training")
        
        for epoch in epoch_iterator:
            # Shuffle interactions
            np.random.shuffle(interactions)
            
            # Process in batches
            total_loss = 0
            n_batches = 0
            
            for i in range(0, n_interactions, self.batch_size):
                batch = interactions[i:i+self.batch_size]
                X, y = self._prepare_batch(batch)
                
                if X is None or X.shape[0] == 0:
                    continue
                
                # Forward pass
                optimizer.zero_grad()
                y_pred = self.model(X).squeeze()
                
                # Calculate loss
                loss = F.binary_cross_entropy(y_pred, y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            # Log epoch results
            if n_batches > 0:
                epoch_loss = total_loss / n_batches
                self.loss_history.append(epoch_loss)
                if self.verbose:
                    epoch_iterator.set_postfix({"loss": f"{epoch_loss:.4f}"})
    
    def fit(self, interactions: List[Tuple]) -> 'DeepFM_base':
        """
        Fit the model with interactions data.
        
        Args:
            interactions: List of (user_id, item_id, features) tuples, where features is optional.
            
        Returns:
            self: The fitted model.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.logger.info(f"Fitting {self.name} model with {len(interactions)} interactions")
        
        # Extract user and item IDs
        user_ids = list(set(user_id for user_id, _, _ in interactions))
        item_ids = list(set(item_id for _, item_id, _ in interactions))
        
        # Create user and item mappings
        self.user_map = {user_id: i+1 for i, user_id in enumerate(user_ids)}
        self.item_map = {item_id: i+1 for i, item_id in enumerate(item_ids)}
        self.reverse_user_map = {i: user_id for user_id, i in self.user_map.items()}
        self.reverse_item_map = {i: item_id for item_id, i in self.item_map.items()}
        
        # Extract features
        self._extract_features(interactions)
        
        # Build model
        self._build_model()
        
        # Train model
        self._train(interactions)
        
        self.is_fitted = True
        self.logger.info(f"Model fitting completed with final loss: {self.loss_history[-1]:.4f}")
        
        return self
    
    def predict(self, user_id: Any, item_id: Any, features: Optional[Dict[str, Any]] = None) -> float:
        """
        Predict the likelihood of interaction between a user and an item.
        
        Args:
            user_id: User ID.
            item_id: Item ID.
            features: Optional features dictionary.
            
        Returns:
            Prediction score.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        # Encode input
        try:
            X = self._encode_features(user_id, item_id, features)
        except ValueError as e:
            self.logger.warning(str(e))
            return 0.0
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(X).item()
        
        return prediction
    
    def recommend(self, user_id: Any, top_n: int = 10, exclude_seen: bool = True, features: Optional[Dict[str, Any]] = None) -> List[Tuple[Any, float]]:
        """
        Recommend items for a user.
        
        Args:
            user_id: User ID.
            top_n: Number of recommendations to return.
            exclude_seen: Whether to exclude seen items.
            features: Optional features dictionary.
            
        Returns:
            List of (item_id, score) tuples.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        if user_id not in self.user_map:
            self.logger.warning(f"User {user_id} not found in user map")
            return []
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get items the user has seen
        seen_items = set()
        if exclude_seen:
            # This implementation assumes we have an interactions attribute,
            # which might not be available in practice. In a real implementation,
            # we would need to track this information during training.
            if hasattr(self, 'interactions'):
                for u, i, _ in self.interactions:
                    if u == user_id:
                        seen_items.add(i)
        
        # Generate predictions for all items
        predictions = []
        for item_id in self.item_map:
            if exclude_seen and item_id in seen_items:
                continue
            
            score = self.predict(user_id, item_id, features)
            predictions.append((item_id, score))
        
        # Sort by score and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]
    
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
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'feature_encoders': self.feature_encoders,
            'numerical_means': self.numerical_means,
            'numerical_stds': self.numerical_stds,
            'field_dims': self.field_dims,
            'loss_history': self.loss_history
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> 'DeepFM_base':
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from.
            device: Device to load the model on.
            
        Returns:
            Loaded model.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Create new instance with the saved config
        config = checkpoint['config']
        instance = cls(
            name=config['name'],
            embed_dim=config['embed_dim'],
            mlp_dims=config['mlp_dims'],
            dropout=config['dropout'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            num_epochs=config['num_epochs'],
            weight_decay=config.get('weight_decay', 1e-6),
            device=device,
            seed=config['seed'],
            verbose=config['verbose']
        )
        
        # Load instance variables
        instance.user_map = checkpoint['user_map']
        instance.item_map = checkpoint['item_map']
        instance.reverse_user_map = checkpoint['reverse_user_map']
        instance.reverse_item_map = checkpoint['reverse_item_map']
        instance.feature_names = checkpoint['feature_names']
        instance.categorical_features = checkpoint['categorical_features']
        instance.numerical_features = checkpoint['numerical_features']
        instance.feature_encoders = checkpoint['feature_encoders']
        instance.numerical_means = checkpoint['numerical_means']
        instance.numerical_stds = checkpoint['numerical_stds']
        instance.field_dims = checkpoint['field_dims']
        instance.loss_history = checkpoint['loss_history']
        
        # Build model
        instance._build_model()
        
        # Load model state
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set fitted flag
        instance.is_fitted = True
        
        instance.logger.info(f"Model loaded from {filepath}")
        
        return instance
    
    def get_user_embeddings(self) -> Dict[Any, np.ndarray]:
        """
        Get user embeddings.
        
        Returns:
            Dictionary of user embeddings.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get user embeddings
                # Get user embeddings
        embeddings = {}
        with torch.no_grad():
            for user_id, user_idx in self.user_map.items():
                # Create a dummy input with just the user ID
                X = torch.zeros(1, len(self.field_dims), dtype=torch.long, device=self.device)
                X[0, 0] = user_idx
                
                # Get embedding from the embedding layer
                user_emb = self.model.embedding(X)[:, 0, :].cpu().numpy()[0]
                embeddings[user_id] = user_emb
        
        return embeddings
    
    def get_item_embeddings(self) -> Dict[Any, np.ndarray]:
        """
        Get item embeddings.
        
        Returns:
            Dictionary of item embeddings.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get item embeddings
        embeddings = {}
        with torch.no_grad():
            for item_id, item_idx in self.item_map.items():
                # Create a dummy input with just the item ID
                X = torch.zeros(1, len(self.field_dims), dtype=torch.long, device=self.device)
                X[0, 1] = item_idx
                
                # Get embedding from the embedding layer
                item_emb = self.model.embedding(X)[:, 1, :].cpu().numpy()[0]
                embeddings[item_id] = item_emb
        
        return embeddings
    
    def export_feature_importance(self) -> Dict[str, float]:
        """
        Export feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get weights from first-order component
        with torch.no_grad():
            # Linear weights
            linear_weights = self.model.fm.linear.fc.weight.data.cpu().numpy().squeeze()
            
            # Get importance from weights
            importance = {}
            
            # Add user and item importance
            importance['user'] = float(np.abs(linear_weights[:len(self.user_map)]).mean())
            importance['item'] = float(np.abs(linear_weights[len(self.user_map):len(self.user_map) + len(self.item_map)]).mean())
            
            # Add feature importance
            start_idx = len(self.user_map) + len(self.item_map)
            for i, feature in enumerate(self.feature_names):
                feature_weights = linear_weights[start_idx:start_idx + self.field_dims[i+2]]
                importance[feature] = float(np.abs(feature_weights).mean())
                start_idx += self.field_dims[i+2]
            
            # Normalize to sum to 1
            total = sum(importance.values())
            if total > 0:
                for feature in importance:
                    importance[feature] /= total
        
        return importance
    
    def set_device(self, device: str) -> None:
        """
        Set the device to run the model on.
        
        Args:
            device: Device to run the model on.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.device = torch.device(device)
        if hasattr(self, 'model') and self.model is not None:
            self.model.to(self.device)
    
    def update_incremental(self, new_interactions: List[Tuple], new_users: List = None, new_items: List = None) -> 'DeepFM_base':
        """
        Update the model with new data.
        
        Args:
            new_interactions: List of new interactions.
            new_users: List of new users.
            new_items: List of new items.
            
        Returns:
            Updated model.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            self.logger.warning("Model not fitted yet. Fitting with new data.")
            return self.fit(new_interactions)
        
        # Create new users and items if provided
        if new_users:
            for user_id in new_users:
                if user_id not in self.user_map:
                    self.user_map[user_id] = len(self.user_map) + 1  # +1 for padding
                    self.reverse_user_map[len(self.user_map)] = user_id
        
        if new_items:
            for item_id in new_items:
                if item_id not in self.item_map:
                    self.item_map[item_id] = len(self.item_map) + 1  # +1 for padding
                    self.reverse_item_map[len(self.item_map)] = item_id
        
        # Process new interactions
        if new_interactions:
            # Update field dimensions
            self._extract_features(new_interactions)
            
            # Save old model state
            old_state_dict = None
            old_num_users = 0
            old_num_items = 0
            
            if hasattr(self, 'model') and self.model is not None:
                old_state_dict = self.model.state_dict()
                old_num_users = len(self.user_map) - len(new_users) if new_users else len(self.user_map)
                old_num_items = len(self.item_map) - len(new_items) if new_items else len(self.item_map)
            
            # Build new model
            self._build_model()
            
            # Load old weights where possible
            if old_state_dict is not None:
                with torch.no_grad():
                    # Load params with matching shapes
                    for name, param in self.model.named_parameters():
                        if name in old_state_dict and param.shape == old_state_dict[name].shape:
                            param.copy_(old_state_dict[name])
            
            # Fine-tune with new data
            self._train(new_interactions, num_epochs=min(5, self.num_epochs))
        
        return self

