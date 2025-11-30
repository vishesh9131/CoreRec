import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from corerec.base_recommender import BaseCorerec


class DeepFM(BaseCorerec):
    """
    Deep Factorization Machine (DeepFM)

    Combines factorization machines for recommendation with deep neural networks.
    It jointly learns a factorization machine for recommendation and deep representations
    of features through a neural network.

    Reference:
    Guo et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction" (IJCAI 2017)
    """

    def __init__(
        self,
        name: str = "DeepFM",
        embedding_dim: int = 16,
        hidden_layers: List[int] = [400, 400, 400],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        trainable: bool = True,
        verbose: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        self.feature_map = {}
        self.field_dims = []
        self.model = None

    def _build_model(self, field_dims: List[int]):
        class FMLayer(nn.Module):
            def __init__(self, field_dims, embedding_dim):
                super().__init__()
                self.field_dims = field_dims
                self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
                self.embedding = nn.Embedding(sum(field_dims), 1)
                self.feature_embedding = nn.Embedding(sum(field_dims), embedding_dim)
                nn.init.xavier_uniform_(self.embedding.weight)
                nn.init.xavier_uniform_(self.feature_embedding.weight)

            def forward(self, x):
                # First-order term
                first_order = self.embedding(x + self.offsets.reshape(1, -1)).squeeze(-1).sum(dim=1)

                # Second-order term
                embeddings = self.feature_embedding(x + self.offsets.reshape(1, -1))
                square_sum = torch.sum(embeddings, dim=1) ** 2
                sum_square = torch.sum(embeddings**2, dim=1)
                second_order = 0.5 * (square_sum - sum_square).sum(1)

                return first_order, second_order, embeddings

        class DeepFMModel(nn.Module):
            def __init__(self, field_dims, embedding_dim, hidden_layers, dropout):
                super().__init__()
                self.fm_layer = FMLayer(field_dims, embedding_dim)

                # Deep component
                input_dim = len(field_dims) * embedding_dim
                self.deep_layers = nn.ModuleList()
                for hidden_dim in hidden_layers:
                    self.deep_layers.append(
                        nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            nn.BatchNorm1d(hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                        )
                    )
                    input_dim = hidden_dim

                self.output_layer = nn.Linear(input_dim + 1, 1)  # +1 for FM component

            def forward(self, x):
                # FM component
                first_order, second_order, embeddings = self.fm_layer(x)

                # Deep component
                deep_input = embeddings.view(embeddings.size(0), -1)
                for layer in self.deep_layers:
                    deep_input = layer(deep_input)

                # Combine FM and Deep components
                fm_output = first_order + second_order
                output = self.output_layer(torch.cat([deep_input, fm_output.unsqueeze(1)], dim=1))

                return torch.sigmoid(output).squeeze(1)

        return DeepFMModel(field_dims, self.embedding_dim, self.hidden_layers, self.dropout).to(
            self.device
        )

    def fit(
        self,
        user_ids: List[int],
        item_ids: List[int],
        ratings: List[float],
        user_features: Optional[Dict[int, Dict[str, Any]]] = None,
        item_features: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> None:
        # Create feature mapping
        # First field: user IDs
        unique_users = sorted(set(user_ids))
        self.feature_map["user"] = {user: idx for idx, user in enumerate(unique_users)}
        self.field_dims.append(len(unique_users))

        # Second field: item IDs
        unique_items = sorted(set(item_ids))
        self.feature_map["item"] = {item: idx for idx, item in enumerate(unique_items)}
        self.field_dims.append(len(unique_items))

        # Additional fields for user features
        if user_features:
            user_feature_types = set()
            for features in user_features.values():
                user_feature_types.update(features.keys())

            for feature_type in sorted(user_feature_types):
                feature_values = set()
                for user, features in user_features.items():
                    if feature_type in features:
                        feature_values.add(features[feature_type])

                self.feature_map[f"user_{feature_type}"] = {
                    val: idx for idx, val in enumerate(sorted(feature_values))
                }
                self.field_dims.append(len(feature_values))

        # Additional fields for item features
        if item_features:
            item_feature_types = set()
            for features in item_features.values():
                item_feature_types.update(features.keys())

            for feature_type in sorted(item_feature_types):
                feature_values = set()
                for item, features in item_features.items():
                    if feature_type in features:
                        feature_values.add(features[feature_type])

                self.feature_map[f"item_{feature_type}"] = {
                    val: idx for idx, val in enumerate(sorted(feature_values))
                }
                self.field_dims.append(len(feature_values))

        # Build model
        self.model = self._build_model(self.field_dims)

        # Create training data
        X = []
        y = []

        for user, item, rating in zip(user_ids, item_ids, ratings):
            # Create feature vector
            x = [self.feature_map["user"][user], self.feature_map["item"][item]]

            # Add user features
            if user_features and user in user_features:
                for feature_type in sorted(user_feature_types):
                    if feature_type in user_features[user]:
                        value = user_features[user][feature_type]
                        x.append(self.feature_map[f"user_{feature_type}"][value])
                    else:
                        x.append(0)  # Default value for missing features

            # Add item features
            if item_features and item in item_features:
                for feature_type in sorted(item_feature_types):
                    if feature_type in item_features[item]:
                        value = item_features[item][feature_type]
                        x.append(self.feature_map[f"item_{feature_type}"][value])
                    else:
                        x.append(0)  # Default value for missing features

            X.append(x)
            y.append(1.0 if rating > 0 else 0.0)  # Convert to binary for implicit feedback

        # Convert to tensors
        X = torch.LongTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)

        # Define optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        # Train the model
        self.model.train()
        n_batches = len(X) // self.batch_size + (1 if len(X) % self.batch_size != 0 else 0)

        for epoch in range(self.epochs):
            total_loss = 0

            # Shuffle data
            indices = torch.randperm(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(X))

                batch_X = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]

                # Forward pass
                outputs = self.model(batch_X)

                # Compute loss
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/n_batches:.4f}")

    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        if user_id not in self.feature_map["user"]:
            return []

        # Get all items
        all_items = list(self.feature_map["item"].keys())

        # Get items the user has already interacted with
        seen_items = set()
        if exclude_seen:
            # This would need to be implemented based on your data structure
            # For demonstration purposes, we'll assume we have this information
            pass

        # Generate predictions for all items
        user_idx = self.feature_map["user"][user_id]
        predictions = []

        # Process in batches for efficiency
        batch_size = 1024
        for i in range(0, len(all_items), batch_size):
            batch_items = all_items[i : i + batch_size]
            batch_X = []

            for item in batch_items:
                if item in seen_items:
                    continue

                # Create feature vector
                x = [user_idx, self.feature_map["item"][item]]

                # Add user features (if available)
                # This would need to be adapted based on your data structure

                # Add item features (if available)
                # This would need to be adapted based on your data structure

                batch_X.append(x)

            if not batch_X:
                continue

            # Convert to tensor
            batch_X = torch.LongTensor(batch_X).to(self.device)

            # Get predictions
            self.model.eval()
            with torch.no_grad():
                batch_preds = self.model(batch_X).cpu().numpy()

            # Add to predictions
            for j, item in enumerate(batch_items):
                if item not in seen_items and j < len(batch_preds):
                    predictions.append((item, batch_preds[j]))

        # Sort predictions and get top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_items = [item for item, _ in predictions[:top_n]]

        return top_items
