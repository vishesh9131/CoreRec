import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from corerec.api.base_recommender import BaseRecommender
from corerec.api.exceptions import ModelNotFittedError, InvalidParameterError
import pickle
from pathlib import Path
from typing import Union
import logging

logger = logging.getLogger(__name__)


class NASRec(BaseRecommender):
    """
    Neural Architecture Search for Recommendation (NASRec)

    Uses neural architecture search to automatically discover optimal architectures
    for recommendation tasks. It searches over different operation combinations to
    find the best architecture for a given dataset.

    Reference:
    Zoph & Le. "Neural Architecture Search with Reinforcement Learning" (ICLR 2017)
    """

    def __init__(
        self,
        name: str = "NASRec",
        embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 64],
        num_cells: int = 3,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        trainable: bool = True,
        verbose: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.num_cells = num_cells
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        self.user_map = {}
        self.item_map = {}
        self.model = None

    def _build_model(self, num_users: int, num_items: int):
        class NASRecCell(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim

                # Define operations discovered by NAS
                self.op1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())

                self.op2 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())

                self.op3 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

                # Attention weights for combining operations
                self.attention = nn.Parameter(torch.ones(3) / 3)

            def forward(self, x):
                # Apply operations
                out1 = self.op1(x)
                out2 = self.op2(x)
                out3 = self.op3(x)

                # Normalize attention weights
                weights = F.softmax(self.attention, dim=0)

                # Combine outputs using attention
                output = weights[0] * out1 + weights[1] * out2 + weights[2] * out3
                return output

        class NASRecModel(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim, hidden_dims, num_cells):
                super().__init__()
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)

                # Stack of NAS cells
                # Input dimension is 2 * embedding_dim because we concatenate user and item embeddings
                self.cells = nn.ModuleList()
                input_dim = embedding_dim * 2
                for hidden_dim in hidden_dims:
                    self.cells.append(NASRecCell(input_dim, hidden_dim))
                    input_dim = hidden_dim

                # Final prediction layer
                self.predictor = nn.Linear(input_dim, 1)

            def forward(self, user_ids, item_ids):
                user_emb = self.user_embedding(user_ids)
                item_emb = self.item_embedding(item_ids)

                # Concatenate user and item embeddings
                x = torch.cat([user_emb, item_emb], dim=1)

                # Pass through NAS cells
                for cell in self.cells:
                    x = cell(x)

                # Predict
                score = self.predictor(x)
                return torch.sigmoid(score).squeeze(1)

        return NASRecModel(
            num_users, num_items, self.embedding_dim, self.hidden_dims, self.num_cells
        ).to(self.device)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "NASRec":
        """Load model from disk."""
        import pickle

        with open(path, "rb") as f:
            model = pickle.load(f)

        if hasattr(model, "verbose") and model.verbose:
            logger.info(f"Model loaded from {path}")

        return model

    def fit(
        self, user_ids: List[int], item_ids: List[int], ratings: List[float], **kwargs
    ) -> "NASRec":
        """Train the NASRec model."""
        from corerec.utils.validation import validate_fit_inputs

        # Validate inputs
        validate_fit_inputs(user_ids, item_ids, ratings)

        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))

        self.user_map = {user: idx for idx, user in enumerate(unique_users)}
        self.item_map = {item: idx for idx, item in enumerate(unique_items)}

        # Build model
        self.model = self._build_model(len(unique_users), len(unique_items))

        # Create training data
        user_indices = [self.user_map[user] for user in user_ids]
        item_indices = [self.item_map[item] for item in item_ids]

        # Convert to tensors
        train_users = torch.LongTensor(user_indices).to(self.device)
        train_items = torch.LongTensor(item_indices).to(self.device)
        train_labels = torch.FloatTensor(ratings).to(self.device)

        # Define optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        # Training loop
        self.model.train()
        n_batches = len(train_users) // self.batch_size + (
            1 if len(train_users) % self.batch_size != 0 else 0
        )

        for epoch in range(self.epochs):
            total_loss = 0

            # Shuffle data
            indices = torch.randperm(len(train_users))
            users_shuffled = train_users[indices]
            items_shuffled = train_items[indices]
            labels_shuffled = train_labels[indices]

            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(train_users))

                batch_users = users_shuffled[start_idx:end_idx]
                batch_items = items_shuffled[start_idx:end_idx]
                batch_labels = labels_shuffled[start_idx:end_idx]

                # Forward pass
                outputs = self.model(batch_users, batch_items)

                # Compute loss
                loss = criterion(outputs, batch_labels)

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

        validate_model_fitted(self.is_fitted, self.name)

        if self.model is None or user_id not in self.user_map or item_id not in self.item_map:
            return 0.0

        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]

        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        item_tensor = torch.LongTensor([item_idx]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            score = self.model(user_tensor, item_tensor)
            return float(score.item())

    def recommend(self, user_id: int, top_k: int = 10, **kwargs) -> List[int]:
        """Generate top-K recommendations for a user."""
        from corerec.utils.validation import validate_model_fitted, validate_user_id, validate_top_k

        # Validate inputs
        validate_model_fitted(self.is_fitted, self.name)
        validate_user_id(user_id, self.user_map if hasattr(self, "user_map") else {})
        validate_top_k(top_k)

        if self.model is None:
            raise ValueError("Model has not been trained yet")

        if user_id not in self.user_map:
            return []

        exclude_seen = kwargs.get("exclude_seen", True)

        # Get all items
        all_items = list(self.item_map.keys())

        # Get items the user has already interacted with (simplified)
        seen_items = set()
        if exclude_seen:
            # This would need to be implemented based on your data structure
            pass

        # Generate predictions for all items
        user_idx = self.user_map[user_id]
        user_tensor = torch.LongTensor([user_idx] * len(all_items)).to(self.device)
        item_indices = [self.item_map[item] for item in all_items]
        item_tensor = torch.LongTensor(item_indices).to(self.device)

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(user_tensor, item_tensor).cpu().numpy()

        # Create list of (item_id, score) tuples
        item_scores = [
            (item, float(pred))
            for item, pred in zip(all_items, predictions)
            if item not in seen_items
        ]

        # Sort by score and get top-K
        item_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = [item for item, _ in item_scores[:top_k]]

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
