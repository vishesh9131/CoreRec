import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from corerec.api.base_recommender import BaseRecommender
from corerec.utils.validation import (
    validate_fit_inputs,
    validate_user_id,
    validate_top_k,
    validate_model_fitted,
)
import logging

logger = logging.getLogger(__name__)


class MIND(BaseRecommender):
    """
    Multi-Interest Network with Dynamic routing for Recommendation (MIND)

    Represents a user with multiple interest vectors to better capture diverse user preferences.
    Uses a dynamic routing mechanism inspired by capsule networks to extract user interests.

    Reference:
    Li et al. "Multi-Interest Network with Dynamic Routing for Recommendation at Tmall" (CIKM 2019)
    """

    def __init__(
        self,
        name: str = "MIND",
        embedding_dim: int = 64,
        num_interests: int = 4,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2,
        routing_iterations: int = 3,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        max_seq_length: int = 50,
        trainable: bool = True,
        verbose: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.embedding_dim = embedding_dim
        self.num_interests = num_interests
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.routing_iterations = routing_iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_seq_length = max_seq_length
        self.device = device

        self.user_history = {}
        self.user_map = {}
        self.item_map = {}
        self.reverse_item_map = {}
        self.model = None

    def _build_model(self, num_items: int):
        class MultiInterestExtractor(nn.Module):
            def __init__(self, embedding_dim, num_interests, routing_iterations):
                super().__init__()
                self.embedding_dim = embedding_dim
                self.num_interests = num_interests
                self.routing_iterations = routing_iterations

                # Transformation matrix for dynamic routing
                self.routing_weights = nn.Parameter(
                    torch.randn(num_interests, embedding_dim, embedding_dim) * 0.01
                )

            def forward(self, item_embeddings, mask=None):
                """
                Extract multiple interest vectors from item embeddings.

                Args:
                    item_embeddings: [batch_size, seq_length, embedding_dim]
                    mask: Optional mask for padding [batch_size, seq_length]

                Returns:
                    interest_vectors: [batch_size, num_interests, embedding_dim]
                """
                batch_size, seq_length, embedding_dim = item_embeddings.size()

                # Apply mask if provided
                if mask is not None:
                    item_embeddings = item_embeddings * mask.unsqueeze(-1)

                # Initialize routing logits
                routing_logits = torch.zeros(batch_size, seq_length, self.num_interests).to(
                    item_embeddings.device
                )

                # Dynamic routing
                for _ in range(self.routing_iterations):
                    # Calculate routing weights
                    routing_weights = F.softmax(routing_logits, dim=-1)

                    # Initialize capsules
                    capsules = torch.zeros(batch_size, self.num_interests, self.embedding_dim).to(
                        item_embeddings.device
                    )

                    # Calculate weighted sum for each interest
                    for i in range(self.num_interests):
                        # Transform item embeddings
                        transformed = torch.matmul(item_embeddings, self.routing_weights[i])
                        # Weighted sum
                        capsules[:, i, :] = torch.sum(
                            routing_weights[:, :, i : i + 1] * transformed, dim=1
                        )

                    # Update routing logits
                    if _ < self.routing_iterations - 1:
                        routing_logits = torch.sum(
                            capsules.unsqueeze(1) * item_embeddings.unsqueeze(2), dim=-1
                        )

                return capsules

        class MINDModel(nn.Module):
            def __init__(self, num_items, embedding_dim, num_interests, hidden_dims, dropout, routing_iterations):
                super().__init__()
                self.item_embedding = nn.Embedding(num_items + 1, embedding_dim)  # +1 for padding
                self.multi_interest = MultiInterestExtractor(embedding_dim, num_interests, routing_iterations)

                # Prediction layers for label-aware attention
                self.label_predictor = nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )

            def forward(self, sequences, target_items=None):
                """
                Forward pass for MIND model.
                
                Args:
                    sequences: [batch_size, seq_length] item sequences
                    target_items: Optional [batch_size] target items for prediction
                
                Returns:
                    If target_items is None: user_interests [batch_size, num_interests, embedding_dim]
                    If target_items is provided: prediction scores [batch_size]
                """
                batch_size, seq_length = sequences.size()
                
                # Get item embeddings
                item_emb = self.item_embedding(sequences)  # [batch_size, seq_length, embedding_dim]
                
                # Create mask for padding (0 is padding)
                mask = (sequences > 0).float()  # [batch_size, seq_length]
                
                # Extract multiple interests
                user_interests = self.multi_interest(item_emb, mask)  # [batch_size, num_interests, embedding_dim]
                
                if target_items is None:
                    # Return interests for recommendation
                    return user_interests
                
                # For prediction: compute similarity between interests and target item
                target_emb = self.item_embedding(target_items)  # [batch_size, embedding_dim]
                target_emb = target_emb.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                
                # Compute similarity with each interest
                similarities = (user_interests * target_emb).sum(dim=-1)  # [batch_size, num_interests]
                
                # Take maximum similarity across interests
                scores = similarities.max(dim=1)[0]  # [batch_size]
                
                return torch.sigmoid(scores)

        return MINDModel(
            num_items, self.embedding_dim, self.num_interests, self.hidden_dims, self.dropout, self.routing_iterations
        ).to(self.device)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs):
        """Load model from disk."""
        import pickle

        with open(path, "rb") as f:
            model = pickle.load(f)

        if hasattr(model, "verbose") and model.verbose:
            logger.info(f"Model loaded from {path}")

        return model

    def fit(
        self,
        user_ids: List[int],
        item_ids: List[int],
        ratings: List[float],
        timestamps: Optional[List[int]] = None,
        **kwargs,
    ) -> "MIND":
        """Train the MIND model."""
        # Validate inputs
        validate_fit_inputs(user_ids, item_ids, ratings)

        if timestamps is None:
            timestamps = list(range(len(user_ids)))

        unique_items = sorted(set(item_ids))
        unique_users = sorted(set(user_ids))
        
        # Create user map
        self.user_map = {user: idx for idx, user in enumerate(unique_users)}
        
        # Create item map
        self.item_map = {item: idx + 1 for idx, item in enumerate(unique_items)}  # +1 for padding
        self.reverse_item_map = {idx: item for item, idx in self.item_map.items()}
        self.reverse_item_map[0] = 0  # Padding item

        # Create user histories
        user_item_timestamps = sorted(
            zip(user_ids, item_ids, timestamps), key=lambda x: (x[0], x[2])
        )
        current_user = None
        current_history = []

        for user, item, _ in user_item_timestamps:
            if user != current_user:
                if current_user is not None:
                    self.user_history[current_user] = current_history
                current_user = user
                current_history = []

            current_history.append(self.item_map[item])

        # Add the last user's history
        if current_user is not None:
            self.user_history[current_user] = current_history

        # Build model
        self.model = self._build_model(len(unique_items))

        # Create training sequences and targets
        train_sequences = []
        train_targets = []

        for user, history in self.user_history.items():
            for i in range(1, len(history)):
                # Use items up to position i-1 to predict item at position i
                seq = history[:i]

                # Truncate or pad sequence
                if len(seq) > self.max_seq_length:
                    seq = seq[-self.max_seq_length :]
                else:
                    seq = [0] * (self.max_seq_length - len(seq)) + seq

                train_sequences.append(seq)
                train_targets.append(history[i])

        # Convert to tensors
        train_sequences = torch.LongTensor(train_sequences).to(self.device)
        train_targets = torch.LongTensor(train_targets).to(self.device)

        # Define optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        # Training loop
        self.model.train()

        # Check if we have training data
        if len(train_sequences) == 0:
            if self.verbose:
                logger.warning("Warning: No training sequences found. Skipping training.")
            self.is_fitted = True
            return self

        n_batches = len(train_sequences) // self.batch_size + (
            1 if len(train_sequences) % self.batch_size != 0 else 0
        )

        # Ensure at least 1 batch
        if n_batches == 0:
            n_batches = 1

        for epoch in range(self.epochs):
            total_loss = 0

            # Shuffle data
            indices = torch.randperm(len(train_sequences))
            shuffled_sequences = train_sequences[indices]
            shuffled_targets = train_targets[indices]

            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(train_sequences))

                # Skip if batch is empty
                if start_idx >= end_idx:
                    continue

                batch_sequences = shuffled_sequences[start_idx:end_idx]
                batch_targets = shuffled_targets[start_idx:end_idx]

                # Create positive and negative examples
                pos_targets = batch_targets
                neg_targets = torch.randint(1, len(self.item_map) + 1, size=pos_targets.size()).to(
                    self.device
                )

                # Forward pass for positive examples
                pos_scores = self.model(batch_sequences, pos_targets)
                pos_labels = torch.ones_like(pos_scores)

                # Forward pass for negative examples
                neg_scores = self.model(batch_sequences, neg_targets)
                neg_labels = torch.zeros_like(neg_scores)

                # Combine positive and negative examples
                all_scores = torch.cat([pos_scores, neg_scores])
                all_labels = torch.cat([pos_labels, neg_labels])

                # Compute loss
                loss = criterion(all_scores, all_labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Safe division to avoid ZeroDivisionError
            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            if self.verbose:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    def predict(self, user_id: int, item_id: int, **kwargs) -> float:
        """Predict rating for a user-item pair."""
        from corerec.utils.validation import validate_model_fitted

        validate_model_fitted(self.is_fitted, self.name)

        if self.model is None or user_id not in self.user_history:
            return 0.0

        # Get user history
        history = self.user_history[user_id]
        if len(history) == 0:
            return 0.0

        # Truncate or pad sequence
        if len(history) > self.max_seq_length:
            history = history[-self.max_seq_length :]
        else:
            history = [0] * (self.max_seq_length - len(history)) + history

        # Convert to tensor
        history_tensor = torch.LongTensor([history]).to(self.device)
        item_idx = self.item_map.get(item_id, 0)
        item_tensor = torch.LongTensor([item_idx]).to(self.device)

        # Get prediction
        self.model.eval()
        with torch.no_grad():
            score = self.model(history_tensor, item_tensor)
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

        if user_id not in self.user_history:
            return []

        exclude_seen = kwargs.get("exclude_seen", True)

        # Get user history
        history = self.user_history[user_id]

        # Truncate or pad sequence
        if len(history) > self.max_seq_length:
            history = history[-self.max_seq_length :]
        else:
            history = [0] * (self.max_seq_length - len(history)) + history

        # Convert to tensor
        history_tensor = torch.LongTensor([history]).to(self.device)

        # Get user interests
        self.model.eval()
        with torch.no_grad():
            user_interests = self.model(history_tensor, target_items=None)  # [1, num_interests, embedding_dim]

        # Get all item embeddings (skip padding at index 0)
        all_item_indices = torch.arange(1, len(self.item_map) + 1, dtype=torch.long, device=self.device)
        item_embeddings = self.model.item_embedding(all_item_indices)  # [num_items, embedding_dim]

        # Compute scores for all items
        user_interests_flat = user_interests.squeeze(0)  # [num_interests, embedding_dim]
        
        # Compute similarity between each interest and all items
        # user_interests_flat: [num_interests, embedding_dim]
        # item_embeddings: [num_items, embedding_dim]
        similarities = torch.matmul(user_interests_flat, item_embeddings.T)  # [num_interests, num_items]
        
        # Take maximum similarity across interests for each item
        scores = similarities.max(dim=0)[0].cpu()  # [num_items]

        # Exclude seen items if requested
        if exclude_seen:
            seen_item_indices = [idx - 1 for idx in history if idx > 0]  # -1 to account for padding
            for item_idx in seen_item_indices:
                if 0 <= item_idx < len(scores):
                    scores[item_idx] = float("-inf")

        # Get top-K item indices
        top_indices = torch.argsort(scores, descending=True)[:top_k].numpy()

        # Convert indices back to original item IDs (add 1 to account for padding, then map)
        recommended_items = []
        for idx in top_indices:
            item_idx = idx + 1  # +1 to account for padding
            if item_idx in self.reverse_item_map:
                recommended_items.append(self.reverse_item_map[item_idx])

        return recommended_items

    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save model to disk."""
        import pickle

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.verbose:
            logger.info(f"{self.name} model saved to {path}")
