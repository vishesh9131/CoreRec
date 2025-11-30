"""
Attention Mechanisms for Hybrid Recommendation Systems

This module provides implementations of attention mechanisms tailored for hybrid
recommendation systems. Attention mechanisms allow the model to focus on relevant
parts of the input data, improving the quality of recommendations by considering
contextual and sequential information.

Key Features:
- Implements self-attention and multi-head attention mechanisms.
- Enhances hybrid models by integrating attention layers.
- Supports attention-based feature extraction and representation learning.

Classes:
- ATTENTION_MECHANISMS: Core class for implementing attention mechanisms in hybrid
  recommendation systems.

Usage:
Instantiate the ATTENTION_MECHANISMS class to integrate attention layers into your
hybrid recommendation model. Use the provided methods to train and apply attention
mechanisms to your data.

Example:
    attention_model = ATTENTION_MECHANISMS()
    attention_model.train(user_item_matrix, attention_features)
    enhanced_recommendations = attention_model.recommend(user_id, top_n=10)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings


class ATTENTION_MECHANISMS:
    """
    Attention-based hybrid recommendation system.

    This class implements various attention mechanisms including self-attention
    and multi-head attention for combining multiple recommendation sources.
    """

    def __init__(
        self,
        num_heads: int = 4,
        embedding_dim: int = 64,
        dropout_rate: float = 0.1,
        attention_type: str = "scaled_dot_product",
        normalize: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize attention mechanism for hybrid recommendations.

        Parameters:
        -----------
        num_heads : int
            Number of attention heads for multi-head attention
        embedding_dim : int
            Dimension of embeddings
        dropout_rate : float
            Dropout rate for regularization
        attention_type : str
            Type of attention ('scaled_dot_product', 'additive', 'multiplicative')
        normalize : bool
            Whether to normalize attention weights
        random_state : int
            Random seed for reproducibility
        """
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.attention_type = attention_type
        self.normalize = normalize
        self.random_state = random_state

        np.random.seed(random_state)

        # Initialize parameters
        self.user_embeddings = None
        self.item_embeddings = None
        self.attention_weights = None
        self.feature_weights = None
        self.is_trained = False

        # multi-head attention params
        self.head_dim = embedding_dim // num_heads
        self.query_weights = []
        self.key_weights = []
        self.value_weights = []

    def _initialize_embeddings(self, n_users: int, n_items: int):
        """Initialize user and item embeddings"""
        self.user_embeddings = np.random.randn(n_users, self.embedding_dim) * 0.01
        self.item_embeddings = np.random.randn(n_items, self.embedding_dim) * 0.01

        # initialize multi-head attention weights
        for _ in range(self.num_heads):
            self.query_weights.append(np.random.randn(self.embedding_dim, self.head_dim) * 0.01)
            self.key_weights.append(np.random.randn(self.embedding_dim, self.head_dim) * 0.01)
            self.value_weights.append(np.random.randn(self.embedding_dim, self.head_dim) * 0.01)

    def _scaled_dot_product_attention(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray
    ) -> np.ndarray:
        """Compute scaled dot-product attention"""
        d_k = query.shape[-1]
        scores = np.dot(query, key.T) / np.sqrt(d_k)

        # apply softmax
        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)

        # apply dropout during training
        if self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, attention_weights.shape)
            attention_weights = attention_weights * mask / (1 - self.dropout_rate)

        output = np.dot(attention_weights, value)
        return output, attention_weights

    def _multi_head_attention(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray
    ) -> np.ndarray:
        """Apply multi-head attention mechanism"""
        outputs = []
        all_attention_weights = []

        for i in range(self.num_heads):
            # project query, key, value for this head
            q = np.dot(query, self.query_weights[i])
            k = np.dot(key, self.key_weights[i])
            v = np.dot(value, self.value_weights[i])

            # compute attention for this head
            head_output, head_weights = self._scaled_dot_product_attention(q, k, v)
            outputs.append(head_output)
            all_attention_weights.append(head_weights)

        # concatenate all heads
        multi_head_output = np.concatenate(outputs, axis=-1)
        avg_attention_weights = np.mean(all_attention_weights, axis=0)

        return multi_head_output, avg_attention_weights

    def train(
        self,
        user_item_matrix: np.ndarray,
        attention_features: Optional[Dict[str, np.ndarray]] = None,
        epochs: int = 10,
        learning_rate: float = 0.01,
        verbose: bool = True,
    ):
        """
        Train the attention-based hybrid model.

        Parameters:
        -----------
        user_item_matrix : np.ndarray
            User-item interaction matrix
        attention_features : dict, optional
            Additional features for attention computation
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate for optimization
        verbose : bool
            Whether to print training progress
        """
        n_users, n_items = user_item_matrix.shape
        self._initialize_embeddings(n_users, n_items)

        # extract positive interactions
        user_indices, item_indices = np.where(user_item_matrix > 0)
        n_samples = len(user_indices)

        if verbose:
            print(f"Training attention mechanism with {n_samples} interactions...")

        for epoch in range(epochs):
            total_loss = 0.0

            # shuffle training data
            shuffle_idx = np.random.permutation(n_samples)

            for idx in shuffle_idx:
                user_id = user_indices[idx]
                item_id = item_indices[idx]
                rating = user_item_matrix[user_id, item_id]

                # get embeddings
                user_emb = self.user_embeddings[user_id]
                item_emb = self.item_embeddings[item_id]

                # apply attention mechanism
                attended_output, attention_wts = self._multi_head_attention(
                    user_emb.reshape(1, -1), item_emb.reshape(1, -1), item_emb.reshape(1, -1)
                )

                # compute prediction
                prediction = np.dot(user_emb, attended_output.flatten())

                # compute loss (MSE)
                error = rating - prediction
                loss = error**2
                total_loss += loss

                # gradient update (simplified)
                self.user_embeddings[user_id] += learning_rate * error * attended_output.flatten()
                self.item_embeddings[item_id] += learning_rate * error * user_emb

            if verbose and (epoch + 1) % 2 == 0:
                avg_loss = total_loss / n_samples
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_trained = True
        if verbose:
            print("Training completed!")

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_known: bool = True,
        known_items: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate recommendations using attention mechanism.

        Parameters:
        -----------
        user_id : int
            User ID for recommendations
        top_n : int
            Number of top recommendations to return
        exclude_known : bool
            Whether to exclude known items
        known_items : np.ndarray, optional
            Array of known item indices to exclude

        Returns:
        --------
        item_ids : np.ndarray
            Array of recommended item IDs
        scores : np.ndarray
            Array of recommendation scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating recommendations")

        user_emb = self.user_embeddings[user_id]
        n_items = self.item_embeddings.shape[0]

        # compute attention-weighted scores for all items
        scores = np.zeros(n_items)
        for item_id in range(n_items):
            item_emb = self.item_embeddings[item_id]

            # apply attention
            attended_output, _ = self._multi_head_attention(
                user_emb.reshape(1, -1), item_emb.reshape(1, -1), item_emb.reshape(1, -1)
            )

            scores[item_id] = np.dot(user_emb, attended_output.flatten())

        # exclude known items
        if exclude_known and known_items is not None:
            scores[known_items] = -np.inf

        # get top-N recommendations
        top_indices = np.argsort(scores)[::-1][:top_n]
        top_scores = scores[top_indices]

        return top_indices, top_scores

    def get_attention_weights(self, user_id: int, item_id: int) -> np.ndarray:
        """
        Get attention weights for a specific user-item pair.

        Parameters:
        -----------
        user_id : int
            User ID
        item_id : int
            Item ID

        Returns:
        --------
        attention_weights : np.ndarray
            Attention weight distribution
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        user_emb = self.user_embeddings[user_id]
        item_emb = self.item_embeddings[item_id]

        _, attention_weights = self._multi_head_attention(
            user_emb.reshape(1, -1), item_emb.reshape(1, -1), item_emb.reshape(1, -1)
        )

        return attention_weights

    def save_model(self, filepath: str):
        """Save model parameters to file"""
        np.savez(
            filepath,
            user_embeddings=self.user_embeddings,
            item_embeddings=self.item_embeddings,
            num_heads=self.num_heads,
            embedding_dim=self.embedding_dim,
        )

    def load_model(self, filepath: str):
        """Load model parameters from file"""
        data = np.load(filepath)
        self.user_embeddings = data["user_embeddings"]
        self.item_embeddings = data["item_embeddings"]
        self.is_trained = True
