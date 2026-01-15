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
import random
from tqdm import tqdm
from corerec.base_recommender import BaseCorerec
from datetime import datetime
from corerec.utils import hook_manager


class HookManager:
    """Manager for model hooks to inspect internal states."""

    def __init__(self):
        """Initialize the hook manager."""
        self.hooks = {}
        self.activations = {}

    def _get_activation(self, name):
        """Get activation for a specific layer."""

        def hook(model, input, output):
            self.activations[name] = output.detach()

        return hook

    def register_hook(self, model, layer_name, callback=None):
        """Register a hook for a specific layer."""
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
        """Remove a hook for a specific layer."""
        if layer_name in self.hooks:
            self.hooks[layer_name].remove()
            del self.hooks[layer_name]
            return True
        return False

    def get_activation(self, layer_name):
        """Get the activation for a specific layer."""
        return self.activations.get(layer_name, None)

    def clear_activations(self):
        """Clear all stored activations."""
        self.activations.clear()


class TokenEmbedding(nn.Module):
    """Embedding layer for tokens (items)."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize the token embedding layer.

        Args:
            vocab_size: Number of unique tokens (items).
            embedding_dim: Dimension of embedding vectors.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dim = embedding_dim

        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim**-0.5)
        nn.init.constant_(self.embedding.weight[0], 0)  # padding idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the token embedding layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len).
                Each value is the token index.

        Returns:
            Embedded tokens of shape (batch_size, seq_len, embedding_dim).
        """
        return self.embedding(x) * (self.embedding_dim**0.5)


class PositionalEmbedding(nn.Module):
    """Embedding layer for positions in a sequence."""

    def __init__(self, max_len: int, embedding_dim: int):
        """
        Initialize the positional embedding layer.

        Args:
            max_len: Maximum sequence length.
            embedding_dim: Dimension of embedding vectors.
        """
        super().__init__()
        self.embedding = nn.Embedding(max_len, embedding_dim)

        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim**-0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional embedding layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len).
                Values are not used, only the shape.

        Returns:
            Positional embeddings of shape (batch_size, seq_len, embedding_dim).
        """
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.embedding(positions)


class GELU(nn.Module):
    """Gaussian Error Linear Unit."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GELU activation function.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    """Layer normalization module."""

    def __init__(self, features: int, eps: float = 1e-6):
        """
        Initialize the layer normalization module.

        Args:
            features: Number of features.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer normalization module.

        Args:
            x: Input tensor of shape (batch_size, seq_len, features).

        Returns:
            Normalized tensor of the same shape.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize the multi-head attention module.

        Args:
            hidden_dim: Dimension of hidden layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the multi-head attention module.

        Args:
            query: Query tensor of shape (batch_size, seq_len, hidden_dim).
            key: Key tensor of shape (batch_size, seq_len, hidden_dim).
            value: Value tensor of shape (batch_size, seq_len, hidden_dim).
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tuple of (output tensor, attention weights).
            Output tensor has shape (batch_size, seq_len, hidden_dim).
            Attention weights has shape (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size = query.shape[0]

        # Linear projections and split into heads
        q = (
            self.q_linear(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_linear(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_linear(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Scale dot-product attention
        self.scale = self.scale.to(query.device)
        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale

        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # Attention weights
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.out_linear(out)

        return out, attention


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, hidden_dim: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize the position-wise feed-forward network.

        Args:
            hidden_dim: Dimension of hidden layers.
            ff_dim: Dimension of feed-forward layer.
            dropout: Dropout probability.
        """
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the position-wise feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim).

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward layers."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize the transformer block.

        Args:
            hidden_dim: Dimension of hidden layers.
            num_heads: Number of attention heads.
            ff_dim: Dimension of feed-forward layer.
            dropout: Dropout probability.
        """
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.feed_forward = PositionwiseFeedForward(hidden_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim).
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tuple of (output tensor, attention weights).
            Output tensor has shape (batch_size, seq_len, hidden_dim).
            Attention weights has shape (batch_size, num_heads, seq_len, seq_len).
        """
        # Self-attention with residual connection and layer normalization
        attn_output, attention = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attention


class BERT4RecModel(nn.Module):
    """BERT4Rec model for sequential recommendation."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        """
        Initialize the BERT4Rec model.

        Args:
            vocab_size: Number of unique items.
            hidden_dim: Dimension of hidden layers.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            ff_dim: Dimension of feed-forward layer.
            max_seq_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.position_embedding = PositionalEmbedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        self.output_layer = nn.Linear(hidden_dim, vocab_size)

        # Initialize output layer
        nn.init.normal_(self.output_layer.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the BERT4Rec model.

        Args:
            x: Input tensor of shape (batch_size, seq_len).
                Each value is the item index.
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tuple of (output tensor, attention weights).
            Output tensor has shape (batch_size, seq_len, vocab_size).
            Attention weights is a list of tensors, each with shape (batch_size, num_heads, seq_len, seq_len).
        """
        # Embedding layers
        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(x)
        x = self.dropout(token_embed + pos_embed)

        # Create attention mask if not provided
        if mask is None:
            batch_size, seq_len = x.size(0), x.size(1)
            mask = torch.ones((batch_size, seq_len, seq_len), device=x.device)
            # Create mask for padding tokens (0)
            padding_mask = (x != 0).unsqueeze(-1).float()
            mask = mask * padding_mask * padding_mask.transpose(1, 2)

        # Transformer blocks
        attention_weights = []
        for transformer in self.transformer_blocks:
            x, attention = transformer(x, mask)
            attention_weights.append(attention)

        # Output layer
        output = self.output_layer(x)

        return output, attention_weights

    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention weights for visualization or analysis.

        Args:
            x: Input tensor of shape (batch_size, seq_len).
                Each value is the item index.

        Returns:
            List of attention weights, each with shape (batch_size, num_heads, seq_len, seq_len).
        """
        _, attention_weights = self.forward(x)
        return attention_weights


class BERT4RecTrainer:
    """Trainer for BERT4Rec model."""

    def __init__(self, model: BERT4RecModel, config: Dict[str, Any], device: torch.device):
        """
        Initialize the BERT4Rec trainer.

        Args:
            model: BERT4Rec model.
            config: Configuration dictionary.
            device: Device to use for training.
        """
        self.model = model
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 0.0),
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
        )

        # Set mask token id (usually the last token in vocab)
        self.mask_token = config.get(
            "mask_token", model.token_embedding.embedding.weight.size(0) - 1
        )

        # Mask probability
        self.mask_prob = config.get("mask_prob", 0.15)

        # Criterion (ignore padding token 0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def _create_masked_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create masked input for MLM training.

        Args:
            x: Input tensor of shape (batch_size, seq_len).
                Each value is the item index.

        Returns:
            Tuple of (masked input, targets).
            Masked input has shape (batch_size, seq_len).
            Targets has shape (batch_size, seq_len).
        """
        # Clone input for targets
        targets = x.clone()

        # Create mask
        mask = torch.rand(x.shape, device=x.device) < self.mask_prob
        # Don't mask padding tokens
        mask = mask & (x != 0)

        # Apply mask
        x_masked = x.clone()
        x_masked[mask] = self.mask_token

        return x_masked, targets

    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            x: Input tensor of shape (batch_size, seq_len).
                Each value is the item index.

        Returns:
            Dictionary of loss values.
        """
        self.model.train()

        # Create masked input
        x_masked, targets = self._create_masked_input(x)

        # Forward pass
        logits, _ = self.model(x_masked)

        # Reshape for loss calculation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        # Calculate loss (only on masked tokens)
        loss = self.criterion(logits_flat, targets_flat)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate_step(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single validation step.

        Args:
            x: Input tensor of shape (batch_size, seq_len).
                Each value is the item index.

        Returns:
            Dictionary of loss values.
        """
        self.model.eval()

        with torch.no_grad():
            # Create masked input
            x_masked, targets = self._create_masked_input(x)

            # Forward pass
            logits, _ = self.model(x_masked)

            # Reshape for loss calculation
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)

            # Calculate loss
            loss = self.criterion(logits_flat, targets_flat)

        return {"loss": loss.item()}


class SequentialDataset:
    """Dataset for sequential recommendation."""

    def __init__(
        self, interactions: Dict[int, List[int]], max_seq_len: int, mask_prob: float = 0.15
    ):
        """
        Initialize the sequential dataset.

        Args:
            interactions: Dictionary mapping user IDs to list of item IDs.
            max_seq_len: Maximum sequence length.
            mask_prob: Probability of masking an item.
        """
        self.interactions = interactions
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.user_ids = list(interactions.keys())

    def __len__(self) -> int:
        """Get the number of users."""
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> List[int]:
        """
        Get a sequence for a user.

        Args:
            idx: User index.

        Returns:
            List of item IDs.
        """
        user_id = self.user_ids[idx]
        seq = self.interactions[user_id][-self.max_seq_len :]

        # Pad sequence
        pad_len = self.max_seq_len - len(seq)
        if pad_len > 0:
            seq = [0] * pad_len + seq

        return seq


class Bert4Rec_base(BaseCorerec):
    """
    BERT4Rec base implementation for sequential recommendation.

    This model uses a BERT-style transformer architecture for sequential
    recommendation tasks. It trains using masked language modeling and
    predicts the next item in a user's sequence.
    """

    def __init__(
        self,
        name: str = "BERT4Rec",
        trainable: bool = True,
        verbose: bool = True,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        """
        Initialize the BERT4Rec base model.

        Args:
            name: Model name.
            trainable: Whether the model is trainable.
            verbose: Whether to print verbose output.
            config: Configuration dictionary.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.name = name
        self.trainable = trainable
        self.verbose = verbose
        self.seed = seed

        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # Default configuration
        default_config = {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 256,
            "max_seq_len": 50,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "batch_size": 256,
            "num_epochs": 100,
            "patience": 10,
            "min_delta": 0.001,
            "mask_prob": 0.15,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        # Update with user config
        self.config = default_config.copy()
        if config is not None:
            self.config.update(config)

        # Set device
        self.device = torch.device(self.config["device"])

        # Initialize model (will be built after fit)
        self.model = None
        self.trainer = None
        self.hooks = HookManager()

        # Attributes to be set in fit
        self.is_fitted = False
        self.user_ids = []
        self.item_ids = []
        self.vocab_size = 0
        self.uid_map = {}
        self.iid_map = {}
        self.num_users = 0
        self.num_items = 0
        self.user_sequences = {}

        if self.verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(self.name)
        else:
            logging.basicConfig(level=logging.WARNING)
            self.logger = logging.getLogger(self.name)

    def _build_model(self):
        """Build the BERT4Rec model."""
        # Add 2 to vocab size for padding (0) and mask tokens
        self.vocab_size = self.num_items + 2
        self.mask_token = self.vocab_size - 1

        # Create model
        self.model = BERT4RecModel(
            vocab_size=self.vocab_size,
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"],
            ff_dim=self.config["ff_dim"],
            max_seq_len=self.config["max_seq_len"],
            dropout=self.config["dropout"],
        ).to(self.device)

        # Create trainer
        self.trainer = BERT4RecTrainer(model=self.model, config=self.config, device=self.device)

        if self.verbose:
            self.logger.info(
                f"Built BERT4Rec model with {sum(p.numel() for p in self.model.parameters())} parameters"
            )

    def _convert_to_sequences(self, interaction_matrix, user_ids, item_ids):
        """
        Convert interaction matrix to user sequences.

        Args:
            interaction_matrix: Interaction matrix.
            user_ids: List of user IDs.
            item_ids: List of item IDs.

        Returns:
            Dictionary mapping user indices to list of item indices.
        """
        # Create mappings
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.uid_map = {uid: i for i, uid in enumerate(user_ids)}
        self.iid_map = {iid: i + 1 for i, iid in enumerate(item_ids)}  # Start from 1, 0 is padding
        self.num_users = len(user_ids)
        self.num_items = len(item_ids)

        # Convert to sequences
        sequences = defaultdict(list)
        interaction_matrix = interaction_matrix.tocoo()

        # Sort interactions by user and timestamp (if available)
        data = list(zip(interaction_matrix.row, interaction_matrix.col, interaction_matrix.data))
        data.sort()  # Sort by row, then col

        for user_idx, item_idx, _ in data:
            item_mapped = self.iid_map[item_ids[item_idx]]
            sequences[user_idx].append(item_mapped)

        return sequences

    def fit(self, interaction_matrix, user_ids, item_ids):
        """
        Fit the BERT4Rec model.

        Args:
            interaction_matrix: Interaction matrix.
            user_ids: List of user IDs.
            item_ids: List of item IDs.

        Returns:
            Dictionary of training history.
        """
        if not self.trainable:
            raise RuntimeError("Model is not trainable.")

        # Convert interaction matrix to sequences
        self.user_sequences = self._convert_to_sequences(interaction_matrix, user_ids, item_ids)

        # Build model if not already built
        if self.model is None:
            self._build_model()

        # Create dataset
        dataset = SequentialDataset(
            interactions=self.user_sequences,
            max_seq_len=self.config["max_seq_len"],
            mask_prob=self.config["mask_prob"],
        )

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=0
        )

        # Training loop
        best_loss = float("inf")
        patience_counter = 0
        history = defaultdict(list)

        for epoch in range(self.config["num_epochs"]):
            # Training
            train_loss = 0
            num_batches = 0

            for batch in tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{self.config['num_epochs']}",
                disable=not self.verbose,
            ):
                batch = torch.tensor(batch, dtype=torch.long).to(self.device)
                batch_result = self.trainer.train_step(batch)
                train_loss += batch_result["loss"]
                num_batches += 1

            train_loss /= num_batches
            history["loss"].append(train_loss)

            # Validation (using training data)
            val_loss = 0
            num_batches = 0

            for batch in dataloader:
                batch = torch.tensor(batch, dtype=torch.long).to(self.device)
                batch_result = self.trainer.validate_step(batch)
                val_loss += batch_result["loss"]
                num_batches += 1

            val_loss /= num_batches
            history["val_loss"].append(val_loss)

            # Print progress
            if self.verbose:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config['num_epochs']}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            # Early stopping
            if val_loss < best_loss - self.config["min_delta"]:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config["patience"]:
                    if self.verbose:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        self.is_fitted = True
        return history

    def predict(self, user_id, item_id):
        """
        Predict the score for a user-item pair.

        Args:
            user_id: User ID.
            item_id: Item ID.

        Returns:
            Predicted score.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if user_id not in self.uid_map:
            raise ValueError(f"User {user_id} not found in training data.")

        if item_id not in self.iid_map:
            raise ValueError(f"Item {item_id} not found in training data.")

        user_idx = self.uid_map[user_id]
        item_idx = self.iid_map[item_id]

        # Get user sequence
        sequence = self.user_sequences[user_idx][-self.config["max_seq_len"] :]

        # Pad sequence
        pad_len = self.config["max_seq_len"] - len(sequence)
        if pad_len > 0:
            sequence = [0] * pad_len + sequence

        # Convert to tensor
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(sequence_tensor)
            # Get score for target item at last position
            score = logits[0, -1, item_idx].item()

        return score

    def recommend(self, user_id, k=10, exclude_seen=True, top_n=None):
        """
        Recommend items for a user.

        Args:
            user_id: User ID.
            k: Number of candidate items to consider per user.
            exclude_seen: Whether to exclude seen items.
            top_n: Number of items to recommend.

        Returns:
            List of (item ID, score) tuples.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if user_id not in self.uid_map:
            raise ValueError(f"User {user_id} not found in training data.")

        if top_n is None:
            top_n = k

        user_idx = self.uid_map[user_id]

        # Get user sequence
        sequence = self.user_sequences[user_idx][-self.config["max_seq_len"] :]

        # Pad sequence
        pad_len = self.config["max_seq_len"] - len(sequence)
        if pad_len > 0:
            sequence = [0] * pad_len + sequence

        # Convert to tensor
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(sequence_tensor)
            # Get scores for all items at last position
            scores = (
                logits[0, -1, 1 : self.vocab_size - 1].cpu().numpy()
            )  # Exclude padding and mask tokens

        # Get seen items
        seen_items = set(self.user_sequences[user_idx]) if exclude_seen else set()

        # Get recommendations
        item_scores = []
        for i, score in enumerate(scores):
            item_idx = i + 1  # Adjust for 1-indexing in iid_map
            if exclude_seen and item_idx in seen_items:
                continue
            item_id = self.item_ids[i]
            item_scores.append((item_id, float(score)))

        # Sort by score
        item_scores.sort(key=lambda x: x[1], reverse=True)

        return item_scores[:top_n]

    def register_hook(self, layer_name, callback=None):
        """
        Register a hook for a specific layer.

        Args:
            layer_name: Name of the layer.
            callback: Optional callback function.

        Returns:
            True if hook was registered, False otherwise.
        """
        if self.model is None:
            raise RuntimeError("Model is not built yet. Call fit() first.")

        return self.hooks.register_hook(self.model, layer_name, callback)

    def remove_hook(self, layer_name):
        """
        Remove a hook for a specific layer.

        Args:
            layer_name: Name of the layer.

        Returns:
            True if hook was removed, False otherwise.
        """
        return self.hooks.remove_hook(layer_name)

    def get_attention_weights(self, user_id):
        """
        Get attention weights for a user sequence.

        Args:
            user_id: User ID.

        Returns:
            List of attention weight matrices from each layer.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if user_id not in self.uid_map:
            raise ValueError(f"User {user_id} not found in training data.")

        user_idx = self.uid_map[user_id]

        # Get user sequence
        sequence = self.user_sequences[user_idx][-self.config["max_seq_len"] :]

        # Pad sequence
        pad_len = self.config["max_seq_len"] - len(sequence)
        if pad_len > 0:
            sequence = [0] * pad_len + sequence

        # Convert to tensor
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)

        # Get attention weights
        self.model.eval()
        with torch.no_grad():
            attention_weights = self.model.get_attention_weights(sequence_tensor)

        # Convert to numpy arrays
        return [w.cpu().numpy() for w in attention_weights]

    def save(self, path):
        """
        Save the model to the given path.

        Args:
            path: Path to save the model.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Create directory if it doesn't exist
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_state = {
            "config": self.config,
            "state_dict": self.model.state_dict(),
            "user_ids": self.user_ids,
            "item_ids": self.item_ids,
            "uid_map": self.uid_map,
            "iid_map": self.iid_map,
            "user_sequences": self.user_sequences,
            "name": self.name,
            "trainable": self.trainable,
            "verbose": self.verbose,
            "seed": self.seed,
        }

        # Save model
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(model_state, f)

        # Save metadata
        with open(f"{path}.meta", "w") as f:
            yaml.dump(
                {
                    "name": self.name,
                    "type": "BERT4Rec",
                    "version": "1.0",
                    "num_users": self.num_users,
                    "num_items": self.num_items,
                    "vocab_size": self.vocab_size,
                    "embedding_dim": self.config["hidden_dim"],
                    "num_layers": self.config["num_layers"],
                    "num_heads": self.config["num_heads"],
                    "max_seq_len": self.config["max_seq_len"],
                    "created_at": str(datetime.now()),
                },
                f,
            )

    @classmethod
    def load(cls, path):
        """
        Load the model from the given path.

        Args:
            path: Path to load the model from.

        Returns:
            Loaded model.
        """
        # Load model state
        with open(path, "rb") as f:
            model_state = pickle.load(f)

        # Create new model
        model = cls(
            name=model_state["name"],
            config=model_state["config"],
            trainable=model_state["trainable"],
            verbose=model_state["verbose"],
            seed=model_state["seed"],
        )

        # Restore model state
        model.user_ids = model_state["user_ids"]
        model.item_ids = model_state["item_ids"]
        model.uid_map = model_state["uid_map"]
        model.iid_map = model_state["iid_map"]
        model.user_sequences = model_state["user_sequences"]
        model.num_users = len(model.user_ids)
        model.num_items = len(model.item_ids)
        model.vocab_size = model.num_items + 2  # Add padding and mask tokens

        # Rebuild model architecture
        model._build_model()

        # Load model weights
        model.model.load_state_dict(model_state["state_dict"])

        # Set fitted flag
        model.is_fitted = True

        return model

    def update_incremental(self, new_interactions, new_user_ids=None, new_item_ids=None):
        """
        Update model incrementally with new interactions.

        Args:
            new_interactions: New interaction data (user-item-timestamp).
            new_user_ids: New user IDs (if any).
            new_item_ids: New item IDs (if any).

        Returns:
            Updated model.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Check if we have new users
        if new_user_ids is not None:
            # Add new users to mappings
            new_users = [uid for uid in new_user_ids if uid not in self.uid_map]
            for uid in new_users:
                self.uid_map[uid] = len(self.uid_map)
                self.user_ids.append(uid)
            self.num_users = len(self.user_ids)

        # Check if we have new items
        if new_item_ids is not None:
            # Add new items to mappings
            new_items = [iid for iid in new_item_ids if iid not in self.iid_map]
            if new_items:
                for iid in new_items:
                    self.iid_map[iid] = len(self.iid_map)
                    self.item_ids.append(iid)
                self.num_items = len(self.item_ids)
                self.vocab_size = self.num_items + 2  # Add padding and mask tokens

                # Rebuild model with new vocabulary size
                old_state_dict = self.model.state_dict()
                self._build_model()

                # Copy weights for existing parameters
                new_state_dict = self.model.state_dict()
                for name, param in old_state_dict.items():
                    if name in new_state_dict:
                        if "token_embedding" in name and "embedding.weight" in name:
                            # Copy existing item embeddings
                            old_size = param.shape[0]
                            new_size = new_state_dict[name].shape[0]
                            if old_size < new_size:
                                new_state_dict[name][:old_size, :] = param
                        elif param.shape == new_state_dict[name].shape:
                            new_state_dict[name] = param

                self.model.load_state_dict(new_state_dict)

        # Update user sequences with new interactions
        self._update_user_sequences(new_interactions)

        # Fine-tune on new data
        self.fit(new_interactions, epochs=self.config.get("incremental_epochs", 5))

        return self

    def _update_user_sequences(self, interactions):
        """
        Update user sequences with new interactions.

        Args:
            interactions: New interactions (user-item-timestamp).
        """
        # Process interactions into timestamped sequences
        timestamp_data = []
        for u, i, t in interactions:
            if u in self.uid_map and i in self.iid_map:
                user_idx = self.uid_map[u]
                item_idx = self.iid_map[i]
                timestamp_data.append((user_idx, item_idx, t))

        # Sort by user_idx and timestamp
        timestamp_data.sort(key=lambda x: (x[0], x[2]))

        # Update user sequences
        current_user = -1
        for user_idx, item_idx, _ in timestamp_data:
            if user_idx != current_user:
                current_user = user_idx
                if user_idx >= len(self.user_sequences):
                    # Add new user sequence
                    self.user_sequences.append([])
            self.user_sequences[user_idx].append(item_idx)

    def export_embeddings(self):
        """
        Export item embeddings.

        Returns:
            Dict mapping item IDs to embeddings.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Get item embeddings
        item_embeddings = self.model.token_embedding.embedding.weight.detach().cpu().numpy()

        # Create mapping from item ID to embedding
        embeddings = {}
        for i, iid in enumerate(self.item_ids):
            # Skip padding and mask tokens
            item_idx = i + 1  # Adjust for 1-indexing in iid_map
            if item_idx < self.vocab_size - 1:  # Skip mask token
                embeddings[iid] = item_embeddings[item_idx].tolist()

        return embeddings

    def set_device(self, device):
        """
        Set the device to run the model on.

        Args:
            device: Device to run the model on.
        """
        self.device = device
        if hasattr(self, "model") and self.model is not None:
            self.model.to(device)
