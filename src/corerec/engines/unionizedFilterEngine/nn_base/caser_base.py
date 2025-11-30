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


class HorizontalConvolution(nn.Module):
    """Horizontal convolutional layer for sequence patterns."""

    def __init__(self, num_filters: int, embedding_dim: int):
        """
        Initialize the horizontal convolutional layer.

        Args:
            num_filters: Number of horizontal convolutional filters.
            embedding_dim: Dimension of item embeddings.
        """
        super(HorizontalConvolution, self).__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(1, num_filters, (i, embedding_dim))
                for i in range(1, 3)  # Default window sizes 1 and 2
            ]
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim].

        Returns:
            Tensor of shape [batch_size, num_filters * 2].
        """
        # Add channel dimension [batch_size, 1, seq_len, embedding_dim]
        x = x.unsqueeze(1)

        # Apply horizontal convolutional layers
        outputs = []
        for conv in self.conv_layers:
            # Apply convolution
            out = conv(x)
            # Apply activation
            out = F.relu(out)
            # Apply max pooling over the sequence dimension
            out = F.max_pool2d(out, (out.size(2), 1))
            # Flatten
            out = out.squeeze(3).squeeze(2)
            outputs.append(out)

        # Concatenate outputs [batch_size, num_filters * num_window_sizes]
        return torch.cat(outputs, 1)


class VerticalConvolution(nn.Module):
    """Vertical convolutional layer for embedding dimension patterns."""

    def __init__(self, num_filters: int, seq_len: int):
        """
        Initialize the vertical convolutional layer.

        Args:
            num_filters: Number of vertical convolutional filters.
            seq_len: Maximum sequence length.
        """
        super(VerticalConvolution, self).__init__()
        self.conv = nn.Conv2d(1, num_filters, (seq_len, 1))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim].

        Returns:
            Tensor of shape [batch_size, num_filters * embedding_dim].
        """
        # Add channel dimension [batch_size, 1, seq_len, embedding_dim]
        x = x.unsqueeze(1)

        # Apply vertical convolution
        out = self.conv(x)
        out = F.relu(out)

        # Flatten
        return out.squeeze(2).permute(0, 1, 2).reshape(x.size(0), -1)


class CaserModel(nn.Module):
    """
    Convolutional Sequence Embedding Recommendation (Caser) model.

    Architecture diagram:
    -------------------
    Input Sequence -> Item Embedding -> [Horizontal Conv, Vertical Conv] -> Concat -> FC -> Output
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        max_seq_len: int,
        num_h_filters: int,
        num_v_filters: int,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize the Caser model.

        Args:
            num_items: Number of items in the dataset.
            embedding_dim: Dimension of item embeddings.
            max_seq_len: Maximum sequence length.
            num_h_filters: Number of horizontal convolutional filters.
            num_v_filters: Number of vertical convolutional filters.
            dropout_rate: Dropout probability for regularization.
        """
        super(CaserModel, self).__init__()

        # Item embedding layer
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        # Horizontal convolution
        self.h_conv = HorizontalConvolution(num_h_filters, embedding_dim)

        # Vertical convolution
        self.v_conv = VerticalConvolution(num_v_filters, max_seq_len)

        # Fully connected layers
        fc_dim = num_h_filters * 2 + num_v_filters * embedding_dim
        self.fc1 = nn.Linear(fc_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(embedding_dim, num_items + 1)  # +1 for padding at index 0

    def forward(self, seq_var):
        """
        Forward pass.

        Args:
            seq_var: Input sequence tensor of shape [batch_size, seq_len].

        Returns:
            Output prediction tensor of shape [batch_size, num_items].
        """
        # Embedding lookup
        item_embs = self.item_embedding(seq_var)

        # Apply horizontal and vertical convolutions
        h_out = self.h_conv(item_embs)
        v_out = self.v_conv(item_embs)

        # Concatenate outputs
        out = torch.cat([h_out, v_out], 1)

        # Apply fully connected layers
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class Caser_base(BaseCorerec):
    """
    Base class for Convolutional Sequence Embedding Recommendation (Caser) model.

    This model uses horizontal and vertical convolutional filters to capture sequential patterns
    in user behavior sequences for next-item recommendation.

    Architecture diagram:
    -------------------
    User Sequences -> Embeddings -> [Horizontal Conv, Vertical Conv] -> Concat -> FC -> Prediction
    """

    def __init__(
        self,
        name: str = "Caser",
        config: Optional[Dict[str, Any]] = None,
        trainable: bool = True,
        verbose: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the Caser base class.

        Args:
            name: Name of the model.
            config: Configuration dictionary.
            trainable: Whether the model is trainable.
            verbose: Whether to print verbose output.
            seed: Random seed for reproducibility.
        """
        super().__init__(name, trainable, verbose)
        self.seed = seed

        # Set random seeds
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set default configuration
        self.config = {
            "embedding_dim": 64,
            "num_h_filters": 16,
            "num_v_filters": 4,
            "max_seq_len": 50,
            "dropout_rate": 0.5,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "batch_size": 256,
            "num_epochs": 30,
            "device": "cpu",
            "optimizer": "adam",
            "loss": "bce",
            "negative_samples": 3,
        }

        # Update configuration with provided config
        if config is not None:
            self.config.update(config)

        # Initialize model attributes
        # Note: user_ids and item_ids are properties in base class, don't set directly
        self._caser_user_ids = []
        self._caser_item_ids = []
        self.uid_map = {}
        self.iid_map = {}
        self.num_users = 0
        self.num_items = 0
        self.user_sequences = []
        self.model = None
        self.device = torch.device(self.config["device"])
        self.is_fitted = False

        # Initialize hook manager
        self.hooks = HookManager()

        # Initialize logger
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(self.name)

    def _build_model(self):
        """Build the Caser model architecture."""
        self.model = CaserModel(
            num_items=self.num_items,
            embedding_dim=self.config["embedding_dim"],
            max_seq_len=self.config["max_seq_len"],
            num_h_filters=self.config["num_h_filters"],
            num_v_filters=self.config["num_v_filters"],
            dropout_rate=self.config["dropout_rate"],
        ).to(self.device)

    def _get_optimizer(self):
        """
        Get optimizer based on configuration.

        Returns:
            Optimizer instance.
        """
        if self.config["optimizer"].lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"].lower() == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")

    def _get_loss_function(self):
        """
        Get loss function based on configuration.

        Returns:
            Loss function.
        """
        if self.config["loss"].lower() == "bce":
            return nn.BCEWithLogitsLoss()
        elif self.config["loss"].lower() == "ce":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config['loss']}")

    def _prepare_sequences(self, interactions, user_ids=None, item_ids=None):
        """
        Prepare user sequences from interactions.

        Args:
            interactions: List of (user_id, item_id, timestamp) tuples.
            user_ids: List of user IDs.
            item_ids: List of item IDs.

        Returns:
            Prepared sequences.
        """
        if user_ids is not None:
            # Create mappings
            self._caser_user_ids = list(user_ids)
            self.uid_map = {uid: i for i, uid in enumerate(self._caser_user_ids)}
            self.num_users = len(self._caser_user_ids)

        if item_ids is not None:
            # Create mappings
            self._caser_item_ids = list(item_ids)
            self.iid_map = {
                iid: i + 1 for i, iid in enumerate(self._caser_item_ids)
            }  # +1 for padding at 0
            self.num_items = len(self._caser_item_ids)

        # Sort interactions by timestamp
        interactions_sorted = sorted(interactions, key=lambda x: (x[0], x[2]))

        # Group interactions by user
        user_sequences = defaultdict(list)
        for user_id, item_id, _ in interactions_sorted:
            if user_id in self.uid_map and item_id in self.iid_map:
                user_idx = self.uid_map[user_id]
                item_idx = self.iid_map[item_id]
                user_sequences[user_idx].append(item_idx)
            else:
                # Auto-add users and items not in mappings
                if user_id not in self.uid_map:
                    # Add new user
                    if not self._caser_user_ids:
                        self._caser_user_ids = [user_id]
                        self.uid_map = {user_id: 0}
                        self.num_users = 1
                    else:
                        self._caser_user_ids.append(user_id)
                        self.uid_map[user_id] = len(self._caser_user_ids) - 1
                        self.num_users = len(self._caser_user_ids)

                if item_id not in self.iid_map:
                    # Add new item
                    if not self._caser_item_ids:
                        self._caser_item_ids = [item_id]
                        self.iid_map = {item_id: 1}  # +1 for padding
                        self.num_items = 1
                    else:
                        self._caser_item_ids.append(item_id)
                        self.iid_map[item_id] = len(
                            self._caser_item_ids
                        )  # +1 for padding since we start at 1
                        self.num_items = len(self._caser_item_ids)

                # Add to sequence
                user_idx = self.uid_map[user_id]
                item_idx = self.iid_map[item_id]
                user_sequences[user_idx].append(item_idx)

        # Convert to list of sequences
        sequences = [user_sequences.get(i, []) for i in range(self.num_users)]
        self.user_sequences = sequences

        return sequences

    def _generate_training_examples(self, sequences, negative_samples=None):
        """
        Generate training examples from sequences.

        Args:
            sequences: List of user sequences.
            negative_samples: Number of negative samples per positive sample.

        Returns:
            Training examples (X, y).
        """
        if negative_samples is None:
            negative_samples = self.config["negative_samples"]

        X = []
        y = []
        for sequence in sequences:
            if len(sequence) < 2:
                continue

            # For each position in the sequence
            for i in range(1, len(sequence)):
                # Get the sequence up to position i (limited by max_seq_len)
                seq = sequence[max(0, i - self.config["max_seq_len"]) : i]

                # Pad sequence if needed
                if len(seq) < self.config["max_seq_len"]:
                    seq = [0] * (self.config["max_seq_len"] - len(seq)) + seq

                # Add positive example
                X.append(seq)
                y.append([sequence[i], 1])

                # Add negative examples
                for _ in range(negative_samples):
                    neg_item = np.random.randint(1, self.num_items + 1)
                    while neg_item in sequence:
                        neg_item = np.random.randint(1, self.num_items + 1)
                    X.append(seq)
                    y.append([neg_item, 0])

        return np.array(X), np.array(y)

    def fit(self, interactions, user_ids=None, item_ids=None, epochs=None, batch_size=None):
        """
        Train the Caser model using the provided data.

        Args:
            interactions: List of (user_id, item_id, timestamp) tuples.
            user_ids: List of all user IDs.
            item_ids: List of all item IDs.
            epochs: Number of training epochs (overrides config).
            batch_size: Batch size (overrides config).

        Returns:
            self
        """
        # Prepare sequences
        self._prepare_sequences(interactions, user_ids, item_ids)

        # Build model if not already built
        if self.model is None:
            self._build_model()

        # Set epochs and batch size
        if epochs is None:
            epochs = self.config["num_epochs"]
        if batch_size is None:
            batch_size = self.config["batch_size"]

        # Get optimizer and loss function
        optimizer = self._get_optimizer()
        loss_fn = self._get_loss_function()

        # Generate training examples
        X, y = self._generate_training_examples(self.user_sequences)

        # Convert to tensors
        X_tensor = torch.LongTensor(X).to(self.device)
        y_items = torch.LongTensor(y[:, 0]).to(self.device)
        y_labels = torch.FloatTensor(y[:, 1]).to(self.device)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_items, y_labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train model
        self.model.train()
        progress_bar = tqdm(range(epochs), desc="Training", disable=not self.verbose)

        for epoch in progress_bar:
            total_loss = 0
            for batch_x, batch_y_items, batch_y_labels in dataloader:
                # Forward pass
                logits = self.model(batch_x)

                # Get predictions for the target items
                item_logits = logits.gather(1, batch_y_items.unsqueeze(1)).squeeze()

                # Compute loss
                loss = loss_fn(item_logits, batch_y_labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Update progress bar
            avg_loss = total_loss / len(dataloader)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Log epoch loss
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    def predict(self, user_id, item_id):
        """
        Predict the likelihood of a user interacting with an item.

        Args:
            user_id: User ID.
            item_id: Item ID.

        Returns:
            Prediction score.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Get user index
        if user_id not in self.uid_map:
            raise ValueError(f"User {user_id} not found in training data.")
        user_idx = self.uid_map[user_id]

        # Get item index
        if item_id not in self.iid_map:
            raise ValueError(f"Item {item_id} not found in training data.")
        item_idx = self.iid_map[item_id]

        # Get user sequence
        sequence = self.user_sequences[user_idx]

        # If sequence is empty, return 0
        if not sequence:
            return 0.0

        # Get the last max_seq_len items
        seq = sequence[-self.config["max_seq_len"] :]

        # Pad sequence if needed
        if len(seq) < self.config["max_seq_len"]:
            seq = [0] * (self.config["max_seq_len"] - len(seq)) + seq

        # Convert to tensor
        seq_tensor = torch.LongTensor([seq]).to(self.device)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            logits = self.model(seq_tensor)
            score = torch.sigmoid(logits[0, item_idx]).item()

        return score

    def recommend(self, user_id, top_n=10, exclude_seen=True, items_to_recommend=None):
        """
        Generate top-N recommendations for a user.

        Args:
            user_id: User ID.
            top_n: Number of items to recommend.
            exclude_seen: Whether to exclude seen items.
            items_to_recommend: Specific items to score for recommendation. If None, all items are considered.

        Returns:
            List of (item_id, score) tuples.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Get user index
        if user_id not in self.uid_map:
            raise ValueError(f"User {user_id} not found in training data.")
        user_idx = self.uid_map[user_id]

        # Get user sequence
        sequence = self.user_sequences[user_idx]

        # If sequence is empty, return empty list
        if not sequence:
            return []

        # Get the last max_seq_len items
        seq = sequence[-self.config["max_seq_len"] :]

        # Pad sequence if needed
        if len(seq) < self.config["max_seq_len"]:
            seq = [0] * (self.config["max_seq_len"] - len(seq)) + seq

        # Convert to tensor
        seq_tensor = torch.LongTensor([seq]).to(self.device)

        # Items to score
        if items_to_recommend is None:
            items_to_score = list(range(1, self.num_items + 1))
        else:
            items_to_score = [self.iid_map.get(iid, 0) for iid in items_to_recommend]
            items_to_score = [idx for idx in items_to_score if idx > 0]

        # Exclude seen items if requested
        if exclude_seen:
            items_to_score = [idx for idx in items_to_score if idx not in sequence]

        # If no items to score, return empty list
        if not items_to_score:
            return []

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            logits = self.model(seq_tensor)
            scores = torch.sigmoid(logits[0, items_to_score]).cpu().numpy()

        # Get item IDs
        item_ids = [self._caser_item_ids[idx - 1] for idx in items_to_score]

        # Create (item_id, score) tuples
        item_scores = list(zip(item_ids, scores))

        # Sort by score in descending order
        item_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-n items
        return item_scores[:top_n]

    def save(self, path):
        """
        Save model to file.

        Args:
            path: Path to save model.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Create directory if it doesn't exist
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_state = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "user_ids": self._caser_user_ids,
            "item_ids": self._caser_item_ids,
            "uid_map": self.uid_map,
            "iid_map": self.iid_map,
            "user_sequences": self.user_sequences,
            "name": self.name,
            "trainable": self.trainable,
            "verbose": self.verbose,
            "seed": self.seed,
        }

        # Save model
        torch.save(model_state, f"{path}.pt")

        # Save metadata
        with open(f"{path}.meta", "w") as f:
            yaml.dump(
                {
                    "name": self.name,
                    "type": "Caser",
                    "version": "1.0",
                    "num_users": self.num_users,
                    "num_items": self.num_items,
                    "embedding_dim": self.config["embedding_dim"],
                    "num_h_filters": self.config["num_h_filters"],
                    "num_v_filters": self.config["num_v_filters"],
                    "max_seq_len": self.config["max_seq_len"],
                    "created_at": str(datetime.now()),
                },
                f,
            )

    @classmethod
    def load(cls, path):
        """
        Load model from file.

        Args:
            path: Path to load model from.

        Returns:
            Loaded model.
        """
        # Load model state
        model_state = torch.load(path, map_location="cpu")

        # Create model instance
        instance = cls(
            name=model_state["name"],
            config=model_state["config"],
            trainable=model_state["trainable"],
            verbose=model_state["verbose"],
            seed=model_state["seed"],
        )

        # Restore model attributes
        instance.user_ids = model_state["user_ids"]
        instance.item_ids = model_state["item_ids"]
        instance.uid_map = model_state["uid_map"]
        instance.iid_map = model_state["iid_map"]
        instance.user_sequences = model_state["user_sequences"]
        instance.num_users = len(instance.user_ids)
        instance.num_items = len(instance.item_ids)

        # Build model
        instance._build_model()

        # Load model weights
        instance.model.load_state_dict(model_state["model_state_dict"])

        # Set model to eval mode
        instance.model.eval()

        # Set fitted flag
        instance.is_fitted = True

        return instance

    def update_incremental(self, new_interactions, new_user_ids=None, new_item_ids=None):
        """
        Update model incrementally with new data.

        Args:
            new_interactions: List of (user_id, item_id, timestamp) tuples.
            new_user_ids: List of new user IDs.
            new_item_ids: List of new item IDs.

        Returns:
            self
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Add new users if provided
        if new_user_ids is not None:
            new_users = [uid for uid in new_user_ids if uid not in self.uid_map]
            for uid in new_users:
                self.uid_map[uid] = len(self.uid_map)
                self._caser_user_ids.append(uid)
            self.num_users = len(self._caser_user_ids)

            # Extend user sequences if needed
            while len(self.user_sequences) < self.num_users:
                self.user_sequences.append([])

        # Add new items if provided
        rebuild_model = False
        if new_item_ids is not None:
            new_items = [iid for iid in new_item_ids if iid not in self.iid_map]
            if new_items:
                # Save old model state
                old_state_dict = self.model.state_dict()

                # Update item mappings
                for iid in new_items:
                    self.iid_map[iid] = len(self.iid_map) + 1  # +1 for padding at 0
                    self._caser_item_ids.append(iid)
                self.num_items = len(self._caser_item_ids)

                # Rebuild model
                rebuild_model = True

        # Update user sequences with new interactions
        self._prepare_sequences(new_interactions)

        # Rebuild model if needed
        if rebuild_model:
            self._build_model()

            # Transfer weights for existing parameters
            for name, param in old_state_dict.items():
                if name in self.model.state_dict():
                    new_param = self.model.state_dict()[name]
                    if param.shape == new_param.shape:
                        self.model.state_dict()[name].copy_(param)

        # Fine-tune model on new data
        self.fit(new_interactions, epochs=self.config.get("incremental_epochs", 5))

        return self

    def export_embeddings(self):
        """
        Export item embeddings.

        Returns:
            Dict mapping item IDs to embeddings.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Get item embeddings
        self.model.eval()
        with torch.no_grad():
            # +1 to skip padding embedding at index 0
            indices = torch.arange(1, self.num_items + 1, dtype=torch.long).to(self.device)
            embeddings = self.model.item_embedding(indices).cpu().numpy()

        # Create mapping from item ID to embedding
        item_embeddings = {}
        for i, iid in enumerate(self._caser_item_ids):
            item_embeddings[iid] = embeddings[i].tolist()

        return item_embeddings

    def set_device(self, device):
        """
        Set the device to run the model on.

        Args:
            device: Device to run the model on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.config["device"] = device
        if hasattr(self, "model") and self.model is not None:
            self.model.to(self.device)
