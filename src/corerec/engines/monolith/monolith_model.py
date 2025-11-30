"""
Monolith model for large-scale distributed training of recommendation models.

This module implements a monolith model architecture designed for efficient
distributed training of large-scale recommendation models.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Tuple, Any, Union, Optional
import logging
import time
import os

from corerec.core.base_model import BaseModel
from corerec.core.towers import Tower, MLPTower
from corerec.core.embedding_tables.collisionless import CollisionlessEmbedding


class MonolithModel(BaseModel):
    """
    Monolith model for large-scale recommendation systems.

    This model is designed for distributed training of large recommendation models,
    with support for:
    - Efficient embedding lookups for sparse features
    - Sharded embeddings across multiple devices
    - Multi-tower architecture for different feature types
    - Large-scale distributed training

    Attributes:
        name (str): Name of the model
        config (Dict[str, Any]): Model configuration
        embedding_tables (nn.ModuleDict): Dictionary of embedding tables
        sparse_feature_towers (nn.ModuleDict): Dictionary of towers for sparse features
        dense_feature_towers (nn.ModuleDict): Dictionary of towers for dense features
        interaction_layer (nn.Module): Layer for feature interactions
        prediction_tower (nn.Module): Tower for final prediction
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the monolith model.

        Args:
            name (str): Name of the model
            config (Dict[str, Any]): Model configuration including:
                - embedding_dim (int): Dimension of embedding vectors
                - sparse_features (Dict[str, Dict]): Configuration for sparse features
                - dense_features (Dict[str, Dict]): Configuration for dense features
                - tower_hidden_dims (List[int]): Hidden dimensions for towers
                - prediction_hidden_dims (List[int]): Hidden dimensions for prediction tower
                - use_collisionless (bool): Whether to use collisionless embeddings
                - sharded_embeddings (bool): Whether to shard embeddings
        """
        super().__init__(name, config)

        # Get configuration
        self.embedding_dim = config.get("embedding_dim", 128)
        sparse_features = config.get("sparse_features", {})
        dense_features = config.get("dense_features", {})
        tower_hidden_dims = config.get("tower_hidden_dims", [256, 128])
        prediction_hidden_dims = config.get("prediction_hidden_dims", [128, 64])
        self.use_collisionless = config.get("use_collisionless", False)
        self.sharded_embeddings = config.get("sharded_embeddings", False)

        # Set up distributed training
        self.world_size = 1
        self.rank = 0
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.logger.info(f"Initialized distributed model: rank {self.rank}/{self.world_size}")

        # Create embedding tables for sparse features
        self.embedding_tables = nn.ModuleDict()
        for feature_name, feature_config in sparse_features.items():
            # Check if this embedding table should be on this rank
            if (
                self.sharded_embeddings
                and feature_config.get("shard_id", 0) % self.world_size != self.rank
            ):
                continue

            vocab_size = feature_config.get("vocab_size", 100000)
            dim = feature_config.get("dim", self.embedding_dim)

            if self.use_collisionless and vocab_size > 1000000:
                # Use collisionless embeddings for very large vocabularies
                self.embedding_tables[feature_name] = CollisionlessEmbedding(
                    num_embeddings=vocab_size // 10,  # Reduce size via hashing
                    embedding_dim=dim,
                    num_hash_functions=2,
                )
            else:
                # Use standard embeddings
                self.embedding_tables[feature_name] = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=dim,
                    sparse=True,  # Use sparse gradients for efficient updates
                )

        # Create towers for sparse features
        self.sparse_feature_towers = nn.ModuleDict()
        for feature_name, feature_config in sparse_features.items():
            # Skip if not on this rank
            if (
                self.sharded_embeddings
                and feature_config.get("shard_id", 0) % self.world_size != self.rank
            ):
                continue

            dim = feature_config.get("dim", self.embedding_dim)

            self.sparse_feature_towers[feature_name] = MLPTower(
                name=f"{feature_name}_tower",
                input_dim=dim,
                output_dim=self.embedding_dim,
                config={
                    "hidden_dims": tower_hidden_dims,
                    "dropout": 0.1,
                    "activation": "relu",
                    "norm": "batch",
                },
            )

        # Create towers for dense features
        self.dense_feature_towers = nn.ModuleDict()
        for feature_name, feature_config in dense_features.items():
            dim = feature_config.get("dim", 13)  # Default for numeric features

            self.dense_feature_towers[feature_name] = MLPTower(
                name=f"{feature_name}_tower",
                input_dim=dim,
                output_dim=self.embedding_dim,
                config={
                    "hidden_dims": tower_hidden_dims,
                    "dropout": 0.1,
                    "activation": "relu",
                    "norm": "batch",
                },
            )

        # Calculate total dimension for concatenated features
        self.total_feature_dim = (
            len(sparse_features) * self.embedding_dim + len(dense_features) * self.embedding_dim
        )

        # Create interaction layer (simple concatenation for now)
        self.interaction_layer = nn.Identity()

        # Create prediction tower
        self.prediction_tower = nn.Sequential(
            nn.Linear(self.total_feature_dim, prediction_hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(prediction_hidden_dims[0]),
            nn.Dropout(0.1),
            *[
                nn.Sequential(
                    nn.Linear(prediction_hidden_dims[i], prediction_hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(prediction_hidden_dims[i + 1]),
                    nn.Dropout(0.1),
                )
                for i in range(len(prediction_hidden_dims) - 1)
            ],
            nn.Linear(prediction_hidden_dims[-1], 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the monolith model.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data including:
                - Sparse features as tensors with name matching embedding tables
                - Dense features as tensors with name matching dense feature towers

        Returns:
            torch.Tensor: Prediction scores
        """
        sparse_feature_embeddings = []
        for feature_name, embedding_table in self.embedding_tables.items():
            if feature_name in batch:
                # Look up embeddings
                embeddings = embedding_table(batch[feature_name])

                # Apply tower
                tower_output = self.sparse_feature_towers[feature_name](embeddings)

                # Add to list
                sparse_feature_embeddings.append(tower_output)

        dense_feature_embeddings = []
        for feature_name, tower in self.dense_feature_towers.items():
            if feature_name in batch:
                # Apply tower
                tower_output = tower(batch[feature_name])

                # Add to list
                dense_feature_embeddings.append(tower_output)

        # Concatenate all feature embeddings
        all_embeddings = sparse_feature_embeddings + dense_feature_embeddings

        if not all_embeddings:
            raise ValueError("No valid features found in batch")

        concatenated_embeddings = torch.cat(all_embeddings, dim=1)

        # Apply interaction layer
        interaction_output = self.interaction_layer(concatenated_embeddings)

        # Apply prediction tower
        predictions = self.prediction_tower(interaction_output)

        return predictions.squeeze(-1)

    def train_step(
        self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data including:
                - Sparse features as tensors with name matching embedding tables
                - Dense features as tensors with name matching dense feature towers
                - labels (torch.Tensor): Target labels
            optimizer (torch.optim.Optimizer): Optimizer instance

        Returns:
            Dict[str, float]: Dictionary with loss values
        """
        # Extract labels
        labels = batch.pop("labels")

        # Forward pass
        predictions = self.forward(batch)

        # Compute loss
        if predictions.shape != labels.shape:
            labels = labels.view(predictions.shape)

        loss = nn.BCEWithLogitsLoss()(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single validation step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data including:
                - Sparse features as tensors with name matching embedding tables
                - Dense features as tensors with name matching dense feature towers
                - labels (torch.Tensor): Target labels

        Returns:
            Dict[str, float]: Dictionary with validation metrics
        """
        # Extract labels
        labels = batch.pop("labels")

        # Forward pass
        with torch.no_grad():
            predictions = self.forward(batch)

            # Compute loss
            if predictions.shape != labels.shape:
                labels = labels.view(predictions.shape)

            loss = nn.BCEWithLogitsLoss()(predictions, labels)

            # Compute accuracy
            binary_predictions = (torch.sigmoid(predictions) > 0.5).float()
            accuracy = (binary_predictions == labels).float().mean()

        return {"val_loss": loss.item(), "val_accuracy": accuracy.item()}

    def get_state_dict_for_save(self) -> Dict[str, Any]:
        """Get a state dict for saving that handles distributed training.

        In distributed training, we need to gather parameters from all ranks.
        This method handles that process.

        Returns:
            Dict[str, Any]: State dict for saving
        """
        if not self.sharded_embeddings or self.world_size == 1:
            # No need for special handling
            return self.state_dict()

        # Placeholder for gathered state dict
        gathered_state_dict = {}

        # Gather state dict from all ranks
        local_state_dict = self.state_dict()

        if self.rank == 0:
            # Master process gathers all state dicts
            all_state_dicts = [local_state_dict]
            for rank in range(1, self.world_size):
                rank_state_dict = {}
                dist.recv(rank_state_dict, src=rank)
                all_state_dicts.append(rank_state_dict)

            # Merge state dicts
            for k in local_state_dict.keys():
                if "embedding_tables" in k:
                    # Find which rank has this embedding table
                    for rank_dict in all_state_dicts:
                        if k in rank_dict:
                            gathered_state_dict[k] = rank_dict[k]
                            break
                else:
                    # For non-embedding parameters, use the local ones
                    gathered_state_dict[k] = local_state_dict[k]
        else:
            # Send state dict to master process
            dist.send(local_state_dict, dst=0)

        return gathered_state_dict
