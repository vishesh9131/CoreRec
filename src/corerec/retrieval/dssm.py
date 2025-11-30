"""
Deep Structured Semantic Model (DSSM) for recommendation.

This module implements the DSSM retriever for recommendation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Union, Optional

from corerec.retrieval.base_retriever import BaseRetriever
from corerec.core.towers import MLPTower


class DSSM(BaseRetriever):
    """
    Deep Structured Semantic Model (DSSM) for recommendation.

    DSSM uses two separate towers to encode users and items into a
    common embedding space, where relevance is measured by dot product.

    Attributes:
        name (str): Name of the retriever
        config (Dict[str, Any]): Retriever configuration
        user_tower (MLPTower): Tower for encoding users
        item_tower (MLPTower): Tower for encoding items
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the DSSM retriever.

        Args:
            name (str): Name of the retriever
            config (Dict[str, Any]): Retriever configuration including:
                - embedding_dim (int): Dimension of the embedding space
                - user_input_dim (int): Dimension of user features
                - item_input_dim (int): Dimension of item features
                - hidden_dims (List[int]): List of hidden dimensions for the towers
                - dropout (float): Dropout rate
        """
        super().__init__(name, config)

        # Get configuration
        user_input_dim = config.get("user_input_dim", 128)
        item_input_dim = config.get("item_input_dim", 128)
        hidden_dims = config.get("hidden_dims", [256, 128])
        dropout = config.get("dropout", 0.1)

        # Create tower configs
        tower_config = {
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "activation": "relu",
            "norm": "batch",
        }

        # Create user and item towers
        self.user_tower = MLPTower(
            name="user_tower",
            input_dim=user_input_dim,
            output_dim=self.embedding_dim,
            config=tower_config,
        )

        self.item_tower = MLPTower(
            name="item_tower",
            input_dim=item_input_dim,
            output_dim=self.embedding_dim,
            config=tower_config,
        )

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DSSM model.

        Args:
            user_features (torch.Tensor): User features of shape [batch_size, user_input_dim]
            item_features (torch.Tensor): Item features of shape [batch_size, item_input_dim]

        Returns:
            torch.Tensor: Similarity scores of shape [batch_size]
        """
        # Encode user and item
        user_embedding = self.user_tower(user_features)
        item_embedding = self.item_tower(item_features)

        # Compute similarity score
        return self.score(user_embedding, item_embedding)

    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        """Encode a query (user) into the embedding space.

        Args:
            query (torch.Tensor): User features

        Returns:
            torch.Tensor: User embedding
        """
        return self.user_tower(query)

    def encode_item(self, item: torch.Tensor) -> torch.Tensor:
        """Encode an item into the embedding space.

        Args:
            item (torch.Tensor): Item features

        Returns:
            torch.Tensor: Item embedding
        """
        return self.item_tower(item)

    def train_step(
        self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data including:
                - user_features (torch.Tensor): User features
                - item_features (torch.Tensor): Item features
                - labels (torch.Tensor): Binary labels
            optimizer (torch.optim.Optimizer): Optimizer instance

        Returns:
            Dict[str, float]: Dictionary with loss values
        """
        user_features = batch["user_features"]
        item_features = batch["item_features"]
        labels = batch["labels"]

        # Forward pass
        scores = self.forward(user_features, item_features)

        # Compute loss (binary cross entropy)
        loss = nn.BCEWithLogitsLoss()(scores, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single validation step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data including:
                - user_features (torch.Tensor): User features
                - item_features (torch.Tensor): Item features
                - labels (torch.Tensor): Binary labels

        Returns:
            Dict[str, float]: Dictionary with validation metrics
        """
        user_features = batch["user_features"]
        item_features = batch["item_features"]
        labels = batch["labels"]

        # Forward pass
        with torch.no_grad():
            scores = self.forward(user_features, item_features)

            # Compute loss
            loss = nn.BCEWithLogitsLoss()(scores, labels)

            # Compute accuracy
            predictions = (torch.sigmoid(scores) > 0.5).float()
            accuracy = (predictions == labels).float().mean()

        return {"val_loss": loss.item(), "val_accuracy": accuracy.item()}
