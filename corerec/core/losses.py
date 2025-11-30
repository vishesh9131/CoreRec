"""
Losses module for CoreRec framework.

This module contains loss functions for recommendation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class DotProductLoss(nn.Module):
    """
    Dot product loss for recommendation tasks.

    This loss maximizes the dot product between positive user-item pairs
    and minimizes it between negative pairs.
    """

    def __init__(self, margin: float = 0.0, reduction: str = "mean"):
        """Initialize the dot product loss.

        Args:
            margin (float): Margin for the loss
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
            self,
            user_embeddings: torch.Tensor,
            item_embeddings: torch.Tensor,
            labels: torch.Tensor) -> torch.Tensor:
        """Forward pass of the dot product loss.

        Args:
            user_embeddings (torch.Tensor): User embeddings of shape [batch_size, embedding_dim]
            item_embeddings (torch.Tensor): Item embeddings of shape [batch_size, embedding_dim]
            labels (torch.Tensor): Binary labels of shape [batch_size]

        Returns:
            torch.Tensor: Loss value
        """
        # Compute dot product
        dot_product = torch.sum(user_embeddings * item_embeddings, dim=1)

        # Compute loss
        if self.margin > 0:
            loss = torch.where(
                labels > 0.5,
                torch.clamp(
                    self.margin -
                    dot_product,
                    min=0.0),
                # Positive pairs
                torch.clamp(
                    dot_product -
                    self.margin,
                    min=0.0),
                # Negative pairs
            )
        else:
            loss = -labels * dot_product + (1 - labels) * dot_product

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class CosineLoss(nn.Module):
    """
    Cosine similarity loss for recommendation tasks.

    This loss maximizes the cosine similarity between positive user-item pairs
    and minimizes it between negative pairs.
    """

    def __init__(self, margin: float = 0.0, reduction: str = "mean"):
        """Initialize the cosine loss.

        Args:
            margin (float): Margin for the loss
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
            self,
            user_embeddings: torch.Tensor,
            item_embeddings: torch.Tensor,
            labels: torch.Tensor) -> torch.Tensor:
        """Forward pass of the cosine loss.

        Args:
            user_embeddings (torch.Tensor): User embeddings of shape [batch_size, embedding_dim]
            item_embeddings (torch.Tensor): Item embeddings of shape [batch_size, embedding_dim]
            labels (torch.Tensor): Binary labels of shape [batch_size]

        Returns:
            torch.Tensor: Loss value
        """
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(
            user_embeddings, item_embeddings, dim=1)

        # Compute loss
        if self.margin > 0:
            loss = torch.where(
                labels > 0.5,
                torch.clamp(
                    self.margin -
                    cosine_sim,
                    min=0.0),
                # Positive pairs
                torch.clamp(
                    cosine_sim -
                    self.margin,
                    min=0.0),
                # Negative pairs
            )
        else:
            loss = -labels * cosine_sim + (1 - labels) * cosine_sim

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class InfoNCE(nn.Module):
    """
    InfoNCE loss for contrastive learning in recommendation tasks.

    This loss is used for contrastive learning to maximize the similarity
    between positive pairs while minimizing it between negative pairs.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        """Initialize the InfoNCE loss.

        Args:
            temperature (float): Temperature parameter for scaling
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the InfoNCE loss.

        Args:
            anchor_embeddings (torch.Tensor): Anchor embeddings of shape [batch_size, embedding_dim]
            positive_embeddings (torch.Tensor): Positive embeddings of shape [batch_size, embedding_dim]
            negative_embeddings (Optional[torch.Tensor]): Negative embeddings of shape [num_negatives, embedding_dim].
                If None, all other examples in the batch are treated as negatives (in-batch negatives).

        Returns:
            torch.Tensor: Loss value
        """
        batch_size = anchor_embeddings.size(0)

        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)

        # Compute positive similarity
        pos_sim = torch.sum(
            anchor_embeddings * positive_embeddings,
            dim=1) / self.temperature

        if negative_embeddings is None:
            # Use in-batch negatives
            # Create similarity matrix
            sim_matrix = torch.matmul(
                anchor_embeddings,
                positive_embeddings.t()) / self.temperature

            # Create labels (identity matrix)
            labels = torch.eye(batch_size, device=anchor_embeddings.device)

            # Compute log softmax
            log_softmax = F.log_softmax(sim_matrix, dim=1)

            # Compute loss
            loss = -torch.sum(log_softmax * labels, dim=1)
        else:
            # Normalize negative embeddings
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)

            # Compute negative similarity
            neg_sim = torch.matmul(
                anchor_embeddings,
                negative_embeddings.t()) / self.temperature

            # Combine positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

            # Labels are all 0 (for positives)
            labels = torch.zeros(
                batch_size,
                device=anchor_embeddings.device,
                dtype=torch.long)

            # Compute cross entropy loss
            loss = F.cross_entropy(logits, labels, reduction="none")

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
