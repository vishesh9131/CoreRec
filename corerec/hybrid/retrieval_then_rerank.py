"""
Retrieval-then-rerank model for recommendation systems.

This module provides a two-stage recommendation model that first retrieves
candidates using a retrieval model and then reranks them using a more
sophisticated model.
"""

import torch
from corerec.api.model_bundle import require_legacy_pickle, save_bundle, load_bundle, is_safe_bundle
from corerec.api.exceptions import SaveLoadError
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional, Callable
import logging
import time
import os

from corerec.core.base_model import BaseModel


class RetrievalThenRerank(BaseModel):
    """
    Two-stage recommendation model: retrieval followed by reranking.

    This model first retrieves candidates using a retrieval model and then
    reranks them using a more sophisticated model.

    Attributes:
        name (str): Name of the model
        config (Dict[str, Any]): Model configuration
        retriever (nn.Module): Retrieval model
        reranker (nn.Module): Reranking model
    """

    def __init__(
        self, name: str, config: Dict[str, Any], retriever: nn.Module, reranker: nn.Module
    ):
        """Initialize the retrieval-then-rerank model.

        Args:
            name (str): Name of the model
            config (Dict[str, Any]): Model configuration including:
                - retriever_config (Dict[str, Any]): Configuration for the retriever
                - reranker_config (Dict[str, Any]): Configuration for the reranker
                - num_candidates (int): Number of candidates to retrieve
            retriever (nn.Module): Retrieval model
            reranker (nn.Module): Reranking model
        """
        super().__init__(name, config)

        if not isinstance(retriever, nn.Module) or not isinstance(reranker, nn.Module):
            raise TypeError("RetrievalThenRerank requires torch.nn.Module components; use RecommendationPipeline for candidate rankers")
        self.retriever = retriever
        self.reranker = reranker

        # Additional configuration
        self.num_candidates = config.get("num_candidates", 100)
        self.rerank_all = config.get("rerank_all", False)
        self.combined_score_weight = config.get("combined_score_weight", 0.5)
        self.use_combined_score = config.get("use_combined_score", False)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass of the retrieval-then-rerank model.

        Args:
            batch (Dict[str, Any]): Batch of data

        Returns:
            torch.Tensor: Scores for each item
        """
        return self._score_candidates(batch)[0]

    def _score_candidates(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keep the candidate set from this exact forward pass for recommendation.
        # Retrieve candidates
        retrieval_scores = self.retriever(batch)

        # Get top-k candidates
        if self.rerank_all:
            # Rerank all items
            candidate_indices = torch.arange(
                retrieval_scores.shape[1], device=retrieval_scores.device
            )
            candidate_indices = candidate_indices.expand(retrieval_scores.shape[0], -1)
        else:
            # Get top-k candidates
            _, candidate_indices = torch.topk(
                retrieval_scores, k=min(self.num_candidates, retrieval_scores.shape[1]), dim=1
            )

        # Prepare batch for reranker
        reranker_batch = self._prepare_reranker_batch(batch, candidate_indices)

        # Rerank candidates
        reranking_scores = self.reranker(reranker_batch)

        # Create final scores
        if self.use_combined_score:
            # Combine retrieval and reranking scores
            final_scores = torch.zeros_like(retrieval_scores)

            for i in range(retrieval_scores.shape[0]):
                # Get scores for this instance
                r_scores = retrieval_scores[i]
                rr_scores = reranking_scores[i]
                indices = candidate_indices[i]

                # Normalize scores
                r_scores = (r_scores - r_scores.min()) / (r_scores.max() - r_scores.min() + 1e-8)
                rr_scores = (rr_scores - rr_scores.min()) / (
                    rr_scores.max() - rr_scores.min() + 1e-8
                )

                # Combine scores
                combined_scores = (1 - self.combined_score_weight) * r_scores[
                    indices
                ] + self.combined_score_weight * rr_scores

                # Update final scores
                final_scores[i, indices] = combined_scores
        else:
            # Use only reranking scores
            final_scores = torch.zeros_like(retrieval_scores)

            for i in range(retrieval_scores.shape[0]):
                # Get scores for this instance
                rr_scores = reranking_scores[i]
                indices = candidate_indices[i]

                # Update final scores
                final_scores[i, indices] = rr_scores

        return final_scores, candidate_indices

    def _prepare_reranker_batch(
        self, batch: Dict[str, Any], candidate_indices: torch.Tensor
    ) -> Dict[str, Any]:
        """Prepare batch for reranker.

        Args:
            batch (Dict[str, Any]): Original batch
            candidate_indices (torch.Tensor): Indices of candidate items

        Returns:
            Dict[str, Any]: Batch for reranker
        """
        # Create a new batch for the reranker
        reranker_batch = {}

        # Copy user data
        for key, value in batch.items():
            if key.startswith("user_"):
                reranker_batch[key] = value

        # Copy item data for candidates
        for key, value in batch.items():
            if key.startswith("item_"):
                if isinstance(value, torch.Tensor) and value.dim() > 1 and value.shape[1] > 1:
                    # For item features/embeddings, select candidates
                    reranker_batch[key] = torch.stack(
                        [value[i, candidate_indices[i]] for i in range(value.shape[0])]
                    )

        # Copy other data
        for key, value in batch.items():
            if not key.startswith("user_") and not key.startswith("item_"):
                reranker_batch[key] = value

        # Add candidate indices
        reranker_batch["candidate_indices"] = candidate_indices

        return reranker_batch

    def recommend(
        self, user_data: Dict[str, Any], item_data: Dict[str, Any], top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Generate recommendations for a user.

        Args:
            user_data (Dict[str, Any]): User data
            item_data (Dict[str, Any]): Item data
            top_k (int): Number of recommendations to generate

        Returns:
            List[Tuple[int, float]]: List of recommended item IDs and scores
        """
        # Create batch
        batch = {**user_data, **item_data}

        # Forward pass
        scores, candidates = self._score_candidates(batch)

        # Only retrieved items are eligible, even when their scores are negative.
        candidate_scores = scores[0, candidates[0]]
        top_k_scores, positions = torch.topk(candidate_scores, k=min(top_k, candidates.shape[1]))
        top_k_indices = candidates[0, positions]

        # Convert to list of (id, score) tuples
        recommendations = []
        for i in range(len(top_k_indices)):
            idx = top_k_indices[i].item()
            score = top_k_scores[i].item()
            recommendations.append((idx, score))

        return recommendations

    def train_step(
        self, batch: Dict[str, Any], optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch (Dict[str, Any]): Batch of data
            optimizer (torch.optim.Optimizer): Optimizer instance

        Returns:
            Dict[str, float]: Dictionary with loss values
        """
        # Extract labels
        labels = batch.get("labels", None)
        if labels is None:
            self.logger.warning("No labels found in batch")
            return {"loss": 0.0}

        # Forward pass
        scores = self.forward(batch)

        # Compute loss
        retriever_loss = self.retriever.compute_loss(scores, labels)
        reranker_loss = self.reranker.compute_loss(scores, labels)

        # Combine losses
        retriever_weight = self.config.get("retriever_loss_weight", 0.5)
        loss = retriever_weight * retriever_loss + (1 - retriever_weight) * reranker_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
            "retriever_loss": retriever_loss.item(),
            "reranker_loss": reranker_loss.item(),
        }

    def validate_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single validation step.

        Args:
            batch (Dict[str, Any]): Batch of data

        Returns:
            Dict[str, float]: Dictionary with validation metrics
        """
        # Extract labels
        labels = batch.get("labels", None)
        if labels is None:
            self.logger.warning("No labels found in batch")
            return {"val_loss": 0.0}

        # Forward pass
        with torch.no_grad():
            scores = self.forward(batch)

            # Compute loss
            retriever_loss = self.retriever.compute_loss(scores, labels)
            reranker_loss = self.reranker.compute_loss(scores, labels)

            # Combine losses
            retriever_weight = self.config.get("retriever_loss_weight", 0.5)
            loss = retriever_weight * retriever_loss + (1 - retriever_weight) * reranker_loss

            # Compute metrics
            binary_preds = (torch.sigmoid(scores) > 0.5).float()
            accuracy = (binary_preds == labels).float().mean()

        return {
            "val_loss": loss.item(),
            "val_retriever_loss": retriever_loss.item(),
            "val_reranker_loss": reranker_loss.item(),
            "val_accuracy": accuracy.item(),
        }

    def save(self, path: str) -> str:
        """Save both neural components in one atomic safe bundle."""
        return str(save_bundle(
            path, model_class=type(self).__name__, config=self.config,
            state={"name": self.name, "training": self.training},
            state_dict=self.state_dict(),
        ))

    @classmethod
    def load(cls, path: str, *, retriever: Optional[nn.Module] = None,
             reranker: Optional[nn.Module] = None,
             allow_pickle: bool = False) -> "RetrievalThenRerank":
        """Restore weights into explicitly supplied neural architectures.

        Legacy pickle checkpoints require explicit trust. Component classes
        are never imported from checkpoint metadata.
        """
        if is_safe_bundle(path):
            checkpoint = load_bundle(path, map_location="cpu")
            name = checkpoint["state"]["name"]
            training = checkpoint["state"].get("training", True)
            weights = checkpoint["state_dict"]
        else:
            require_legacy_pickle(path, allow_pickle)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            name = checkpoint.get("model_name", checkpoint.get("name"))
            training = True
            weights = checkpoint.get("model_state_dict")
            if weights is None:
                raise SaveLoadError("This legacy hybrid checkpoint has no combined weights; save a new bundle from the original components")
        if retriever is None or reranker is None:
            raise ValueError("Supply retriever= and reranker= with the original neural architectures")
        model = cls(name=name, config=checkpoint["config"],
                    retriever=retriever, reranker=reranker)
        model.load_state_dict(weights)
        model.train(training)
        return model
