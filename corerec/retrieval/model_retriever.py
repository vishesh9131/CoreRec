"""
Base retriever class for CoreRec framework.

This module provides the BaseRetriever abstract class that all retrievers should inherit from.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
from abc import ABC, abstractmethod
import logging
import time

from corerec.core.base_model import BaseModel


class BaseRetriever(BaseModel, ABC):
    """
    Base class for all retrieval models in CoreRec.

    Retrievers are responsible for efficiently generating candidates from a large corpus.
    They typically encode queries and items into a common embedding space and perform
    efficient similarity search.

    Attributes:
        name (str): Name of the retriever
        config (Dict[str, Any]): Retriever configuration
        embedding_dim (int): Dimension of the embedding space
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the base retriever.

        Args:
            name (str): Name of the retriever
            config (Dict[str, Any]): Retriever configuration
        """
        super().__init__(name, config)
        self.embedding_dim = config.get("embedding_dim", 128)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def encode_query(self, query: Any) -> torch.Tensor:
        """Encode a query into the embedding space.

        Args:
            query (Any): Query to encode

        Returns:
            torch.Tensor: Query embedding
        """
        pass

    @abstractmethod
    def encode_item(self, item: Any) -> torch.Tensor:
        """Encode an item into the embedding space.

        Args:
            item (Any): Item to encode

        Returns:
            torch.Tensor: Item embedding
        """
        pass

    def score(self, query_embedding: torch.Tensor, item_embedding: torch.Tensor) -> torch.Tensor:
        """Score the similarity between query and item embeddings.

        Args:
            query_embedding (torch.Tensor): Query embedding
            item_embedding (torch.Tensor): Item embedding

        Returns:
            torch.Tensor: Similarity score
        """
        # Default scoring is dot product
        return torch.sum(query_embedding * item_embedding, dim=-1)

    def batch_score(
        self, query_embeddings: torch.Tensor, item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Score the similarity between batches of query and item embeddings.

        Args:
            query_embeddings (torch.Tensor): Query embeddings of shape [batch_size, embedding_dim]
            item_embeddings (torch.Tensor): Item embeddings of shape [num_items, embedding_dim]

        Returns:
            torch.Tensor: Similarity scores of shape [batch_size, num_items]
        """
        # Default batch scoring is matrix multiplication
        return torch.matmul(query_embeddings, item_embeddings.t())

    def retrieve(
        self, query: Any, item_embeddings: torch.Tensor, item_ids: List[Any], top_k: int = 10
    ) -> Tuple[List[Any], List[float]]:
        """Retrieve top-k items for a query.

        Args:
            query (Any): Query to retrieve items for
            item_embeddings (torch.Tensor): Item embeddings of shape [num_items, embedding_dim]
            item_ids (List[Any]): List of item IDs corresponding to item_embeddings
            top_k (int): Number of items to retrieve

        Returns:
            Tuple[List[Any], List[float]]: Tuple containing lists of item IDs and scores
        """
        # Encode query
        query_embedding = self.encode_query(query)

        # Score items
        scores = self.batch_score(query_embedding.unsqueeze(0), item_embeddings).squeeze(0)

        # Get top-k items
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(item_ids)))

        # Convert to lists
        top_item_ids = [item_ids[i] for i in top_indices.cpu().numpy()]
        top_item_scores = top_scores.cpu().numpy().tolist()

        return top_item_ids, top_item_scores

    def batch_retrieve(
        self,
        queries: List[Any],
        item_embeddings: torch.Tensor,
        item_ids: List[Any],
        top_k: int = 10,
    ) -> List[Tuple[List[Any], List[float]]]:
        """Retrieve top-k items for multiple queries.

        Args:
            queries (List[Any]): Queries to retrieve items for
            item_embeddings (torch.Tensor): Item embeddings of shape [num_items, embedding_dim]
            item_ids (List[Any]): List of item IDs corresponding to item_embeddings
            top_k (int): Number of items to retrieve per query

        Returns:
            List[Tuple[List[Any], List[float]]]: List of tuples containing lists of item IDs and scores
        """
        # Encode queries
        query_embeddings = torch.stack([self.encode_query(query) for query in queries])

        # Score items
        scores = self.batch_score(query_embeddings, item_embeddings)

        # Get top-k items for each query
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(item_ids)))

        # Convert to lists
        results = []
        for i in range(len(queries)):
            top_item_ids = [item_ids[j] for j in top_indices[i].cpu().numpy()]
            top_item_scores = top_scores[i].cpu().numpy().tolist()
            results.append((top_item_ids, top_item_scores))

        return results

    def index_items(self, items: List[Any]) -> torch.Tensor:
        """Index a list of items by encoding them.

        Args:
            items (List[Any]): List of items to index

        Returns:
            torch.Tensor: Item embeddings of shape [num_items, embedding_dim]
        """
        start_time = time.time()
        self.logger.info(f"Indexing {len(items)} items...")

        # Process in batches to avoid OOM
        batch_size = 128
        embeddings = []

        for i in range(0, len(items), batch_size):
            batch_items = items[i : i + batch_size]
            batch_embeddings = torch.stack([self.encode_item(item) for item in batch_items])
            embeddings.append(batch_embeddings)

        item_embeddings = torch.cat(embeddings, dim=0)

        elapsed = time.time() - start_time
        self.logger.info(f"Indexed {len(items)} items in {elapsed:.2f}s")

        return item_embeddings
