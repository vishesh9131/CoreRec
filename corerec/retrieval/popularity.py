"""
Popularity-based Retrieval

Simple baseline that returns most popular items.
Useful for cold-start users and as a component in ensembles.
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np

from .base import BaseRetriever, Candidate, RetrievalResult


class PopularityRetriever(BaseRetriever):
    """
    Retriever that returns most popular items.
    
    "Popular" can mean different things:
    - Most interactions (views, clicks, purchases)
    - Highest average rating
    - Most recent trending
    
    This is a simple but effective baseline, especially for:
    - Cold-start users with no history
    - Fallback when other retrievers fail
    - Diversity injection in ensembles
    
    Example:
        retriever = PopularityRetriever()
        retriever.fit(item_ids, interaction_counts)
        candidates = retriever.retrieve(user_id=None, top_k=50)
    """
    
    def __init__(
        self,
        name: str = "popularity",
        time_decay: Optional[float] = None,
    ):
        """
        Args:
            name: identifier for this retriever
            time_decay: if set, apply exponential decay based on recency
        """
        super().__init__(name=name)
        self.time_decay = time_decay
        
        # populated by fit()
        self.item_ids: List[Any] = []
        self.popularity_scores: np.ndarray = np.array([])
        self._sorted_indices: np.ndarray = np.array([])
    
    def fit(
        self,
        item_ids: List[Any],
        scores: Optional[List[float]] = None,
        interaction_counts: Optional[List[int]] = None,
        timestamps: Optional[List[float]] = None,
        **kwargs
    ) -> "PopularityRetriever":
        """
        Compute popularity scores for items.
        
        Args:
            item_ids: unique identifiers for items
            scores: pre-computed popularity scores (if available)
            interaction_counts: raw interaction counts to use as popularity
            timestamps: if provided with time_decay, applies recency weighting
        
        Provide either scores or interaction_counts.
        """
        item_ids = list(item_ids)
        arrays = {}
        for name, values in (("scores", scores), ("interaction_counts", interaction_counts),
                             ("timestamps", timestamps)):
            if values is not None:
                array = np.asarray(values, dtype=float)
                if array.ndim != 1 or len(array) != len(item_ids):
                    raise ValueError(f"{name} must be one-dimensional with one value per item")
                if not np.isfinite(array).all():
                    raise ValueError(f"{name} must contain only finite values")
                arrays[name] = array

        popularity_scores = arrays.get("scores", arrays.get("interaction_counts"))
        if popularity_scores is None:
            popularity_scores = np.ones(len(item_ids))

        if self.time_decay is not None:
            if not np.isfinite(self.time_decay) or self.time_decay < 0:
                raise ValueError("time_decay must be finite and non-negative")
            if "timestamps" in arrays and item_ids:
                ts = arrays["timestamps"]
                with np.errstate(over="ignore"):
                    decay = np.exp(-self.time_decay * (ts.max() - ts))
                popularity_scores = popularity_scores * decay

        # Publish fitted state only after every input has been validated.
        self.item_ids = item_ids
        self.popularity_scores = popularity_scores
        self._sorted_indices = np.argsort(popularity_scores)[::-1]

        self._is_fitted = True
        return self
    
    def retrieve(
        self,
        query: Any = None,
        top_k: int = 100,
        exclude_items: Optional[List[Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve most popular items.
        
        Args:
            query: ignored (popularity is query-independent)
            top_k: number of items to return
            exclude_items: items to exclude from results
        
        Returns:
            RetrievalResult with most popular items
        """
        self._check_fitted()
        
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return RetrievalResult(candidates=[], query_id=query, retriever_name=self.name)

        start = time.perf_counter()
        
        exclude_set = set(exclude_items) if exclude_items else set()
        
        candidates = []
        for idx in self._sorted_indices:
            item_id = self.item_ids[idx]
            if item_id in exclude_set:
                continue
            
            candidates.append(Candidate(
                item_id=item_id,
                score=float(self.popularity_scores[idx]),
                source=self.name,
            ))
            
            if len(candidates) >= top_k:
                break
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return RetrievalResult(
            candidates=candidates,
            query_id=query,
            retriever_name=self.name,
            timing_ms=elapsed,
        )
    
    def get_item_popularity(self, item_id: Any) -> float:
        """Get popularity score for a specific item."""
        try:
            idx = self.item_ids.index(item_id)
            return float(self.popularity_scores[idx])
        except ValueError:
            return 0.0
