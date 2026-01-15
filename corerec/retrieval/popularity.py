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
        self.item_ids = list(item_ids)
        
        if scores is not None:
            self.popularity_scores = np.asarray(scores, dtype=float)
        elif interaction_counts is not None:
            self.popularity_scores = np.asarray(interaction_counts, dtype=float)
        else:
            # default: uniform popularity
            self.popularity_scores = np.ones(len(item_ids))
        
        # apply time decay if configured
        if self.time_decay is not None and timestamps is not None:
            ts = np.asarray(timestamps)
            max_ts = ts.max()
            decay = np.exp(-self.time_decay * (max_ts - ts))
            self.popularity_scores = self.popularity_scores * decay
        
        # pre-sort for fast retrieval
        self._sorted_indices = np.argsort(self.popularity_scores)[::-1]
        
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
