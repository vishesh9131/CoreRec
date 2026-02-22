"""
Collaborative Filtering based Retrieval

Wraps traditional CF algorithms (SAR, ALS, etc) as retrievers
for use in multi-stage pipelines.
"""

import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .base import BaseRetriever, Candidate, RetrievalResult


class CollaborativeRetriever(BaseRetriever):
    """
    Retriever based on collaborative filtering signals.
    
    This wraps any CF model that has a recommend() method, making it
    usable in retrieval pipelines. The CF model handles the actual
    similarity computation; this class just adapts the interface.
    
    Example:
        from corerec.collaborative import SAR
        from corerec.retrieval import CollaborativeRetriever
        
        sar = SAR(similarity_type='jaccard')
        sar.fit(train_df)
        
        retriever = CollaborativeRetriever(model=sar)
        candidates = retriever.retrieve(user_id=123, top_k=100)
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        name: str = "collaborative",
        exclude_seen: bool = True,
    ):
        """
        Args:
            model: a fitted CF model with recommend() method
            name: identifier for this retriever
            exclude_seen: whether to exclude items user has seen
        """
        super().__init__(name=name)
        self.model = model
        self.exclude_seen = exclude_seen
        
        # if model already fitted, we're ready
        if model is not None and getattr(model, 'is_fitted', False):
            self._is_fitted = True
    
    def fit(
        self,
        model: Optional[Any] = None,
        **fit_kwargs
    ) -> "CollaborativeRetriever":
        """
        Fit the underlying CF model, or attach a pre-fitted model.
        
        Args:
            model: CF model to use (if not set in __init__)
            **fit_kwargs: passed to model.fit() if model needs fitting
        """
        if model is not None:
            self.model = model
        
        if self.model is None:
            raise ValueError("No model provided. Pass model to __init__ or fit()")
        
        # fit if not already fitted
        if not getattr(self.model, 'is_fitted', False):
            if hasattr(self.model, 'fit'):
                self.model.fit(**fit_kwargs)
            else:
                raise ValueError("Model has no fit() method and is not fitted")
        
        self._is_fitted = True
        return self
    
    def retrieve(
        self,
        query: Any,
        top_k: int = 100,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve candidates for a user.
        
        Args:
            query: user_id to retrieve for
            top_k: number of candidates
        
        Returns:
            RetrievalResult with candidates
        """
        self._check_fitted()
        
        start = time.perf_counter()
        
        user_id = query
        
        # call the underlying model's recommend
        if hasattr(self.model, 'recommend'):
            # most models: recommend(user_id, top_k)
            try:
                items = self.model.recommend(
                    user_id=user_id,
                    top_k=top_k,
                    exclude_seen=self.exclude_seen if hasattr(self.model.recommend, 'exclude_seen') else None,
                    **{k: v for k, v in kwargs.items() if k != 'exclude_seen'}
                )
            except TypeError:
                # simpler signature
                items = self.model.recommend(user_id, top_k)
        else:
            raise ValueError("Model has no recommend() method")
        
        # convert to candidates
        # items might be list of ids or list of (id, score) tuples
        candidates = []
        for i, item in enumerate(items):
            if isinstance(item, tuple):
                item_id, score = item[0], item[1]
            else:
                item_id = item
                # use rank as score if no explicit score
                score = 1.0 / (i + 1)
            
            candidates.append(Candidate(
                item_id=item_id,
                score=float(score),
                source=self.name,
            ))
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return RetrievalResult(
            candidates=candidates,
            query_id=user_id,
            retriever_name=self.name,
            timing_ms=elapsed,
        )
    
    def retrieve_batch(
        self,
        queries: List[Any],
        top_k: int = 100,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Batch retrieval - uses model's batch method if available.
        """
        self._check_fitted()
        
        # check if model supports batch
        if hasattr(self.model, 'batch_recommend'):
            start = time.perf_counter()
            batch_results = self.model.batch_recommend(queries, top_k=top_k)
            elapsed = (time.perf_counter() - start) * 1000 / len(queries)
            
            results = []
            for user_id, items in batch_results.items():
                candidates = []
                for i, item in enumerate(items):
                    if isinstance(item, tuple):
                        item_id, score = item[0], item[1]
                    else:
                        item_id = item
                        score = 1.0 / (i + 1)
                    candidates.append(Candidate(
                        item_id=item_id,
                        score=float(score),
                        source=self.name,
                    ))
                results.append(RetrievalResult(
                    candidates=candidates,
                    query_id=user_id,
                    retriever_name=self.name,
                    timing_ms=elapsed,
                ))
            return results
        
        # fallback to sequential
        return super().retrieve_batch(queries, top_k, **kwargs)
