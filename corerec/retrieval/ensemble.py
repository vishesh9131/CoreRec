"""
Ensemble Retriever - Combines Multiple Retrieval Sources

In production systems, using multiple retrieval channels improves
both coverage and quality. This module provides strategies for
merging results from different retrievers.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from .base import BaseRetriever, Candidate, RetrievalResult


class EnsembleRetriever(BaseRetriever):
    """
    Combines candidates from multiple retrievers.
    
    Different retrievers capture different signals:
    - Collaborative: "users like you bought X"
    - Semantic: "X matches your query meaning"
    - Popularity: "X is trending"
    
    Ensembling these improves recall and handles cases where
    one retriever might fail (cold start, sparse history, etc).
    
    Fusion strategies:
    - ``union`` -- combine all candidates, dedupe, keep max score
    - ``rrf`` -- Reciprocal Rank Fusion (position-based, score-agnostic)
    - ``weighted`` -- weighted sum of normalized scores
    
    Example::

        ensemble = EnsembleRetriever(
            retrievers=[
                ("collab", collaborative_retriever, 1.0),
                ("semantic", semantic_retriever, 0.5),
                ("popular", popularity_retriever, 0.3),
            ],
            strategy="rrf"
        )
        ensemble.fit()  # fits underlying retrievers if needed
        candidates = ensemble.retrieve(query, top_k=100)
    """
    
    def __init__(
        self,
        retrievers: List[Tuple[str, BaseRetriever, float]],
        strategy: str = "rrf",
        k_per_retriever: Optional[int] = None,
        name: str = "ensemble",
    ):
        """
        Args:
            retrievers: list of (name, retriever, weight) tuples
            strategy: 'union', 'rrf', or 'weighted'
            k_per_retriever: candidates to get from each retriever
                            (defaults to 2x final top_k)
            name: identifier for this retriever
        """
        super().__init__(name=name)
        
        self.retrievers = retrievers
        self.strategy = strategy
        self.k_per_retriever = k_per_retriever
        
        # validate strategy
        valid = {"union", "rrf", "weighted"}
        if strategy not in valid:
            raise ValueError(f"strategy must be one of {valid}")
    
    def fit(self, **kwargs) -> "EnsembleRetriever":
        """
        Fit all underlying retrievers.
        
        Pass retriever-specific kwargs as retriever_name__param.
        e.g., collab__model=sar_model
        """
        for name, retriever, _ in self.retrievers:
            if not retriever.is_fitted:
                # extract kwargs for this retriever
                prefix = f"{name}__"
                retriever_kwargs = {
                    k[len(prefix):]: v 
                    for k, v in kwargs.items() 
                    if k.startswith(prefix)
                }
                retriever.fit(**retriever_kwargs)
        
        self._is_fitted = True
        return self
    
    def retrieve(
        self,
        query: Any,
        top_k: int = 100,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve and merge candidates from all sources.
        """
        # all retrievers must be fitted (either externally or via our fit())
        for name, retriever, _ in self.retrievers:
            if not retriever.is_fitted:
                raise RuntimeError(
                    f"Retriever '{name}' is not fitted. "
                    f"Call ensemble.fit() or fit the retriever first."
                )
        
        self._is_fitted = True  # mark ourselves as ready
        
        start = time.perf_counter()
        
        k_each = self.k_per_retriever or (top_k * 2)
        
        # gather candidates from each retriever
        all_results: List[Tuple[str, float, RetrievalResult]] = []
        for name, retriever, weight in self.retrievers:
            try:
                result = retriever.retrieve(query, top_k=k_each, **kwargs)
                all_results.append((name, weight, result))
            except Exception as e:
                # one retriever failing shouldn't kill the ensemble
                # in production you'd log this
                pass
        
        if not all_results:
            return RetrievalResult(
                candidates=[],
                query_id=query,
                retriever_name=self.name,
                timing_ms=0,
            )
        
        # merge using selected strategy
        if self.strategy == "union":
            merged = self._merge_union(all_results)
        elif self.strategy == "rrf":
            merged = self._merge_rrf(all_results)
        else:  # weighted
            merged = self._merge_weighted(all_results)
        
        # sort and truncate
        merged.sort(key=lambda c: c.score, reverse=True)
        candidates = merged[:top_k]
        
        # update source to show ensemble
        for c in candidates:
            c.source = f"{self.name}({c.source})"
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return RetrievalResult(
            candidates=candidates,
            query_id=query,
            retriever_name=self.name,
            timing_ms=elapsed,
        )
    
    def _merge_union(
        self, 
        results: List[Tuple[str, float, RetrievalResult]]
    ) -> List[Candidate]:
        """Simple union - keep candidate with highest score."""
        item_to_candidate: Dict[Any, Candidate] = {}
        
        for name, weight, result in results:
            for c in result.candidates:
                existing = item_to_candidate.get(c.item_id)
                if existing is None or c.score > existing.score:
                    item_to_candidate[c.item_id] = Candidate(
                        item_id=c.item_id,
                        score=c.score * weight,
                        source=name,
                        metadata=c.metadata,
                    )
        
        return list(item_to_candidate.values())
    
    def _merge_rrf(
        self,
        results: List[Tuple[str, float, RetrievalResult]],
        k: int = 60,
    ) -> List[Candidate]:
        """
        Reciprocal Rank Fusion.
        
        Score = sum over retrievers of: weight / (k + rank)
        
        This is score-agnostic and works well when different retrievers
        have incompatible score scales.
        """
        item_scores: Dict[Any, float] = {}
        item_sources: Dict[Any, str] = {}
        
        for name, weight, result in results:
            for rank, c in enumerate(result.candidates):
                rrf_score = weight / (k + rank + 1)  # +1 for 0-indexed
                
                if c.item_id not in item_scores:
                    item_scores[c.item_id] = 0.0
                    item_sources[c.item_id] = name
                
                item_scores[c.item_id] += rrf_score
                
                # track primary source (highest contribution)
                if rrf_score > weight / (k + 1) * 0.5:
                    item_sources[c.item_id] = name
        
        return [
            Candidate(
                item_id=item_id,
                score=score,
                source=item_sources[item_id],
            )
            for item_id, score in item_scores.items()
        ]
    
    def _merge_weighted(
        self,
        results: List[Tuple[str, float, RetrievalResult]]
    ) -> List[Candidate]:
        """
        Weighted sum of normalized scores.
        
        Normalizes each retriever's scores to [0, 1] then takes
        weighted sum. Good when scores are comparable across retrievers.
        """
        item_scores: Dict[Any, float] = {}
        item_sources: Dict[Any, str] = {}
        
        for name, weight, result in results:
            if not result.candidates:
                continue
            
            # normalize scores for this retriever
            scores = np.array([c.score for c in result.candidates])
            min_s, max_s = scores.min(), scores.max()
            
            if max_s > min_s:
                norm_scores = (scores - min_s) / (max_s - min_s)
            else:
                norm_scores = np.ones_like(scores) * 0.5
            
            for c, norm_score in zip(result.candidates, norm_scores):
                if c.item_id not in item_scores:
                    item_scores[c.item_id] = 0.0
                    item_sources[c.item_id] = name
                
                item_scores[c.item_id] += weight * norm_score
        
        return [
            Candidate(
                item_id=item_id,
                score=score,
                source=item_sources[item_id],
            )
            for item_id, score in item_scores.items()
        ]
