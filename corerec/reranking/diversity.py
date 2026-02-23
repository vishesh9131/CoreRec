"""
Diversity Reranking

Prevents recommendation lists from being too homogeneous.
Users get bored seeing the same type of items repeatedly.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from corerec.ranking.base import RankedCandidate, RankingResult
from .base import BaseReranker


class DiversityReranker(BaseReranker):
    """
    Reranks to maximize diversity while preserving relevance.
    
    Uses Maximal Marginal Relevance (MMR) or similar algorithms to
    iteratively select items that are both relevant AND different
    from already selected items.
    
    Trade-off controlled by ``lambda_``:
    - ``lambda_`` = 1.0 -- pure relevance (no diversity)
    - ``lambda_`` = 0.0 -- pure diversity (ignore relevance)
    - ``lambda_`` = 0.5 -- balanced
    
    Example::

        reranker = DiversityReranker(
            lambda_=0.7,
            similarity_fn=cosine_similarity,  # how to measure item similarity
        )
        diverse_recs = reranker.rerank(ranked_candidates)
    """
    
    def __init__(
        self,
        lambda_: float = 0.7,
        similarity_fn: Optional[Callable[[Any, Any], float]] = None,
        category_key: Optional[str] = None,
        name: str = "diversity",
    ):
        """
        Args:
            lambda_: trade-off between relevance (1) and diversity (0)
            similarity_fn: function(item_a, item_b) -> similarity score
            category_key: if set, diversify by this categorical attribute
            name: identifier for this reranker
        """
        super().__init__(name=name)
        
        self.lambda_ = lambda_
        self.similarity_fn = similarity_fn
        self.category_key = category_key
    
    def rerank(
        self,
        ranked: Union[List[RankedCandidate], RankingResult],
        context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RankingResult:
        """
        Rerank for diversity using MMR.
        
        Args:
            ranked: candidates sorted by relevance
            context: optional context (unused by default)
            top_k: number of items to return (default: all)
        """
        start = time.perf_counter()
        
        candidates = self._to_list(ranked)
        n = len(candidates)
        
        if n == 0:
            return RankingResult(candidates=[], ranker_name=self.name)
        
        top_k = top_k or n
        
        # mmr selection
        selected = []
        remaining = list(range(n))
        
        # always pick the top item first
        selected.append(remaining.pop(0))
        
        while len(selected) < top_k and remaining:
            best_idx = None
            best_score = float('-inf')
            
            for idx in remaining:
                # relevance component (normalized rank score)
                relevance = candidates[idx].score
                
                # diversity component (dissimilarity to selected)
                max_sim = 0.0
                for sel_idx in selected:
                    sim = self._similarity(candidates[idx], candidates[sel_idx])
                    max_sim = max(max_sim, sim)
                
                diversity = 1.0 - max_sim
                
                # MMR score
                mmr = self.lambda_ * relevance + (1 - self.lambda_) * diversity
                
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            
            if best_idx is not None:
                remaining.remove(best_idx)
                selected.append(best_idx)
        
        # build result
        reranked = []
        for rank, idx in enumerate(selected, 1):
            rc = candidates[idx]
            reranked.append(RankedCandidate(
                item_id=rc.item_id,
                score=rc.score,
                retrieval_score=rc.retrieval_score,
                rank=rank,
                features=rc.features,
            ))
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return RankingResult(
            candidates=reranked,
            ranker_name=self.name,
            timing_ms=elapsed,
        )
    
    def _similarity(self, a: RankedCandidate, b: RankedCandidate) -> float:
        """Compute similarity between two candidates."""
        # use provided similarity function
        if self.similarity_fn is not None:
            return self.similarity_fn(a.item_id, b.item_id)
        
        # use category if specified
        if self.category_key is not None:
            cat_a = a.features.get(self.category_key)
            cat_b = b.features.get(self.category_key)
            return 1.0 if cat_a == cat_b else 0.0
        
        # fallback: items are dissimilar by default
        return 0.0


class CategorySpreadReranker(BaseReranker):
    """
    Simple diversity by spreading categories.
    
    Ensures no more than max_per_category consecutive items
    from the same category. Simpler than MMR, faster too.
    
    Example::

        reranker = CategorySpreadReranker(
            category_fn=lambda item: item_categories[item],
            max_consecutive=2
        )
    """
    
    def __init__(
        self,
        category_fn: Callable[[Any], Any],
        max_consecutive: int = 2,
        name: str = "category_spread",
    ):
        """
        Args:
            category_fn: function(item_id) -> category
            max_consecutive: max items from same category in a row
            name: identifier
        """
        super().__init__(name=name)
        
        self.category_fn = category_fn
        self.max_consecutive = max_consecutive
    
    def rerank(
        self,
        ranked: Union[List[RankedCandidate], RankingResult],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RankingResult:
        """Rerank to spread categories."""
        start = time.perf_counter()
        
        candidates = self._to_list(ranked)
        
        if not candidates:
            return RankingResult(candidates=[], ranker_name=self.name)
        
        # greedy reordering
        result = []
        remaining = list(candidates)
        recent_categories = []
        
        while remaining:
            # find best candidate that doesn't violate constraint
            for i, c in enumerate(remaining):
                cat = self.category_fn(c.item_id)
                
                # check if adding this would violate max_consecutive
                if len(recent_categories) >= self.max_consecutive:
                    if all(rc == cat for rc in recent_categories[-self.max_consecutive:]):
                        continue  # skip, would be too many in a row
                
                # this one is ok
                result.append(c)
                recent_categories.append(cat)
                remaining.pop(i)
                break
            else:
                # couldn't find a valid one, just take the top remaining
                result.append(remaining.pop(0))
                recent_categories.append(self.category_fn(result[-1].item_id))
        
        # update ranks
        for i, rc in enumerate(result):
            rc.rank = i + 1
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return RankingResult(
            candidates=result,
            ranker_name=self.name,
            timing_ms=elapsed,
        )
