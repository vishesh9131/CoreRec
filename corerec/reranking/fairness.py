"""
Fairness Reranking

Ensures fair exposure across item groups (e.g., sellers, brands).
Prevents popularity bias from completely dominating recommendations.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from corerec.ranking.base import RankedCandidate, RankingResult
from .base import BaseReranker


class FairnessReranker(BaseReranker):
    """
    Reranks to ensure fair exposure across item groups.
    
    Without intervention, popular items/sellers dominate recommendations.
    This reranker adjusts exposure to be more equitable while still
    respecting relevance.
    
    Fairness objectives:
    - ``proportional`` -- exposure proportional to group size
    - ``equal`` -- equal exposure across groups
    - ``min_exposure`` -- ensure minimum exposure per group
    
    Example::

        reranker = FairnessReranker(
            group_fn=lambda item: item_to_seller[item],
            objective='proportional',
            fairness_weight=0.3
        )
        fair_recs = reranker.rerank(ranked_candidates)
    """
    
    def __init__(
        self,
        group_fn: Callable[[Any], Any],
        objective: str = "proportional",
        fairness_weight: float = 0.3,
        group_targets: Optional[Dict[Any, float]] = None,
        name: str = "fairness",
    ):
        """
        Args:
            group_fn: function(item_id) -> group_id
            objective: 'proportional', 'equal', or 'min_exposure'
            fairness_weight: how much to weight fairness vs relevance
            group_targets: explicit target exposure per group
            name: identifier for this reranker
        """
        super().__init__(name=name)
        
        self.group_fn = group_fn
        self.objective = objective
        self.fairness_weight = fairness_weight
        self.group_targets = group_targets or {}
    
    def rerank(
        self,
        ranked: Union[List[RankedCandidate], RankingResult],
        context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RankingResult:
        """
        Rerank for fairness.
        
        Uses a greedy algorithm that balances relevance with
        exposure fairness.
        """
        start = time.perf_counter()
        
        candidates = self._to_list(ranked)
        n = len(candidates)
        
        if n == 0:
            return RankingResult(candidates=[], ranker_name=self.name)
        
        top_k = top_k or n
        
        # compute group memberships
        groups = [self.group_fn(c.item_id) for c in candidates]
        unique_groups = list(set(groups))
        
        # compute target exposure per group
        if self.group_targets:
            targets = self.group_targets
        elif self.objective == "equal":
            targets = {g: 1.0 / len(unique_groups) for g in unique_groups}
        else:  # proportional
            group_counts = {}
            for g in groups:
                group_counts[g] = group_counts.get(g, 0) + 1
            total = sum(group_counts.values())
            targets = {g: c / total for g, c in group_counts.items()}
        
        # greedy selection with fairness adjustment
        selected = []
        remaining = list(range(n))
        group_exposure = {g: 0.0 for g in unique_groups}
        
        for position in range(min(top_k, n)):
            best_idx = None
            best_score = float('-inf')
            
            # position-weighted exposure (earlier = more exposure)
            position_weight = 1.0 / (position + 1)
            
            for idx in remaining:
                c = candidates[idx]
                g = groups[idx]
                
                # relevance component
                relevance = c.score
                
                # fairness component: how much does this group need exposure?
                current_exposure = group_exposure[g]
                target_exposure = targets.get(g, 1.0 / len(unique_groups))
                
                # boost underexposed groups
                exposure_gap = target_exposure - current_exposure / (position + 1)
                fairness_bonus = max(0, exposure_gap)
                
                # combined score
                score = (
                    (1 - self.fairness_weight) * relevance + 
                    self.fairness_weight * fairness_bonus
                )
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None:
                remaining.remove(best_idx)
                selected.append(best_idx)
                group_exposure[groups[best_idx]] += position_weight
        
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
