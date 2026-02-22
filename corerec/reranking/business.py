"""
Business Rules Reranking

Apply custom business logic: promotions, filtering, boosting, etc.
This is where domain-specific rules live.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Set, Union

from corerec.ranking.base import RankedCandidate, RankingResult
from .base import BaseReranker


class BusinessRulesReranker(BaseReranker):
    """
    Apply configurable business rules to rerank items.
    
    Rules can:
    - Boost specific items (promotions)
    - Filter out items (blocklist)
    - Pin items to positions (sponsored)
    - Apply custom transformations
    
    Example:
        reranker = BusinessRulesReranker()
        reranker.add_boost(item_id=123, multiplier=2.0)
        reranker.add_filter(lambda item: item not in blocked_items)
        reranker.add_pin(item_id=456, position=1)  # sponsored slot
        
        result = reranker.rerank(ranked_candidates)
    """
    
    def __init__(self, name: str = "business_rules"):
        super().__init__(name=name)
        
        self._boosts: Dict[Any, float] = {}
        self._filters: List[Callable[[Any], bool]] = []
        self._pins: Dict[int, Any] = {}  # position -> item_id
        self._blocklist: Set[Any] = set()
    
    def add_boost(self, item_id: Any, multiplier: float) -> "BusinessRulesReranker":
        """
        Boost an item's score by a multiplier.
        
        Args:
            item_id: item to boost
            multiplier: score multiplier (>1 = boost, <1 = demote)
        """
        self._boosts[item_id] = multiplier
        return self
    
    def add_filter(self, filter_fn: Callable[[Any], bool]) -> "BusinessRulesReranker":
        """
        Add a filter function.
        
        Items where filter_fn(item_id) returns False are removed.
        
        Args:
            filter_fn: function(item_id) -> keep (True/False)
        """
        self._filters.append(filter_fn)
        return self
    
    def add_pin(self, item_id: Any, position: int) -> "BusinessRulesReranker":
        """
        Pin an item to a specific position.
        
        Useful for sponsored/promoted slots.
        
        Args:
            item_id: item to pin
            position: 1-indexed position to pin to
        """
        self._pins[position] = item_id
        return self
    
    def add_blocklist(self, item_ids: List[Any]) -> "BusinessRulesReranker":
        """
        Block items from appearing in results.
        
        Args:
            item_ids: items to block
        """
        self._blocklist.update(item_ids)
        return self
    
    def clear_rules(self) -> "BusinessRulesReranker":
        """Clear all rules."""
        self._boosts.clear()
        self._filters.clear()
        self._pins.clear()
        self._blocklist.clear()
        return self
    
    def rerank(
        self,
        ranked: Union[List[RankedCandidate], RankingResult],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RankingResult:
        """
        Apply business rules and rerank.
        """
        start = time.perf_counter()
        
        candidates = self._to_list(ranked)
        
        # step 1: filter
        filtered = []
        for c in candidates:
            # check blocklist
            if c.item_id in self._blocklist:
                continue
            
            # check filter functions
            if not all(f(c.item_id) for f in self._filters):
                continue
            
            filtered.append(c)
        
        # step 2: apply boosts
        for c in filtered:
            if c.item_id in self._boosts:
                c.score *= self._boosts[c.item_id]
        
        # step 3: sort by (potentially boosted) score
        filtered.sort(key=lambda x: x.score, reverse=True)
        
        # step 4: apply pins
        # extract pinned items from the list
        pinned_items = {}
        for pos, item_id in self._pins.items():
            # find the candidate with this item_id
            for i, c in enumerate(filtered):
                if c.item_id == item_id:
                    pinned_items[pos] = filtered.pop(i)
                    break
        
        # insert pinned items at their positions
        result = []
        filtered_iter = iter(filtered)
        
        for pos in range(1, len(filtered) + len(pinned_items) + 1):
            if pos in pinned_items:
                result.append(pinned_items[pos])
            else:
                try:
                    result.append(next(filtered_iter))
                except StopIteration:
                    break
        
        # step 5: update ranks
        for i, rc in enumerate(result):
            rc.rank = i + 1
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return RankingResult(
            candidates=result,
            ranker_name=self.name,
            timing_ms=elapsed,
        )
