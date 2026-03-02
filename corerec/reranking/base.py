"""
Base classes for reranking stage.

Rerankers modify the ranked list to satisfy constraints beyond
pure relevance: diversity, fairness, business rules, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from corerec.ranking.base import RankedCandidate, RankingResult


class BaseReranker(ABC):
    """
    Abstract base class for all rerankers.
    
    A reranker takes a ranked list and modifies it to satisfy
    additional constraints. Unlike rankers which focus on relevance,
    rerankers balance relevance with other objectives.
    
    Common reranking objectives:
    - Diversity: avoid showing too-similar items
    - Fairness: ensure exposure across item groups
    - Freshness: boost recently added items
    - Business: promote certain items, apply filters
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def rerank(
        self,
        ranked: Union[List[RankedCandidate], RankingResult],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RankingResult:
        """
        Rerank the candidates.
        
        Args:
            ranked: ranked candidates from Stage 2
            context: user context, constraints, etc.
            **kwargs: reranker-specific parameters
        
        Returns:
            RankingResult with reranked candidates
        """
        pass
    
    def _to_list(
        self,
        ranked: Union[List[RankedCandidate], RankingResult]
    ) -> List[RankedCandidate]:
        """Convert input to list."""
        if isinstance(ranked, RankingResult):
            return list(ranked.candidates)
        return list(ranked)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
