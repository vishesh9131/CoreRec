"""
Base classes for ranking stage.

Rankers score candidates retrieved in Stage 1. They can use
richer features and more complex models since they only
process hundreds of items, not millions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np

from corerec.retrieval.base import Candidate, RetrievalResult


@dataclass
class RankedCandidate:
    """A candidate with ranking score and optional auxiliary predictions."""

    #: unique identifier
    item_id: Any
    #: ranking score (higher = better)
    score: float
    #: original score from retrieval
    retrieval_score: float = 0.0
    #: position in ranked list (1-indexed)
    rank: int = 0
    #: multi-task predictions (e.g., p(click), p(purchase))
    predictions: Dict[str, float] = field(default_factory=dict)
    #: feature values used for scoring
    features: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: "RankedCandidate") -> bool:
        return self.score < other.score
    
    def __repr__(self) -> str:
        return f"RankedCandidate(id={self.item_id}, score={self.score:.4f}, rank={self.rank})"


@dataclass
class RankingResult:
    """Output from ranking operation."""

    #: ranked list (best first)
    candidates: List[RankedCandidate]
    #: identifier for the query
    query_id: Any = None
    #: which ranker produced this
    ranker_name: str = "unknown"
    #: ranking latency
    timing_ms: float = 0.0
    
    def __len__(self) -> int:
        return len(self.candidates)
    
    def __iter__(self):
        return iter(self.candidates)
    
    def top_k(self, k: int) -> List[RankedCandidate]:
        """Return top k candidates."""
        return self.candidates[:k]
    
    def item_ids(self) -> List[Any]:
        """Extract item IDs in ranked order."""
        return [c.item_id for c in self.candidates]
    
    def scores(self) -> np.ndarray:
        """Extract scores as numpy array."""
        return np.array([c.score for c in self.candidates])


class BaseRanker(ABC):
    """
    Abstract base class for all rankers.
    
    A ranker takes a list of candidates (from retrieval) and a context
    (user features, session info, etc) and produces a ranked list.
    
    Design goals:
    - Precision-focused: quality over speed
    - Feature-rich: can use detailed user/item features
    - Multi-objective: can predict multiple targets
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._is_fitted = False
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @abstractmethod
    def fit(self, **kwargs) -> "BaseRanker":
        """
        Train or configure the ranker.
        
        What "fit" means depends on the ranker:
        - PointwiseRanker: train a scoring model
        - FeatureCrossRanker: configure feature interactions
        - PairwiseRanker: train pairwise preference model
        
        Returns self for method chaining.
        """
        pass
    
    @abstractmethod
    def rank(
        self,
        candidates: Union[List[Candidate], RetrievalResult],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RankingResult:
        """
        Rank candidates for a given context.
        
        Args:
            candidates: list of Candidate or RetrievalResult from retrieval
            context: user features, session info, etc.
            **kwargs: ranker-specific parameters
        
        Returns:
            RankingResult with candidates sorted by score
        """
        pass
    
    def rank_batch(
        self,
        candidate_batches: List[Union[List[Candidate], RetrievalResult]],
        contexts: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[RankingResult]:
        """
        Batch ranking. Default implementation just loops.
        Subclasses can override for efficiency.
        """
        if contexts is None:
            contexts = [None] * len(candidate_batches)
        
        return [
            self.rank(candidates, context, **kwargs)
            for candidates, context in zip(candidate_batches, contexts)
        ]
    
    def _candidates_to_list(
        self,
        candidates: Union[List[Candidate], RetrievalResult]
    ) -> List[Candidate]:
        """Helper to normalize input."""
        if isinstance(candidates, RetrievalResult):
            return candidates.candidates
        return candidates
    
    def _check_fitted(self) -> None:
        """Raise if not fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.name} must be fitted before ranking. Call fit() first."
            )

    @staticmethod
    def _is_recommender(model: Any) -> bool:
        """Is this a CoreRec recommender rather than an sklearn-style estimator?

        The two use incompatible predict() conventions -- a recommender takes
        predict(user_id, item_id), an estimator takes predict(X) -- and the
        rankers were written for the estimator one only, so passing a CoreRec
        model to a ranker raised TypeError. Dispatch on the type instead.
        """
        from corerec.api.base_recommender import BaseRecommender

        return isinstance(model, BaseRecommender)

    def _score_recommender(self, model: Any, item_id: Any,
                           context: Dict[str, Any]) -> float:
        """Score one item with a CoreRec recommender.

        Needs the user, which lives in the ranking context rather than in the
        feature dict -- pass context={"user_id": ...} to rank().
        """
        user_id = (context or {}).get("user_id")
        if user_id is None:
            raise ValueError(
                f"{self.name} was given a CoreRec model, which scores a "
                "(user, item) pair, but no user was supplied. Call "
                'rank(candidates, context={"user_id": ...}).'
            )
        try:
            return float(model.predict(user_id, item_id))
        except Exception:
            # Unknown user or item: fall back to the retrieval score's ordering
            # rather than failing the whole request over one candidate.
            return 0.0
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name={self.name}, {status})"
