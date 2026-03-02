"""
Base classes for retrieval stage.

Retrievers are responsible for quickly narrowing down millions of items
to hundreds of candidates. Speed matters more than precision here.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class Candidate:
    """A single candidate item from retrieval."""

    #: unique identifier for the item
    item_id: Any
    #: retrieval score (higher = more relevant)
    score: float
    #: which retriever produced this candidate
    source: str = "unknown"
    #: optional extra info (embeddings, features, etc)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: "Candidate") -> bool:
        # for sorting - higher score = better
        return self.score < other.score
    
    def __repr__(self) -> str:
        return f"Candidate(id={self.item_id}, score={self.score:.4f}, src={self.source})"


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    #: list of retrieved candidates
    candidates: List[Candidate]
    #: identifier for the query (user_id, session_id, etc)
    query_id: Any = None
    #: name of the retriever that produced this
    retriever_name: str = "unknown"
    #: how long retrieval took in milliseconds
    timing_ms: float = 0.0
    
    def __len__(self) -> int:
        return len(self.candidates)
    
    def __iter__(self):
        return iter(self.candidates)
    
    def top_k(self, k: int) -> List[Candidate]:
        """Return top k candidates by score."""
        return sorted(self.candidates, reverse=True)[:k]
    
    def item_ids(self) -> List[Any]:
        """Extract just the item IDs."""
        return [c.item_id for c in self.candidates]
    
    def scores(self) -> np.ndarray:
        """Extract scores as numpy array."""
        return np.array([c.score for c in self.candidates])


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    
    A retriever takes a query (user, context, or text) and returns
    a set of candidate items. Subclasses implement different
    retrieval strategies: collaborative filtering, semantic search,
    popularity-based, etc.
    
    Design goals:
    - Fast: O(log n) or O(1) lookup, not O(n)
    - Recall-focused: better to include irrelevant than miss relevant
    - Composable: multiple retrievers can be ensembled
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._is_fitted = False
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @abstractmethod
    def fit(self, **kwargs) -> "BaseRetriever":
        """
        Fit the retriever on training data.
        
        What "fit" means depends on the retriever type:
        - CollaborativeRetriever: build item-item similarity matrix
        - SemanticRetriever: encode items into embeddings
        - PopularityRetriever: count item frequencies
        
        Returns self for method chaining.
        """
        pass
    
    @abstractmethod
    def retrieve(
        self,
        query: Any,
        top_k: int = 100,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve candidate items for a query.
        
        Args:
            query: the query to retrieve for. Can be:
                   - user_id for collaborative retrieval
                   - text string for semantic retrieval
                   - embedding vector for ANN search
            top_k: number of candidates to return
            **kwargs: retriever-specific parameters
        
        Returns:
            RetrievalResult containing ranked candidates
        """
        pass
    
    def retrieve_batch(
        self,
        queries: List[Any],
        top_k: int = 100,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve for multiple queries. Default impl just loops.
        Subclasses can override for batched efficiency.
        """
        return [self.retrieve(q, top_k, **kwargs) for q in queries]
    
    def _check_fitted(self) -> None:
        """Raise if not fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.name} must be fitted before retrieval. Call fit() first."
            )
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name={self.name}, {status})"
