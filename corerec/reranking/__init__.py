"""
Reranking Module - Post-processing (Stage 3)

Rerankers apply business logic and constraints after scoring.
They adjust the ranked list for diversity, fairness, freshness,
or other non-relevance objectives.

Usage:
    from corerec.reranking import DiversityReranker, FairnessReranker
    
    reranker = DiversityReranker(lambda_=0.3)
    final = reranker.rerank(ranked_candidates)
"""

from .base import BaseReranker
from .diversity import DiversityReranker
from .fairness import FairnessReranker
from .business import BusinessRulesReranker

__all__ = [
    "BaseReranker",
    "DiversityReranker",
    "FairnessReranker",
    "BusinessRulesReranker",
]
