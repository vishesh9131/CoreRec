"""
Ranking Module - Candidate Scoring (Stage 2)

Rankers take candidates from retrieval and score them with
more sophisticated (and expensive) models. Unlike retrieval
which prioritizes speed, ranking prioritizes precision.

Usage:
    from corerec.ranking import PointwiseRanker, CrossEncoderRanker
    
    ranker = PointwiseRanker(model=your_model)
    ranked = ranker.rank(candidates, user_context)
"""

from .base import BaseRanker, RankedCandidate, RankingResult
from .pointwise import PointwiseRanker
from .pairwise import PairwiseRanker
from .feature_cross import FeatureCrossRanker

__all__ = [
    "BaseRanker",
    "RankedCandidate",
    "RankingResult",
    "PointwiseRanker",
    "PairwiseRanker",
    "FeatureCrossRanker",
]
