"""
Pipelines module for modern recommendation systems.

Provides multi-stage pipeline architecture for production RecSys.
"""

try:
    from corerec.pipelines.recommendation_pipeline import (
        RecommendationPipeline,
        PipelineStage,
        RetrievalStage,
        RankingStage,
        RerankingStage,
        DiversityRule,
        FreshnessRule,
    )
except ImportError:
    RecommendationPipeline = None
    PipelineStage = None
    RetrievalStage = None
    RankingStage = None
    RerankingStage = None
    DiversityRule = None
    FreshnessRule = None

__all__ = [
    "RecommendationPipeline",
    "PipelineStage",
    "RetrievalStage",
    "RankingStage",
    "RerankingStage",
    "DiversityRule",
    "FreshnessRule",
]
