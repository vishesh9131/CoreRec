"""
Pipelines module for modern recommendation systems.

Provides multi-stage pipeline architecture for production RecSys:
- RecommendationPipeline: Full orchestrator integrating retrieval/ranking/reranking
- StagePipeline: Simpler stage-based pipeline for custom stage chains
- PipelineStage: Abstract base for building custom pipeline stages
- Data pipeline and feature transformer utilities
"""

# Main orchestrator (production-ready, integrates with retrieval/ranking/reranking modules)
try:
    from corerec.pipelines.orchestrator import (
        RecommendationPipeline,
        PipelineConfig,
        PipelineResult,
    )
except ImportError:
    RecommendationPipeline = None
    PipelineConfig = None
    PipelineResult = None

# Config loading (YAML/JSON)
try:
    from corerec.pipelines.config import (
        load_pipeline_config,
        build_pipeline_from_config,
    )
except ImportError:
    load_pipeline_config = None
    build_pipeline_from_config = None

# Stage abstractions (for building custom pipelines)
try:
    from corerec.pipelines.recommendation_pipeline import (
        StagePipeline,
        PipelineStage,
        RetrievalStage,
        RankingStage,
        RerankingStage,
        DiversityRule,
        FreshnessRule,
    )
except ImportError:
    StagePipeline = None
    PipelineStage = None
    RetrievalStage = None
    RankingStage = None
    RerankingStage = None
    DiversityRule = None
    FreshnessRule = None

__all__ = [
    # Main orchestrator
    "RecommendationPipeline",
    "PipelineConfig",
    "PipelineResult",
    # Config
    "load_pipeline_config",
    "build_pipeline_from_config",
    # Stage abstractions
    "StagePipeline",
    "PipelineStage",
    "RetrievalStage",
    "RankingStage",
    "RerankingStage",
    "DiversityRule",
    "FreshnessRule",
]
