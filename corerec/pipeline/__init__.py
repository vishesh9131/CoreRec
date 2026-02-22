"""
Pipeline Module - Orchestration

Chains retrieval, ranking, and reranking stages into a cohesive
recommendation pipeline. Handles data flow between stages.

Usage:
    from corerec.pipeline import RecommendationPipeline
    
    pipeline = RecommendationPipeline()
    pipeline.add_retriever(semantic_retriever)
    pipeline.add_retriever(collaborative_retriever)
    pipeline.set_ranker(pointwise_ranker)
    pipeline.add_reranker(diversity_reranker)
    
    recs = pipeline.recommend(user_id=123, top_k=10)
"""

from .base import RecommendationPipeline, PipelineConfig
from .config import load_pipeline_config

__all__ = [
    "RecommendationPipeline",
    "PipelineConfig",
    "load_pipeline_config",
]
