"""
Recommendation Pipeline

Orchestrates the multi-stage recommendation process:
1. Retrieval (multiple sources)
2. Ranking (scoring)
3. Reranking (post-processing)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from corerec.retrieval.base import BaseRetriever, Candidate, RetrievalResult
from corerec.retrieval.ensemble import EnsembleRetriever
from corerec.ranking.base import BaseRanker, RankingResult
from corerec.reranking.base import BaseReranker

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a recommendation pipeline."""

    #: candidates to retrieve per retriever
    retrieval_k: int = 500
    #: candidates to pass to ranking
    ranking_k: int = 100
    #: final recommendations to return
    final_k: int = 10
    #: how to merge retrieval results
    fusion_strategy: str = "rrf"


@dataclass
class PipelineResult:
    """
    Output from a pipeline recommendation.
    
    Includes the final recommendations plus debugging info
    about what happened at each stage.
    """
    items: List[Any]
    scores: List[float]
    
    # detailed per-stage info
    retrieval_candidates: int = 0
    ranking_candidates: int = 0
    final_candidates: int = 0
    
    # timing breakdown
    retrieval_ms: float = 0.0
    ranking_ms: float = 0.0
    reranking_ms: float = 0.0
    total_ms: float = 0.0
    
    # for debugging
    stage_results: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __iter__(self):
        return iter(zip(self.items, self.scores))
    
    def to_list(self) -> List[Tuple[Any, float]]:
        """Return as list of (item_id, score) tuples."""
        return list(zip(self.items, self.scores))


class RecommendationPipeline:
    """
    Multi-stage recommendation pipeline.
    
    Combines multiple retrievers, a ranker, and multiple rerankers
    into a single cohesive pipeline. Handles data flow, timing,
    and provides debugging info.
    
    Example::

        pipeline = RecommendationPipeline()
        
        # add retrieval sources (multiple is fine)
        pipeline.add_retriever(collaborative_retriever, weight=1.0)
        pipeline.add_retriever(semantic_retriever, weight=0.5)
        
        # set the ranker (only one)
        pipeline.set_ranker(pointwise_ranker)
        
        # add rerankers (applied in order)
        pipeline.add_reranker(diversity_reranker)
        pipeline.add_reranker(business_rules_reranker)
        
        # get recommendations
        result = pipeline.recommend(
            user_id=123,
            context={'session_id': 'abc'},
            top_k=10
        )
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        name: str = "pipeline",
    ):
        """
        Args:
            config: pipeline configuration
            name: identifier for this pipeline
        """
        self.config = config or PipelineConfig()
        self.name = name
        
        self._retrievers: List[Tuple[BaseRetriever, float]] = []
        self._ranker: Optional[BaseRanker] = None
        self._rerankers: List[BaseReranker] = []
        
        self._ensemble: Optional[EnsembleRetriever] = None
    
    def add_retriever(
        self,
        retriever: BaseRetriever,
        weight: float = 1.0
    ) -> "RecommendationPipeline":
        """
        Add a retrieval source.
        
        Args:
            retriever: the retriever to add
            weight: weight for ensemble fusion (higher = more influence)
        """
        self._retrievers.append((retriever, weight))
        self._ensemble = None  # invalidate cached ensemble
        return self
    
    def set_ranker(self, ranker: BaseRanker) -> "RecommendationPipeline":
        """
        Set the ranking model.
        
        Args:
            ranker: the ranker to use for Stage 2
        """
        self._ranker = ranker
        return self
    
    def add_reranker(self, reranker: BaseReranker) -> "RecommendationPipeline":
        """
        Add a reranking step.
        
        Rerankers are applied in the order they're added.
        
        Args:
            reranker: the reranker to add
        """
        self._rerankers.append(reranker)
        return self
    
    def recommend(
        self,
        query: Any,
        context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> PipelineResult:
        """
        Generate recommendations.
        
        Args:
            query: user_id, text query, or embedding
            context: user features, session info, etc.
            top_k: number of final recommendations (overrides config)
        
        Returns:
            PipelineResult with recommendations and timing info
        """
        total_start = time.perf_counter()
        context = context or {}
        context['query'] = query
        
        top_k = top_k or self.config.final_k
        stage_results = {}
        
        # Stage 1: Retrieval
        retrieval_start = time.perf_counter()
        retrieval_result = self._retrieve(query, **kwargs)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        
        stage_results['retrieval'] = {
            'candidates': len(retrieval_result),
            'timing_ms': retrieval_ms,
        }
        
        # limit candidates for ranking
        ranking_candidates = retrieval_result.candidates[:self.config.ranking_k]
        
        # Stage 2: Ranking
        ranking_start = time.perf_counter()
        if self._ranker is not None:
            ranking_result = self._ranker.rank(ranking_candidates, context, **kwargs)
        else:
            # no ranker - just use retrieval scores
            from corerec.ranking.base import RankedCandidate, RankingResult
            ranking_result = RankingResult(
                candidates=[
                    RankedCandidate(
                        item_id=c.item_id,
                        score=c.score,
                        retrieval_score=c.score,
                        rank=i+1,
                    )
                    for i, c in enumerate(ranking_candidates)
                ]
            )
        ranking_ms = (time.perf_counter() - ranking_start) * 1000
        
        stage_results['ranking'] = {
            'candidates': len(ranking_result),
            'timing_ms': ranking_ms,
        }
        
        # Stage 3: Reranking
        reranking_start = time.perf_counter()
        current_result = ranking_result
        
        for reranker in self._rerankers:
            current_result = reranker.rerank(current_result, context, top_k=top_k)
        
        reranking_ms = (time.perf_counter() - reranking_start) * 1000
        
        stage_results['reranking'] = {
            'num_rerankers': len(self._rerankers),
            'timing_ms': reranking_ms,
        }
        
        # extract final results
        final_candidates = current_result.candidates[:top_k]
        items = [c.item_id for c in final_candidates]
        scores = [c.score for c in final_candidates]
        
        total_ms = (time.perf_counter() - total_start) * 1000
        
        return PipelineResult(
            items=items,
            scores=scores,
            retrieval_candidates=len(retrieval_result),
            ranking_candidates=len(ranking_result),
            final_candidates=len(final_candidates),
            retrieval_ms=retrieval_ms,
            ranking_ms=ranking_ms,
            reranking_ms=reranking_ms,
            total_ms=total_ms,
            stage_results=stage_results,
        )
    
    def _retrieve(self, query: Any, **kwargs) -> RetrievalResult:
        """Run retrieval stage."""
        if not self._retrievers:
            raise RuntimeError("No retrievers added. Call add_retriever() first.")
        
        # build ensemble if needed
        if self._ensemble is None:
            self._ensemble = EnsembleRetriever(
                retrievers=[
                    (r.name, r, w) for r, w in self._retrievers
                ],
                strategy=self.config.fusion_strategy,
                k_per_retriever=self.config.retrieval_k,
            )
        
        return self._ensemble.retrieve(query, top_k=self.config.retrieval_k, **kwargs)
    
    def recommend_batch(
        self,
        queries: List[Any],
        contexts: Optional[List[Dict[str, Any]]] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[PipelineResult]:
        """
        Batch recommendation for multiple queries.
        
        Default implementation loops; can be optimized in subclasses.
        """
        if contexts is None:
            contexts = [None] * len(queries)
        
        return [
            self.recommend(q, ctx, top_k, **kwargs)
            for q, ctx in zip(queries, contexts)
        ]
    
    def __repr__(self) -> str:
        return (
            f"RecommendationPipeline("
            f"retrievers={len(self._retrievers)}, "
            f"ranker={self._ranker is not None}, "
            f"rerankers={len(self._rerankers)})"
        )
