"""Integration tests for recommendation pipeline."""
import unittest

from corerec.pipelines.orchestrator import PipelineOrchestrator, RecommendationPipeline
from corerec.ranking.base import RankedCandidate, RankingResult
from corerec.retrieval.base import BaseRetriever, Candidate, RetrievalResult
from corerec.ranking.base import BaseRanker
from corerec.reranking.base import BaseReranker


class _MockRetriever(BaseRetriever):
    def __init__(self, items):
        super().__init__(name="mock")
        self._items = items
        self._is_fitted = True

    def fit(self, **kwargs):
        self._is_fitted = True
        return self

    def retrieve(self, query, top_k=100, **kwargs):
        cands = [
            Candidate(item_id=i, score=1.0 / (idx + 1), source=self.name)
            for idx, i in enumerate(self._items[:top_k])
        ]
        return RetrievalResult(candidates=cands, query_id=query, retriever_name=self.name)


class _MockRanker(BaseRanker):
    def __init__(self):
        super().__init__(name="mock_ranker")
        self._is_fitted = True

    def fit(self, **kwargs):
        self._is_fitted = True
        return self

    def rank(self, candidates, context=None, **kwargs):
        if isinstance(candidates, RetrievalResult):
            cands = candidates.candidates
        else:
            cands = candidates
        ranked = [
            RankedCandidate(item_id=c.item_id, score=c.score, retrieval_score=c.score, rank=i + 1)
            for i, c in enumerate(sorted(cands, reverse=True))
        ]
        return RankingResult(candidates=ranked)


class _MockReranker(BaseReranker):
    def rerank(self, ranked, context=None, **kwargs):
        if isinstance(ranked, RankingResult):
            return ranked
        return RankingResult(candidates=list(ranked))


class TestPipelineIntegration(unittest.TestCase):
    def test_alias_and_recommend(self):
        self.assertIs(PipelineOrchestrator, RecommendationPipeline)

        pipeline = RecommendationPipeline()
        pipeline.add_retriever(_MockRetriever([1, 2, 3, 4, 5]))
        pipeline.set_ranker(_MockRanker())
        pipeline.add_reranker(_MockReranker())

        result = pipeline.recommend(query=1, top_k=3)
        self.assertGreaterEqual(len(result.items), 1)
        self.assertEqual(len(result.items), len(result.scores))


if __name__ == "__main__":
    unittest.main()
