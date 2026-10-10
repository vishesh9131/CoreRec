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


class TestBuildPipelineFromConfig(unittest.TestCase):
    config = {"pipeline": {
        "retrieval": {"sources": [{"type": "popularity"}]},
        "ranking": {"type": "pointwise"},
        "final_k": 3,
    }}
    pop_fit = {"popularity": {"item_ids": [1, 2, 3, 4], "interaction_counts": [5, 9, 1, 7]}}

    def test_fit_kwargs_make_config_pipeline_recommend(self):
        from corerec.pipelines import build_pipeline_from_config

        pipe = build_pipeline_from_config(self.config, fit=self.pop_fit)
        items = [i for i, _ in pipe.recommend(query=0).to_list()]
        self.assertEqual(items, [2, 4, 1])

    def test_unknown_fit_key_raises(self):
        from corerec.pipelines import build_pipeline_from_config

        with self.assertRaises(ValueError):
            build_pipeline_from_config(self.config, fit={"populrity": {}})


class TestConfigRejectsUnknownStages(unittest.TestCase):
    """A misspelled stage used to vanish, so a blocklist stopped applying (#103)."""

    fit = {"popularity": {"item_ids": [10, 20, 30], "interaction_counts": [10, 5, 1]}}

    def _build(self, **stages):
        from corerec.pipelines import build_pipeline_from_config

        config = {"retrieval": {"sources": [{"type": "popularity"}]}, **stages}
        return build_pipeline_from_config(config, fit=self.fit)

    def test_valid_blocklist_still_applies(self):
        pipe = self._build(reranking=[{"type": "business", "blocklist": [10]}])
        self.assertEqual([i for i, _ in pipe.recommend(query=1, top_k=3).to_list()], [20, 30])

    def test_misspelled_stages_raise(self):
        from corerec.pipelines import build_pipeline_from_config

        for stages, word in [({"reranking": [{"type": "buisness", "blocklist": [10]}]}, "buisness"),
                             ({"ranking": {"type": "pointwize"}}, "pointwize")]:
            with self.assertRaisesRegex(ValueError, f"'{word}' is not supported"):
                self._build(**stages)
        with self.assertRaisesRegex(ValueError, "'popularty' is not supported"):
            build_pipeline_from_config({"retrieval": {"sources": [{"type": "popularty"}]}})

    def test_fairness_says_what_it_needs(self):
        with self.assertRaisesRegex(ValueError, "group_fn"):
            self._build(reranking=[{"type": "fairness"}])
