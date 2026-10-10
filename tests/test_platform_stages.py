"""Smoke tests for retrieval, ranking, and reranking platform stages."""
import unittest
from itertools import permutations

import numpy as np

from corerec.ranking.base import RankedCandidate, RankingResult
from corerec.ranking.pointwise import PointwiseRanker
from corerec.reranking.diversity import DiversityReranker
from corerec.retrieval.base import Candidate, RetrievalResult
from corerec.retrieval.ensemble import EnsembleRetriever
from corerec.retrieval.popularity import PopularityRetriever


class TestRetrievalStage(unittest.TestCase):
    def test_popularity_retrieve(self):
        retriever = PopularityRetriever()
        retriever.fit(item_ids=[10, 11, 12], interaction_counts=[100, 50, 200])
        result = retriever.retrieve(user_id=None, top_k=2)
        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.candidates[0].item_id, 12)

    def test_union_compares_weighted_scores_in_every_source_order(self):
        high = PopularityRetriever().fit([42], scores=[10])
        low = PopularityRetriever().fit([42], scores=[20])
        for sources in permutations([("high", high, 10), ("low", low, 1)]):
            with self.subTest(sources=[source[0] for source in sources]):
                result = EnsembleRetriever(list(sources), strategy="union").retrieve(None)
                self.assertEqual(result.item_ids(), [42])
                self.assertEqual(result.scores().tolist(), [100])
                self.assertEqual(result.candidates[0].source, "ensemble(high)")

    def test_invalid_popularity_inputs_leave_fitted_state_unchanged(self):
        for field in ("scores", "interaction_counts", "timestamps"):
            for values in ([1], [1, 2, 3], [[1, 2]], [1, np.nan], [1, np.inf]):
                with self.subTest(field=field, values=values):
                    retriever = PopularityRetriever()
                    with self.assertRaisesRegex(ValueError, field):
                        retriever.fit([10, 20], **{field: values})
                    self.assertFalse(retriever.is_fitted)
                    retriever.fit([30, 40], scores=[2, 1])
                    with self.assertRaisesRegex(ValueError, field):
                        retriever.fit([10, 20], **{field: values})
                    self.assertEqual(retriever.retrieve().item_ids(), [30, 40])
                    self.assertEqual(retriever.retrieve().scores().tolist(), [2, 1])

    def test_popularity_empty_catalog_and_zero_limit(self):
        retriever = PopularityRetriever(time_decay=1).fit([], timestamps=[])
        self.assertEqual(retriever.retrieve().item_ids(), [])
        retriever.fit([10, 20], timestamps=[0, 1])
        self.assertEqual(retriever.retrieve(top_k=0).item_ids(), [])
        self.assertEqual(retriever.retrieve().item_ids(), [20, 10])
        with self.assertRaisesRegex(ValueError, "top_k"):
            retriever.retrieve(top_k=-1)

    def test_invalid_time_decay(self):
        for value in (-1, np.nan, np.inf):
            with self.subTest(value=value):
                retriever = PopularityRetriever(time_decay=value)
                with self.assertRaisesRegex(ValueError, "time_decay"):
                    retriever.fit([10], timestamps=[0])
                self.assertFalse(retriever.is_fitted)


class TestRankingStage(unittest.TestCase):
    def test_pointwise_rank(self):
        ranker = PointwiseRanker(
            score_fn=lambda feats: feats.get("retrieval_score", 0.0),
        )
        ranker.fit()
        retrieval = RetrievalResult(
            candidates=[
                Candidate(item_id=10, score=0.2),
                Candidate(item_id=11, score=0.9),
                Candidate(item_id=12, score=0.5),
            ],
            retriever_name="test",
        )
        ranked = ranker.rank(retrieval)
        self.assertIsInstance(ranked, RankingResult)
        self.assertEqual(ranked.candidates[0].item_id, 11)
        self.assertGreaterEqual(len(ranked.candidates), 2)


class TestRerankingStage(unittest.TestCase):
    def test_diversity_rerank(self):
        ranked = RankingResult(
            candidates=[
                RankedCandidate(item_id=10, score=0.9, rank=1, features={"category": "a"}),
                RankedCandidate(item_id=11, score=0.85, rank=2, features={"category": "a"}),
                RankedCandidate(item_id=12, score=0.8, rank=3, features={"category": "b"}),
            ],
            ranker_name="test",
        )
        reranker = DiversityReranker(lambda_=0.5, category_key="category")
        out = reranker.rerank(ranked, top_k=2)
        self.assertEqual(len(out.candidates), 2)
        item_ids = [c.item_id for c in out.candidates]
        self.assertIn(10, item_ids)
        self.assertIn(12, item_ids)


if __name__ == "__main__":
    unittest.main()
