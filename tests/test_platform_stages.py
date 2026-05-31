"""Smoke tests for retrieval, ranking, and reranking platform stages."""
import unittest

from corerec.ranking.base import RankedCandidate, RankingResult
from corerec.ranking.pointwise import PointwiseRanker
from corerec.reranking.diversity import DiversityReranker
from corerec.retrieval.base import Candidate, RetrievalResult
from corerec.retrieval.popularity import PopularityRetriever


class TestRetrievalStage(unittest.TestCase):
    def test_popularity_retrieve(self):
        retriever = PopularityRetriever()
        retriever.fit(item_ids=[10, 11, 12], interaction_counts=[100, 50, 200])
        result = retriever.retrieve(user_id=None, top_k=2)
        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.candidates[0].item_id, 12)


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
