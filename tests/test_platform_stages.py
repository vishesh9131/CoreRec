"""Smoke tests for retrieval, ranking, and reranking platform stages."""
import unittest

import pytest

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


def test_union_merge_does_not_depend_on_source_order():
    """Union compared a raw score with a stored weighted one (#105)."""
    import itertools

    from corerec.retrieval import EnsembleRetriever, PopularityRetriever

    a = PopularityRetriever().fit([42, 1, 2], scores=[10, 9, 1])
    b = PopularityRetriever().fit([42, 2, 3], scores=[20, 30, 5])
    c = PopularityRetriever().fit([1, 3, 42], scores=[4, 8, 2])
    sources = [("a", a, 10), ("b", b, 1), ("c", c, 3)]
    seen = set()
    for order in itertools.permutations(sources):
        res = EnsembleRetriever(list(order), strategy="union").fit().retrieve(None, top_k=4)
        seen.add(tuple((x.item_id, round(x.score, 9)) for x in res.candidates))
    assert len(seen) == 1, seen
    # each item gets its best weighted contribution: 42 -> max(100, 20, 6)
    assert dict(next(iter(seen)))[42] == 100.0


@pytest.mark.parametrize("kwargs,match", [
    ({"scores": [1, 2]}, "one value per item_id"),
    ({"scores": [1, 2, 3, 4]}, "one value per item_id"),
    ({"scores": [[1, 2, 3]]}, "one value per item_id"),
    ({"scores": [1, float("nan"), 2]}, "NaN/inf"),
    ({"interaction_counts": [1, float("inf"), 2]}, "NaN/inf"),
    ({"scores": [1, 2, 3], "timestamps": [1, 2]}, "timestamps"),
])
def test_popularity_retriever_rejects_bad_inputs(kwargs, match):
    """Misaligned or NaN scores used to fit, then drop items or crash (#107)."""
    from corerec.retrieval import PopularityRetriever

    r = PopularityRetriever(time_decay=0.1)
    with pytest.raises(ValueError, match=match):
        r.fit([10, 20, 30], **kwargs)
    assert not r.is_fitted


def test_popularity_retriever_keeps_every_item():
    from corerec.retrieval import PopularityRetriever

    got = PopularityRetriever().fit([10, 20, 30], scores=[1, 3, 2]).retrieve(None, top_k=10)
    assert [c.item_id for c in got.candidates] == [20, 30, 10]
    assert len(PopularityRetriever().fit([10, 20, 30]).retrieve(None, top_k=10).candidates) == 3
