"""Rerankers have to compose, because that's what the pipeline is for.

Findings/bug.md #13: BusinessRulesReranker re-sorted every list by `.score`,
which is still the original relevance score after Diversity/Fairness reorder
things. So any chain ending in business rules -- including the one
examples/pipeline_example.py builds -- silently came back in plain relevance
order. #4: the same reranker swallowed top_k through **kwargs and ignored it.
Also found while fixing those: boosts were applied with `c.score *= m` on the
caller's own objects, so reranking the same input twice compounded the boost.
"""

import pytest

from corerec.ranking.base import RankedCandidate, RankingResult
from corerec.reranking import BusinessRulesReranker, DiversityReranker, FairnessReranker


def _ranked(n=20):
    return RankingResult(
        candidates=[RankedCandidate(item_id=i, score=1.0 / (i + 1)) for i in range(n)],
        ranker_name="test",
    )


def _grouped():
    rows = [(1, 1.0, "A"), (2, 0.9, "A"), (3, 0.8, "A"),
            (4, 0.5, "B"), (5, 0.4, "B"), (6, 0.3, "B")]
    return RankingResult(
        candidates=[RankedCandidate(item_id=i, score=s, features={"g": g}) for i, s, g in rows],
        ranker_name="test",
    )


def _fair(result):
    return FairnessReranker(
        group_fn=lambda i: "A" if i <= 3 else "B", objective="equal", fairness_weight=0.9
    ).rerank(result)


RERANKERS = {
    "diversity": lambda: DiversityReranker(lambda_=0.7),
    "fairness": lambda: FairnessReranker(group_fn=lambda i: i % 2),
    "business": lambda: BusinessRulesReranker(),
}


@pytest.mark.parametrize("name", sorted(RERANKERS))
def test_every_reranker_honours_top_k(name):
    out = RERANKERS[name]().rerank(_ranked(20), top_k=5)
    assert len(out.candidates) == 5, f"{name} returned {len(out.candidates)} for top_k=5"


def test_business_rules_with_no_rules_is_a_no_op_after_fairness():
    fair = _fair(_grouped())
    after = BusinessRulesReranker().rerank(fair)
    assert [c.item_id for c in after.candidates] == [c.item_id for c in fair.candidates]


def test_fairness_actually_reordered_something():
    """Guard: if fairness stops reordering, the no-op test above proves nothing."""
    fair = _fair(_grouped())
    assert [c.item_id for c in fair.candidates] != [1, 2, 3, 4, 5, 6]


def test_boost_moves_only_the_boosted_item():
    fair = _fair(_grouped())
    before = [c.item_id for c in fair.candidates]
    after = [c.item_id for c in BusinessRulesReranker().add_boost(6, 3.0).rerank(fair).candidates]

    assert after.index(6) < before.index(6), "boosted item did not move up"
    assert [i for i in after if i != 6] == [i for i in before if i != 6], (
        "unboosted items lost the order fairness gave them"
    )


def test_diversity_then_business_keeps_diversity_order():
    feats = [{"category": "x"}, {"category": "x"}, {"category": "x"},
             {"category": "y"}, {"category": "y"}, {"category": "z"}]
    ranked = RankingResult(
        candidates=[RankedCandidate(item_id=i, score=1.0 - i * 0.05, features=f)
                    for i, f in enumerate(feats)],
        ranker_name="test",
    )
    div = DiversityReranker(lambda_=0.3, category_key="category").rerank(ranked)
    after = BusinessRulesReranker().add_blocklist([999]).rerank(div)
    assert [c.item_id for c in after.candidates] == [c.item_id for c in div.candidates]


def test_single_stage_boost_unchanged():
    """On relevance-ordered input a boost still lands where its score puts it."""
    out = BusinessRulesReranker().add_boost(3, 10.0).rerank(_ranked(5))
    assert [c.item_id for c in out.candidates] == [3, 0, 1, 2, 4]


def test_rerank_does_not_mutate_its_input():
    ranked = _ranked(5)
    reranker = BusinessRulesReranker().add_boost(3, 10.0)
    for _ in range(3):
        reranker.rerank(ranked)
    assert ranked.candidates[3].score == pytest.approx(0.25)


def test_a_later_filter_can_fall_back_to_lower_ranked_candidates():
    """diversity -> business returned [] when the blocklist hit diversity's top_k (#104)."""
    from corerec.pipelines import build_pipeline_from_config

    fit = {"popularity": {"item_ids": [1, 2, 3, 4], "interaction_counts": [40, 30, 20, 10]}}
    for stages in ([{"type": "business", "blocklist": [1, 2]}],
                   [{"type": "diversity", "lambda": 1.0}, {"type": "business", "blocklist": [1, 2]}]):
        pipe = build_pipeline_from_config(
            {"retrieval": {"sources": [{"type": "popularity"}]}, "reranking": stages}, fit=fit)
        assert [i for i, _ in pipe.recommend(query=1, top_k=2).to_list()] == [3, 4]
