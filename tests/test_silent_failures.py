"""Failures that used to disappear without a trace (Findings/bug.md #5, #9, #10, #11)."""

import math

import numpy as np
import pandas as pd
import pytest


class _Broken:
    def recommend(self, user_id, top_k=10):
        raise RuntimeError("kaboom")


def test_evaluator_reports_nan_and_error_count_for_broken_model():
    from corerec.evaluation import Evaluator

    out = Evaluator(metrics=["ndcg@10"]).evaluate(_Broken(), {1: [1], 2: [2]})
    assert math.isnan(out["ndcg@10"]), "a crashed model must not score 0.0"
    assert out["n_errors"] == 2 and out["n_users"] == 0


def test_evaluator_strict_reraises():
    from corerec.evaluation import Evaluator

    with pytest.raises(RuntimeError, match="kaboom"):
        Evaluator(metrics=["ndcg@10"]).evaluate(_Broken(), {1: [1]}, strict=True)


def test_evaluator_asks_for_enough_items_for_deep_metrics():
    """Evaluator always requested 20 items, so recall@50 counted ranks 21..50
    as misses (#50). Both relevant items here are in the top 50."""
    from corerec.evaluation import Evaluator

    asked = []

    class Model:
        def recommend(self, user_id, top_k=10, **kw):
            asked.append(top_k)
            return list(range(top_k))

    out = Evaluator(metrics=["recall@50", "ndcg@10"]).evaluate(Model(), {0: [30, 40]})
    assert out["recall@50"] == 1.0
    assert asked == [50]


def test_cross_validate_runs_the_documented_call():
    from corerec.engines import ItemKNN
    from corerec.evaluation import CrossValidator

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"user_id": rng.integers(0, 20, 400),
                       "item_id": rng.integers(0, 30, 400), "rating": 1.0})
    out = CrossValidator(n_folds=3).cross_validate(ItemKNN, df, metric="ndcg@10")
    assert len(out["folds"]) == 3
    assert 0.0 <= out["mean"] <= 1.0


def test_retrieval_then_rerank_is_importable():
    from corerec.hybrid import RetrievalThenRerank

    assert RetrievalThenRerank is not None


def test_ensemble_records_a_failing_child(caplog):
    from corerec.retrieval import EnsembleRetriever
    from corerec.retrieval.base import BaseRetriever, Candidate, RetrievalResult

    class Ok(BaseRetriever):
        def fit(self, **kw):
            self._is_fitted = True
            return self

        def retrieve(self, query, top_k=10, **kw):
            return RetrievalResult(candidates=[Candidate(item_id=1, score=1.0, source="ok")])

    class Bad(Ok):
        def retrieve(self, query, top_k=10, **kw):
            raise ValueError("index missing")

    ens = EnsembleRetriever(retrievers=[("ok", Ok().fit(), 1.0), ("bad", Bad().fit(), 1.0)])
    with caplog.at_level("WARNING"):
        res = ens.retrieve(0, top_k=5)
    assert [c.item_id for c in res.candidates] == [1]
    assert "bad" in ens.last_errors
    assert "index missing" in caplog.text


@pytest.mark.parametrize("name,kw", [
    ("ALS", {}), ("Item2Vec", {}), ("LightGCN", {"epochs": 2}),
    ("TwoTower", {"embedding_dim": 8, "epochs": 2, "verbose": False}),
])
def test_online_from_model_serves_what_the_model_recommends(name, kw):
    """#7: from_model raised NotImplementedError on every CoreRec model."""
    import corerec.engines as E
    from corerec.serving.online import OnlineRecommender

    rng = np.random.default_rng(0)
    u = rng.integers(0, 20, 300).tolist()
    i = (rng.integers(0, 30, 300) * 7 + 100).tolist()  # non-contiguous ids
    m = getattr(E, name)(**kw)
    m.fit(u, i, [1.0] * 300)
    online = OnlineRecommender.from_model(m, index_type="flat")
    for user in set(u):
        assert online.recommend(user, top_k=5) == m.recommend(user, top_k=5)


def test_online_from_model_refuses_non_dot_product_models():
    from corerec.engines import ItemKNN
    from corerec.serving.online import OnlineRecommender

    m = ItemKNN()
    m.fit([0, 0, 1], [0, 1, 1], [1.0, 1.0, 1.0])
    with pytest.raises(NotImplementedError, match="dot product"):
        OnlineRecommender.from_model(m)


def test_tfidf_recommend_by_text_takes_top_k_like_recommend():
    """#12: recommend(top_k=) worked, recommend_by_text(top_k=) raised TypeError."""
    from corerec.engines import TFIDFRecommender

    m = TFIDFRecommender()
    m.fit([1, 2, 3, 4], {1: "red apple", 2: "green apple", 3: "red car", 4: "blue sky"})
    assert len(m.recommend_by_text("apple", top_k=2)) == 2
    assert m.recommend_by_text("apple", top_n=2) == m.recommend_by_text("apple", top_k=2)


def test_evaluator_and_evaluate_agree():
    """Two evaluation APIs reported different keys and could differ in protocol (#46)."""
    from corerec.engines import ItemKNN
    from corerec.evaluation import Evaluator, evaluate

    rng = np.random.default_rng(0)
    u, i = rng.integers(0, 60, 1500).tolist(), rng.integers(0, 90, 1500).tolist()
    train = list(zip(u[:1200], i[:1200]))
    test = list(zip(u[1200:], i[1200:]))
    model = ItemKNN().fit(u[:1200], i[:1200])
    truth = {}
    for a, b in test:
        truth.setdefault(a, []).append(b)

    ref = evaluate(model, test, train_interactions=train, k=[5, 20])
    out = Evaluator(metrics=["ndcg@5", "Recall@20", "hit_rate@5"]).evaluate(
        model, truth, train_interactions=train)
    for key in ("NDCG@5", "Recall@20", "HitRate@5"):
        assert out[key] == pytest.approx(ref[key])
    # the requested spelling works too
    assert out["ndcg@5"] == out["NDCG@5"] and out["hit_rate@5"] == out["HitRate@5"]
    # one k or several: same numbers
    assert evaluate(model, test, train_interactions=train, k=5)["NDCG@5"] == pytest.approx(ref["NDCG@5"])


def test_evaluator_rejects_unknown_metric():
    from corerec.evaluation import Evaluator

    with pytest.raises(ValueError, match="unknown metric"):
        Evaluator(metrics=["ndgc@10"])


@pytest.mark.parametrize("recs,truth", [
    ([1, 1, 1], [1]),
    ([7, 7, 7, 7, 7], [7]),
    ([1, 2, 1, 2, 3], [1, 2, 3]),
    ([1, 2, 3], [1, 1, 2]),          # duplicate ground truth
])
def test_ranking_metrics_stay_in_unit_range_with_repeats(recs, truth):
    """A repeated item counted once per occurrence: NDCG 2.1, recall 3.0 (#82)."""
    from corerec.evaluation.metrics import RankingMetrics as R

    for fn in (R.ndcg_at_k, R.map_at_k, R.mrr_at_k, R.precision_at_k, R.recall_at_k, R.hit_rate_at_k):
        assert 0.0 <= fn(recs, truth, 5) <= 1.0, fn.__name__


def test_a_repeat_is_a_wasted_slot_not_a_promotion():
    from corerec.evaluation.metrics import RankingMetrics as R

    # [a, a, b]: b was shown third, so it gets third-place credit
    assert R.mrr_at_k(["a", "a", "b"], ["b"], 3) == pytest.approx(1 / 3)
    assert R.precision_at_k([7, 7, 7, 7, 7], [7], 5) == pytest.approx(1 / 5)


def test_evaluate_does_not_reward_repeating_the_answer(caplog):
    from corerec.evaluation import evaluate

    class Repeats:
        def recommend(self, user_id, top_k=10, **kw):
            return [7] * top_k

    with caplog.at_level("WARNING"):
        r = evaluate(Repeats(), [(0, 7)], k=5)
    assert r["NDCG@5"] == pytest.approx(1.0) and r["Recall@5"] == 1.0 and r["MAP@5"] == 1.0
    assert "same item twice" in caplog.text
