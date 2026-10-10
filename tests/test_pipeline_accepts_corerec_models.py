"""A ranker must accept a CoreRec model, not just an sklearn estimator.

The pipeline layer (retrieval -> ranking -> reranking) is the thing CoreRec has
that a single-model library does not, and it could not consume CoreRec's own
models. The rankers called `model.predict(X)` -- the sklearn convention -- while
every CoreRec recommender takes `predict(user_id, item_id)`, so

    PointwiseRanker(model=als).fit().rank(candidates)

raised TypeError: predict() missing 1 required positional argument: 'item_id'.

Nothing caught it because no test ran a model through a ranker; the two layers
were written against different assumptions and never met. Same defect as
ModelServer's /recommend passing exclude_items to models that did not accept it.
"""

import numpy as np
import pytest

from corerec.engines.matrix_factorization import ALS
from corerec.ranking import FeatureCrossRanker, PointwiseRanker
from corerec.reranking import DiversityReranker
from corerec.retrieval import CollaborativeRetriever

RANKERS = [PointwiseRanker, FeatureCrossRanker]


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    users = rng.integers(0, 40, 300).tolist()
    items = rng.integers(0, 60, 300).tolist()
    ratings = rng.uniform(1, 5, 300).tolist()
    model = ALS(factors=16, iterations=10)
    model.fit(user_ids=users, item_ids=items, ratings=ratings)
    return model, users[0]


@pytest.mark.parametrize("ranker_cls", RANKERS, ids=[r.__name__ for r in RANKERS])
def test_ranker_accepts_a_corerec_model(ranker_cls, fitted):
    """The whole point: hand it a CoreRec model and it ranks."""
    model, user = fitted
    candidates = CollaborativeRetriever(model=model).retrieve(user, top_k=20)

    ranked = ranker_cls(model=model).fit().rank(candidates, context={"user_id": user})

    assert len(ranked.candidates) == 20
    scores = [c.score for c in ranked.candidates]
    assert scores == sorted(scores, reverse=True), "ranker did not sort by score"
    assert len(set(c.item_id for c in ranked.candidates)) == 20


@pytest.mark.parametrize("ranker_cls", RANKERS, ids=[r.__name__ for r in RANKERS])
def test_model_path_matches_explicit_score_fn(ranker_cls, fitted):
    """model= and an equivalent score_fn must produce the same ranking.

    If they diverge, one of the two paths is scoring something other than the
    model's own prediction.
    """
    model, user = fitted
    candidates = CollaborativeRetriever(model=model).retrieve(user, top_k=15)

    via_model = ranker_cls(model=model).fit().rank(
        candidates, context={"user_id": user}
    )
    via_fn = PointwiseRanker(
        feature_extractor=lambda item_id, ctx: {"item_id": item_id},
        score_fn=lambda f: model.predict(user, f["item_id"]),
    ).fit().rank(candidates, context={"user_id": user})

    assert [c.item_id for c in via_model.candidates] == [
        c.item_id for c in via_fn.candidates
    ]


def test_missing_user_says_so(fitted):
    """A recommender scores a (user, item) pair; without a user, say that."""
    model, user = fitted
    candidates = CollaborativeRetriever(model=model).retrieve(user, top_k=5)

    with pytest.raises(ValueError, match="user"):
        PointwiseRanker(model=model).fit().rank(candidates)  # no context


def test_item_id_reaches_score_fn_without_a_feature_extractor(fitted):
    """The item being scored should be visible by default.

    Previously the default feature dict was {"retrieval_score": ...} only, so a
    score_fn that needed the item had to supply a feature_extractor purely to
    pass through a value the ranker already had.
    """
    model, user = fitted
    candidates = CollaborativeRetriever(model=model).retrieve(user, top_k=5)

    seen = []
    ranked = PointwiseRanker(
        score_fn=lambda f: seen.append(f["item_id"]) or float(f["retrieval_score"])
    ).fit().rank(candidates, context={"user_id": user})

    assert len(seen) == 5
    assert len(ranked.candidates) == 5


def test_three_stage_pipeline_end_to_end(fitted):
    """retrieval -> ranking -> reranking, with a CoreRec model throughout."""
    model, user = fitted

    candidates = CollaborativeRetriever(model=model).retrieve(user, top_k=20)
    ranked = PointwiseRanker(model=model).fit().rank(
        candidates, context={"user_id": user}
    )
    final = DiversityReranker(lambda_=0.7).rerank(ranked, top_k=5)

    assert len(final.candidates) == 5
    ids = [c.item_id for c in final.candidates]
    assert len(set(ids)) == 5, f"duplicate items survived reranking: {ids}"


@pytest.mark.parametrize("estimator", ["regressor", "classifier"])
def test_sklearn_features_do_not_depend_on_identifier_type(estimator):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from corerec.retrieval.base import Candidate

    model = LinearRegression() if estimator == "regressor" else LogisticRegression()
    model.fit(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float), [0, 1, 0, 1])
    ranker = PointwiseRanker(
        model=model,
        feature_extractor=lambda item, context: {"quality": 0.5, "retrieval_score": 0.2},
    ).fit()
    ranked = ranker.rank([Candidate("7", 0.2), Candidate(7, 0.2)])
    assert len(ranked.candidates) == 2
    expected = (model.predict_proba([[0.5, 0.2]])[0, 1] if estimator == "classifier"
                else model.predict([[0.5, 0.2]])[0])
    assert [c.score for c in ranked.candidates] == pytest.approx([expected, expected])
    assert all(list(c.features) == ["quality", "retrieval_score"] for c in ranked.candidates)


def test_callable_model_still_receives_item_id():
    from corerec.retrieval.base import Candidate

    seen = []
    ranker = PointwiseRanker(model=lambda features: seen.append(features["item_id"]) or 1).fit()
    ranker.rank([Candidate(7, 0.2), Candidate("8", 0.3)])
    assert seen == [7, "8"]


def test_explicit_numeric_item_feature_is_preserved():
    from sklearn.linear_model import LinearRegression
    from corerec.retrieval.base import Candidate

    model = LinearRegression().fit([[0, 0], [1, 0], [0, 1]], [0, 1, 0])
    ranker = PointwiseRanker(
        model=model, feature_extractor=lambda item, context: {"item_id": item},
    ).fit()
    assert ranker.rank([Candidate(7, 0.2)]).candidates[0].score == pytest.approx(7)
