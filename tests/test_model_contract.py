"""One calling convention, checked against every production model.

tests/test_api_uniformity.py is named for this job but only exercises
FastRecommender, so the zoo drifted without anything noticing: two-tower and
BERT4Rec took a dense [n_users, n_items] matrix while everything else took the
triple fit(user_ids, item_ids, ratings), and neither accepted the exclude_items
argument BaseRecommender declares. ModelServer passes exclude_items on every
/recommend call, so those models returned HTTP 500 in production serving.

This file drives all of them through the same calls. When a model cannot meet
the contract, add it to KNOWN_DIVERGENT with a reason rather than loosening the
assertions -- the point is that divergence stays visible.
"""

import numpy as np
import pytest

from corerec.engines import MODELS as REGISTRY

# Small constructor kwargs so CI stays fast. Models not listed use defaults.
_FAST_KWARGS = {
    "TwoTower": {"embedding_dim": 16, "epochs": 3, "verbose": False},
    "SASRec": {"hidden_units": 16, "num_blocks": 1, "epochs": 1,
               "batch_size": 32, "max_seq_length": 20, "verbose": False},
    "HSTU": {"embedding_dim": 16, "epochs": 2, "num_negatives": 8, "batch_size": 16},
    "DCN": {"embedding_dim": 16, "epochs": 2},
    "DeepFM": {"embedding_dim": 16, "epochs": 2},
    "LightGCN": {"epochs": 5},
    "MultVAE": {"epochs": 5},
    "MultiDAE": {"epochs": 5},
}

# (test id, import path, class name, constructor kwargs). Every interaction
# model in the registry is checked; content-based models take item text
# instead of interactions and are covered by their own tests.
MODELS = [
    (name.lower(), "corerec.engines" + module, name, _FAST_KWARGS.get(name, {}))
    for name, (module, family, _) in REGISTRY.items()
    if family != "content"
]

# model id -> why it cannot meet the common contract yet.
# These are DataFrame-first models with their own documented entry points
# (fit_from_lists, fit_from_dataset). Changing a released fit() signature is its
# own decision, not a drive-by in a test file -- so the divergence is recorded
# here and shows up as xfail rather than being silently tolerated.
KNOWN_DIVERGENT = {}


def _interactions(n_users=25, n_items=40, seed=0):
    """Parallel (users, items, ratings) with group structure, one row per event."""
    rng = np.random.default_rng(seed)
    user_group = rng.integers(0, 3, n_users)
    item_group = rng.integers(0, 3, n_items)
    users, items, ratings = [], [], []
    for u in range(n_users):
        for i in np.flatnonzero(item_group == user_group[u])[:6]:
            users.append(int(u))
            items.append(int(i))
            ratings.append(5.0)
    return users, items, ratings


def _build(module_path, cls_name, kwargs):
    module = pytest.importorskip(module_path)
    cls = getattr(module, cls_name, None)
    if cls is None:
        pytest.skip(f"{cls_name} not exported from {module_path}")
    # No fallback to cls(): a model that rejects a shared knob such as
    # epochs= must fail here, not silently train with its defaults.
    return cls(**kwargs)


@pytest.fixture(scope="module")
def data():
    return _interactions()


@pytest.mark.parametrize("model_id,module_path,cls_name,kwargs", MODELS,
                         ids=[m[0] for m in MODELS])
def test_fit_accepts_the_triple(model_id, module_path, cls_name, kwargs, data):
    """fit(user_ids, item_ids, ratings) -- the form the README documents."""
    if model_id in KNOWN_DIVERGENT:
        pytest.xfail(KNOWN_DIVERGENT[model_id])
    users, items, ratings = data
    model = _build(module_path, cls_name, kwargs)
    model.fit(users, items, ratings)
    assert getattr(model, "is_fitted", True), f"{cls_name}.fit left is_fitted False"


@pytest.mark.parametrize("model_id,module_path,cls_name,kwargs", MODELS,
                         ids=[m[0] for m in MODELS])
def test_fit_accepts_ratings_keyword(model_id, module_path, cls_name, kwargs, data):
    """fit(user_ids=..., item_ids=..., ratings=...) — kwargs form callers use."""
    if model_id in KNOWN_DIVERGENT:
        pytest.xfail(KNOWN_DIVERGENT[model_id])
    users, items, ratings = data
    model = _build(module_path, cls_name, kwargs)
    model.fit(user_ids=users, item_ids=items, ratings=ratings)
    assert getattr(model, "is_fitted", True), f"{cls_name}.fit left is_fitted False"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("model_id,module_path,cls_name,kwargs", MODELS,
                         ids=[m[0] for m in MODELS])
def test_fit_rejects_non_finite_ratings(model_id, module_path, cls_name, kwargs, data, bad):
    """One NaN made every ALS/EASE/KNN score NaN with no error (#83)."""
    from corerec.api.exceptions import InvalidDataError

    users, items, ratings = data
    ratings = list(ratings)
    ratings[3] = bad
    model = _build(module_path, cls_name, kwargs)
    with pytest.raises(InvalidDataError, match="1 of .* ratings are NaN or infinite"):
        model.fit(users, items, ratings)
    with pytest.raises(InvalidDataError):
        model.fit(user_ids=users, item_ids=items, ratings=ratings)


@pytest.mark.parametrize("model_id,module_path,cls_name,kwargs", MODELS,
                         ids=[m[0] for m in MODELS])
def test_recommend_returns_ranked_ids(model_id, module_path, cls_name, kwargs, data):
    """recommend(user_id, top_k) -> at most top_k distinct item IDs."""
    if model_id in KNOWN_DIVERGENT:
        pytest.xfail(KNOWN_DIVERGENT[model_id])
    users, items, ratings = data
    model = _build(module_path, cls_name, kwargs)
    model.fit(users, items, ratings)

    recs = model.recommend(users[0], top_k=5)
    assert isinstance(recs, list), f"{cls_name}.recommend returned {type(recs).__name__}"
    assert len(recs) <= 5
    assert len(set(recs)) == len(recs), f"{cls_name} returned duplicates: {recs}"


@pytest.mark.parametrize("model_id,module_path,cls_name,kwargs", MODELS,
                         ids=[m[0] for m in MODELS])
def test_recommend_accepts_exclude_items(model_id, module_path, cls_name, kwargs, data):
    """BaseRecommender declares exclude_items, and ModelServer always sends it."""
    if model_id in KNOWN_DIVERGENT:
        pytest.xfail(KNOWN_DIVERGENT[model_id])
    users, items, ratings = data
    model = _build(module_path, cls_name, kwargs)
    model.fit(users, items, ratings)

    baseline = model.recommend(users[0], top_k=5)
    if not baseline:
        pytest.skip(f"{cls_name} returned no recommendations to exclude")

    banned = baseline[:2]
    filtered = model.recommend(users[0], top_k=5, exclude_items=banned)
    assert not set(banned) & set(filtered), (
        f"{cls_name} ignored exclude_items={banned}, returned {filtered}"
    )


@pytest.mark.parametrize("model_id,module_path,cls_name,kwargs", MODELS,
                         ids=[m[0] for m in MODELS])
def test_save_load_preserves_recommendations(model_id, module_path, cls_name, kwargs,
                                             data, tmp_path):
    """A reloaded model must recommend what the original did.

    Persistence bugs hide easily: two-tower kept its per-user seen-set out of
    save(), so exclude_seen=True quietly stopped excluding anything after a
    round-trip. Nothing in the suite noticed, because no test compared
    recommendations across save/load.
    """
    if model_id in KNOWN_DIVERGENT:
        pytest.xfail(KNOWN_DIVERGENT[model_id])
    users, items, ratings = data
    model = _build(module_path, cls_name, kwargs)
    model.fit(users, items, ratings)

    before = model.recommend(users[0], top_k=5)
    path = tmp_path / f"{model_id}.pkl"
    try:
        model.save(str(path))
    except (NotImplementedError, AttributeError) as exc:
        pytest.skip(f"{cls_name} does not implement save(): {exc}")

    reloaded = type(model).load(str(path))
    assert reloaded.recommend(users[0], top_k=5) == before, (
        f"{cls_name} recommends differently after save/load"
    )


@pytest.mark.parametrize("model_id,module_path,cls_name,kwargs", MODELS,
                         ids=[m[0] for m in MODELS])
def test_model_loader_reconstructs_without_knowing_the_class(model_id, module_path, cls_name,
                                                             kwargs, data, tmp_path):
    """serving.ModelLoader is how another process picks a model up.

    It used to return a raw dict for pickled CF state and FileNotFoundError for
    models that write <stem>.meta.json next to the path (Findings/bug.md #8).
    """
    if model_id in KNOWN_DIVERGENT:
        pytest.xfail(KNOWN_DIVERGENT[model_id])
    from corerec.serving import ModelLoader

    users, items, ratings = data
    model = _build(module_path, cls_name, kwargs)
    model.fit(users, items, ratings)
    path = tmp_path / f"{model_id}.model"
    model.save(str(path))

    loaded = ModelLoader().load(str(path))
    assert type(loaded) is type(model)
    assert loaded.recommend(users[0], top_k=5) == model.recommend(users[0], top_k=5)


@pytest.mark.parametrize("model_id,module_path,cls_name,kwargs", MODELS,
                         ids=[m[0] for m in MODELS])
def test_batch_predict_matches_predict(model_id, module_path, cls_name, kwargs, data):
    """batch_predict must agree with predict, pair for pair.

    BaseRecommender.batch_predict defaults to a list comprehension over
    predict(), which for a torch model is one forward pass per pair. NCF
    overrides it to score a whole request in a single pass -- 262ms/user down to
    single digits on ML-100K. This checks the override stays correct; the point
    of batching is lost if it quietly returns different numbers.
    """
    users, items, ratings = data
    model = _build(module_path, cls_name, kwargs)
    if model_id in KNOWN_DIVERGENT:
        import pandas as pd
        model.fit(pd.DataFrame({"user_id": users, "item_id": items, "rating": ratings}))
    else:
        model.fit(users, items, ratings)

    if not hasattr(model, "predict"):
        pytest.skip(f"{cls_name} has no predict()")

    pairs = [(users[0], i) for i in sorted(set(items))[:10]]
    try:
        batched = model.batch_predict(pairs)
        one_at_a_time = [model.predict(u, i) for u, i in pairs]
    except (NotImplementedError, ValueError) as exc:
        pytest.skip(f"{cls_name} cannot score these pairs: {exc}")

    assert len(batched) == len(pairs)
    for got, want in zip(batched, one_at_a_time):
        assert got == pytest.approx(want, abs=1e-5), (
            f"{cls_name}.batch_predict disagrees with predict: {batched} vs {one_at_a_time}"
        )


@pytest.mark.parametrize("model_id,module_path,cls_name,kwargs", MODELS,
                         ids=[m[0] for m in MODELS])
def test_training_length_is_called_epochs(model_id, module_path, cls_name, kwargs):
    """Iterative models take epochs=; num_epochs= survives only as a deprecated alias."""
    import inspect

    params = inspect.signature(_build(module_path, cls_name, {}).__class__.__init__).parameters
    if "num_epochs" in params:
        assert "epochs" in params, f"{cls_name} takes num_epochs but not epochs"


@pytest.mark.parametrize("model_id,module_path,cls_name,kwargs", MODELS,
                         ids=[m[0] for m in MODELS])
def test_fit_accepts_a_corerec_data_dataset(model_id, module_path, cls_name, kwargs, data):
    """corerec.data datasets used to be unusable with every model (Findings/bug.md #6)."""
    import pandas as pd

    from corerec.data import RecommendationDataset

    users, items, ratings = data
    ds = RecommendationDataset(pd.DataFrame({"user_id": users, "item_id": items,
                                             "rating": ratings}))
    model = _build(module_path, cls_name, kwargs)
    model.fit(ds)
    assert len(model.recommend(users[0], top_k=5)) > 0


# models with no training loop; everything else must take epochs=
_NO_EPOCHS = {"SAR", "ItemKNN", "UserKNN", "EASE", "SLIM", "TFIDFRecommender"}


@pytest.mark.parametrize("cls_name", sorted(set(REGISTRY) - _NO_EPOCHS))
def test_every_iterative_model_takes_epochs(cls_name):
    """One config dict must work across the zoo (Findings/bug.md #2).

    ALS and Item2Vec spelled it iterations=, and epochs= went into **kwargs and
    was silently ignored -- they trained for their default length.
    """
    import corerec.engines as engines

    assert getattr(engines, cls_name)(epochs=2).epochs == 2


@pytest.mark.parametrize("cls_name", ["SASRec", "TwoTower"])
def test_num_epochs_alias_warns_and_applies(cls_name):
    import corerec.engines as engines

    with pytest.warns(DeprecationWarning, match="num_epochs"):
        model = getattr(engines, cls_name)(num_epochs=3)
    assert model.epochs == 3



@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_misspelled_constructor_argument_raises(name):
    """ALS(factor=8) used to become a stray attribute and train with factors=64,
    so `corerec train --param factor=8` was silently ignored (#85)."""
    import corerec.engines as engines

    with pytest.raises(TypeError):
        getattr(engines, name)(definitely_not_a_param=1)


def test_embedding_cf_still_takes_its_own_parameters(tmp_path):
    from corerec.engines import ALS, Item2Vec

    assert ALS(factors=8, alpha=2.0, epochs=3, seed=1).iterations == 3
    assert Item2Vec(num_negatives=2, learning_rate=0.1).num_negatives == 2
    m = ALS(factors=8, iterations=2).fit([0, 0, 1, 2], [1, 2, 2, 3])
    m.save(str(tmp_path / "als.pkl"))
    assert ALS.load(str(tmp_path / "als.pkl")).factors == 8


@pytest.mark.parametrize("model_id,module_path,cls_name,kwargs", MODELS, ids=[m[0] for m in MODELS])
def test_unknown_users_have_no_personalized_recommendations(model_id, module_path, cls_name, kwargs):
    model = _build(module_path, cls_name, kwargs)
    users, items, ratings = _interactions()
    model.fit(users, items, ratings)
    assert model.recommend("__unknown_user__", top_k=3) == []
    assert len(model.recommend(users[0], top_k=3)) <= 3


@pytest.mark.parametrize("model_id,module_path,cls_name,kwargs",
                         [m for m in MODELS if m[0] in {"sar", "deepfm", "sasrec"}])
def test_unknown_user_does_not_hide_unfitted_model(model_id, module_path, cls_name, kwargs):
    from corerec.api.exceptions import ModelNotFittedError

    model = _build(module_path, cls_name, kwargs)
    with pytest.raises(ModelNotFittedError):
        model.recommend("__unknown_user__", top_k=3)
