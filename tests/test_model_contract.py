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

# (test id, import path, class name, constructor kwargs kept small for speed)
MODELS = [
    ("two_tower", "corerec.engines.two_tower", "TwoTower",
     {"embedding_dim": 16, "num_epochs": 3, "verbose": False}),
    ("bert4rec", "corerec.engines.bert4rec", "BERT4Rec", {}),
    ("dcn", "corerec.engines.dcn", "DCN", {"embedding_dim": 16, "epochs": 2}),
    ("deepfm", "corerec.engines.deepfm", "DeepFM", {"embedding_dim": 16, "epochs": 2}),
    ("ncf", "corerec.engines.collaborative.nn_base.ncf", "NCF", {}),
    ("lightgcn", "corerec.engines.collaborative.graph_based_base.lightgcn",
     "LightGCN", {}),
]

# model id -> why it cannot meet the common contract yet.
# These are DataFrame-first models with their own documented entry points
# (fit_from_lists, fit_from_dataset). Changing a released fit() signature is its
# own decision, not a drive-by in a test file -- so the divergence is recorded
# here and shows up as xfail rather than being silently tolerated.
KNOWN_DIVERGENT = {
    "sar": "fit() takes a DataFrame; use fit_from_lists() for the triple form",
    "ncf": "fit(data, validation_data) takes a DataFrame; see fit_from_dataset()",
}


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
    try:
        return cls(**kwargs)
    except TypeError:
        return cls()  # constructor kwargs are a convenience, not the contract


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
