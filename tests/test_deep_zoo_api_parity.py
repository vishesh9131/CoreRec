"""API parity smoke for every model in the zoo.

Source of truth: corerec.engines.MODELS

Checks that every named model:
  - imports and subclasses BaseRecommender
  - exposes fit / predict / recommend / save / load
  - accepts fit(user_ids=..., item_ids=..., ratings=...)
  - round-trips save/load without blowing up on predict/recommend

Keep this tight. Ranking-loss gaps (BPR hooks etc.) are documented in PRs,
not asserted here — those are feature work, not contract breakages.
"""

import tempfile
from pathlib import Path

import pytest

from corerec.api.base_recommender import BaseRecommender
from corerec.engines import MODELS, list_models

# Models whose fit() is not the (user_ids, item_ids, ratings) triple, with why.
NOT_TRIPLE = {
    "SAR": "fit() takes a DataFrame; use fit_from_lists() for the triple form",
    "TFIDFRecommender": "content-based: fit(items, docs) takes item text, not interactions",
}
TRIPLE_MODELS = sorted(n for n in MODELS if n not in NOT_TRIPLE)


REQUIRED = ("fit", "predict", "recommend", "save", "load")


def _tiny():
    users = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    items = [10, 11, 12, 10, 13, 14, 11, 12, 15, 10, 14, 15]
    ratings = [1.0] * len(users)
    return users, items, ratings


def _build(name):
    import importlib
    import inspect

    mod = importlib.import_module(MODELS[name][0], "corerec.engines")
    cls = getattr(mod, name)
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    # tiny configs so CI stays cheap; only pass knobs the ctor actually takes
    candidates = {
        "embedding_dim": 8,
        "hidden_dims": [16],
        "hidden_units": 8,
        "epochs": 1,
        "batch_size": 4,
        "verbose": False,
        "max_seq_length": 8,
        "max_len": 8,
        "max_seq_len": 8,
        "num_blocks": 1,
        "num_layers": 1,
        "num_heads": 1,
    }
    kw = {k: v for k, v in candidates.items() if k in params}
    try:
        return cls(**kw)
    except TypeError:
        return cls()


def test_registry_and_list_models_agree():
    assert list_models() == list(MODELS)
    assert sorted(n for f in {m[1] for m in MODELS.values()} for n in list_models(f)) == sorted(MODELS)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_is_base_recommender_with_contract(name):
    import importlib

    mod = importlib.import_module(MODELS[name][0], "corerec.engines")
    cls = getattr(mod, name)
    assert issubclass(cls, BaseRecommender), f"{name} does not inherit BaseRecommender"
    for method in REQUIRED:
        assert hasattr(cls, method), f"{name} missing {method}"


@pytest.mark.parametrize("name", TRIPLE_MODELS)
def test_fit_predict_recommend_save_load_keyword_ratings(name):
    users, items, ratings = _tiny()
    model = _build(name)
    model.fit(user_ids=users, item_ids=items, ratings=ratings)

    score = model.predict(users[0], items[0])
    assert isinstance(score, (int, float))

    recs = model.recommend(users[0], top_k=3)
    assert isinstance(recs, list)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / name
        model.save(path)
        loaded = type(model).load(path)
        _ = loaded.predict(users[0], items[0])
        _ = loaded.recommend(users[0], top_k=2)
