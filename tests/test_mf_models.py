"""Production-contract tests for the embedding-CF family
(`corerec.engines.matrix_factorization`): ALS (WMF) and Item2Vec.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

MODELS = {"ALS": {"iterations": 10}, "Item2Vec": {"iterations": 10}}


def _data(n_users=60, n_items=80, seed=0):
    rng = np.random.RandomState(seed)
    u, i = [], []
    for usr in range(n_users):
        center = rng.randint(0, n_items)
        for _ in range(12):
            u.append(usr); i.append(int((center + rng.randint(-6, 7)) % n_items))
    return np.array(u), np.array(i), np.ones(len(u), dtype=float)


@pytest.fixture(params=list(MODELS))
def fitted(request):
    from corerec import engines
    u, i, r = _data()
    model = getattr(engines, request.param)(**MODELS[request.param])
    model.fit(u, i, r)
    return request.param, model, (u, i)


def test_importable(request_fixture=None):
    from corerec import engines
    for m in MODELS:
        assert getattr(engines, m) is not None


def test_predict_float(fitted):
    _, model, (u, i) = fitted
    assert isinstance(model.predict(int(u[0]), int(i[0])), float)


def test_recommend_known_items(fitted):
    name, model, _ = fitted
    recs = model.recommend(0, top_k=10)
    assert 0 < len(recs) <= 10 and all(it in model.item_map for it in recs)


def test_no_collapse(fitted):
    name, model, _ = fitted
    assert float(np.std(model._score_all_items(0))) > 1e-6, f"{name} collapsed"


def test_save_load_roundtrip(fitted, tmp_path):
    name, model, (u, i) = fitted
    base = str(tmp_path / name)
    before = model.predict(int(u[0]), int(i[0]))
    model.save(base)
    after = type(model).load(base).predict(int(u[0]), int(i[0]))
    assert abs(before - after) < 1e-4, f"{name} save/load mismatch"


@pytest.mark.parametrize("cls_name,options", [
    ("ALS", dict(factors=4, iterations=2, alpha=9, reg=0.5, seed=7)),
    ("Item2Vec", dict(factors=4, iterations=2, num_negatives=2, learning_rate=0.001, seed=7)),
])
def test_save_load_keeps_every_training_setting(cls_name, options, tmp_path):
    """load() rebuilt alpha, seed, num_negatives, learning_rate at defaults (#102)."""
    import corerec.engines as engines

    cls = getattr(engines, cls_name)
    users, items = [1, 1, 2, 2, 3], [10, 20, 20, 30, 10]
    model = cls(**options).fit(users, items)
    model.save(tmp_path / "m")
    loaded = cls.load(tmp_path / "m")
    assert {k: getattr(loaded, k) for k in options} == options
    # refitting the loaded object trains the same model as the original config
    assert loaded.fit(users, items).recommend(1, top_k=2) == cls(**options).fit(users, items).recommend(1, top_k=2)
