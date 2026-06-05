"""Production-contract tests for the deep CTR / feature-interaction model family
(`corerec.engines.deep_ctr`). Every model must: fit on implicit data without
collapsing, predict a float, recommend `top_k` known items, and round-trip through
save/load with identical predictions.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

MODELS = ["FM", "AFM", "NFM", "DeepFMCTR", "DCNCTR", "AutoInt",
          "xDeepFM", "FiBiNet", "PNN", "WideDeep", "GMF", "MLP"]


def _data(n_users=60, n_items=80, seed=0):
    rng = np.random.RandomState(seed)
    u, i = [], []
    for usr in range(n_users):
        center = rng.randint(0, n_items)
        for _ in range(12):
            u.append(usr)
            i.append(int((center + rng.randint(-6, 7)) % n_items))
    return np.array(u), np.array(i), np.ones(len(u), dtype=float)


@pytest.fixture(params=MODELS)
def fitted(request):
    import corerec.engines.deep_ctr as dc
    u, i, r = _data()
    model = getattr(dc, request.param)(embedding_dim=16, epochs=15, device="cpu")
    model.fit(u, i, r)
    return request.param, model, (u, i, r)


def test_importable_from_engines():
    from corerec import engines
    for m in MODELS:
        assert getattr(engines, m) is not None, f"{m} not exported from corerec.engines"


def test_predict_returns_float(fitted):
    _, model, (u, i, _) = fitted
    s = model.predict(int(u[0]), int(i[0]))
    assert isinstance(s, float)


def test_recommend_returns_known_items(fitted):
    name, model, _ = fitted
    recs = model.recommend(0, top_k=10)
    assert isinstance(recs, list) and 0 < len(recs) <= 10
    assert all(it in model.item_map for it in recs), f"{name} returned unknown items"


def test_no_output_collapse(fitted):
    name, model, _ = fitted
    scores = model._score_all_items(0)
    # true collapse (the old sigmoid+BCE-on-all-positives bug) gives std ~0;
    # this catches that while tolerating low-but-nonzero variance on tiny data.
    assert float(np.std(scores)) > 1e-5, f"{name} output collapsed"


def test_save_load_roundtrip(fitted, tmp_path):
    name, model, (u, i, _) = fitted
    base = str(tmp_path / name)
    before = model.predict(int(u[0]), int(i[0]))
    model.save(base)
    reloaded = type(model).load(base)
    after = reloaded.predict(int(u[0]), int(i[0]))
    assert abs(before - after) < 1e-4, f"{name} save/load prediction mismatch"
    assert len(reloaded.recommend(0, top_k=5)) == 5
