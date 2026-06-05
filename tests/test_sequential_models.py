"""Production-contract tests for the sequential model family
(`corerec.engines.sequential`): GRU4Rec, Caser, BST, DIN, DIEN. Each must fit on
chronological interactions without collapsing, predict, recommend `top_k` known
items, and round-trip through save/load with identical predictions.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

MODELS = ["GRU4Rec", "Caser", "BST", "DIN", "DIEN", "NARM"]


def _seq_data(n_users=60, n_items=80, seed=0):
    """Coherent per-user sequences so next-item training is learnable."""
    rng = np.random.RandomState(seed)
    u, i, ts = [], [], []
    t = 0
    for usr in range(n_users):
        center = rng.randint(0, n_items)
        for _ in range(15):
            u.append(usr)
            i.append(int((center + rng.randint(-5, 6)) % n_items))
            ts.append(t); t += 1
    return np.array(u), np.array(i), np.array(ts)


@pytest.fixture(params=MODELS)
def fitted(request):
    import corerec.engines.sequential as sq
    u, i, ts = _seq_data()
    model = getattr(sq, request.param)(embedding_dim=32, epochs=10, device="cpu")
    model.fit(u, i, ratings=np.ones(len(u)), timestamps=ts)
    return request.param, model, (u, i)


def test_importable_from_engines():
    from corerec import engines
    for m in MODELS:
        assert getattr(engines, m) is not None, f"{m} not exported from corerec.engines"


def test_predict_returns_float(fitted):
    _, model, (u, i) = fitted
    assert isinstance(model.predict(int(u[0]), int(i[0])), float)


def test_recommend_returns_known_items(fitted):
    name, model, _ = fitted
    recs = model.recommend(0, top_k=10)
    assert isinstance(recs, list) and 0 < len(recs) <= 10
    assert all(it in model.item_map for it in recs), f"{name} returned unknown items"


def test_no_output_collapse(fitted):
    name, model, _ = fitted
    scores = model._score_all_items(0)
    assert float(np.std(scores)) > 1e-5, f"{name} output collapsed"


def test_save_load_roundtrip(fitted, tmp_path):
    name, model, (u, i) = fitted
    base = str(tmp_path / name)
    before = model.predict(int(u[0]), int(i[0]))
    model.save(base)
    reloaded = type(model).load(base)
    after = reloaded.predict(int(u[0]), int(i[0]))
    assert abs(before - after) < 1e-4, f"{name} save/load prediction mismatch"
    assert len(reloaded.recommend(0, top_k=5)) == 5
