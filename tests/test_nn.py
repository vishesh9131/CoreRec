"""corerec.nn: a researcher's own torch module gets the whole CoreRec lifecycle."""

import math

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from corerec.nn import (HSTUBlock, MatrixFactorization, Recommender, SequentialTransformer,
                        causal_mask)


class PlainDot(nn.Module):
    """A user-written module with only forward(): exercises the score_all fallback."""

    def __init__(self, n_users, n_items, dim=8):
        super().__init__()
        self.u = nn.Embedding(n_users, dim)
        self.i = nn.Embedding(n_items + 1, dim, padding_idx=0)

    def forward(self, users, items):
        return (self.u(users).unsqueeze(1) * self.i(items)).sum(-1)


class HSTUEncoder(nn.Module):
    """A custom sequential model assembled from an exported block."""

    def __init__(self, n_items, max_len, dim=16):
        super().__init__()
        self.items = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.block = HSTUBlock(dim, 1, dim, dim, 0.0, max_len, 8, False)

    def forward(self, history, items):
        x = self.block(self.items(history), causal_mask(history.shape[1], history.device), None)
        return (x[:, -1].unsqueeze(1) * self.items(items)).sum(-1)


def _groups(n_users=120, seed=0):
    """Users in group g interact with items in group g."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        g = u % 4
        for it in rng.choice(np.arange(g * 25, (g + 1) * 25), 8, replace=False):
            rows.append((f"u{u}", int(it) * 7 + 3))  # non-contiguous item ids
    return pd.DataFrame(rows, columns=["user_id", "item_id"])


def _group_of(item):
    return ((item - 3) // 7) // 25


def _chains(n_users=300, seed=0):
    rng = np.random.default_rng(seed)
    rows, nxt = [], {}
    for u in range(n_users):
        s, n = int(rng.integers(0, 40)), int(rng.integers(3, 9))
        rows += [(u, s + k, k) for k in range(n)]
        nxt[u] = s + n
    return pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"]), nxt


@pytest.fixture(scope="module")
def mf():
    return Recommender(MatrixFactorization, {"dim": 16}, epochs=30, batch_size=256, lr=0.01,
                       device="cpu").fit(_groups())


def test_matrix_factorization_learns_the_groups(mf):
    users = _groups().user_id.unique()[:40]
    in_group = [all(_group_of(i) == int(u[1:]) % 4 for i in mf.recommend(u, top_k=5)) for u in users]
    assert np.mean(in_group) > 0.9


def test_recommend_contract(mf):
    df = _groups()
    u = df.user_id.iloc[0]
    recs = mf.recommend(u, top_k=5)
    assert len(recs) == 5 and len(set(recs)) == 5
    assert not set(recs) & set(df[df.user_id == u].item_id), "seen items must be excluded"
    assert not set(recs[:2]) & set(mf.recommend(u, top_k=5, exclude_items=recs[:2]))
    assert mf.recommend("nobody", top_k=5) == []
    assert isinstance(mf.predict(u, recs[0]), float)


def test_forward_only_module_uses_the_fallback():
    df = _groups()
    rec = Recommender(PlainDot, epochs=2, device="cpu").fit(df)
    u = df.user_id.iloc[0]
    with torch.no_grad():
        q = torch.tensor([rec.user_map[u]])
        manual = rec.model(q, torch.arange(1, rec.num_items + 1).unsqueeze(0))[0].numpy()
    seen = {rec.item_map[i] - 1 for i in df[df.user_id == u].item_id}
    manual[list(seen)] = -np.inf
    want = [rec._items[j] for j in np.argsort(-manual)[:5]]
    assert rec.recommend(u, top_k=5) == want


def test_wrong_output_shape_is_reported():
    class Bad(PlainDot):
        def forward(self, users, items):
            return super().forward(users, items).sum(-1)

    with pytest.raises(ValueError, match="expected"):
        Recommender(Bad, epochs=1, device="cpu").fit(_groups())


def test_sequential_transformer_learns_next_item():
    df, nxt = _chains()
    rec = Recommender(SequentialTransformer, {"dim": 32, "num_blocks": 1}, inputs="history",
                      loss="sampled_softmax", num_negatives=20, epochs=15, batch_size=128,
                      lr=0.005, max_len=20, device="cpu").fit(df)
    hits = [nxt[u] in rec.recommend(u, top_k=1) for u in nxt if nxt[u] < 48]
    assert np.mean(hits) > 0.9


def test_custom_block_model_trains():
    df, _ = _chains(100)
    rec = Recommender(HSTUEncoder, inputs="history", epochs=1, max_len=10, device="cpu").fit(df)
    assert len(rec.recommend(0, top_k=3)) == 3


def test_save_load_and_model_loader(mf, tmp_path):
    from corerec.serving import ModelLoader

    u = _groups().user_id.iloc[0]
    path = tmp_path / "mf.pt"
    mf.save(path)
    assert Recommender.load(path, device="cpu").recommend(u, top_k=5) == mf.recommend(u, top_k=5)
    loaded = ModelLoader().load(str(path))
    assert isinstance(loaded, Recommender)
    assert loaded.recommend(u, top_k=5) == mf.recommend(u, top_k=5)


def test_evaluator_runs_on_it(mf):
    from corerec.evaluation import Evaluator

    df = _groups()
    truth = {u: df[df.user_id == u].item_id.tolist() for u in df.user_id.unique()[:20]}
    out = Evaluator(metrics=["ndcg@10"]).evaluate(mf, truth)
    assert out["n_errors"] == 0 and not math.isnan(out["ndcg@10"])


def test_model_server_serves_it(mf):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from corerec.serving import ModelServer

    u = _groups().user_id.iloc[0]
    body = TestClient(ModelServer(mf).app).post("/recommend", json={"user_id": u, "top_k": 5}).json()
    assert body["recommendations"] == mf.recommend(u, top_k=5)


@pytest.mark.parametrize("mode", ["user", "history"])
def test_onnx_export_matches(mode, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    import json

    from corerec.export import to_onnx

    if mode == "user":
        df = _groups()
        rec = Recommender(MatrixFactorization, {"dim": 8}, epochs=2, device="cpu").fit(df)
    else:
        df, _ = _chains(100)
        rec = Recommender(SequentialTransformer, {"dim": 16, "num_blocks": 1}, inputs="history",
                          epochs=1, max_len=10, device="cpu").fit(df)
    sess = ort.InferenceSession(str(to_onnx(rec, tmp_path / "m.onnx")))
    items = json.loads(sess.get_modelmeta().custom_metadata_map["item_ids"])
    u = df.user_id.iloc[0]
    k = rec.user_map[u]
    x = (np.array([k, k], np.int64) if mode == "user"
         else rec._histories([rec._sequences[k]] * 2))
    scores = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    assert scores.shape == (2, rec.num_items)
    top = [items[j] for j in np.argsort(-scores[0])[:5]]
    assert top == rec.recommend(u, top_k=5, exclude_seen=False)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_trains_on_mps():
    rec = Recommender(MatrixFactorization, {"dim": 8}, epochs=1, device="mps").fit(_groups())
    assert rec.device.type == "mps"
    assert len(rec.recommend(_groups().user_id.iloc[0], top_k=5)) == 5
