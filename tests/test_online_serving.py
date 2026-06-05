"""Tests for the online serving engine (corerec.serving.OnlineRecommender):
ANN-backed recommend, graceful cold-start, and freshness (incremental add +
user fold-in) without retraining.
"""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _toy_interactions(n_users=80, n_items=120, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    # give each user a coherent "taste" cluster so embeddings are learnable
    for u in range(n_users):
        center = rng.randint(0, n_items)
        for _ in range(15):
            it = (center + rng.randint(-8, 9)) % n_items
            rows.append((u, int(it)))
    return pd.DataFrame(rows, columns=["user_id", "item_id"])


@pytest.fixture(scope="module")
def rec():
    from corerec.serving import OnlineRecommender
    df = _toy_interactions()
    return OnlineRecommender.from_interactions(df, dim=32, epochs=10, device="cpu")


def test_builds_with_faiss_or_fallback(rec):
    s = rec.stats()
    assert s["n_items"] > 0 and s["n_users"] > 0


def test_ann_recommend_returns_valid_items(rec):
    recs = rec.recommend(0, top_k=10)
    assert isinstance(recs, list) and len(recs) <= 10
    assert all(it in rec._item_pos for it in recs)


def test_recommend_excludes_seen(rec):
    seen = rec._seen.get(0, set())
    recs = rec.recommend(0, top_k=10, exclude_seen=True)
    assert not (set(recs) & seen), "recommendations must exclude already-seen items"


def test_cold_start_unknown_user_does_not_raise(rec):
    # the original failure mode: unknown user raised. Now -> popularity fallback.
    out = rec.recommend("user_that_never_existed", top_k=5)
    assert isinstance(out, list) and len(out) == 5


def test_freshness_add_items_without_retrain(rec):
    before = rec.stats()["n_items"]
    rec.add_items(["FRESH_A", "FRESH_B"], np.random.RandomState(1).randn(2, 32).astype("float32"))
    assert rec.stats()["n_items"] == before + 2
    assert "FRESH_A" in rec._item_pos


def test_freshness_fold_in_new_user_without_retrain(rec):
    # a brand-new user, described only by a few interactions, becomes serveable
    items = list(rec._item_pos)[:6]
    rec.fold_in_user("FOLD_USER", item_ids=items)
    recs = rec.recommend("FOLD_USER", top_k=5)
    assert isinstance(recs, list) and len(recs) == 5


def test_inner_product_path_serves(rec):
    # dot-product models (MF/ALS) need IP-metric ANN, not cosine. Build a small
    # IP index and confirm it serves without error and ranks the obvious match.
    from corerec.serving import OnlineRecommender
    item_emb = np.eye(4, dtype="float32") * np.array([1, 2, 3, 4], dtype="float32")[:, None]
    user_emb = np.array([[0, 0, 0, 5]], dtype="float32")  # most aligned with item 3
    r = OnlineRecommender.from_embeddings(
        item_ids=["a", "b", "c", "d"], item_emb=item_emb,
        user_ids=["u"], user_emb=user_emb, index_type="hnsw", metric="ip")
    assert r.recommend("u", top_k=1, exclude_seen=False)[0] == "d"


def test_latency_stats_present(rec):
    for _ in range(20):
        rec.recommend(1, top_k=10)
    s = rec.stats()
    assert s["queries_served"] >= 20
    assert s["latency_ms_p50"] >= 0.0 and s["latency_ms_p99"] >= s["latency_ms_p50"]
