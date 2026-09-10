"""LightGCN / GNNRec graph-path regressions.

Pins down hangs and API mismatches found while auditing the bipartite path:
saturated-user negative sampling, recommend leaking seen items when top_k is
larger than the unseen pool, cold users raising instead of returning [], and
the adj builder densifying the graph via diag().
"""

import os
import signal

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


class _Timeout(BaseException):
    """BaseException so a stuck loop can't swallow it via bare except Exception."""


def _alarm_handler(signum, frame):
    raise _Timeout("timed out")


def _fit_lightgcn(users, items, **kwargs):
    from corerec.engines.collaborative.graph_based_base.lightgcn import LightGCN

    defaults = dict(
        n_factors=8,
        n_layers=2,
        epochs=2,
        batch_size=16,
        verbose=False,
        seed=0,
        device="cpu",
        early_stopping_patience=5,
    )
    defaults.update(kwargs)
    model = LightGCN(**defaults)
    model.fit(users, items, [1.0] * len(users))
    return model


def test_lightgcn_adj_stays_sparse():
    from corerec.engines.collaborative.graph_based_base.lightgcn import LightGCN
    from scipy.sparse import csr_matrix

    model = LightGCN(n_factors=4, n_layers=1, epochs=1, verbose=False, device="cpu", seed=0)
    model.n_users = 5
    model.n_items = 7
    mat = csr_matrix(([1.0, 1.0, 1.0], ([0, 1, 2], [0, 3, 5])), shape=(5, 7))
    adj = model._create_adjacency_matrix(mat)
    assert adj.is_sparse, "normalized bipartite adj densified; use edge-wise D^-1/2"
    assert adj.shape == (12, 12)


def test_lightgcn_saturated_user_fit_does_not_hang():
    # user 0 touches every item in a tiny catalog -- old rejection sampler spun
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(3)
    try:
        model = _fit_lightgcn([0, 0, 0, 1], [0, 1, 2, 0], n_layers=1, epochs=1, n_factors=4)
    finally:
        signal.alarm(0)
    assert model.is_fitted
    # only item left to recommend for user 0 is... nothing
    assert model.recommend(0, top_k=5) == []


def test_lightgcn_single_item_catalog_fit_does_not_hang():
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(3)
    try:
        model = _fit_lightgcn([0, 1], [5, 5], n_layers=1, epochs=1, n_factors=4)
    finally:
        signal.alarm(0)
    assert model.is_fitted
    assert model.recommend(0, top_k=3) == []


def test_lightgcn_recommend_does_not_leak_seen_items():
    # 4 items, user 0 saw two of them; top_k=3 used to pad with a seen id
    model = _fit_lightgcn(
        [0, 0, 1, 1, 2, 2],
        [10, 11, 10, 12, 11, 13],
    )
    recs = model.recommend(0, top_k=3)
    assert 10 not in recs and 11 not in recs
    assert len(recs) <= 2
    assert set(recs).issubset({12, 13})


def test_lightgcn_cold_user_and_item_are_graceful():
    model = _fit_lightgcn([0, 1], [10, 11])
    assert model.recommend(999, top_k=5) == []
    assert model.predict(999, 10) == 0.0
    assert model.predict(0, 999) == 0.0


def test_lightgcn_predict_recommend_rank_agree():
    model = _fit_lightgcn(
        [0, 0, 0, 1, 1, 2, 2],
        [0, 1, 2, 0, 3, 1, 4],
        epochs=3,
        seed=2,
    )
    seen = {0, 1, 2}
    scored = []
    for idx in range(model.n_items):
        oid = model.reverse_item_map[idx]
        if oid in seen:
            continue
        scored.append((oid, model.predict(0, oid)))
    scored.sort(key=lambda x: -x[1])
    from_pred = [oid for oid, _ in scored[:3]]
    assert model.recommend(0, top_k=3) == from_pred


def test_lightgcn_save_load_keeps_seen_exclusion(tmp_path):
    model = _fit_lightgcn(
        [0, 0, 1, 1, 2, 2, 3],
        [10, 11, 10, 12, 11, 13, 14],
        seed=1,
    )
    before = model.recommend(0, top_k=2)
    path = tmp_path / "lightgcn_bundle"
    model.save(path)
    from corerec.engines.collaborative.graph_based_base.lightgcn import LightGCN

    loaded = LightGCN.load(path)
    assert loaded.recommend(0, top_k=2) == before
    assert 10 not in before and 11 not in before


def test_gnnrec_cold_user_returns_empty_not_error():
    from corerec.engines.gnnrec import GNNRec

    model = GNNRec(
        embedding_dim=8,
        num_gnn_layers=1,
        epochs=1,
        batch_size=16,
        verbose=False,
        device="cpu",
        num_negatives=1,
    )
    model.fit([0, 0, 1, 1], [0, 1, 0, 2], [1.0, 1.0, 1.0, 1.0])
    assert model.recommend(0, top_k=2)
    assert model.recommend(999, top_k=2) == []
    assert model.predict(999, 0) == 0.0


def test_gnnrec_saturated_user_fit_skips_false_negatives():
    from corerec.engines.gnnrec import GNNRec

    model = GNNRec(
        embedding_dim=4,
        num_gnn_layers=1,
        epochs=1,
        batch_size=8,
        verbose=False,
        device="cpu",
        num_negatives=3,
    )
    # one user, three items, all observed -- negatives must not be forced
    model.fit([0, 0, 0], [0, 1, 2], [1.0, 1.0, 1.0])
    assert model.is_fitted
    assert model.recommend(0, top_k=5) == []
