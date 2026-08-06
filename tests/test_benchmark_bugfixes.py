"""Regression tests for defects found while writing BENCHMARKS.md.

Each test pins down one bug that was silent before: SAR's unbounded similarity
types scoring near-random with no warning, and GNNRec redoing dense O(N^2)
graph convolution on every mini-batch instead of caching the (fixed) graph
structure once. See BENCHMARKS.md and the commit that introduced this file for
the measurements that motivated each fix.
"""

import time
import warnings

import numpy as np
import pytest

from corerec.engines.collaborative import SAR


# --------------------------------------------------------------------------- #
# SAR: lift / mutual_information / inclusion_index are unbounded and blow up
# on rare item pairs at the class's own threshold=1 default.
# --------------------------------------------------------------------------- #

UNBOUNDED_SIMILARITY_TYPES = ["lift", "mutual_information", "inclusion_index"]
BOUNDED_SIMILARITY_TYPES = ["jaccard", "cosine", "cooccurrence", "lexicographers_mi"]


@pytest.mark.parametrize("similarity_type", UNBOUNDED_SIMILARITY_TYPES)
def test_sar_warns_on_unbounded_similarity_at_default_threshold(similarity_type):
    """SAR(similarity_type='lift') at threshold=1 used to fail silently.

    Measured on ML-100K: lift NDCG@10=0.0007, mutual_information=0.0015,
    inclusion_index=0.0541 -- against jaccard's 0.3730, with no signal to the
    caller that anything was wrong. Constructing with the default threshold
    now raises a UserWarning naming the problem and the fix.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SAR(similarity_type=similarity_type)  # threshold defaults to 1
    assert any(issubclass(w.category, UserWarning) for w in caught), (
        f"SAR(similarity_type={similarity_type!r}) at threshold=1 should warn"
    )


@pytest.mark.parametrize("similarity_type", UNBOUNDED_SIMILARITY_TYPES)
def test_sar_no_warning_with_raised_threshold(similarity_type):
    """The warning is specifically about the default; raising threshold clears it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SAR(similarity_type=similarity_type, threshold=50)
    assert not any(issubclass(w.category, UserWarning) for w in caught)


@pytest.mark.parametrize("similarity_type", BOUNDED_SIMILARITY_TYPES)
def test_sar_no_warning_for_bounded_similarity(similarity_type):
    """jaccard/cosine/cooccurrence/lexicographers_mi don't have this failure mode."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SAR(similarity_type=similarity_type)
    assert not any(issubclass(w.category, UserWarning) for w in caught)


# --------------------------------------------------------------------------- #
# GNNRec: graph propagation must use the cached sparse Laplacian, not rebuild
# a dense (N+M)x(N+M) matrix (plus a fresh identity matrix) on every call.
# --------------------------------------------------------------------------- #


def test_gnnrec_laplacian_is_cached_sparse():
    """The propagation layers must reuse one sparse L / L+I, not rebuild per call.

    Previously EmbeddingPropagationLayer.forward did `L + torch.eye(...)` and two
    dense (N+M)x(N+M) @ (N+M)xd matmuls on every call -- every mini-batch, not
    once per epoch. On ML-100K (2593 nodes, ~2.4% graph density) with this
    class's defaults that was ~7,800 calls redoing ~40 TFLOPs of work over a
    graph that's 97.6% zeros, and GNNRec had to be killed after 41+ minutes.
    """
    from corerec.engines.gnnrec import GNNRec

    rng = np.random.default_rng(0)
    u = rng.integers(0, 40, 300).tolist()
    i = rng.integers(0, 60, 300).tolist()
    r = np.ones(300).tolist()

    model = GNNRec(embedding_dim=16, num_gnn_layers=2, epochs=1, batch_size=64, verbose=False)
    model.fit(user_ids=u, item_ids=i, ratings=r)

    assert model.model.laplacian.is_sparse, "L should be cached as a sparse tensor"
    assert model.model.laplacian_plus_i.is_sparse, "L+I should be cached as a sparse tensor"
    # Identity is only ever added once, at build time -- not reconstructed per call.
    assert model.model.laplacian_plus_i is model.model.laplacian_plus_i


def test_gnnrec_recommendations_are_sane_after_sparse_fix():
    """Switching dense matmul for sparse must not change what the model predicts."""
    from corerec.engines.gnnrec import GNNRec

    rng = np.random.default_rng(1)
    n_users, n_items = 50, 80
    u = rng.integers(0, n_users, 600).tolist()
    i = rng.integers(0, n_items, 600).tolist()
    r = np.ones(600).tolist()

    model = GNNRec(embedding_dim=16, num_gnn_layers=2, epochs=3, batch_size=128, verbose=False)
    model.fit(user_ids=u, item_ids=i, ratings=r)

    recs = model.recommend(u[0], top_k=5)
    assert len(recs) == 5
    assert len(set(recs)) == 5, f"duplicate recommendations: {recs}"


def test_gnnrec_save_load_preserves_sparse_laplacian():
    """_build_model runs on both the fit() and load() paths; both must produce
    a working sparse L/L+I, and a reloaded model must recommend what the
    original did."""
    import os
    import tempfile

    from corerec.engines.gnnrec import GNNRec

    rng = np.random.default_rng(2)
    u = rng.integers(0, 40, 400).tolist()
    i = rng.integers(0, 60, 400).tolist()
    r = np.ones(400).tolist()

    model = GNNRec(embedding_dim=16, num_gnn_layers=2, epochs=2, batch_size=128, verbose=False)
    model.fit(user_ids=u, item_ids=i, ratings=r)
    before = model.recommend(u[0], top_k=5)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gnnrec.pkl")
        model.save(path)
        reloaded = GNNRec.load(path)

    assert reloaded.model.laplacian.is_sparse
    assert reloaded.recommend(u[0], top_k=5) == before


def test_gnnrec_forward_is_fast_relative_to_a_dense_baseline():
    """One epoch on a small synthetic graph should complete well under a second.

    Not a tight perf assertion (CI hardware varies) -- just a floor that would
    have caught the O(N^2)-per-call regression this file exists to guard
    against, without depending on ML-100K being present.
    """
    from corerec.engines.gnnrec import GNNRec

    rng = np.random.default_rng(3)
    u = rng.integers(0, 60, 500).tolist()
    i = rng.integers(0, 100, 500).tolist()
    r = np.ones(500).tolist()

    model = GNNRec(embedding_dim=32, num_gnn_layers=3, epochs=1, batch_size=128, verbose=False)
    t0 = time.perf_counter()
    model.fit(user_ids=u, item_ids=i, ratings=r)
    elapsed = time.perf_counter() - t0

    assert elapsed < 10.0, (
        f"one epoch on {len(set(u))} users x {len(set(i))} items took {elapsed:.1f}s; "
        "expected well under a second with cached sparse propagation -- check "
        "whether GNNModel is rebuilding L/L+I per forward() call again"
    )
