"""Regression tests for defects found while writing BENCHMARKS.md.

Each test pins down one bug that was silent before: SAR's unbounded similarity
types scoring near-random with no warning, and GNNRec rebuilding the (fixed)
graph structure on every mini-batch instead of caching it once. See BENCHMARKS.md and the commit that introduced this file for
the measurements that motivated each fix.
"""

import time
import warnings

import numpy as np
import pytest
import torch

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
# GNNRec: graph propagation must reuse a cached L / L+I rather than
# reconstructing the identity matrix on every call.
# --------------------------------------------------------------------------- #


def test_gnnrec_caches_laplacian_and_identity():
    """L and L+I must be built once per model, not rebuilt inside forward().

    EmbeddingPropagationLayer.forward used to do `L + torch.eye(...)` on every
    call -- every mini-batch, ~7,800 times over a 20-epoch ML-100K run. Both are
    fixed for the life of the model, so both are cached at build time.

    Deliberately does NOT assert sparsity: sparse was measured at 14.3s against
    8.1s dense for a 3-epoch fit, so the dense form is the correct one here.
    See the comment in GNNModel.__init__.
    """
    from corerec.engines.gnnrec import GNNRec

    rng = np.random.default_rng(0)
    u = rng.integers(0, 40, 300).tolist()
    i = rng.integers(0, 60, 300).tolist()
    r = np.ones(300).tolist()

    model = GNNRec(embedding_dim=16, num_gnn_layers=2, epochs=1, batch_size=64, verbose=False)
    model.fit(user_ids=u, item_ids=i, ratings=r)

    assert model.model.laplacian is not None
    assert model.model.laplacian_plus_i is not None
    # L+I must actually be L plus an identity, computed once.
    n = model.model.laplacian.size(0)
    expected = model.model.laplacian + torch.eye(n, device=model.model.laplacian.device)
    assert torch.allclose(model.model.laplacian_plus_i, expected)


def test_gnnrec_recommendations_are_sane():
    """Caching the graph must not change what the model predicts."""
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


def test_gnnrec_save_load_preserves_laplacian():
    """_build_model runs on both the fit() and load() paths; both must produce
    a working cached L/L+I, and a reloaded model must recommend what the
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

    assert reloaded.model.laplacian is not None
    assert reloaded.recommend(u[0], top_k=5) == before


def test_gnnrec_forward_is_fast():
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
        "expected well under a second with cached propagation -- check "
        "whether GNNModel is rebuilding L/L+I per forward() call again"
    )


# --------------------------------------------------------------------------- #
# Reproducibility: same seed, same data, same model.
# --------------------------------------------------------------------------- #

# Models that draw from numpy's *global* RNG without ever seeding it, so two
# runs of identical code on identical data produce different models. Found by
# running the benchmark on two machines: LightGCN's NDCG@10 moved 2.5% between
# them, which is what prompted giving it a private Generator. These three have
# the same defect and are recorded rather than fixed here -- each needs a seed
# threaded through its constructor and persistence, which is a per-model change.
KNOWN_NONREPRODUCIBLE = {
    "bert4rec": "uses np.random.* with no seeding; needs a seed parameter",
    "two_tower": "uses np.random.* with no seeding; needs a seed parameter",
    "sasrec": "uses np.random.* with no seeding; needs a seed parameter",
}


def test_lightgcn_is_reproducible_across_runs():
    """Same seed must give the same model.

    LightGCN sampled negatives with np.random.randint and shuffled epochs with
    np.random.shuffle, both against the global RNG that nothing seeded. It now
    owns a numpy Generator seeded from its `seed` argument.
    """
    from corerec.engines.collaborative.graph_based_base.lightgcn import LightGCN

    rng = np.random.default_rng(7)
    u = rng.integers(0, 40, 400).tolist()
    i = rng.integers(0, 70, 400).tolist()
    r = [1.0] * 400

    def run(seed):
        m = LightGCN(n_factors=16, n_layers=2, epochs=4, verbose=False, seed=seed)
        m.fit(user_ids=u, item_ids=i, ratings=r)
        return m.recommend(u[0], top_k=5)

    assert run(42) == run(42), "same seed produced different recommendations"
    assert run(42) != run(7), "different seeds produced identical output; seed ignored"


def test_lightgcn_seed_survives_save_load():
    """The seed is part of the model's configuration, so it must persist."""
    import os
    import tempfile

    from corerec.engines.collaborative.graph_based_base.lightgcn import LightGCN

    rng = np.random.default_rng(3)
    u = rng.integers(0, 30, 300).tolist()
    i = rng.integers(0, 50, 300).tolist()
    r = [1.0] * 300

    m = LightGCN(n_factors=16, n_layers=2, epochs=3, verbose=False, seed=123)
    m.fit(user_ids=u, item_ids=i, ratings=r)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "lightgcn.pkl")
        m.save(p)
        reloaded = LightGCN.load(p)
    assert reloaded.seed == 123
