"""Regression tests for defects found while writing BENCHMARKS.md.

Each test pins down one bug that was silent before, such as SAR's unbounded
similarity types scoring near-random with no warning. See BENCHMARKS.md and the commit that introduced this file for
the measurements that motivated each fix.
"""

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
# Reproducibility: same seed, same data, same model.
# --------------------------------------------------------------------------- #

# Models that draw from numpy's *global* RNG without ever seeding it, so two
# runs of identical code on identical data produce different models. Found by
# running the benchmark on two machines: LightGCN's NDCG@10 moved 2.5% between
# them, which is what prompted giving it a private Generator. These three have
# the same defect and are recorded rather than fixed here -- each needs a seed
# threaded through its constructor and persistence, which is a per-model change.
KNOWN_NONREPRODUCIBLE = {
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
