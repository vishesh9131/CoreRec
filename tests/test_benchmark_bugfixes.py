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


def _seed_data(seed=7, n_users=40, n_items=70, n=400):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, n_users, n).tolist(), rng.integers(0, n_items, n).tolist(), [1.0] * n)


@pytest.mark.parametrize("cls_name,kwargs", [
    ("TwoTower", {"embedding_dim": 16, "epochs": 3, "verbose": False}),
    ("DCN", {"embedding_dim": 8, "epochs": 2}),
    ("DeepFM", {"embedding_dim": 8, "hidden_layers": [16], "epochs": 2}),
    ("SASRec", {"hidden_units": 16, "num_blocks": 1, "epochs": 2, "batch_size": 32,
                "max_seq_length": 20, "verbose": False}),
])
def test_torch_models_are_reproducible_across_runs(cls_name, kwargs):
    """TwoTower and SASRec shuffled and sampled negatives from the unseeded global
    np.random, so two fits on the same data recommended different items."""
    import corerec.engines as engines

    u, i, r = _seed_data()

    def run(seed):
        m = getattr(engines, cls_name)(seed=seed, **kwargs)
        m.fit(u, i, r)
        return m.recommend(u[0], top_k=10)

    assert run(42) == run(42), f"{cls_name}: same seed produced different recommendations"
    assert run(42) != run(7), f"{cls_name}: different seeds gave identical output; seed ignored"


@pytest.mark.parametrize("cls_name,kwargs", [
    ("TwoTower", {"embedding_dim": 16, "epochs": 1, "verbose": False}),
    ("DCN", {"embedding_dim": 8, "epochs": 1}),
    ("DeepFM", {"embedding_dim": 8, "hidden_layers": [16], "epochs": 1}),
    ("SASRec", {"hidden_units": 16, "num_blocks": 1, "epochs": 1, "batch_size": 32,
                "max_seq_length": 20, "verbose": False}),
])
def test_torch_model_seed_survives_save_load(cls_name, kwargs, tmp_path):
    import corerec.engines as engines

    u, i, r = _seed_data(seed=3, n_users=30, n_items=50, n=300)
    cls = getattr(engines, cls_name)
    m = cls(seed=123, **kwargs)
    m.fit(u, i, r)
    m.save(str(tmp_path / "m.model"))
    assert cls.load(str(tmp_path / "m.model")).seed == 123


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
