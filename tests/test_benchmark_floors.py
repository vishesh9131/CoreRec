"""Production guardrails for the deep recommenders.

Two layers:

1. ``test_no_output_collapse_*`` (fast, always on): trains each deep model on a
   small synthetic dataset and asserts the per-item score variance is non-trivial.
   This is a regression guard for the sigmoid+BCE-on-all-positives collapse that
   previously made DCN/DeepFM/GNNRec emit a constant score.

2. ``test_ndcg_floor_ml100k`` (slower, skipped if the dataset is absent): trains
   on the canonical MovieLens-100K split and asserts NDCG@10 clears a floor, so a
   future refactor cannot silently re-break a model's accuracy.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "Findings", "bench")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # deterministic CPU for CI


def _synthetic(n_users=60, n_items=80, seed=0):
    """A small dataset with latent structure so a working model learns variance."""
    rng = np.random.RandomState(seed)
    users, items, ratings = [], [], []
    for u in range(n_users):
        liked = rng.choice(n_items, size=12, replace=False)
        for it in liked:
            users.append(u)
            items.append(int(it))
            ratings.append(float(rng.randint(1, 6)))  # raw 1-5, the trap case
    return np.array(users), np.array(items), np.array(ratings, dtype=float)


def _score_std(model, n_items, user=0):
    pairs = [(user, i) for i in range(n_items)]
    scores = np.asarray(model.batch_predict(pairs), dtype=float)
    return float(scores.std())


@pytest.mark.parametrize("model_name", ["DCN", "DeepFM"])
def test_no_output_collapse_ctr(model_name):
    """DCN/DeepFM trained on raw 1-5 ratings must NOT collapse to a constant."""
    from corerec.engines import DCN, DeepFM

    u, i, r = _synthetic()
    n_items = int(i.max()) + 1
    cls = {"DCN": DCN, "DeepFM": DeepFM}[model_name]
    model = cls(embedding_dim=16, epochs=5, verbose=False, device="cpu")
    model.fit(user_ids=u, item_ids=i, ratings=r)
    std = _score_std(model, n_items)
    assert std > 1e-3, f"{model_name} output collapsed (score std={std:.2e})"


def test_no_output_collapse_gnnrec():
    from corerec.engines import GNNRec

    u, i, r = _synthetic()
    n_items = int(i.max()) + 1
    model = GNNRec(embedding_dim=16, epochs=5, verbose=False, device="cpu")
    model.fit(u, i, r)
    std = _score_std(model, n_items)
    assert std > 1e-3, f"GNNRec output collapsed (score std={std:.2e})"


def test_rating_task_predicts_in_range():
    """task='rating' should regress the rating scale, not emit a [0,1] score."""
    from corerec.engines import DCN

    u, i, r = _synthetic()
    model = DCN(embedding_dim=16, epochs=8, verbose=False, device="cpu", task="rating")
    model.fit(user_ids=u, item_ids=i, ratings=r)
    preds = [model.predict(0, it) for it in range(int(i.max()) + 1)]
    # a rating-task model should produce values around the 1-5 scale, not ~1.0
    assert max(preds) > 1.5, f"rating-task predictions look like [0,1] scores: max={max(preds):.3f}"


@pytest.mark.parametrize("model_name", ["DCN", "DeepFM"])
def test_rating_task_persistence_roundtrip(tmp_path, model_name):
    """A task='rating' model must reload with its linear head (no sigmoid), so
    predictions are identical after save/load."""
    from corerec.engines import DCN, DeepFM

    u, i, r = _synthetic()
    cls = {"DCN": DCN, "DeepFM": DeepFM}[model_name]
    m = cls(embedding_dim=16, epochs=5, verbose=False, device="cpu", task="rating")
    m.fit(user_ids=u, item_ids=i, ratings=r)
    base = str(tmp_path / f"{model_name}_rating")
    before = m.predict(0, 5)
    m.save(base)
    reloaded = cls.load(base)
    assert reloaded._fit_task == "rating"
    after = reloaded.predict(0, 5)
    assert abs(before - after) < 1e-5, f"{model_name} rating round-trip mismatch: {before} vs {after}"


# ---- slower accuracy floor on the real ML-100K split ---------------------- #
_FLOORS = {  # NDCG@10 floors; the collapsed models used to score < 0.02
    ("corerec", "SAR"): 0.25,
    ("corerec", "DCN"): 0.05,
    ("corerec", "DeepFM"): 0.05,
    ("corerec", "NCF"): 0.05,
    ("corerec", "LightGCN"): 0.05,
}


def _ml100k_available():
    return os.path.isfile(os.path.join(
        REPO, "cr_learn_setup", "cr_learn", "CRDS", "ml_100k", "u1.base"))


@pytest.mark.skipif(not _ml100k_available(), reason="ML-100K data not present")
@pytest.mark.parametrize("framework,model", list(_FLOORS.keys()))
def test_ndcg_floor_ml100k(framework, model):
    sys.path.insert(0, BENCH)
    import datautil
    import metrics as M
    import runner as R

    R.DEVICE = "cpu"
    R.EPOCHS = 8  # keep CI quick; floors set accordingly
    data = datautil.load_split()
    seen = datautil.train_seen(data)
    rel = datautil.test_relevant(data)
    eu = datautil.eval_users(data, n=150, seed=42)

    score_fn, _, _ = R.DISPATCH[framework](model, data)
    res = M.ranking_metrics(score_fn, data["n_items"], seen, rel, k=10, user_subset=eu)
    ndcg = res["NDCG@10"]
    assert ndcg >= _FLOORS[(framework, model)], (
        f"{framework}/{model} NDCG@10={ndcg:.4f} below floor "
        f"{_FLOORS[(framework, model)]}")
