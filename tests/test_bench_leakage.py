"""Leakage / cold-start guards for the shared Findings/bench eval path.

These run without MovieLens on disk. Floor tests still need the real split.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "Findings", "bench")
sys.path.insert(0, BENCH)

import datautil  # noqa: E402
import metrics as M  # noqa: E402


def test_bench_path_is_findings_not_under_corerec():
    # regression for the old <repo>/corerec/Findings/bench pointer
    assert BENCH.endswith(os.path.join("Findings", "bench"))
    assert os.path.isdir(BENCH)
    assert not os.path.isdir(os.path.join(REPO, "corerec", "Findings", "bench"))


def test_ml100k_dir_default_resolves_under_repo_root():
    d = os.path.abspath(datautil.ml100k_dir())
    # two ups from Findings/bench lands in the repo, not above it
    assert d.startswith(os.path.abspath(REPO))
    assert d.endswith(os.path.join("cr_learn_setup", "cr_learn", "CRDS", "ml_100k"))


def test_ml100k_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("COREC_ML100K_DIR", str(tmp_path))
    assert datautil.ml100k_dir() == str(tmp_path)
    assert datautil.ml100k_available() is False
    (tmp_path / "u1.base").write_text("1\t1\t5\t0\n")
    (tmp_path / "u1.test").write_text("1\t2\t5\t0\n")
    assert datautil.ml100k_available() is True


def test_index_drops_cold_users_and_items():
    train = pd.DataFrame({
        "user": [10, 10, 20],
        "item": [100, 101, 100],
        "rating": [5.0, 4.0, 3.0],
        "ts": [1, 2, 3],
    })
    # (10,100) overlaps train; 99 is cold user; 999 is cold item;
    # (20,101) is warm+novel (item exists in train, just not for user 20)
    test = pd.DataFrame({
        "user": [10, 20, 99, 20],
        "item": [100, 101, 100, 999],
        "rating": [5.0, 5.0, 5.0, 5.0],
        "ts": [4, 5, 6, 7],
    })
    out = datautil._index_from_train(train, test)
    pairs = set(zip(out["test"]["user"].tolist(), out["test"]["item"].tolist()))
    assert (10, 100) not in pairs          # overlap stripped
    assert (99, 100) not in pairs          # cold user
    assert (20, 999) not in pairs          # cold item
    assert (20, 101) in pairs              # only warm novel pair survives
    assert out["n_users"] == 2
    assert out["n_items"] == 2


def test_ranking_metrics_masks_train_items():
    # item 0 is train-seen and also wrongly marked relevant; if masking works
    # it cannot appear in the top-1, so NDCG stays 0 unless we pick item 1.
    n_items = 4
    seen = {0: {0}}
    relevant = {0: {0, 1}}  # 0 is leaked; 1 is the real holdout

    def score_fn(u):
        # model prefers the leaked train item over everything
        return np.array([10.0, 1.0, 0.5, 0.0])

    res = M.ranking_metrics(score_fn, n_items, seen, relevant, k=1, user_subset=[0])
    # after mask, top-1 must be item 1 -> hit
    assert res["HitRate@1"] == 1.0
    assert res["NDCG@1"] == 1.0
    assert res["n_eval_users"] == 1


def test_ranking_metrics_ignores_leaked_relevant_only():
    # if the ONLY "relevant" item was also in train, user is skipped (not a free hit)
    n_items = 3
    seen = {0: {1}}
    relevant = {0: {1}}

    def score_fn(u):
        return np.array([0.0, 9.0, 0.0])

    res = M.ranking_metrics(score_fn, n_items, seen, relevant, k=1, user_subset=[0])
    assert res["n_eval_users"] == 0
    assert res["NDCG@1"] == 0.0


def test_load_split_missing_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("COREC_ML100K_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="COREC_ML100K_DIR|GroupLens"):
        datautil.load_split()
