"""The DCN, DeepFM and LightGCN tutorials run and evaluate what the models predict (#227).

They reported RMSE of click probabilities (DCN/DeepFM, task="auto") and BPR
scores (LightGCN) against 1-5 ratings. Every block runs here with
ml_1m.load() replaced by a small frame of the same shape, and epochs cut, so
nothing is downloaded.
"""

import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TUTORIALS = Path(__file__).resolve().parents[1] / "docs" / "source" / "tutorials"


def _fake_ml_1m():
    rng = np.random.default_rng(0)
    n = 3000
    users = rng.integers(1, 120, n)
    ratings = pd.DataFrame({"user_id": users, "movie_id": (users % 10) * 20 + rng.integers(1, 20, n),
                            "rating": rng.integers(1, 6, n), "timestamp": np.arange(n)})
    return types.SimpleNamespace(load=lambda: {"ratings": ratings})


@pytest.mark.parametrize("name", ["dcn", "deepfm", "lightgcn"])
def test_tutorial_runs_and_evaluates_ranking(name, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cr_learn", types.SimpleNamespace(ml_1m=_fake_ml_1m()))
    monkeypatch.chdir(tmp_path)
    text = (TUTORIALS / f"{name}_tutorial.md").read_text()
    assert "Test RMSE" not in text.split("task=\"rating\"")[0], "RMSE before switching to task='rating'"
    ns = {}
    for i, block in enumerate(re.findall(r"```python\n(.*?)```", text, re.S)):
        block = re.sub(r"epochs=\d+", "epochs=1", block)
        try:
            exec(compile(block, f"{name}_tutorial[block {i}]", "exec"), ns)
        except Exception as e:
            pytest.fail(f"{name}_tutorial.md block {i}: {type(e).__name__}: {e}\n{block}")
    assert 0.0 <= ns["metrics"]["NDCG@10"] <= 1.0
