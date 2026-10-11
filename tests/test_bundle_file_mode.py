"""Saved models are readable by other users, like any file the process writes (#210).

The safe bundle wrote through NamedTemporaryFile, which is always 0600, so a
model copied into an image and run as a non-root user failed to load.
"""

import os
import stat
import sys

import numpy as np
import pytest

from corerec.api.model_bundle import _UMASK
from corerec.engines import DCN

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")


def _modes(directory):
    return {p.name: stat.S_IMODE(os.stat(p).st_mode) for p in directory.iterdir() if p.is_file()}


def test_bundle_files_get_the_umask_mode(tmp_path):
    rng = np.random.default_rng(0)
    model = DCN(embedding_dim=4, epochs=1, device="cpu").fit(
        rng.integers(0, 20, 200).tolist(), rng.integers(0, 30, 200).tolist(), [1.0] * 200)
    model.save(tmp_path / "dcn" / "model")
    modes = _modes(tmp_path / "dcn")
    assert any(n.endswith(".meta.json") for n in modes) and any(n.endswith(".npz") for n in modes)
    assert set(modes.values()) == {0o666 & ~_UMASK}
    if _UMASK == 0o022:  # the usual default: group and others can read
        assert set(modes.values()) == {0o644}
    assert DCN.load(tmp_path / "dcn" / "model").recommend(0, top_k=3)


def test_cli_artifacts_get_the_umask_mode(tmp_path):
    import pandas as pd

    from corerec.serving.from_csv import save_artifact, train_from_csv

    pd.DataFrame({"user_id": [1, 1, 2, 2, 3], "item_id": [1, 2, 2, 3, 1], "rating": 1.0}) \
        .to_csv(tmp_path / "e.csv", index=False)
    out = save_artifact(train_from_csv(tmp_path / "e.csv", model="ItemKNN", evaluate=False), tmp_path / "m")
    assert set(_modes(out).values()) == {0o666 & ~_UMASK}
