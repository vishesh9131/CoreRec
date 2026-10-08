"""device="auto" picks CUDA, then Apple MPS, then CPU, and saved models load anywhere."""

import warnings

import numpy as np
import pytest
import torch

import corerec.device as D


def _fake(monkeypatch, cuda=False, mps=False):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(D, "mps_available", lambda: mps)


@pytest.mark.parametrize("cuda,mps,want", [(True, True, "cuda"), (False, True, "mps"),
                                           (False, False, "cpu")])
def test_auto_order(monkeypatch, cuda, mps, want):
    _fake(monkeypatch, cuda, mps)
    assert D.resolve_device("auto").type == want
    assert D.resolve_device(None).type == want


def test_sparse_models_never_get_mps(monkeypatch):
    _fake(monkeypatch, mps=True)
    assert D.resolve_device("auto", needs_sparse=True).type == "cpu"
    with pytest.warns(UserWarning, match="sparse"):
        assert D.resolve_device("mps", needs_sparse=True).type == "cpu"


@pytest.mark.parametrize("asked", ["cuda", "mps"])
def test_missing_device_falls_back_to_cpu_with_warning(monkeypatch, asked):
    _fake(monkeypatch)
    with pytest.warns(UserWarning, match="using CPU"):
        assert D.resolve_device(asked).type == "cpu"


def test_model_saved_on_a_gpu_loads_on_a_cpu_box(monkeypatch, tmp_path):
    """DCN records its device in the save; a Mac/GPU model must load on a CPU server."""
    from corerec.engines import DCN

    rng = np.random.default_rng(0)
    u, i = rng.integers(0, 20, 200).tolist(), rng.integers(0, 30, 200).tolist()
    m = DCN(embedding_dim=8, epochs=1, verbose=False, device="cpu")
    m.fit(u, i, [1.0] * 200)
    m.device = "mps"  # what a Mac would have written
    path = str(tmp_path / "dcn")
    m.save(path)
    m.device = "cpu"

    _fake(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = DCN.load(path)
    assert loaded.device == "cpu"
    assert loaded.recommend(u[0], top_k=5) == m.recommend(u[0], top_k=5)


@pytest.mark.skipif(not D.mps_available(), reason="needs Apple MPS")
@pytest.mark.parametrize("name", ["TwoTower", "SASRec", "HSTU", "DCN", "DeepFM", "MultVAE"])
def test_trains_on_mps(name):
    import corerec.engines as E

    rng = np.random.default_rng(0)
    u, i = rng.integers(0, 30, 300).tolist(), rng.integers(0, 50, 300).tolist()
    m = getattr(E, name)(device="mps", epochs=1)
    m.fit(u, i, [1.0] * 300)
    assert len(m.recommend(u[0], top_k=5)) == 5
