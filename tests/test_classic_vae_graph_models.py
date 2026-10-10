"""Production-contract tests for Batch-3 families:
classic CF (ItemKNN, UserKNN, EASE, SLIM) and auto-encoder CF (MultVAE,
MultiDAE). Each must fit without collapsing, predict, recommend top_k known
items, and round-trip through save/load with identical predictions.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

MODELS = {
    "ItemKNN": {}, "UserKNN": {}, "EASE": {"reg": 50.0}, "SLIM": {"alpha": 0.1},
    "MultVAE": {"epochs": 15, "device": "cpu"},
    "MultiDAE": {"epochs": 15, "device": "cpu"},
}


def _data(n_users=60, n_items=80, seed=0):
    rng = np.random.RandomState(seed)
    u, i = [], []
    for usr in range(n_users):
        center = rng.randint(0, n_items)
        for _ in range(12):
            u.append(usr); i.append(int((center + rng.randint(-6, 7)) % n_items))
    return np.array(u), np.array(i), np.ones(len(u), dtype=float)


@pytest.fixture(params=list(MODELS))
def fitted(request):
    from corerec import engines
    u, i, r = _data()
    model = getattr(engines, request.param)(**MODELS[request.param])
    model.fit(u, i, r)
    return request.param, model, (u, i)


def test_importable_from_engines():
    from corerec import engines
    for m in MODELS:
        assert getattr(engines, m) is not None, f"{m} not exported"


def test_predict_returns_float(fitted):
    _, model, (u, i) = fitted
    assert isinstance(model.predict(int(u[0]), int(i[0])), float)


def test_recommend_returns_known_items(fitted):
    name, model, _ = fitted
    recs = model.recommend(0, top_k=10)
    assert isinstance(recs, list) and 0 < len(recs) <= 10
    assert all(it in model.item_map for it in recs), f"{name} returned unknown items"


def test_no_output_collapse(fitted):
    name, model, _ = fitted
    assert float(np.std(model._score_all_items(0))) > 1e-5, f"{name} output collapsed"


def test_save_load_roundtrip(fitted, tmp_path):
    name, model, (u, i) = fitted
    base = str(tmp_path / name)
    before = model.predict(int(u[0]), int(i[0]))
    model.save(base)
    reloaded = type(model).load(base)
    after = reloaded.predict(int(u[0]), int(i[0]))
    assert abs(before - after) < 1e-4, f"{name} save/load mismatch"
    assert len(reloaded.recommend(0, top_k=5)) == 5


@pytest.mark.parametrize("cls_name", ["MultVAE", "MultiDAE"])
def test_model_loader_finds_vae_saved_with_a_custom_name(cls_name, tmp_path):
    """ModelLoader picked the class from cfg["name"], the display name, so
    MultVAE(name="my_vae") saved fine and then couldn't be loaded (#51)."""
    import corerec.engines as engines
    from corerec.serving import ModelLoader

    cls = getattr(engines, cls_name)
    m = cls(epochs=1, hidden_dim=16, latent_dim=4, name="my_vae")
    m.fit([0, 0, 1, 1, 2], [1, 2, 2, 3, 1])
    path = tmp_path / "v.pt"
    m.save(str(path))

    loaded = ModelLoader().load(str(path), allow_pickle=True)
    assert type(loaded) is cls
    assert loaded.name == "my_vae"
    assert loaded.recommend(0, top_k=2) == m.recommend(0, top_k=2)


def _legacy_vae_checkpoint(m, path, with_cls=True, with_binarize=True):
    """The torch.save checkpoint VAEs wrote before the safe bundle (#75)."""
    import torch

    cfg = {"name": m.name, "hidden_dim": m.hidden_dim, "latent_dim": m.latent_dim,
           "dropout": m.dropout, "learning_rate": m.learning_rate, "batch_size": m.batch_size,
           "epochs": m.epochs, "beta": m.beta, "reg": m.reg, "device": m.device, "seed": m.seed}
    if with_binarize:
        cfg["binarize"] = m.binarize
    ckpt = {"cfg": cfg, "user_map": m.user_map, "item_map": m.item_map,
            "num_users": m.num_users, "num_items": m.num_items, "R": m.R,
            "state_dict": m.model.state_dict()}
    if with_cls:
        ckpt["cls"] = type(m).__name__
    torch.save(ckpt, path)


def test_model_loader_still_reads_vae_files_saved_before_cls_was_written(tmp_path):
    from corerec.engines import MultiDAE
    from corerec.serving import ModelLoader

    m = MultiDAE(epochs=1, hidden_dim=16, latent_dim=4)
    m.fit([0, 0, 1, 1, 2], [1, 2, 2, 3, 1])
    path = tmp_path / "old.pt"
    _legacy_vae_checkpoint(m, path, with_cls=False)  # what save() wrote before "cls"

    with pytest.warns(UserWarning, match="execute"):
        assert type(ModelLoader().load(str(path), allow_pickle=True)) is MultiDAE


@pytest.mark.parametrize("cls_name", ["MultVAE", "MultiDAE"])
def test_vae_trains_on_a_binary_matrix_unless_told_not_to(cls_name, tmp_path):
    """Repeat events used to sum into the cell (#42); the paper's input is 0/1."""
    import torch

    import corerec.engines as engines

    cls = getattr(engines, cls_name)
    u, i = [0, 0, 0, 0, 1, 1, 2], [1, 1, 1, 2, 2, 3, 1]  # user 0 saw item 1 three times
    assert cls(epochs=1, hidden_dim=16, latent_dim=4).fit(u, i).R.max() == 1
    counts = cls(epochs=1, hidden_dim=16, latent_dim=4, binarize=False).fit(u, i)
    assert counts.R.max() == 3

    path = tmp_path / "v.pt"
    counts.save(str(path))
    assert cls.load(str(path)).binarize is False
    # checkpoints written before binarize existed trained on counts
    _legacy_vae_checkpoint(counts, tmp_path / "old.pt", with_binarize=False)
    with pytest.warns(UserWarning, match="execute"):
        assert cls.load(str(tmp_path / "old.pt"), allow_pickle=True).binarize is False


@pytest.mark.parametrize("cls_name", ["MultVAE", "MultiDAE"])
def test_vae_saves_a_safe_bundle_that_loads_without_pickle(cls_name, tmp_path, monkeypatch):
    """torch.load(weights_only=False) on a VAE file ran arbitrary code (#75)."""
    import json
    import pickle

    import corerec.engines as engines
    from corerec.serving import ModelLoader

    cls = getattr(engines, cls_name)
    m = cls(epochs=1, hidden_dim=16, latent_dim=4, name="custom").fit(
        ["a", "a", "b", "b", "c"], ["x", "y", "y", "z", "x"])
    m.save(tmp_path / "v")
    meta = json.loads((tmp_path / "v.meta.json").read_text())
    assert meta["format"] == "corerec_safe_v1"

    def no_unpickling(*a, **k):
        raise AssertionError("loading a safe bundle must not unpickle")
    monkeypatch.setattr(pickle, "load", no_unpickling)
    monkeypatch.setattr(pickle, "loads", no_unpickling)
    for loaded in (cls.load(tmp_path / "v"), ModelLoader().load(str(tmp_path / "v"))):
        assert type(loaded) is cls and loaded.name == "custom"
        assert loaded.recommend("a", top_k=2) == m.recommend("a", top_k=2)
        assert (loaded.R != m.R).nnz == 0


def test_a_malicious_legacy_vae_file_is_rejected_before_running(tmp_path):
    """Reject an untrusted legacy file before its payload can execute."""
    import torch

    from corerec.engines import MultVAE

    class Boom:
        def __reduce__(self):
            return (exec, ("import pathlib; pathlib.Path(%r).touch()" % str(tmp_path / "pwned"),))

    torch.save({"cfg": {}, "payload": Boom()}, tmp_path / "evil.pt")
    from corerec.api.exceptions import SaveLoadError
    with pytest.raises(SaveLoadError, match="allow_pickle=True"):
        MultVAE.load(tmp_path / "evil.pt")
    assert not (tmp_path / "pwned").exists()



@pytest.mark.parametrize("cls_name", ["MultVAE", "MultiDAE"])
def test_unfitted_vae_save_leaves_the_existing_bundle_intact(cls_name, tmp_path):
    """The VAE case of #101: fail before writing, so the saved model survives."""
    import corerec.engines as engines
    from corerec.api.exceptions import ModelNotFittedError

    cls = getattr(engines, cls_name)
    m = cls(epochs=1, hidden_dim=16, latent_dim=4).fit([1, 1, 2, 2, 3], [10, 20, 20, 30, 10])
    m.save(tmp_path / "v")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ModelNotFittedError):
        cls(epochs=1).save(tmp_path / "v")
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert cls.load(tmp_path / "v").recommend(1, top_k=2) == m.recommend(1, top_k=2)
