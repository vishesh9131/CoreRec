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

    loaded = ModelLoader().load(str(path))
    assert type(loaded) is cls
    assert loaded.name == "my_vae"
    assert loaded.recommend(0, top_k=2) == m.recommend(0, top_k=2)


def test_model_loader_still_reads_vae_files_saved_before_cls_was_written(tmp_path):
    import torch

    from corerec.engines import MultiDAE
    from corerec.serving import ModelLoader

    m = MultiDAE(epochs=1, hidden_dim=16, latent_dim=4)
    m.fit([0, 0, 1, 1, 2], [1, 2, 2, 3, 1])
    path = tmp_path / "old.pt"
    m.save(str(path))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    del ckpt["cls"]  # what save() wrote before
    torch.save(ckpt, path)

    assert type(ModelLoader().load(str(path))) is MultiDAE
