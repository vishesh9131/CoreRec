"""Saved CF artifacts survive failed writes and retain training configuration."""
import pickle
from unittest.mock import patch

import numpy as np
import pytest

from corerec.api.exceptions import ModelNotFittedError
from corerec.engines import ALS, Item2Vec, ItemKNN

USERS = [1, 1, 2, 2]
ITEMS = [10, 20, 20, 30]


def _snapshot(directory):
    """Every file save() wrote (a safe bundle is several), by name."""
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


@pytest.mark.parametrize("model_class", [ItemKNN, ALS, Item2Vec])
def test_unfitted_save_preserves_existing_artifact(model_class, tmp_path):
    path = tmp_path / "model.pkl"
    model = model_class().fit(USERS, ITEMS)
    model.save(path)
    original = _snapshot(tmp_path)
    with pytest.raises(ModelNotFittedError):
        model_class().save(path)
    assert _snapshot(tmp_path) == original
    assert model_class.load(path).predict(1, 10) == pytest.approx(model.predict(1, 10))
    new_path = tmp_path / "new" / "model.pkl"
    with pytest.raises(ModelNotFittedError):
        model_class().save(new_path)
    assert not new_path.parent.exists()


@pytest.mark.parametrize("model_class", [ItemKNN, ALS])
@pytest.mark.parametrize("failure", ["serialization", "replace"])
def test_failed_save_preserves_artifact_and_removes_temporary_file(model_class, failure, tmp_path):
    path = tmp_path / "model.pkl"
    model = model_class().fit(USERS, ITEMS)
    model.save(path)
    original = _snapshot(tmp_path)
    if failure == "serialization":
        def fail_dump(stream, *args, **kwargs):
            stream.write(b"partial output")
            raise OSError("write failed")
        target, effect = "numpy.savez_compressed", fail_dump
    else:
        target, effect = "os.replace", OSError("replace failed")
    with patch(target, side_effect=effect), pytest.raises(OSError):
        model.save(path)
    assert _snapshot(tmp_path) == original  # same files, same bytes, no temp left
    assert model_class.load(path).predict(1, 10) == pytest.approx(model.predict(1, 10))


@pytest.mark.parametrize("model_class,params", [
    (ALS, {"alpha": 9, "seed": 7}),
    (Item2Vec, {"num_negatives": 2, "learning_rate": 0.001, "seed": 7}),
])
def test_embedding_configuration_and_refit_roundtrip(model_class, params, tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr("corerec.device.resolve_device", lambda _: torch.device("cpu"))
    params = {"name": "custom", "factors": 3, "reg": 0.02, "iterations": 2,
              "verbose": True, "trainable": True, **params}
    model = model_class(**params).fit(USERS, ITEMS)
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = model_class.load(path)
    from corerec.serving.model_loader import ModelLoader
    assert isinstance(ModelLoader().load(str(path)), model_class)
    for key, value in params.items():
        assert getattr(loaded, key) == value
    np.testing.assert_allclose(loaded.U, model.U)
    np.testing.assert_allclose(loaded.V, model.V)
    loaded.fit(USERS, ITEMS)
    model.fit(USERS, ITEMS)
    np.testing.assert_allclose(loaded.U, model.U)
    np.testing.assert_allclose(loaded.V, model.V)


@pytest.mark.parametrize("model_class", [ALS, Item2Vec])
def test_older_embedding_artifacts_load_with_constructor_defaults(model_class, tmp_path):
    model = model_class(factors=3, iterations=1).fit(USERS, ITEMS)
    payload = {"U": model.U, "V": model.V, "R": model.R,
               "user_map": model.user_map, "item_map": model.item_map,
               "params": {"name": model.name, "factors": model.factors,
                          "reg": model.reg, "iterations": model.iterations}}
    path = tmp_path / "old.pkl"
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    loaded = model_class.load(path, allow_pickle=True)
    assert loaded.seed == 42
    assert loaded.predict(1, 10) == pytest.approx(model.predict(1, 10))


@pytest.mark.parametrize("name", ["ItemKNN", "UserKNN", "EASE", "SLIM", "ALS", "Item2Vec"])
def test_saves_a_safe_bundle_that_loads_without_pickle(name, tmp_path, monkeypatch):
    """These six pickled their whole state, so loading a shared file ran code (#75)."""
    import json

    import corerec.engines as engines
    from corerec.serving.model_loader import ModelLoader

    cls = getattr(engines, name)
    users, items = ["a", "a", "b", "b", "c", "c"], ["x", "y", "y", "z", "x", "z"]
    model = cls(name="custom").fit(users, items)
    model.save(tmp_path / "m.pkl")
    assert json.loads((tmp_path / "m.meta.json").read_text())["format"] == "corerec_safe_v1"

    def refuse(*a, **k):
        raise AssertionError("loading a safe bundle must not unpickle")
    monkeypatch.setattr(pickle, "load", refuse)
    monkeypatch.setattr(pickle, "loads", refuse)
    for loaded in (cls.load(tmp_path / "m.pkl"), ModelLoader().load(str(tmp_path / "m.pkl"))):
        assert type(loaded) is cls and loaded.name == "custom"
        assert loaded.user_map == model.user_map and loaded.item_map == model.item_map
        assert loaded.recommend("a", top_k=2) == model.recommend("a", top_k=2)


def test_legacy_pickles_still_load_with_a_warning(tmp_path):
    from corerec.engines import ItemKNN

    model = ItemKNN().fit(USERS, ITEMS)
    payload = {"cls": "ItemKNN", "user_map": model.user_map, "item_map": model.item_map,
               "R": model.R, "state": {"S": model.S},
               "params": {"top_k_neighbors": model.top_k_neighbors, "reg": model.reg,
                          "shrink": model.shrink, "name": model.name}}
    path = tmp_path / "old.pkl"
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    with pytest.warns(UserWarning, match="execute"):
        assert ItemKNN.load(path, allow_pickle=True).recommend(1, top_k=2) == model.recommend(1, top_k=2)
