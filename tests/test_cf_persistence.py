"""Saved CF artifacts survive failed writes and retain training configuration."""
import pickle
from unittest.mock import patch

import numpy as np
import pytest

from corerec.api.exceptions import ModelNotFittedError
from corerec.engines import ALS, Item2Vec, ItemKNN

USERS = [1, 1, 2, 2]
ITEMS = [10, 20, 20, 30]


@pytest.mark.parametrize("model_class", [ItemKNN, ALS, Item2Vec])
def test_unfitted_save_preserves_existing_artifact(model_class, tmp_path):
    path = tmp_path / "model.pkl"
    model = model_class().fit(USERS, ITEMS)
    model.save(path)
    original = path.read_bytes()
    with pytest.raises(ModelNotFittedError):
        model_class().save(path)
    assert path.read_bytes() == original
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
    original = path.read_bytes()
    if failure == "serialization":
        def fail_dump(payload, stream, **kwargs):
            stream.write(b"partial output")
            raise OSError("write failed")
        target, effect = "pickle.dump", fail_dump
    else:
        target, effect = "os.replace", OSError("replace failed")
    with patch(target, side_effect=effect), pytest.raises(OSError):
        model.save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
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
    loaded = model_class.load(path)
    assert loaded.seed == 42
    assert loaded.predict(1, 10) == pytest.approx(model.predict(1, 10))
