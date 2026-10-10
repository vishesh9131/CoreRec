"""Untrusted artifacts cannot authorize Python execution during model loading."""
from contextlib import nullcontext
import importlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from corerec import engines
from corerec.api.exceptions import SaveLoadError
from corerec.api.model_bundle import bundle_meta_path, is_safe_bundle, load_bundle, save_bundle
from corerec.nn import Recommender
from corerec.nn.models import MatrixFactorization
from corerec.serving import ModelLoader


class Payload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return exec, (f"open({str(self.marker)!r}, 'w').write('executed')",)


@pytest.mark.parametrize("name", list(engines.MODELS))
def test_every_production_loader_rejects_malicious_pickle_without_execution(name, tmp_path):
    path, marker = tmp_path / "model.pkl", tmp_path / "executed"
    path.write_bytes(pickle.dumps(Payload(marker)))
    with pytest.raises(SaveLoadError, match="allow_pickle=True"):
        getattr(engines, name).load(path)
    assert not marker.exists()


def test_wrapper_and_serving_reject_malicious_files_without_execution(tmp_path):
    from corerec.api.mixins import ModelPersistenceMixin
    from corerec.api.torch_recommender import TorchRecommender

    path, marker = tmp_path / "model.pkl", tmp_path / "executed"
    path.write_bytes(pickle.dumps(Payload(marker)))
    for load in (Recommender.load, ModelLoader().load,
                 ModelPersistenceMixin.load, TorchRecommender.load):
        with pytest.raises(SaveLoadError, match="allow_pickle=True"):
            load(path)
        assert not marker.exists()


def test_safe_weights_reject_a_malicious_torch_payload(tmp_path):
    marker = tmp_path / "executed"
    save_bundle(tmp_path / "model", model_class="test.Model", config={}, state={},
                state_dict={"weight": torch.zeros(1)})
    meta = json.loads(bundle_meta_path(tmp_path / "model").read_text())
    torch.save(Payload(marker), tmp_path / "evil.weights.pt")
    meta["weights_file"] = "evil.weights.pt"
    meta.pop("tensor_state")
    bundle_meta_path(tmp_path / "model").write_text(json.dumps(meta))
    with pytest.raises((SaveLoadError, pickle.UnpicklingError)):
        load_bundle(tmp_path / "model")
    assert not marker.exists()


def test_pickle_sidecar_requires_explicit_opt_in(tmp_path):
    from corerec.api.safe_persistence import load_artifact, save_artifact

    path, marker = tmp_path / "model", tmp_path / "executed"
    with pytest.raises(SaveLoadError, match="allow_pickle=True"):
        save_artifact(path, sklearn_payload={"test": 1})
    assert not list(tmp_path.iterdir())
    (tmp_path / "model.meta.json").write_text(json.dumps({"sklearn_file": "model.skops"}))
    (tmp_path / "model.skops").write_bytes(pickle.dumps(Payload(marker)))
    with pytest.raises(SaveLoadError, match="allow_pickle=True"):
        load_artifact(path)
    assert not marker.exists()


def test_safe_metadata_cannot_import_an_arbitrary_model_module(tmp_path, monkeypatch):
    path = tmp_path / "model"
    save_bundle(path, model_class="untrusted_module.Model", config={}, state={})
    def refuse_import(name, *args, **kwargs):
        raise AssertionError(f"artifact attempted to import {name}")
    monkeypatch.setattr(importlib, "import_module", refuse_import)
    with pytest.raises(ValueError, match="registered"):
        ModelLoader().load(path)


@pytest.mark.parametrize("component", ["arrays_file", "weights_file"])
@pytest.mark.parametrize("filename", ["../outside", "/tmp/outside"])
def test_bundle_components_cannot_escape_artifact_directory(component, filename, tmp_path):
    path = tmp_path / "model"
    save_bundle(path, model_class="test.Model", config={}, state={})
    meta_path = bundle_meta_path(path)
    meta = json.loads(meta_path.read_text())
    meta[component] = filename
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(SaveLoadError, match="filename"):
        load_bundle(path)


def test_symlink_component_cannot_escape_directory(tmp_path):
    directory = tmp_path / "artifact"
    directory.mkdir()
    outside = tmp_path / "outside.npz"
    np.savez(outside, value=[1])
    (directory / "linked.npz").symlink_to(outside)
    path = directory / "model"
    save_bundle(path, model_class="test.Model", config={}, state={})
    meta_path = bundle_meta_path(path)
    meta = json.loads(meta_path.read_text())
    meta["arrays_file"] = "linked.npz"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(SaveLoadError, match="remain"):
        load_bundle(path)


def test_legacy_opt_in_cannot_be_bypassed_by_a_serving_cache_hit(tmp_path):
    model = engines.ItemKNN().fit([1, 1, 2, 2], [10, 20, 20, 30])
    path = tmp_path / "old.pkl"
    import pickle
    with path.open("wb") as stream:
        pickle.dump({"cls": "ItemKNN", "params": {}, "user_map": model.user_map,
                     "item_map": model.item_map, "R": model.R, "state": model._state()}, stream)
    loader = ModelLoader()
    with pytest.warns(UserWarning, match="execute"):
        loaded = loader.load(path, allow_pickle=True)
    assert loaded.predict(1, 10) == pytest.approx(model.predict(1, 10))
    with pytest.raises(SaveLoadError, match="allow_pickle=True"):
        loader.load(path)


@pytest.mark.parametrize("failure", ["weights", "arrays", "metadata"])
def test_failed_bundle_save_preserves_previous_generation(failure, tmp_path, monkeypatch):
    path = tmp_path / "model.v1"
    save_bundle(path, model_class="test.Model", config={"version": 1}, state={},
                arrays={"x": np.array([1])}, state_dict={"weight": torch.ones(1)})
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    original_save = np.savez_compressed
    calls = 0
    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if failure == "metadata" or calls == (1 if failure == "weights" else 2):
            raise OSError("injected failure")
        return original_save(*args, **kwargs)
    target = "os.replace" if failure == "metadata" else "numpy.savez_compressed"
    with monkeypatch.context() as patch:
        patch.setattr(target, fail)
        with pytest.raises(OSError, match="injected"):
            save_bundle(path, model_class="test.Model", config={"version": 2}, state={},
                        arrays={"x": np.array([2])}, state_dict={"weight": torch.zeros(1)})
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == original
    assert load_bundle(path)["config"] == {"version": 1}
    assert load_bundle(path)["arrays"]["x"].tolist() == [1]


def test_save_replaces_generation_and_preserves_dotted_names(tmp_path):
    path = tmp_path / "model.v1.pt"
    for version in (1, 2):
        save_bundle(path, model_class="test.Model", config={"version": version}, state={},
                    arrays={"x": np.array([version])})
    assert len(list(tmp_path.iterdir())) == 2
    assert bundle_meta_path(path).name == "model.v1.meta.json"
    bundle = load_bundle(path)
    assert bundle["config"] == {"version": 2}
    assert is_safe_bundle(tmp_path / bundle["metadata"]["arrays_file"])


def test_load_retries_if_a_concurrent_save_removes_old_components(tmp_path, monkeypatch):
    path = tmp_path / "model"
    save_bundle(path, model_class="test.Model", config={"version": 1}, state={},
                arrays={"x": np.array([1])})
    original_load = np.load
    replaced = False
    def load_after_replacement(filename, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            save_bundle(path, model_class="test.Model", config={"version": 2}, state={},
                        arrays={"x": np.array([2])})
        return original_load(filename, **kwargs)
    monkeypatch.setattr(np, "load", load_after_replacement)
    assert load_bundle(path)["arrays"]["x"].tolist() == [2]


class CustomMF(MatrixFactorization):
    pass


def test_custom_nn_module_requires_an_explicit_class(tmp_path):
    model = Recommender(CustomMF, {"dim": 3}, epochs=1, device="cpu").fit(
        [1, 1, 2, 2], [10, 20, 20, 30])
    path = tmp_path / "custom"
    model.save(path)
    with pytest.raises(ValueError, match="module_cls="):
        Recommender.load(path)
    loaded = Recommender.load(path, module_cls=CustomMF, device="cpu")
    assert loaded.predict(1, 10) == pytest.approx(model.predict(1, 10))


@pytest.mark.parametrize("name,params", [
    ("ItemKNN", {}), ("UserKNN", {}), ("EASE", {}), ("SLIM", {"max_iter": 5}),
    ("ALS", {"iterations": 1, "factors": 3}), ("Item2Vec", {"iterations": 1, "factors": 3}),
    ("MultVAE", {"epochs": 1, "hidden_dim": 4, "latent_dim": 2, "device": "cpu"}),
    ("MultiDAE", {"epochs": 1, "hidden_dim": 4, "latent_dim": 2, "device": "cpu"}),
])
def test_new_safe_models_preserve_string_ids_and_sparse_state(name, params, tmp_path):
    model = getattr(engines, name)(**params).fit(["01", "01", "02", "02"],
                                               ["010", "020", "020", "030"])
    path = tmp_path / name
    model.save(path)
    assert is_safe_bundle(path)
    assert not path.exists()
    loaded = ModelLoader().load(path)
    assert set(loaded.user_map) == {"01", "02"}
    assert set(loaded.item_map) == {"010", "020", "030"}
    assert loaded.predict("01", "010") == pytest.approx(model.predict("01", "010"))
    arrays = load_bundle(path)["arrays"]
    assert any(key in arrays for key in ("R.indptr", "R_indptr")) and "R" not in arrays
    assert all(not array.dtype.hasobject for array in arrays.values())


@pytest.mark.parametrize("metadata", [[], "invalid", None])
def test_malformed_metadata_is_rejected_and_can_be_replaced(metadata, tmp_path):
    path = tmp_path / "model"
    bundle_meta_path(path).write_text(json.dumps(metadata))
    assert not is_safe_bundle(path)
    with pytest.raises(SaveLoadError, match="JSON object"):
        load_bundle(path)
    save_bundle(path, model_class="test.Model", config={}, state={})
    assert is_safe_bundle(path)


def test_object_arrays_are_refused_without_serializing_python(tmp_path):
    path, marker = tmp_path / "model", tmp_path / "executed"
    with pytest.raises(SaveLoadError, match="Python objects"):
        save_bundle(path, model_class="test.Model", config={}, state={},
                    arrays={"payload": np.array([Payload(marker)], dtype=object)})
    assert not marker.exists()
    assert not is_safe_bundle(path)
    assert not list(tmp_path.iterdir())



def test_existing_fixed_filename_bundle_remains_readable(tmp_path):
    path = tmp_path / "old"
    np.savez_compressed(tmp_path / "old.arrays.npz", x=np.array([1, 2]))
    torch.save({"weight": torch.ones(1)}, tmp_path / "old.weights.pt")
    (tmp_path / "old.meta.json").write_text(json.dumps({
        "format": "corerec_safe_v1", "model_class": "test.Model", "config": {}, "state": {},
        "arrays_file": "old.arrays.npz", "weights_file": "old.weights.pt",
    }))
    assert is_safe_bundle(path)
    with pytest.warns(UserWarning, match="execute") if tuple(map(int, torch.__version__.split(".")[:2])) < (2, 10) else nullcontext():
        loaded = load_bundle(path, allow_pickle=True)
    assert loaded["arrays"]["x"].tolist() == [1, 2]
    assert loaded["state_dict"]["weight"].item() == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64, torch.bool])
def test_numeric_weights_roundtrip_without_torch_load(dtype, tmp_path, monkeypatch):
    from collections import OrderedDict
    from corerec.api.safe_persistence import save_artifact, load_artifact
    weights = OrderedDict(scalar=torch.tensor(1, dtype=dtype),
                          empty=torch.empty((0, 2), dtype=dtype))
    weights._metadata = {"": {"version": 1}}
    def forbidden(*args, **kwargs):
        raise AssertionError("Numeric bundles must not use torch.load")
    monkeypatch.setattr(torch, "load", forbidden)
    for save, load in ((lambda path: save_bundle(path, model_class="test.Model", config={},
                                                state={}, state_dict=weights), load_bundle),
                       (lambda path: save_artifact(path, state_dict=weights,
                                                   metadata={"version": 1}), load_artifact)):
        path = tmp_path / "weights"
        save(path)
        restored = load(path)["state_dict"]
        assert restored._metadata == weights._metadata
        for key in weights:
            assert torch.equal(restored[key], weights[key])
            assert restored[key].dtype == dtype


def test_numeric_weights_refuse_callable_map_location(tmp_path):
    path = tmp_path / "weights"
    save_bundle(path, model_class="test.Model", config={}, state={},
                state_dict={"x": torch.ones(1)})
    with pytest.raises(SaveLoadError, match="device map_location"):
        load_bundle(path, map_location=lambda storage, location: storage)


@pytest.mark.parametrize("filename", ["artifact", "artifact.pt"])
def test_mixed_numeric_weights_and_python_payload_require_trust(filename, tmp_path):
    from corerec.api.safe_persistence import save_artifact, load_artifact
    path = tmp_path / filename
    save_artifact(path, state_dict={"weight": torch.ones(1)},
                  sklearn_payload={"value": 2}, allow_pickle=True)
    with pytest.raises(SaveLoadError, match="allow_pickle=True"):
        load_artifact(path)
    with pytest.warns(UserWarning, match="execute"):
        restored = load_artifact(path, allow_pickle=True)
    assert restored["sklearn_payload"] == {"value": 2}
    assert restored["state_dict"]["weight"].item() == 1
