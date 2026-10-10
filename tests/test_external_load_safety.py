"""Data and generic object loaders enforce the same trust boundary as models."""
import json
import pickle

import numpy as np
import pytest

from corerec.api.exceptions import SaveLoadError
from corerec.data.data import BaseDataset
from corerec.embeddings.pretrained import PretrainedEmbeddings
from corerec.serialization import Serializable, SerializableRegistry, deserialize, load_from_file


class Payload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return exec, (f"open({str(self.marker)!r}, 'w').write('executed')",)


@pytest.mark.parametrize("loader", [BaseDataset.load, PretrainedEmbeddings.load,
                                   load_from_file, deserialize, Serializable.load])
def test_external_loaders_reject_untrusted_pickle_before_execution(loader, tmp_path):
    path, marker = tmp_path / "evil.pkl", tmp_path / "executed"
    path.write_bytes(pickle.dumps(Payload(marker)))
    with pytest.raises(SaveLoadError, match="allow_pickle=True"):
        loader(path)
    assert not marker.exists()


def test_trusted_datasets_embeddings_and_objects_remain_loadable(tmp_path):
    dataset = BaseDataset(seed=7)
    path = tmp_path / "dataset.pkl"
    dataset.save(path)
    with pytest.warns(UserWarning, match="execute"):
        assert BaseDataset.load(path, allow_pickle=True).seed == 7
    values = {"embeddings": np.ones((2, 3)), "ids": ["01", "1"]}
    path.write_bytes(pickle.dumps(values))
    with pytest.warns(UserWarning, match="execute"):
        embeddings = PretrainedEmbeddings.load(path, allow_pickle=True)
    assert embeddings.ids == ["01", "1"]
    np.testing.assert_array_equal(embeddings.embeddings, values["embeddings"])
    for loader in (load_from_file, deserialize, Serializable.load):
        with pytest.warns(UserWarning, match="execute"):
            restored = loader(path, allow_pickle=True)
        assert restored["ids"] == values["ids"]


def test_numeric_embeddings_need_no_pickle_permission(tmp_path):
    path = tmp_path / "embeddings.npz"
    np.savez(path, embeddings=np.ones((2, 3)), ids=np.array(["01", "1"]))
    restored = PretrainedEmbeddings.load(path)
    assert restored.ids == ["01", "1"]
    assert restored.dim == 3


def test_json_metadata_cannot_import_a_module(tmp_path, monkeypatch):
    marker = tmp_path / "executed"
    (tmp_path / "evil_metadata.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nclass Object: pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    data = {"_type": "Object", "_module": "evil_metadata"}
    for value in (data, json.dumps(data)):
        with pytest.raises(ValueError, match="unregistered"):
            deserialize(value)
    path = tmp_path / "object.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unregistered"):
        load_from_file(path)
    assert not marker.exists()


class Scalar(Serializable):
    def __init__(self, value):
        self.value = value


class Container(Serializable):
    def __init__(self, values):
        self.values = values


def test_registered_nested_lists_roundtrip_without_dynamic_import(monkeypatch):
    monkeypatch.setattr(SerializableRegistry, "_registry", {"Scalar": Scalar, "Container": Container})
    original = Container([Scalar(1), Scalar(2)])
    restored = deserialize(original.to_dict())
    assert [value.value for value in restored.values] == [1, 2]
    nested = Container({"group": [Scalar(3).to_dict(), {"child": Scalar(4).to_dict()}]})
    restored_nested = deserialize(nested.to_dict())
    assert restored_nested.values["group"][0].value == 3
    assert restored_nested.values["group"][1]["child"].value == 4
    with pytest.raises(ValueError, match="unregistered"):
        deserialize({"_type": "Scalar", "_module": "other.module", "value": 1})


def test_failed_dataset_save_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "dataset.pkl"
    path.write_bytes(b"original")
    def fail(*args, **kwargs):
        raise OSError("injected failure")
    monkeypatch.setattr(pickle, "dump", fail)
    with pytest.raises(OSError, match="injected"):
        BaseDataset().save(path)
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]
