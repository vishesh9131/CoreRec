import pytest
"""Tests for safe model bundle persistence."""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from corerec.api.model_bundle import is_safe_bundle, load_bundle, save_bundle
from corerec.engines.collaborative import SAR
from corerec.engines.dcn import DCN


class TestSafePersistence(unittest.TestCase):
    def test_bundle_roundtrip_primitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "artifact")
            save_bundle(
                base,
                model_class="test.Model",
                config={"a": 1},
                state={"b": 2},
                arrays={"x": np.array([1.0, 2.0])},
            )
            self.assertTrue(is_safe_bundle(base))
            loaded = load_bundle(base)
            self.assertEqual(loaded["config"]["a"], 1)
            self.assertEqual(loaded["arrays"]["x"].tolist(), [1.0, 2.0])

    def test_dcn_safe_save_load(self):
        model = DCN(embedding_dim=8, num_cross_layers=1, deep_layers=[8], epochs=1, batch_size=4)
        users, items, ratings = [0, 0, 1], [10, 11, 10], [5.0, 4.0, 3.0]
        model.fit(users, items, ratings)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dcn")
            model.save(path, safe=True)
            self.assertTrue(is_safe_bundle(path))
            loaded = DCN.load(path)
            self.assertTrue(loaded.is_fitted)
            recs = loaded.recommend(0, top_k=2)
            self.assertIsInstance(recs, list)

    def test_dcn_predict_parity_after_safe_load(self):
        model = DCN(embedding_dim=8, num_cross_layers=1, deep_layers=[8], epochs=1, batch_size=4)
        users, items, ratings = [0, 0, 1], [10, 11, 10], [5.0, 4.0, 3.0]
        model.fit(users, items, ratings)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dcn")
            model.save(path, safe=True)
            loaded = DCN.load(path)
            self.assertIsInstance(next(iter(loaded.user_map.keys())), int)
            self.assertAlmostEqual(model.predict(0, 10), loaded.predict(0, 10), delta=1e-2)
        df = pd.DataFrame(
            {"userID": [0, 0, 1, 1], "itemID": [10, 11, 10, 12], "rating": [5, 4, 3, 5]}
        )
        model = SAR()
        model.fit(df)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sar")
            model.save(path, safe=True)
            self.assertTrue(is_safe_bundle(path))
            loaded = SAR.load(path)
            self.assertTrue(loaded.is_fitted)
            recs = loaded.recommend(0, top_k=2)
            self.assertGreaterEqual(len(recs), 1)

    def test_legacy_dcn_still_loads(self):
        model = DCN(embedding_dim=8, num_cross_layers=1, deep_layers=[8], epochs=1, batch_size=4)
        model.fit([0, 1], [10, 11], [5.0, 4.0])
        with tempfile.TemporaryDirectory() as tmp:
            legacy = os.path.join(tmp, "legacy.pt")
            model.save(legacy, safe=False)
            self.assertFalse(is_safe_bundle(legacy))
            loaded = DCN.load(legacy, allow_pickle=True)
            self.assertTrue(loaded.is_fitted)

    def test_deepfm_safe_save_load(self):
        from corerec.engines.deepfm import DeepFM

        model = DeepFM(embedding_dim=8, hidden_layers=[8], epochs=1, batch_size=4, verbose=False)
        users, items, ratings = [0, 0, 1], [10, 11, 10], [5.0, 4.0, 3.0]
        model.fit(users, items, ratings)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "deepfm")
            model.save(path, safe=True)
            self.assertTrue(is_safe_bundle(path))
            loaded = DeepFM.load(path)
            self.assertTrue(loaded.is_fitted)

    def test_sasrec_safe_save_load(self):
        from corerec.engines.sasrec import SASRec

        user_ids, item_ids, mat = list(range(5)), list(range(8)), None
        import numpy as np

        rng = np.random.RandomState(0)
        mat = (rng.rand(5, 8) < 0.4).astype(np.float32)
        model = SASRec(
            hidden_units=8,
            num_blocks=1,
            epochs=1,
            batch_size=4,
            verbose=False,
        )
        model.fit(user_ids, item_ids, mat)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sasrec")
            model.save(path, safe=True)
            self.assertTrue(is_safe_bundle(path))
            loaded = SASRec.load(path)
            self.assertTrue(loaded.is_fitted)
            self.assertGreater(len(loaded.user_sequences), 0)
            user_with_history = next(iter(loaded.user_sequences))
            recs = loaded.recommend(user_with_history, top_k=2)
            self.assertIsInstance(recs, list)

    def test_tfidf_safe_save_load(self):
        from corerec.engines.content_based.tfidf_recommender import TFIDFRecommender

        items = [0, 1, 2]
        docs = {i: f"document text for item {i}" for i in items}
        model = TFIDFRecommender(verbose=False)
        model.fit(items, docs)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tfidf")
            model.save(path, safe=True)
            self.assertTrue(is_safe_bundle(path))
            loaded = TFIDFRecommender.load(path)
            self.assertTrue(loaded.is_fitted)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("fail_at", [2, 3])
def test_bundle_replacement_failure_preserves_all_previous_files(tmp_path, monkeypatch, fail_at):
    import os
    import torch

    base = tmp_path / "m"
    save_bundle(base, model_class="test.Model", config={}, state={},
                state_dict={"w": torch.tensor([1.0])}, arrays={"x": np.array([2.0])})
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    replace = os.replace
    calls = 0

    def fail_later(src, dst):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("replacement failed")
        return replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_later)
    with pytest.raises(OSError, match="replacement failed"):
        save_bundle(base, model_class="test.Model", config={}, state={},
                    state_dict={"w": torch.tensor([9.0])}, arrays={"x": np.array([8.0])})
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    loaded = load_bundle(base)
    assert loaded["state_dict"]["w"].item() == 1.0
    assert loaded["arrays"]["x"].item() == 2.0


class _Marker:
    """Unpickling this creates a file: stands in for an arbitrary-code payload."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, "w"))


def test_vector_index_load_never_unpickles_without_trust(tmp_path):
    """NumpyIndex.load used np.load(allow_pickle=True), so a crafted index file
    ran code on load (#180)."""
    import numpy as np
    import pytest

    from corerec.api.exceptions import SaveLoadError
    from corerec.retrieval.vector_store import NumpyIndex

    marker = tmp_path / "pwned"
    evil = tmp_path / "evil.npz"
    np.savez(evil, vectors=np.zeros((1, 2)), ids=np.array([_Marker(str(marker))], dtype=object),
             dim=2, metric="dot")

    with pytest.raises(SaveLoadError, match="allow_pickle=True"):
        NumpyIndex(dim=2).load(str(evil))
    assert not marker.exists()

    with pytest.warns(UserWarning, match="arbitrary code"):
        NumpyIndex(dim=2).load(str(evil), allow_pickle=True)  # explicit trust
    assert marker.exists()


def test_vector_index_round_trips_ids_without_pickle(tmp_path):
    import numpy as np

    from corerec.retrieval.vector_store import NumpyIndex

    for ids in (["a", "b", "c"], [10, 20, 30], [1, None, "x"]):
        idx = NumpyIndex(dim=2, metric="dot")
        idx.add(np.eye(3, 2), ids=ids)
        path = str(tmp_path / "ix.npz")
        idx.save(path)
        assert "ids" in np.load(path).files or "ids_json" in np.load(path).files
        back = NumpyIndex(dim=2)
        back.load(path)  # no allow_pickle needed for files we write
        assert list(back.ids) == list(np.asarray(ids).tolist())
        assert list(back.search(np.array([1.0, 0.0]), k=1)[1]) == [ids[0]]
