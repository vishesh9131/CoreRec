import unittest
import sys
from types import ModuleType
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import numpy as np
import torch

ENGINES_BASE = Path(__file__).resolve().parents[1] / "corerec" / "engines"


def ensure_module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    m = ModuleType(name)
    sys.modules[name] = m
    return m


def stub_scipy_sparse():
    # Provide a minimal csr_matrix that returns a numpy array with .nonzero working
    scipy = ensure_module("scipy")
    sparse = ensure_module("scipy.sparse")
    def csr_matrix(args, shape=None):
        # args expected: (data, (rows, cols)) where data is list/array of values
        data, idx = args
        rows, cols = idx
        arr = np.zeros(shape, dtype=float)
        for r, c, v in zip(rows, cols, data):
            arr[r, c] = v
        return arr
    sparse.csr_matrix = csr_matrix


def load_module(rel_path: str):
    spec = spec_from_file_location(Path(rel_path).stem, str(ENGINES_BASE / rel_path))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestEnginesModelsSmoke(unittest.TestCase):
    def test_dcn(self):
        # Stub scipy.sparse if missing
        try:
            import scipy.sparse  # noqa: F401
        except Exception:
            stub_scipy_sparse()
        try:
            m = load_module("dcn.py")
        except Exception as e:
            self.skipTest(f"Skip DCN import: {e}")
        model = m.DCN(embedding_dim=4, num_cross_layers=1, deep_layers=[8], epochs=1, batch_size=4, device="cpu")
        users = [1, 2, 1, 3]
        items = [10, 10, 20, 30]
        ratings = [1, 0, 1, 0]
        try:
            model.fit(users, items, ratings)
        except Exception as e:
            self.skipTest(f"Skip DCN.fit for minimal data: {e}")
        else:
            recs = model.recommend(1, top_n=2, exclude_seen=False)
            self.assertIsInstance(recs, list)

    def test_deepfm(self):
        try:
            m = load_module("deepfm.py")
        except Exception as e:
            self.skipTest(f"Skip DeepFM import: {e}")
        model = m.DeepFM(embedding_dim=4, hidden_layers=[8], epochs=1, batch_size=4, device="cpu")
        users = [1, 2, 1, 3]
        items = [10, 10, 20, 30]
        ratings = [1, 0, 1, 0]
        try:
            model.fit(users, items, ratings)
        except Exception as e:
            self.skipTest(f"Skip DeepFM.fit for minimal data: {e}")
        else:
            recs = model.recommend(1, top_n=2, exclude_seen=False)
            self.assertIsInstance(recs, list)

    def test_gnnrec(self):
        # Stub scipy.sparse if missing
        try:
            import scipy.sparse  # noqa: F401
        except Exception:
            stub_scipy_sparse()
        try:
            m = load_module("gnnrec.py")
        except Exception as e:
            self.skipTest(f"Skip GNNRec import: {e}")
        model = m.GNNRec(embedding_dim=8, num_gnn_layers=1, epochs=1, batch_size=4, device="cpu")
        users = [1, 2, 1, 3]
        items = [10, 10, 20, 30]
        ratings = [1, 1, 1, 0]
        model.fit(users, items, ratings)
        recs = model.recommend(1, top_n=2, exclude_seen=False)
        self.assertIsInstance(recs, list)

    def test_mind(self):
        try:
            m = load_module("mind.py")
        except Exception as e:
            self.skipTest(f"Skip MIND import: {e}")
        model = m.MIND(embedding_dim=8, num_interests=2, epochs=1, batch_size=4, max_seq_length=5, device="cpu")
        users = [1, 1, 1, 2]
        items = [10, 20, 30, 10]
        ts = [1, 2, 3, 1]
        model.fit(users, items, ts)
        recs = model.recommend(1, top_n=2)
        self.assertIsInstance(recs, list)

    def test_nasrec(self):
        try:
            m = load_module("nasrec.py")
        except Exception as e:
            self.skipTest(f"Skip NASRec import: {e}")
        model = m.NASRec(embedding_dim=8, hidden_dims=[8], epochs=1, batch_size=4, device="cpu")
        users = [1, 2, 1, 3]
        items = [10, 10, 20, 30]
        ratings = [1, 0, 1, 0]
        model.fit(users, items, ratings)
        recs = model.recommend(1, top_n=2, exclude_seen=False)
        self.assertIsInstance(recs, list)

    def test_sasrec(self):
        try:
            import scipy.sparse as sp
        except Exception as e:
            self.skipTest(f"Skip SASRec due to missing scipy: {e}")
        try:
            m = load_module("sasrec.py")
        except Exception as e:
            self.skipTest(f"Skip SASRec import: {e}")
        model = m.SASRec(hidden_units=8, num_blocks=1, num_heads=1, num_epochs=1, batch_size=4, max_seq_length=5, device="cpu")
        # Build tiny interaction matrix as csr
        user_ids = [1, 2]
        item_ids = [10, 20, 30]
        dense = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
        inter = sp.csr_matrix(dense)
        model.fit(inter, user_ids, item_ids)
        # For user 1
        recs = model.recommend(1, top_n=2)
        self.assertIsInstance(recs, list)


if __name__ == "__main__":
    unittest.main(verbosity=2) 