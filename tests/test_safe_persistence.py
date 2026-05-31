"""Tests for safe model bundle persistence."""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from corerec.api.model_bundle import is_safe_bundle, load_bundle, save_bundle
from corerec.engines.collaborative import FAST, SAR
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

    def test_fast_safe_save_load(self):
        model = FAST(factors=4, iterations=1, batch_size=2, seed=42)
        model.fit([0, 0, 1], [10, 11, 10], [5.0, 4.0, 3.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fast")
            model.save(path, safe=True)
            self.assertTrue(is_safe_bundle(path))
            loaded = FAST.load(path)
            self.assertTrue(loaded.is_fitted)
            self.assertIsNotNone(loaded.user_factors)

    def test_sar_safe_save_load(self):
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
            loaded = DCN.load(legacy)
            self.assertTrue(loaded.is_fitted)


if __name__ == "__main__":
    unittest.main()
