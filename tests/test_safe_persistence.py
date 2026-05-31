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
            self.assertAlmostEqual(model.predict(0, 10), loaded.predict(0, 10), delta=1e-2)

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
            loaded = DCN.load(legacy)
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
            num_epochs=1,
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
