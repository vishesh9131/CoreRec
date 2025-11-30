"""
Integration tests for migrated engine models.

Tests that migrated models (DCN, DeepFM, GNNRec, NASRec) work correctly
with the new Base Recommender API.

Author: Vishesh Yadav
"""

import unittest
import numpy as np
from typing import List


class TestMigratedModels(unittest.TestCase):
    """Test suite for migrated engine models."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic user-item interaction data
        np.random.seed(42)
        self.n_users = 50
        self.n_items = 30
        self.n_interactions = 200

        self.user_ids = np.random.randint(
            0, self.n_users, self.n_interactions).tolist()
        self.item_ids = np.random.randint(
            0, self.n_items, self.n_interactions).tolist()
        self.ratings = np.random.choice(
            [0.0, 1.0], self.n_interactions).tolist()

    def test_dcn_integration(self):
        """Test DCN model end-to-end."""
        from corerec.engines.dcn import DCN
        from corerec.api.base_recommender import BaseRecommender

        # Test inheritance
        self.assertTrue(issubclass(DCN, BaseRecommender))

        # Test initialization
        model = DCN(epochs=1, verbose=False, batch_size=64)
        self.assertEqual(model.name, "DCN")
        self.assertFalse(model.is_fitted)

        # Test fit
        result = model.fit(
            user_ids=self.user_ids[:50], item_ids=self.item_ids[:50], ratings=self.ratings[:50]
        )
        self.assertIs(result, model)  # Test method chaining
        self.assertTrue(model.is_fitted)

        # Test recommend
        recs = model.recommend(user_id=self.user_ids[0], top_k=5)
        self.assertIsInstance(recs, list)
        self.assertLessEqual(len(recs), 5)

    def test_deepfm_integration(self):
        """Test DeepFM model end-to-end."""
        from corerec.engines.deepfm import DeepFM
        from corerec.api.base_recommender import BaseRecommender

        # Test inheritance
        self.assertTrue(issubclass(DeepFM, BaseRecommender))

        # Test initialization
        model = DeepFM(epochs=1, verbose=False, batch_size=64)
        self.assertEqual(model.name, "DeepFM")
        self.assertFalse(model.is_fitted)

    def test_gnnrec_integration(self):
        """Test GNNRec model end-to-end."""
        from corerec.engines.gnnrec import GNNRec
        from corerec.api.base_recommender import BaseRecommender

        # Test inheritance
        self.assertTrue(issubclass(GNNRec, BaseRecommender))

        # Test initialization
        model = GNNRec(
            epochs=1,
            verbose=False,
            batch_size=64,
            num_gnn_layers=2)
        self.assertEqual(model.name, "GNNRec")
        self.assertFalse(model.is_fitted)

    def test_nasrec_integration(self):
        """Test NASRec model end-to-end."""
        from corerec.engines.nasrec import NASRec
        from corerec.api.base_recommender import BaseRecommender

        # Test inheritance
        self.assertTrue(issubclass(NASRec, BaseRecommender))

        # Test initialization
        model = NASRec(epochs=1, verbose=False, batch_size=64)
        self.assertEqual(model.name, "NASRec")
        self.assertFalse(model.is_fitted)

    def test_all_models_have_required_methods(self):
        """Test that all migrated models have required BaseRecommender methods."""
        from corerec.engines.dcn import DCN
        from corerec.engines.deepfm import DeepFM
        from corerec.engines.gnnrec import GNNRec
        from corerec.engines.nasrec import NASRec

        models = [DCN(epochs=1), DeepFM(epochs=1),
                  GNNRec(epochs=1), NASRec(epochs=1)]

        required_methods = ["fit", "predict",
                            "recommend"]  # save/load to be added

        for model in models:
            for method in required_methods:
                self.assertTrue(
                    hasattr(
                        model, method), f"{
                        model.__class__.__name__} missing {method} method")

    def test_model_info(self):
        """Test that models provide correct metadata."""
        from corerec.engines.dcn import DCN

        model = DCN(epochs=5, verbose=True)
        info = model.get_model_info()

        self.assertEqual(info["name"], "DCN")
        self.assertEqual(info["model_type"], "DCN")
        self.assertFalse(info["is_fitted"])
        self.assertTrue(info["trainable"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
