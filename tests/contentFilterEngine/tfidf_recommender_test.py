import unittest
import numpy as np
import sys
import os
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


def load_tfidf_module():
    # Compute absolute path to the module file to avoid importing corerec.engines package
    repo_root = Path(__file__).resolve().parents[2]
    mod_path = repo_root / "corerec" / "engines" / "contentFilterEngine" / "tfidf_recommender.py"
    spec = spec_from_file_location("tfidf_recommender", str(mod_path))
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestTFIDFRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tfidf_mod = load_tfidf_module()
        cls.TFIDFRecommender = cls.tfidf_mod.TFIDFRecommender

    def setUp(self):
        # Build a simple feature matrix with clear similarities
        # items: A, B similar; C dissimilar; D zero vector; E random
        self.features = np.array([
            [1.0, 1.0, 0.0, 0.0],  # A
            [0.9, 1.1, 0.0, 0.0],  # B (similar to A)
            [0.0, 0.0, 1.0, 0.0],  # C (orthogonal)
            [0.0, 0.0, 0.0, 0.0],  # D (zero vector)
            [0.2, 0.1, 0.3, 0.4],  # E (mixed)
        ])
        self.rec = self.TFIDFRecommender(self.features)

    def test_similarity_matrix_shape(self):
        sim = self.rec.similarity_matrix
        self.assertEqual(sim.shape, (self.features.shape[0], self.features.shape[0]))

    def test_similarity_self_highest(self):
        sim = self.rec.similarity_matrix
        # diagonal should be 1 for non-zero rows (cosine similarity), 0 for zero vector row
        self.assertAlmostEqual(sim[0, 0], 1.0, places=6)
        self.assertAlmostEqual(sim[1, 1], 1.0, places=6)
        self.assertAlmostEqual(sim[2, 2], 1.0, places=6)
        self.assertEqual(sim[3, 3], 0.0)  # zero vector normalized to zeros

    def test_recommend_returns_int_indices(self):
        top = self.rec.recommend([0], top_n=3)
        self.assertIsInstance(top, np.ndarray)
        self.assertEqual(len(top), 3)
        for idx in top:
            self.assertIsInstance(int(idx), (int,))

    def test_recommend_similarity_order(self):
        # For item A (index 0), B (index 1) should appear among top similar items
        top = list(self.rec.recommend([0], top_n=5))
        # Remove A itself if present
        if 0 in top:
            top.remove(0)
        self.assertIn(1, top)

    def test_recommend_combined_scores(self):
        # Recommending based on A and C should surface both neighborhoods
        top = self.rec.recommend([0, 2], top_n=5)
        self.assertEqual(len(top), 5)


if __name__ == "__main__":
    unittest.main() 