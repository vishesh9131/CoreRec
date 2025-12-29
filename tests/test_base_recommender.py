"""
Test Suite for BaseRecommender Unified API

Tests the unified BaseRecommender class ensuring all functionality
from BaseCorerec is preserved and works correctly.

Author: Vishesh Yadav (sciencely98@gmail.com)
"""

import unittest
import tempfile
import os
from pathlib import Path
from typing import List, Any
import pickle

from corerec.api.base_recommender import BaseRecommender


class MockRecommender(BaseRecommender):
    """Mock recommender for testing BaseRecommender functionality."""

    def __init__(self, name=None, trainable=True, verbose=False):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.fit_called = False

    def fit(self, *args, **kwargs):
        """Mock fit implementation."""
        self.is_fitted = True
        self.fit_called = True
        # Simulate setting user/item maps
        self.num_users = 100
        self.num_items = 50
        self.uid_map = {i: i for i in range(100)}
        self.iid_map = {i: i for i in range(50)}
        self.global_mean = 3.5
        self.max_rating = 5.0
        self.min_rating = 1.0
        return self

    def predict(self, user_id: Any, item_id: Any, **kwargs) -> float:
        """Mock predict implementation."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return 4.2

    def recommend(
            self,
            user_id: Any,
            top_k: int = 10,
            exclude_items: List[Any] = None,
            **kwargs) -> List[Any]:
        """Mock recommend implementation."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return list(range(top_k))

    def save(self, path, **kwargs):
        """Mock save implementation."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Mock load implementation."""
        with open(path, "rb") as f:
            return pickle.load(f)


class TestBaseRecommender(unittest.TestCase):
    """Test cases for BaseRecommender unified API."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MockRecommender(name="TestModel", verbose=False)

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.name, "TestModel")
        self.assertFalse(self.model.is_fitted)
        self.assertFalse(self.model.verbose)
        self.assertTrue(self.model.trainable)

    def test_default_name(self):
        """Test that default name is class name."""
        model = MockRecommender()
        self.assertEqual(model.name, "MockRecommender")

    def test_fit(self):
        """Test fit method."""
        result = self.model.fit()
        self.assertTrue(self.model.is_fitted)
        self.assertTrue(self.model.fit_called)
        self.assertIs(result, self.model)  # Test method chaining

    def test_predict_before_fit(self):
        """Test that predict raises error before fitting."""
        with self.assertRaises(RuntimeError):
            self.model.predict(1, 10)

    def test_predict_after_fit(self):
        """Test predict after fitting."""
        self.model.fit()
        score = self.model.predict(1, 10)
        self.assertEqual(score, 4.2)

    def test_recommend_before_fit(self):
        """Test that recommend raises error before fitting."""
        with self.assertRaises(RuntimeError):
            self.model.recommend(1)

    def test_recommend_after_fit(self):
        """Test recommend after fitting."""
        self.model.fit()
        recs = self.model.recommend(1, top_k=5)
        self.assertEqual(len(recs), 5)
        self.assertEqual(recs, [0, 1, 2, 3, 4])

    def test_batch_predict(self):
        """Test batch prediction."""
        self.model.fit()
        pairs = [(1, 10), (2, 20), (3, 30)]
        scores = self.model.batch_predict(pairs)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(s == 4.2 for s in scores))

    def test_batch_recommend(self):
        """Test batch recommendation."""
        self.model.fit()
        recs = self.model.batch_recommend([1, 2, 3], top_k=3)
        self.assertEqual(len(recs), 3)
        self.assertIn(1, recs)
        self.assertIn(2, recs)
        self.assertIn(3, recs)
        self.assertEqual(recs[1], [0, 1, 2])

    def test_user_item_properties(self):
        """Test total_users and total_items properties."""
        self.model.fit()
        self.assertEqual(self.model.total_users, 100)
        self.assertEqual(self.model.total_items, 50)
        self.assertEqual(len(self.model.user_ids), 100)
        self.assertEqual(len(self.model.item_ids), 50)

    def test_knows_user(self):
        """Test knows_user method."""
        self.model.fit()
        self.assertTrue(self.model.knows_user(0))
        self.assertTrue(self.model.knows_user(99))
        self.assertFalse(self.model.knows_user(100))
        self.assertFalse(self.model.knows_user(-1))
        self.assertFalse(self.model.knows_user(None))

    def test_knows_item(self):
        """Test knows_item method."""
        self.model.fit()
        self.assertTrue(self.model.knows_item(0))
        self.assertTrue(self.model.knows_item(49))
        self.assertFalse(self.model.knows_item(50))
        self.assertFalse(self.model.knows_item(-1))

    def test_is_unknown_user(self):
        """Test is_unknown_user method."""
        self.model.fit()
        self.assertFalse(self.model.is_unknown_user(0))
        self.assertTrue(self.model.is_unknown_user(100))

    def test_is_unknown_item(self):
        """Test is_unknown_item method."""
        self.model.fit()
        self.assertFalse(self.model.is_unknown_item(0))
        self.assertTrue(self.model.is_unknown_item(50))

    def test_save_load(self):
        """Test save and load functionality."""
        self.model.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")
            self.model.save(path)
            self.assertTrue(os.path.exists(path))

            loaded_model = MockRecommender.load(path)
            self.assertTrue(loaded_model.is_fitted)
            self.assertEqual(loaded_model.num_users, 100)
            self.assertEqual(loaded_model.num_items, 50)

    def test_get_model_info(self):
        """Test get_model_info method."""
        self.model.fit()
        info = self.model.get_model_info()

        self.assertEqual(info["name"], "TestModel")
        self.assertTrue(info["is_fitted"])
        self.assertEqual(info["model_type"], "MockRecommender")
        self.assertEqual(info["num_users"], 100)
        self.assertEqual(info["num_items"], 50)

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.model)
        self.assertIn("MockRecommender", repr_str)
        self.assertIn("TestModel", repr_str)
        self.assertIn("fitted=False", repr_str)

    def test_clone(self):
        """Test model cloning."""
        self.model.fit()
        cloned = self.model.clone()

        self.assertIsNot(cloned, self.model)
        self.assertEqual(cloned.name, self.model.name)
        # Cloned model should not be fitted
        self.assertFalse(cloned.is_fitted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
