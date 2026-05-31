"""Unit tests for corerec.api modules (coverage)."""
import unittest
import warnings

from corerec.api.dataset import RecommenderDataset, coerce_dataset
from corerec.api.recommend_args import normalize_recommend_kwargs
from corerec.api.versioning import API_VERSION, warn_deprecated_arg
from corerec.api.exceptions import InvalidParameterError
import pandas as pd


class TestAPIModules(unittest.TestCase):
    def test_recommender_dataset_triplet(self):
        ds = RecommenderDataset.from_triplet([1, 2], [3, 4], [5.0, 4.0])
        self.assertEqual(ds.infer_mode(), "triplet")

    def test_recommender_dataset_dataframe(self):
        df = pd.DataFrame({"user_id": [1], "item_id": [2], "rating": [5.0]})
        ds = coerce_dataset(df)
        self.assertIsNotNone(ds)
        self.assertEqual(ds.infer_mode(), "dataframe")

    def test_normalize_top_n_deprecation(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            k, ex, _ = normalize_recommend_kwargs(top_k=10, top_n=5)
            self.assertEqual(k, 5)
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))

    def test_normalize_invalid_top_k(self):
        with self.assertRaises(InvalidParameterError):
            normalize_recommend_kwargs(top_k=0)

    def test_api_version(self):
        self.assertTrue(len(API_VERSION) >= 3)

    def test_warn_deprecated_arg(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated_arg("old", "new")
            self.assertTrue(len(w) >= 1)


if __name__ == "__main__":
    unittest.main()
