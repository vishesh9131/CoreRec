"""API uniformity tests for production models."""
import inspect
import unittest
import warnings

from corerec.api.base_recommender import BaseRecommender
from corerec.api.dataset import RecommenderDataset
from corerec.api.exceptions import ModelNotFittedError, RecommendationError
from corerec.engines.collaborative import FAST, FastRecommender, SAR
from corerec.engines.collaborative.nn_base.ncf import NCF
from corerec.engines.dcn import DCN
from corerec.engines.sasrec import SASRec


class TestAPIUniformity(unittest.TestCase):
    def test_fast_recommender_lazy_import(self):
        self.assertIsNotNone(FastRecommender)
        self.assertTrue(issubclass(FastRecommender, BaseRecommender))

    def test_recommend_accepts_top_k(self):
        sig = inspect.signature(NCF.recommend)
        self.assertIn("top_k", sig.parameters)

    def test_top_n_emits_deprecation(self):
        model = FAST(factors=4, iterations=1, batch_size=2, seed=42)
        model.fit([0, 0, 1, 1], [10, 11, 10, 12], [5.0, 4.0, 3.0, 5.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.recommend(0, top_n=2)
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))

    def test_unfitted_raises_model_not_fitted(self):
        model = DCN()
        with self.assertRaises(ModelNotFittedError):
            model.predict(0, 0)

    def test_sar_unknown_user_raises(self):
        import pandas as pd

        df = pd.DataFrame(
            {"userID": [0, 1], "itemID": [10, 11], "rating": [5.0, 4.0]}
        )
        model = SAR()
        model.fit(df)
        with self.assertRaises(RecommendationError):
            model.recommend(99999, top_k=3)

    def test_recommender_dataset_triplet(self):
        ds = RecommenderDataset.from_triplet([0, 0, 1, 1], [10, 11, 10, 12], [5.0, 4.0, 3.0, 5.0])
        model = FAST(factors=4, iterations=1, batch_size=2, seed=42)
        u, i, r = ds.as_triplet()
        model.fit(u, i, r)
        recs = model.recommend(0, top_k=2)
        self.assertIsInstance(recs, list)


if __name__ == "__main__":
    unittest.main()
