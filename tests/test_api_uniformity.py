"""API uniformity tests for production models."""
import inspect
import unittest
import warnings

from corerec.api.base_recommender import BaseRecommender
from corerec.api.dataset import RecommenderDataset
from corerec.api.exceptions import ModelNotFittedError
from corerec.engines import MODELS
from corerec.engines.collaborative import SAR
from corerec.engines.dcn import DCN
from corerec.engines.deepfm import DeepFM
from corerec.engines.sasrec import SASRec


class TestAPIUniformity(unittest.TestCase):
    def test_registry_models_are_base_recommenders(self):
        import corerec.engines as engines

        for name in MODELS:
            cls = getattr(engines, name)
            self.assertTrue(issubclass(cls, BaseRecommender), name)

    def test_recommend_accepts_top_k(self):
        import corerec.engines as engines

        for name in MODELS:
            sig = inspect.signature(getattr(engines, name).recommend)
            self.assertIn("top_k", sig.parameters, name)

    def test_top_n_emits_deprecation(self):
        model = DeepFM(embedding_dim=4, epochs=1, batch_size=2)
        model.fit([0, 0, 1, 1], [10, 11, 10, 12], [5.0, 4.0, 3.0, 5.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.recommend(0, top_n=2)
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))

    def test_unfitted_raises_model_not_fitted(self):
        model = DCN()
        with self.assertRaises(ModelNotFittedError):
            model.predict(0, 0)

    def test_sar_unknown_user_returns_empty(self):
        import pandas as pd

        df = pd.DataFrame(
            {"userID": [0, 1], "itemID": [10, 11], "rating": [5.0, 4.0]}
        )
        model = SAR()
        model.fit(df)
        self.assertEqual(model.recommend(99999, top_k=3), [])

    def test_sar_custom_column_names(self):
        """col_user/col_item used to be renamed away before fit() looked for them."""
        import pandas as pd

        df = pd.DataFrame(
            {"user_id": [0, 0, 1], "item_id": [10, 11, 10], "rating": [5.0, 4.0, 3.0]}
        )
        model = SAR(col_user="user_id", col_item="item_id", col_rating="rating")
        model.fit(df)
        self.assertEqual(model.recommend(1, top_k=1), [11])

    def test_recommender_dataset_triplet(self):
        ds = RecommenderDataset.from_triplet([0, 0, 1, 1], [10, 11, 10, 12], [5.0, 4.0, 3.0, 5.0])
        model = SASRec(hidden_units=8, num_blocks=1, epochs=1, max_seq_length=4)
        model.fit(ds)
        recs = model.recommend(0, top_k=2)
        self.assertIsInstance(recs, list)


if __name__ == "__main__":
    unittest.main()
