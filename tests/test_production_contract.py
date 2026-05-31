"""Production contract tests — all 14 models must follow the unified API."""
import importlib
import inspect
import unittest

from corerec.api.base_recommender import BaseRecommender
from corerec.api.dataset import RecommenderDataset
from corerec.api.exceptions import ModelNotFittedError


PRODUCTION_MODELS = [
    ("corerec.engines.dcn", "DCN"),
    ("corerec.engines.deepfm", "DeepFM"),
    ("corerec.engines.gnnrec", "GNNRec"),
    ("corerec.engines.mind", "MIND"),
    ("corerec.engines.nasrec", "NASRec"),
    ("corerec.engines.sasrec", "SASRec"),
    ("corerec.engines.two_tower", "TwoTower"),
    ("corerec.engines.bert4rec", "BERT4Rec"),
    ("corerec.engines.collaborative.sar", "SAR"),
    ("corerec.engines.collaborative.nn_base.ncf", "NCF"),
    ("corerec.engines.collaborative.fast", "FAST"),
    ("corerec.engines.collaborative.fast_recommender", "FASTRecommender"),
    ("corerec.engines.collaborative.graph_based_base.lightgcn", "LightGCN"),
    ("corerec.engines.content_based.tfidf_recommender", "TFIDFRecommender"),
]


class TestProductionContract(unittest.TestCase):
    def test_all_inherit_base(self):
        for mod_path, cls_name in PRODUCTION_MODELS:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            with self.subTest(model=cls_name):
                self.assertTrue(issubclass(cls, BaseRecommender))

    def test_recommend_accepts_top_k(self):
        for mod_path, cls_name in PRODUCTION_MODELS:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            sig = inspect.signature(cls.recommend)
            with self.subTest(model=cls_name):
                self.assertIn("top_k", sig.parameters)

    def test_unfitted_predict_raises(self):
        for mod_path, cls_name in PRODUCTION_MODELS:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            try:
                model = cls()
            except TypeError:
                model = cls(verbose=False)  # some DL models need no required kwargs
            with self.subTest(model=cls_name):
                with self.assertRaises(ModelNotFittedError):
                    model.predict(0, 0)

    def test_recommender_dataset_importable(self):
        ds = RecommenderDataset.from_triplet([0], [1], [5.0])
        self.assertEqual(ds.infer_mode(), "triplet")


if __name__ == "__main__":
    unittest.main()
