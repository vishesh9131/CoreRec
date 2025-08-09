import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


def load_module(rel_path: str):
    repo_root = Path(__file__).resolve().parents[2]
    mod_path = repo_root / rel_path
    spec = spec_from_file_location(mod_path.stem, str(mod_path))
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestItemProfiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module("corerec/engines/contentFilterEngine/context_personalization/item_profiling.py")
        cls.ItemProfilingRecommender = cls.mod.ItemProfilingRecommender

    def test_fit_builds_profiles(self):
        rec = self.ItemProfilingRecommender()
        data = {1: [10, 11], 2: [10]}
        item_features = {
            10: {"genre": "action", "year": 2020},
            11: {"genre": "drama", "year": 2019},
        }
        rec.fit(data, item_features)
        # Item profiles should exist for seen items
        self.assertIn(10, rec.item_profiles)
        self.assertIn(11, rec.item_profiles)

    def test_recommend_not_implemented_dependencies(self):
        rec = self.ItemProfilingRecommender()
        # The module references vectorizer/lsa_model that are not initialized.
        with self.assertRaises(Exception):
            rec.recommend("query")


if __name__ == "__main__":
    unittest.main(verbosity=2) 