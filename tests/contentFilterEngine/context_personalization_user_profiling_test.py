import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import pandas as pd


def load_module(rel_path: str):
    repo_root = Path(__file__).resolve().parents[2]
    mod_path = repo_root / rel_path
    spec = spec_from_file_location(mod_path.stem, str(mod_path))
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestUserProfiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module(
            "corerec/engines/contentFilterEngine/context_personalization/user_profiling.py"
        )
        cls.UserProfilingRecommender = cls.mod.UserProfilingRecommender

    def test_fit_and_recommend(self):
        users = pd.DataFrame({"user_id": [1, 2], "age": [25, 30]})
        rec = self.UserProfilingRecommender(user_attributes=users)
        interactions = {1: [10, 11], 2: [10]}
        rec.fit(interactions)

        # Profiles created
        self.assertIn(1, rec.user_profiles)
        self.assertIn(2, rec.user_profiles)
        self.assertIn("age", rec.user_profiles[1])
        self.assertEqual(rec.user_profiles[1]["age"], 25)

        # Recommend excludes interacted items
        all_items = {10, 11, 12, 13}
        top = rec.recommend(user_id=1, all_items=all_items, top_n=3)
        self.assertEqual(len(top), 2)  # only 12 and 13 left
        self.assertTrue(set(top).issubset({12, 13}))

        # Unknown user -> empty list
        self.assertEqual(rec.recommend(user_id=99, all_items=all_items), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
