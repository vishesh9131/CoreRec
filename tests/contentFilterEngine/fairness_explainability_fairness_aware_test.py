import unittest
import pandas as pd
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


class TestFairnessAware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module("corerec/engines/contentFilterEngine/fairness_explainability/fairness_aware.py")
        cls.FAIRNESS_AWARE = cls.mod.FAIRNESS_AWARE

    def test_evaluate_and_ensure(self):
        fa = self.FAIRNESS_AWARE()
        recs = {1: [10, 11], 2: [11]}
        users = pd.DataFrame({"user_id": [1, 2], "gender": ["M", "F"]})
        metrics = fa.evaluate_fairness(recs, users)
        self.assertIn("gender_distribution", metrics)
        # Ensure returns same object (placeholder)
        out = fa.ensure_fairness(recs, users)
        self.assertEqual(out, recs)


if __name__ == "__main__":
    unittest.main(verbosity=2) 