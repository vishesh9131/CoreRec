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


class TestPrivacyPreserving(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module(
            "corerec/engines/contentFilterEngine/fairness_explainability/privacy_preserving.py"
        )
        cls.PRIVACY_PRESERVING = cls.mod.PRIVACY_PRESERVING

    def test_anonymize_and_dp(self):
        df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "zip_code": ["12345", "54321"],
                "age": [25, 30],
            }
        )
        pp = self.PRIVACY_PRESERVING()
        anon = pp.anonymize_data(df)
        self.assertNotIn("user_id", anon.columns)
        self.assertNotIn("zip_code", anon.columns)
        self.assertIn("age", anon.columns)

        # DP placeholder returns same data
        out = pp.apply_differential_privacy(anon, epsilon=1.0)
        pd.testing.assert_frame_equal(out, anon)


if __name__ == "__main__":
    unittest.main(verbosity=2)
