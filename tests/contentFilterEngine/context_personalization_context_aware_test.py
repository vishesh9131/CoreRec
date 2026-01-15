import unittest
import json
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


class TestContextAware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module(
            "corerec/engines/content_based/context_personalization/context_aware.py"
        )
        cls.ContextAwareRecommender = cls.mod.ContextAwareRecommender

    def test_init_and_weights(self):
        # Create a temp config
        config = {
            "default": {"time": "morning"},
            "time": {
                "morning": {"genre_action": 1.5, "duration": 0.8},
                "evening": {"genre_action": 0.7, "duration": 1.2},
            },
        }
        tmp = Path(__file__).resolve().parent / "tmp_ctx_config.json"
        tmp.write_text(json.dumps(config))
        try:
            items = {
                10: {"genre": "action", "duration": 100},
                11: {"genre": "drama", "duration": 90},
                12: {"genre": "action", "duration": 110},
            }
            rec = self.ContextAwareRecommender(str(tmp), item_features=items)

            # Default weights from config default
            w = rec._initialize_feature_weights()
            self.assertIsInstance(w, dict)

            # Apply a context and recompute
            w2 = rec._initialize_feature_weights({"time": "evening"})
            self.assertIn("genre_action", w2)

            # Encode an item with current weights
            rec.feature_weights = w2
            enc = rec._encode_item_features(10)
            self.assertTrue(any(k.startswith("genre_") for k in enc.keys()))

            # Fit on interactions and recommend
            data = {1: [10, 11]}
            rec.fit(data)
            out = rec.recommend(
                user_id=1, context={
                    "time": "evening"}, top_n=2)
            self.assertTrue(all(isinstance(x, int) for x in out))
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
