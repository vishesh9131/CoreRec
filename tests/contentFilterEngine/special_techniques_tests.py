import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

BASE = (
    Path(__file__).resolve().parents[2]
    / "corerec"
    / "engines"
    / "contentFilterEngine"
    / "special_techniques"
)


def load(fname: str):
    spec = spec_from_file_location(fname, str(BASE / fname))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class DummyBase:
    def __init__(self):
        self.added = []
        self.removed = []
        self.updated = []

    def add_item(self, item_id, feats):
        self.added.append((item_id, feats))

    def remove_item(self, item_id):
        self.removed.append(item_id)

    def update_item_features(self, item_id, feats):
        self.updated.append((item_id, feats))

    def recommend(self, query, top_n=10):
        return list(range(top_n))

    def update_user_profile(self, user_id, item_id, feedback):
        pass


class TestSpecialTechniques(unittest.TestCase):
    def test_dynamic_filtering(self):
        m = load("dynamic_filtering.py")
        base = DummyBase()
        df = m.DynamicFilteringRecommender(base)
        df.add_item(1, {"g": "a"})
        df.update_item_features(1, {"g": "b"})
        df.remove_item(1)
        self.assertTrue(base.added and base.removed and base.updated)
        out = df.recommend(user_id=1, query="q", top_n=3)
        self.assertEqual(out, [0, 1, 2])

    def test_interactive_filtering(self):
        m = load("interactive_filtering.py")

        class BaseRec:
            def recommend(self, user_id, query, top_n=10):
                return [1, 2, 3, 4]

            def update_user_profile(self, user_id, item_id, feedback):
                self.last = (user_id, item_id, feedback)

        base = BaseRec()
        ir = m.InteractiveFilteringRecommender(base)
        ir.collect_feedback(7, 2, -1.0)
        self.assertEqual(base.last, (7, 2, -1.0))
        out = ir.recommend(7, "q", top_n=3)
        self.assertEqual(out, [1, 3, 4])

    def test_temporal_import(self):
        m = load("temporal_filtering.py")
        self.assertTrue(hasattr(m, "TemporalFilteringRecommender"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
