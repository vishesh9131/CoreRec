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


class TestExplainable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module("corerec/engines/contentFilterEngine/fairness_explainability/explainable.py")
        cls.EXPLAINABLE = cls.mod.EXPLAINABLE

    def test_generate_and_get(self):
        expl = self.EXPLAINABLE()
        txt = expl.generate_explanation(1, 42, context={"reason": "similar taste"})
        self.assertIn("Item 42", txt)
        self.assertIn("User 1", txt)
        got = expl.get_explanation(1, 42)
        self.assertEqual(txt, got)
        self.assertEqual(expl.get_explanation(2, 99), "No explanation available.")


if __name__ == "__main__":
    unittest.main(verbosity=2) 