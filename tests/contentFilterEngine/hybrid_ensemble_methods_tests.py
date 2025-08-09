import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

BASE = Path(__file__).resolve().parents[2] / "corerec" / "engines" / "contentFilterEngine" / "hybrid_ensemble_methods"


def load(mod_name: str, filename: str):
    spec = spec_from_file_location(mod_name, str(BASE / filename))
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestHybridEnsembleModules(unittest.TestCase):
    def test_attention_module_imports(self):
        m = load("attention_mechanisms", "attention_mechanisms.py")
        # File is a placeholder; just ensure it imports
        self.assertTrue(hasattr(m, "__doc__"))

    def test_ensemble_methods_imports(self):
        m = load("ensemble_methods", "ensemble_methods.py")
        self.assertTrue(hasattr(m, "__doc__"))

    def test_hybrid_collaborative_imports(self):
        m = load("hybrid_collaborative", "hybrid_collaborative.py")
        self.assertTrue(hasattr(m, "__doc__"))


if __name__ == "__main__":
    unittest.main(verbosity=2) 