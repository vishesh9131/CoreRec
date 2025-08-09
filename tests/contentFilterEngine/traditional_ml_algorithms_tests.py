import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

BASE = Path(__file__).resolve().parents[2] / "corerec" / "engines" / "contentFilterEngine" / "traditional_ml_algorithms"


def load(fname: str):
    spec = spec_from_file_location(fname, str(BASE / fname))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestTraditionalMLImports(unittest.TestCase):
    def test_imports(self):
        for fname in ["tfidf.py", "vw.py", "decision_tree.py", "lightgbm.py", "svm.py", "LR.py"]:
            try:
                load(fname)
            except Exception as e:
                self.fail(f"Failed to import {fname}: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2) 