import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

BASE = (
    Path(__file__).resolve().parents[2]
    / "corerec"
    / "engines"
    / "collaborative"
    / "sequential_model_base"
)

EXCLUDE = {"__init__.py"}


def discover_files(root: Path):
    for p in root.glob("*.py"):
        if p.name in EXCLUDE:
            continue
        yield p


def load(path: Path):
    spec = spec_from_file_location(path.stem, str(path))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestSequentialModelBaseImports(unittest.TestCase):
    def test_import_all(self):
        self.assertTrue(BASE.exists())
        for f in sorted(discover_files(BASE)):
            with self.subTest(module=str(f.name)):
                try:
                    load(f)
                except ImportError as e:
                    self.skipTest(f"Skip {f.name}: {e}")
                except Exception as e:
                    self.fail(f"Failed importing {f.name}: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
