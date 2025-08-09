import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

SUBDIR = "traditional_ml_algorithms"
ROOT = Path(__file__).resolve().parents[2] / "corerec" / "engines" / "contentFilterEngine" / SUBDIR

EXCLUDE_DIRS = {"__pycache__"}
EXCLUDE_FILES = {"__init__.py"}


def discover_files(root: Path):
    for p in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.name in EXCLUDE_FILES:
            continue
        yield p


def import_file(path: Path):
    spec = spec_from_file_location(path.stem, str(path))
    if not spec or not spec.loader:
        raise ImportError(f"No loader for {path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestImportTraditionalML(unittest.TestCase):
    def test_import_all(self):
        self.assertTrue(ROOT.exists())
        for file_path in sorted(discover_files(ROOT)):
            with self.subTest(module=str(file_path.relative_to(ROOT))):
                try:
                    import_file(file_path)
                except ImportError as e:
                    self.skipTest(f"Optional dep missing for {file_path.name}: {e}")
                except Exception as e:
                    self.fail(f"Failed importing {file_path}: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2) 