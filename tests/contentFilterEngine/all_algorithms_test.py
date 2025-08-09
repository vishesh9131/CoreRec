import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

CONTENT_ENGINE_ROOT = Path(__file__).resolve().parents[2] / "corerec" / "engines" / "contentFilterEngine"

# Files likely safe to import are included; others may have optional heavy deps.
EXCLUDE_DIR_NAMES = {"__pycache__"}
EXCLUDE_FILE_NAMES = {"__init__.py"}


def discover_algorithm_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
            continue
        if path.name in EXCLUDE_FILE_NAMES:
            continue
        # Keep only leaf modules (exclude __init__.py already)
        yield path


def load_module_from_path(path: Path):
    spec = spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAllContentFilterAlgorithms(unittest.TestCase):
    def test_import_all_algorithm_modules(self):
        self.assertTrue(CONTENT_ENGINE_ROOT.exists(), f"Missing path: {CONTENT_ENGINE_ROOT}")
        failures = []
        skipped = []
        passed = []

        for file_path in sorted(discover_algorithm_files(CONTENT_ENGINE_ROOT)):
            with self.subTest(module=str(file_path.relative_to(CONTENT_ENGINE_ROOT))):
                try:
                    mod = load_module_from_path(file_path)
                    passed.append(file_path)
                except ImportError as e:
                    # Likely missing optional dependency; mark as skipped
                    skipped.append((file_path, f"ImportError: {e}"))
                    self.skipTest(f"Optional dep missing for {file_path.name}: {e}")
                except Exception as e:
                    failures.append((file_path, str(e)))
                    self.fail(f"Failed importing {file_path}: {e}")

        # Summary for logs (unittest will reflect individual subTest outcomes)
        # These prints help in CI logs; not assertions
        print(f"\n[ContentFilterEngine Import Summary] Passed: {len(passed)}, Skipped: {len(skipped)}, Failures: {len(failures)}")
        if skipped:
            for p, r in skipped[:5]:
                print(f"  - SKIP {p.name}: {r}")
        if failures:
            for p, r in failures[:5]:
                print(f"  - FAIL {p.name}: {r}")


if __name__ == "__main__":
    unittest.main(verbosity=2) 