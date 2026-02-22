import unittest
import numpy as np
import sys
from types import ModuleType
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
CFE_DIR = REPO_ROOT / "corerec" / "engines" / "content_based"


def ensure_fake_package(package_name: str, path: Path):
    if package_name in sys.modules:
        return sys.modules[package_name]
    pkg = ModuleType(package_name)
    pkg.__path__ = [str(path)]
    sys.modules[package_name] = pkg
    return pkg


def load_as(package_name: str, rel_file: str):
    file_path = CFE_DIR / rel_file
    spec = spec_from_file_location(package_name, str(file_path))
    module = module_from_spec(spec)
    # Set package to parent to enable relative imports in the module
    module.__package__ = package_name.rsplit(".", 1)[0]
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[package_name] = module
    return module


class TestContentFilterFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build minimal package skeleton to satisfy relative imports without
        # executing real __init__
        ensure_fake_package("corerec", REPO_ROOT / "corerec")
        ensure_fake_package(
            "corerec.engines",
            REPO_ROOT /
            "corerec" /
            "engines")
        ensure_fake_package("corerec.engines.content_based", CFE_DIR)

        # Preload tfidf module under the expected fully-qualified name
        cls.tfidf_mod = load_as(
            "corerec.engines.content_based.tfidf_recommender",
            "tfidf_recommender.py")
        # Now load factory with proper __package__ so `from .tfidf_recommender`
        # works
        cls.factory_mod = load_as(
            "corerec.engines.content_based.cr_contentFilterFactory",
            "cr_contentFilterFactory.py",
        )

    def test_get_recommender_tfidf(self):
        feature_matrix = np.eye(5)
        cfg = {"method": "tfidf", "params": {"feature_matrix": feature_matrix}}
        rec = self.factory_mod.ContentFilterFactory.get_recommender(cfg)
        self.assertIsInstance(rec, self.tfidf_mod.TFIDFRecommender)

    def test_get_recommender_unsupported(self):
        with self.assertRaises(ValueError):
            self.factory_mod.ContentFilterFactory.get_recommender(
                {"method": "unknown"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
