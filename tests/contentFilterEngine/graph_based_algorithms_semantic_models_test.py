import unittest
import sys
from types import ModuleType
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = REPO_ROOT / "corerec" / "engines" / \
    "contentFilterEngine" / "graph_based_algorithms"


def ensure_fake_module(name: str):
    mod = ModuleType(name)
    sys.modules[name] = mod
    return mod


def load_module():
    spec = spec_from_file_location(
        "semantic_models", str(
            BASE_DIR / "semantic_models.py"))
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestSemanticModels(unittest.TestCase):
    def setUp(self):
        # Stub corerec.vish_graphs with run_optimal_path capturing calls
        pkg = ensure_fake_module("corerec")
        subpkg = ensure_fake_module("corerec.vish_graphs")
        calls = {}

        def run_optimal_path(graph, start_city):
            calls["graph"] = graph
            calls["start_city"] = start_city
            return [start_city]

        subpkg.run_optimal_path = run_optimal_path
        self.calls = calls
        # Load target module after stubbing
        self.mod = load_module()
        self.SemanticModels = self.mod.SemanticModels

    def test_set_and_find_path(self):
        sm = self.SemanticModels()
        graph = [(0, 1), (1, 2)]
        sm.set_graph(graph)
        sm.find_optimal_path(0)
        self.assertEqual(self.calls.get("graph"), graph)
        self.assertEqual(self.calls.get("start_city"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
