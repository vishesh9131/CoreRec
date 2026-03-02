import unittest
import sys
from types import ModuleType
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = REPO_ROOT / "corerec" / "engines" / \
    "content_based" / "graph_based_algorithms"


def ensure_fake_module(name: str):
    mod = ModuleType(name)
    sys.modules[name] = mod
    return mod


def load_module():
    spec = spec_from_file_location(
        "graph_filtering", str(
            BASE_DIR / "graph_filtering.py"))
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestGraphFiltering(unittest.TestCase):
    def setUp(self):
        # Stub corerec.vish_graphs functions
        ensure_fake_module("corerec")
        vg = ensure_fake_module("corerec.vish_graphs")

        def gen_graph(n, path, seed=None):
            return {"n": n, "path": path, "seed": seed}

        def gen_large(n, path, seed=None):
            return {"large": True, "n": n, "path": path, "seed": seed}

        def scale_save(inp, out_dir, k):
            self.scale_called = (inp, out_dir, k)

        vg.generate_random_graph = gen_graph
        vg.generate_large_random_graph = gen_large
        vg.scale_and_save_matrices = scale_save
        self.mod = load_module()
        self.GraphFiltering = self.mod.GraphFiltering

    def test_generate_and_scale(self):
        gf = self.GraphFiltering()
        g = gf.generate_graph(5, file_path="a.csv", seed=7)
        self.assertEqual(g["n"], 5)
        g2 = gf.generate_large_graph(10, file_path="b.csv", seed=3)
        self.assertTrue(g2["large"])
        gf.scale_and_save("in.csv", "out/", 4)
        self.assertEqual(self.scale_called, ("in.csv", "out/", 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
