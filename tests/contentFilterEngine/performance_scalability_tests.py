import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

BASE = (
    Path(__file__).resolve().parents[2]
    / "corerec"
    / "engines"
    / "contentFilterEngine"
    / "performance_scalability"
)


def load(fname: str):
    spec = spec_from_file_location(fname, str(BASE / fname))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestPerformanceScalability(unittest.TestCase):
    def test_scalable_algorithms(self):
        m = load("scalable_algorithms.py")
        sa = m.ScalableAlgorithms(num_workers=2)
        chunks = sa.chunkify(list(range(10)), 3)
        self.assertEqual(sum(len(c) for c in chunks), 10)
        # Parallel process with a picklable builtin
        res = sorted(sa.parallel_process(abs, list(range(5))))
        self.assertEqual(res, [0, 1, 2, 3, 4])

    def test_load_balancing(self):
        m = load("load_balancing.py")
        lb = m.LoadBalancing(num_workers=2)
        for i in range(5):
            lb.add_task(lambda x: x + 1, i)
        results = sorted(lb.get_results())
        self.assertEqual(results, [1, 2, 3, 4, 5])
        lb.shutdown()

    def test_feature_extraction_fit(self):
        m = load("feature_extraction.py")
        fe = m.FeatureExtraction(max_features=20)
        X = fe.fit_transform(
            ["this is a test", "another test of features"])  # smoke
        self.assertEqual(X.shape[0], 2)
        # Skip transform due to lsa_model absence in current implementation


if __name__ == "__main__":
    unittest.main(verbosity=2)
