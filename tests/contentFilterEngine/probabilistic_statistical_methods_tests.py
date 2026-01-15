import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

BASE = (
    Path(__file__).resolve().parents[2]
    / "corerec"
    / "engines"
    / "content_based"
    / "probabilistic_statistical_methods"
)


def load(fname: str):
    spec = spec_from_file_location(fname, str(BASE / fname))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestProbabilisticStatisticalMethods(unittest.TestCase):
    def test_lsa_fit_transform(self):
        m = load("lsa.py")
        lsa = m.LSA(n_components=2)
        docs = ["apple banana", "banana orange", "cat dog", "dog mouse"]
        lsa.fit(docs)
        X = lsa.transform(["apple cat"])
        self.assertEqual(X.shape[1], 2)

    def test_lda_fit_transform(self):
        m = load("lda.py")
        lda = m.LDA(n_components=2, max_iter=2)
        docs = ["apple banana", "banana orange", "cat dog", "dog mouse"]
        lda.fit(docs)
        X = lda.transform(["apple cat"])
        self.assertEqual(X.shape[1], 2)

    def test_bayesian_fit_predict(self):
        m = load("bayesian.py")
        bay = m.BAYESIAN()
        docs = ["spam ham", "spam eggs", "dog cat", "cat mouse"]
        labels = [1, 1, 0, 0]
        bay.fit(docs, labels)
        pred = bay.predict("spam")
        self.assertIn(pred, [0, 1])

    def test_fuzzy_logic_evaluate(self):
        m = load("fuzzy_logic.py")

        # Simple fuzzification and defuzzification
        def fuzz_x(v):
            return {"low": 1.0 if v < 5 else 0.0}

        def defuzz_out(fs):
            return 0.0 if fs.get("low", 0) > 0.5 else 1.0

        def rule(fs):
            return {"out": fs["x"]["low"]}

        fl = m.FUZZY_LOGIC(
            input_vars={
                "x": fuzz_x}, output_vars={
                "out": defuzz_out}, rules=[rule])
        out = fl.evaluate({"x": 3})
        self.assertIn("out", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
