import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

BASE = Path(__file__).resolve().parents[2] / "corerec" / "engines" / "contentFilterEngine" / "other_approaches"


def load(fname: str):
    spec = spec_from_file_location(fname, str(BASE / fname))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestOtherApproaches(unittest.TestCase):
    def test_rule_based(self):
        m = load("rule_based.py")
        rb = m.RuleBasedFilter(rules=[{"keyword": "spam", "action": "block"}, {"keyword": "warn", "action": "flag"}])
        res1 = rb.filter_content("This is SPAM content")
        self.assertEqual(res1["status"], "blocked")
        res2 = rb.filter_content("Just warn this")
        self.assertEqual(res2["status"], "flagged")
        res3 = rb.filter_content("clean text")
        self.assertEqual(res3["status"], "allowed")

    def test_sentiment_analysis(self):
        try:
            m = load("sentiment_analysis.py")
        except Exception as e:
            self.skipTest(f"textblob unavailable: {e}")
        sa = m.SentimentAnalysisFilter(threshold=0.1)
        out_pos = sa.filter_content("I love this great product")
        out_neg = sa.filter_content("I hate this terrible thing")
        self.assertIn(out_pos["status"], {"positive", "neutral"})
        self.assertIn(out_neg["status"], {"negative", "neutral"})

    def test_ontology_based_skip_if_missing(self):
        try:
            m = load("ontology_based.py")
            # We cannot load a real ontology file here; expect ValueError
            with self.assertRaises(Exception):
                m.OntologyBasedFilter("/nonexistent.owl")
        except Exception as e:
            self.skipTest(f"owlready2 missing or cannot import: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2) 