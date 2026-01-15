import unittest
import numpy as np
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

BASE = (
    Path(__file__).resolve().parents[2]
    / "corerec"
    / "engines"
    / "content_based"
    / "miscellaneous_techniques"
)


def load(fname: str):
    spec = spec_from_file_location(fname, str(BASE / fname))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestMiscTechniques(unittest.TestCase):
    def test_feature_selection_variants(self):
        mod = load("feature_selection.py")
        X = np.abs(np.random.randn(50, 8))
        y = (np.random.rand(50) > 0.5).astype(int)
        # chi2
        fs = mod.FEATURE_SELECTION(k=3, method="chi2")
        Xs = fs.fit_transform(X, y)
        self.assertEqual(Xs.shape[1], 3)
        # variance
        fs2 = mod.FEATURE_SELECTION(k=2, method="variance")
        Xs2 = fs2.fit_transform(X, y)
        self.assertEqual(Xs2.shape[1], 2)
        # correlation
        fs3 = mod.FEATURE_SELECTION(k=4, method="correlation")
        Xs3 = fs3.fit_transform(X, y)
        self.assertEqual(Xs3.shape[1], 4)

    def test_noise_handling_methods(self):
        mod = load("noise_handling.py")
        X = np.random.randn(100, 5)
        nh = mod.NOISE_HANDLING(method="isolation_forest", contamination=0.1)
        Xin, mask = nh.fit_transform(X)
        self.assertEqual(mask.shape[0], 100)
        Xt = nh.transform(X)
        self.assertIsInstance(Xt, np.ndarray)
        # zscore
        nh2 = mod.NOISE_HANDLING(method="zscore")
        Xin2, mask2 = nh2.fit_transform(X)
        self.assertEqual(mask2.shape[0], 100)
        # iqr
        nh3 = mod.NOISE_HANDLING(method="iqr")
        Xin3, mask3 = nh3.fit_transform(X)
        self.assertEqual(mask3.shape[0], 100)

    def test_cold_start(self):
        mod = load("cold_start.py")
        item_features = np.random.randn(30, 6)
        inter = (np.random.rand(10, 30) > 0.7).astype(int)
        cs = mod.COLD_START(method="content_based", n_neighbors=5)
        cs.fit(item_features, inter)
        rec_idx = cs.recommend_for_new_user(user_profile=np.random.randn(6))
        self.assertEqual(len(rec_idx), 5)
        # popularity
        cs2 = mod.COLD_START(method="popularity", n_neighbors=3)
        cs2.fit(item_features, inter)
        rec_idx2 = cs2.recommend_for_new_user()
        self.assertEqual(len(rec_idx2), 3)
        # hybrid
        cs3 = mod.COLD_START(method="hybrid", n_neighbors=4)
        cs3.fit(item_features, inter)
        rec_idx3 = cs3.recommend_for_new_user(user_profile=np.random.randn(6))
        self.assertEqual(len(rec_idx3), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
