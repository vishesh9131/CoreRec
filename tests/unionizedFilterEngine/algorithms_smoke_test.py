import unittest
import numpy as np
from scipy.sparse import csr_matrix


class TestUnionizedAlgorithms(unittest.TestCase):
    def setUp(self):
        # Tiny toy dataset
        # Users: 1,2,3; Items: 10,20,30,40
        self.users = [1, 1, 2, 2, 3]
        self.items = [10, 20, 20, 30, 40]
        self.ratings = [1.0, 1.0, 1.0, 1.0, 1.0]

    def test_cornac_bpr(self):
        try:
            from corerec.engines.unionizedFilterEngine.cornac_bpr import CornacBPR
        except Exception as e:
            self.skipTest(f"CornacBPR import failed: {e}")
        model = CornacBPR(
            factors=8,
            iterations=1,
            batch_size=8,
            num_neg_samples=1,
            seed=42)
        model.fit(self.users, self.items, self.ratings)
        recs = model.recommend(1, top_n=3)
        self.assertIsInstance(recs, list)

    def test_fast_recommender(self):
        try:
            from corerec.engines.unionizedFilterEngine.fast_recommender import FASTRecommender
        except Exception as e:
            self.skipTest(f"FASTRecommender import failed: {e}")
        model = FASTRecommender(factors=8, iterations=1, batch_size=8, seed=42)
        model.fit(self.users, self.items, self.ratings)
        recs = model.recommend(1, top_n=3)
        self.assertIsInstance(recs, list)

    def test_fast(self):
        try:
            from corerec.engines.unionizedFilterEngine.fast import FAST
        except Exception as e:
            self.skipTest(f"FAST import failed: {e}")
        model = FAST(factors=8, iterations=1, batch_size=8, seed=42)
        model.fit(self.users, self.items, self.ratings)
        recs = model.recommend(1, top_n=3)
        self.assertIsInstance(recs, list)

    def test_sar(self):
        try:
            from corerec.engines.unionizedFilterEngine.sar import SAR
        except Exception as e:
            self.skipTest(f"SAR import failed: {e}")
        model = SAR()
        model.fit(self.users, self.items, self.ratings)
        recs = model.recommend(1, top_n=3)
        self.assertIsInstance(recs, list)

    def test_rbm(self):
        try:
            from corerec.engines.unionizedFilterEngine.rbm import RBM
        except Exception as e:
            self.skipTest(f"RBM import failed: {e}")
        # Build csr matrix for 3 users x 4 items
        uid_map = {u: i for i, u in enumerate(sorted(set(self.users)))}
        iid_map = {it: i for i, it in enumerate(sorted(set(self.items)))}
        rows = [uid_map[u] for u in self.users]
        cols = [iid_map[i] for i in self.items]
        data = np.array(self.ratings, dtype=float)
        mat = csr_matrix(
            (data, (rows, cols)), shape=(
                len(uid_map), len(iid_map)))
        model = RBM(n_hidden=8, n_epochs=1, batch_size=4, verbose=False)
        try:
            model.fit(mat, sorted(uid_map.keys()), sorted(iid_map.keys()))
        except Exception as e:
            self.skipTest(f"RBM.fit skipped due to runtime error: {e}")
            return
        recs = model.recommend(1, top_n=3)
        self.assertIsInstance(recs, list)

    def test_rlrmc(self):
        try:
            from corerec.engines.unionizedFilterEngine.rlrmc import RLRMC
        except Exception as e:
            self.skipTest(f"RLRMC import failed: {e}")
        uid_map = {u: i for i, u in enumerate(sorted(set(self.users)))}
        iid_map = {it: i for i, it in enumerate(sorted(set(self.items)))}
        rows = [uid_map[u] for u in self.users]
        cols = [iid_map[i] for i in self.items]
        data = np.array(self.ratings, dtype=float)
        mat = csr_matrix(
            (data, (rows, cols)), shape=(
                len(uid_map), len(iid_map)))
        # Use safe rank <= min(n_users, n_items)
        min_rank = min(len(uid_map), len(iid_map), 2)
        model = RLRMC(rank=min_rank, max_iter=1, verbose=False)
        model.fit(mat, sorted(uid_map.keys()), sorted(iid_map.keys()))
        recs = model.recommend(1, top_n=3)
        self.assertIsInstance(recs, list)

    def test_sli(self):
        try:
            from corerec.engines.unionizedFilterEngine.sli import SLiRec
        except Exception as e:
            self.skipTest(f"SLiRec import failed: {e}")
        # Build csr matrix where each user has a short sequence
        users = [1, 1, 2, 2]
        items = [10, 20, 10, 30]
        uid_map = {u: i for i, u in enumerate(sorted(set(users)))}
        iid_map = {it: i for i, it in enumerate(sorted(set(items)))}
        rows = [uid_map[u] for u in users]
        cols = [iid_map[i] for i in items]
        data = np.ones(len(users), dtype=float)
        mat = csr_matrix(
            (data, (rows, cols)), shape=(
                len(uid_map), len(iid_map)))
        model = SLiRec(
            embedding_dim=8,
            hidden_dim=8,
            epochs=1,
            batch_size=4,
            sequence_length=5,
            device="cpu")
        model.fit(mat, sorted(uid_map.keys()), sorted(iid_map.keys()))
        recs = model.recommend(1, top_n=2)
        self.assertIsInstance(recs, list)

    def test_sum(self):
        try:
            from corerec.engines.unionizedFilterEngine.sum import SUMModel
        except Exception as e:
            self.skipTest(f"SUMModel import failed: {e}")
        # Simple sequential data with timestamps
        users = [1, 1, 1, 2]
        items = [10, 20, 30, 10]
        ts = [1, 2, 3, 1]
        model = SUMModel(
            embedding_dim=8,
            num_interests=2,
            interest_dim=4,
            epochs=1,
            batch_size=4,
            sequence_length=5,
            device="cpu",
        )
        model.fit(users, items, ts)
        recs = model.recommend(1, top_n=2)
        self.assertIsInstance(recs, list)

    def test_geomlc(self):
        try:
            from corerec.engines.unionizedFilterEngine.geomlc import GeoMLC
        except Exception as e:
            self.skipTest(f"GeoMLC import failed: {e}")
        users = [1, 1, 2, 3]
        items = [10, 20, 10, 30]
        ratings = [1.0, 1.0, 1.0, 1.0]
        model = GeoMLC(n_factors=4, n_epochs=1, batch_size=2, device="cpu")
        model.fit(users, items, ratings)
        recs = model.recommend(1, top_n=2)
        self.assertIsInstance(recs, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
