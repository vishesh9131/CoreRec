"""
Comprehensive test suite for ALL production recommender models.

Every model exposed by ``corerec.engines`` must pass:
  1. Import and instantiation
  2. fit() with synthetic data
  3. predict(user_id, item_id) returns a float
  4. recommend(user_id, top_k) returns a list
  5. save() / load() round-trip
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from corerec.api.base_recommender import BaseRecommender


def assert_predict_parity(testcase, model, loaded, user_id, item_id, delta=1e-2):
    """Assert predict() score is unchanged after save/load (catches JSON ID map bugs)."""
    before = float(model.predict(user_id, item_id))
    after = float(loaded.predict(user_id, item_id))
    if abs(before) < 1e-9 and abs(after) < 1e-9:
        return
    testcase.assertAlmostEqual(
        before, after, delta=delta, msg=f"predict score drift: {before} -> {after}"
    )


# ---------------------------------------------------------------------------
# Shared synthetic data generators
# ---------------------------------------------------------------------------

def _triplet_data(n_users=20, n_items=15, n_interactions=100, seed=42):
    """Return (user_ids, item_ids, ratings) lists for triplet-based models."""
    rng = np.random.RandomState(seed)
    user_ids = rng.randint(0, n_users, n_interactions).tolist()
    item_ids = rng.randint(0, n_items, n_interactions).tolist()
    ratings = rng.choice([0.0, 1.0], n_interactions).tolist()
    return user_ids, item_ids, ratings


def _matrix_data(n_users=20, n_items=15, density=0.3, seed=42):
    """Return (user_ids, item_ids, interaction_matrix) for matrix-based models."""
    rng = np.random.RandomState(seed)
    mat = (rng.rand(n_users, n_items) < density).astype(np.float32)
    user_id_list = list(range(n_users))
    item_id_list = list(range(n_items))
    return user_id_list, item_id_list, mat


# ===================================================================
# 1. Deep-learning top-level models
# ===================================================================

class TestDCN(unittest.TestCase):

    def setUp(self):
        self.user_ids, self.item_ids, self.ratings = _triplet_data()

    def test_import_and_init(self):
        from corerec.engines.dcn import DCN
        model = DCN(epochs=1, verbose=False, batch_size=64)
        self.assertIsInstance(model, BaseRecommender)

    def test_fit_predict_recommend(self):
        from corerec.engines.dcn import DCN
        model = DCN(epochs=1, verbose=False, batch_size=64)
        result = model.fit(user_ids=self.user_ids, item_ids=self.item_ids,
                           ratings=self.ratings)
        self.assertIs(result, model)
        self.assertTrue(model.is_fitted)

        score = model.predict(self.user_ids[0], self.item_ids[0])
        self.assertIsInstance(score, float)

        recs = model.recommend(user_id=self.user_ids[0], top_k=5)
        self.assertIsInstance(recs, list)
        self.assertLessEqual(len(recs), 5)

    def test_save_load(self):
        from corerec.engines.dcn import DCN
        model = DCN(epochs=1, verbose=False, batch_size=64)
        model.fit(user_ids=self.user_ids, item_ids=self.item_ids,
                  ratings=self.ratings)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dcn.pkl")
            model.save(path)
            loaded = DCN.load(path)
            self.assertTrue(loaded.is_fitted)
            assert_predict_parity(self, model, loaded, self.user_ids[0], self.item_ids[0])
            recs = loaded.recommend(self.user_ids[0], top_k=3)
            self.assertIsInstance(recs, list)


class TestDeepFM(unittest.TestCase):

    def setUp(self):
        self.user_ids, self.item_ids, self.ratings = _triplet_data()

    def test_import_and_init(self):
        from corerec.engines.deepfm import DeepFM
        model = DeepFM(epochs=1, verbose=False, batch_size=64)
        self.assertIsInstance(model, BaseRecommender)

    def test_fit_predict_recommend(self):
        from corerec.engines.deepfm import DeepFM
        model = DeepFM(epochs=1, verbose=False, batch_size=64)
        model.fit(user_ids=self.user_ids, item_ids=self.item_ids,
                  ratings=self.ratings)
        self.assertTrue(model.is_fitted)

        score = model.predict(self.user_ids[0], self.item_ids[0])
        self.assertIsInstance(score, float)

        recs = model.recommend(user_id=self.user_ids[0], top_k=5)
        self.assertIsInstance(recs, list)

    def test_save_load(self):
        from corerec.engines.deepfm import DeepFM
        model = DeepFM(epochs=1, verbose=False, batch_size=64)
        model.fit(user_ids=self.user_ids, item_ids=self.item_ids,
                  ratings=self.ratings)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "deepfm.pkl")
            model.save(path)
            loaded = DeepFM.load(path)
            self.assertTrue(loaded.is_fitted)
            assert_predict_parity(self, model, loaded, self.user_ids[0], self.item_ids[0])


class TestSASRec(unittest.TestCase):

    def setUp(self):
        self.user_list, self.item_list, self.mat = _matrix_data()

    def test_import_and_init(self):
        from corerec.engines.sasrec import SASRec
        model = SASRec(epochs=1, verbose=False, batch_size=64,
                       max_seq_length=10, hidden_units=16, num_blocks=1)
        self.assertIsInstance(model, BaseRecommender)

    def test_fit_predict_recommend(self):
        from corerec.engines.sasrec import SASRec
        model = SASRec(epochs=1, verbose=False, batch_size=64,
                       max_seq_length=10, hidden_units=16, num_blocks=1)
        model.fit(self.user_list, self.item_list, self.mat)
        self.assertTrue(model.is_fitted)

        uid = next(iter(model.user_sequences))
        iid = next(iter(model.item_to_index))
        score = model.predict(uid, iid)
        self.assertIsInstance(score, float)

        recs = model.recommend(uid, top_k=5)
        self.assertIsInstance(recs, list)

    def test_save_load(self):
        from corerec.engines.sasrec import SASRec
        model = SASRec(epochs=1, verbose=False, batch_size=64,
                       max_seq_length=10, hidden_units=16, num_blocks=1)
        model.fit(self.user_list, self.item_list, self.mat)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sasrec.pkl")
            model.save(path)
            loaded = SASRec.load(path)
            self.assertTrue(loaded.is_fitted)
            uid = next(iter(model.user_sequences))
            iid = next(iter(model.item_to_index))
            assert_predict_parity(self, model, loaded, uid, iid)


class TestTwoTower(unittest.TestCase):

    def setUp(self):
        self.user_list, self.item_list, self.mat = _matrix_data()

    def test_import_and_init(self):
        from corerec.engines.two_tower import TwoTower
        model = TwoTower(epochs=1, verbose=False, batch_size=64,
                         embedding_dim=16, hidden_dims=[32, 16])
        self.assertIsInstance(model, BaseRecommender)

    def test_fit_predict_recommend(self):
        from corerec.engines.two_tower import TwoTower
        model = TwoTower(epochs=1, verbose=False, batch_size=64,
                         embedding_dim=16, hidden_dims=[32, 16])
        model.fit(self.user_list, self.item_list, self.mat)
        self.assertTrue(model.is_fitted)

        uid = self.user_list[0]
        iid = self.item_list[0]
        score = model.predict(uid, iid)
        self.assertIsInstance(score, float)

        recs = model.recommend(uid, top_k=5)
        self.assertIsInstance(recs, list)
        self.assertLessEqual(len(recs), 5)

    def test_save_load(self):
        from corerec.engines.two_tower import TwoTower
        model = TwoTower(epochs=1, verbose=False, batch_size=64,
                         embedding_dim=16, hidden_dims=[32, 16])
        model.fit(self.user_list, self.item_list, self.mat)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "twotower.pt")
            model.save(path)
            loaded = TwoTower.load(path)
            self.assertTrue(loaded.is_fitted)
            assert_predict_parity(self, model, loaded, self.user_list[0], self.item_list[0])
            recs = loaded.recommend(self.user_list[0], top_k=3)
            self.assertIsInstance(recs, list)


class TestSAR(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(42)
        rows = []
        for u in range(20):
            for i in rng.choice(15, size=rng.randint(3, 8), replace=False):
                rows.append({"userID": u, "itemID": int(i),
                             "rating": float(rng.randint(1, 6))})
        self.df = pd.DataFrame(rows)

    def test_import_and_init(self):
        from corerec.engines.collaborative.sar import SAR
        model = SAR(similarity_type="jaccard")
        self.assertIsInstance(model, BaseRecommender)

    def test_fit_recommend(self):
        from corerec.engines.collaborative.sar import SAR
        model = SAR(similarity_type="jaccard")
        model.fit(self.df)
        self.assertTrue(model.is_fitted)

        uid = self.df["userID"].iloc[0]
        recs = model.recommend(uid, top_k=5)
        self.assertIsInstance(recs, list)

    def test_predict(self):
        from corerec.engines.collaborative.sar import SAR
        model = SAR(similarity_type="jaccard")
        model.fit(self.df)

        uid = self.df["userID"].iloc[0]
        iid = self.df["itemID"].iloc[0]
        score = model.predict(uid, iid)
        self.assertIsInstance(score, float)

    def test_save_load(self):
        from corerec.engines.collaborative.sar import SAR
        model = SAR(similarity_type="jaccard")
        model.fit(self.df)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sar.pkl")
            model.save(path)
            loaded = SAR.load(path)
            self.assertTrue(loaded.is_fitted)
            uid = int(self.df["userID"].iloc[0])
            iid = int(self.df["itemID"].iloc[0])
            assert_predict_parity(self, model, loaded, uid, iid)


class TestLightGCN(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(42)
        n = 100
        self.user_ids = rng.randint(0, 20, n).tolist()
        self.item_ids = rng.randint(0, 15, n).tolist()
        self.ratings = [1.0] * n

    def test_import_and_init(self):
        from corerec.engines.collaborative.graph_based_base.lightgcn import LightGCN
        model = LightGCN(n_factors=16, n_layers=2, epochs=2, verbose=False,
                         device="cpu")
        self.assertIsInstance(model, BaseRecommender)

    def test_fit_predict_recommend(self):
        from corerec.engines.collaborative.graph_based_base.lightgcn import LightGCN
        model = LightGCN(n_factors=16, n_layers=2, epochs=2, verbose=False,
                         device="cpu", early_stopping_patience=5)
        model.fit(self.user_ids, self.item_ids, self.ratings)
        self.assertTrue(model.is_fitted)

        uid = self.user_ids[0]
        iid = self.item_ids[0]
        score = model.predict(uid, iid)
        self.assertIsInstance(score, float)

        recs = model.recommend(uid, top_k=5)
        self.assertIsInstance(recs, list)
        self.assertLessEqual(len(recs), 5)

    def test_save_load(self):
        from corerec.engines.collaborative.graph_based_base.lightgcn import LightGCN
        model = LightGCN(n_factors=16, n_layers=2, epochs=2, verbose=False,
                         device="cpu", early_stopping_patience=5)
        model.fit(self.user_ids, self.item_ids, self.ratings)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lightgcn.pkl")
            model.save(path)
            loaded = LightGCN.load(path)
            self.assertTrue(loaded.is_fitted)
            assert_predict_parity(self, model, loaded, self.user_ids[0], self.item_ids[0])
            recs = loaded.recommend(self.user_ids[0], top_k=3)
            self.assertIsInstance(recs, list)


# ===================================================================
# 3. Content-based models
# ===================================================================

class TestTFIDFRecommender(unittest.TestCase):

    def setUp(self):
        self.items = list(range(20))
        self.docs = {
            i: f"item {i} description with keywords topic{i % 5} category{i % 3}"
            for i in self.items
        }

    def test_import_and_init(self):
        from corerec.engines.content_based.tfidf_recommender import TFIDFRecommender
        model = TFIDFRecommender()
        self.assertIsInstance(model, BaseRecommender)

    def test_fit_predict_recommend(self):
        from corerec.engines.content_based.tfidf_recommender import TFIDFRecommender
        model = TFIDFRecommender()
        model.fit(self.items, self.docs)
        self.assertTrue(model.is_fitted)

        score = model.predict(self.items[0], self.items[1])
        self.assertIsInstance(score, float)

        recs = model.recommend(self.items[0], top_k=5)
        self.assertIsInstance(recs, list)
        self.assertLessEqual(len(recs), 5)

    def test_save_load(self):
        from corerec.engines.content_based.tfidf_recommender import TFIDFRecommender
        model = TFIDFRecommender()
        model.fit(self.items, self.docs)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tfidf.pkl")
            model.save(path)
            loaded = TFIDFRecommender.load(path)
            self.assertTrue(loaded.is_fitted)
            assert_predict_parity(self, model, loaded, self.items[0], self.items[1])


# ===================================================================
# 4. Meta-test: all models are BaseRecommender subclasses
# ===================================================================

# Inheritance and required-method checks for every registered model live in
# tests/test_deep_zoo_api_parity.py, driven by corerec.engines.MODELS.


if __name__ == "__main__":
    unittest.main(verbosity=2)
