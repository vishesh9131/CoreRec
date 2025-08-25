#!/usr/bin/env python3
# quickstart for unionizedFilterEngine algorithms

import os
import sys
import numpy as np
from scipy.sparse import csr_matrix

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def build_csr(users, items, ratings):
    uid_map = {u: i for i, u in enumerate(sorted(set(users)))}
    iid_map = {it: i for i, it in enumerate(sorted(set(items)))}
    rows = [uid_map[u] for u in users]
    cols = [iid_map[i] for i in items]
    data = np.array(ratings, dtype=float)
    mat = csr_matrix((data, (rows, cols)), shape=(len(uid_map), len(iid_map)))
    return mat, uid_map, iid_map


def run_fast():
    try:
        from corerec.engines.unionizedFilterEngine.fast import FAST
    except Exception as e:
        print("FAST not available:", e)
        return
    users = [1, 1, 2, 2, 3]
    items = [10, 20, 20, 30, 40]
    ratings = [1, 1, 1, 1, 1]
    model = FAST(factors=8, iterations=1, batch_size=8, seed=42)
    model.fit(users, items, ratings)
    print("FAST recs:", model.recommend(1, top_n=3))


def run_fast_recommender():
    try:
        from corerec.engines.unionizedFilterEngine.fast_recommender import FASTRecommender
    except Exception as e:
        print("FASTRecommender not available:", e)
        return
    users = [1, 1, 2, 2, 3]
    items = [10, 20, 20, 30, 40]
    ratings = [1, 1, 1, 1, 1]
    model = FASTRecommender(factors=8, iterations=1, batch_size=8, seed=42)
    model.fit(users, items, ratings)
    print("FASTRecommender recs:", model.recommend(1, top_n=3))


def run_sar():
    try:
        from corerec.engines.unionizedFilterEngine.sar import SAR
    except Exception as e:
        print("SAR not available:", e)
        return
    users = [1, 1, 2, 2, 3]
    items = [10, 20, 20, 30, 40]
    ratings = [1, 1, 1, 1, 1]
    model = SAR()
    model.fit(users, items, ratings)
    print("SAR recs:", model.recommend(1, top_n=3))


def run_rlrmc():
    try:
        from corerec.engines.unionizedFilterEngine.rlrmc import RLRMC
    except Exception as e:
        print("RLRMC not available:", e)
        return
    users = [1, 1, 2, 2, 3]
    items = [10, 20, 20, 30, 40]
    ratings = [1, 1, 1, 1, 1]
    mat, uid_map, iid_map = build_csr(users, items, ratings)
    rank = min(2, len(uid_map), len(iid_map))
    model = RLRMC(rank=rank, max_iter=1, verbose=False)
    model.fit(mat, sorted(uid_map.keys()), sorted(iid_map.keys()))
    print("RLRMC recs:", model.recommend(1, top_n=3))


def run_rbm():
    try:
        from corerec.engines.unionizedFilterEngine.rbm import RBM
    except Exception as e:
        print("RBM not available:", e)
        return
    users = [1, 1, 2, 2, 3]
    items = [10, 20, 20, 30, 40]
    ratings = [1, 1, 1, 1, 1]
    mat, uid_map, iid_map = build_csr(users, items, ratings)
    model = RBM(n_hidden=8, n_epochs=1, batch_size=4, verbose=False)
    try:
        model.fit(mat, sorted(uid_map.keys()), sorted(iid_map.keys()))
        print("RBM recs:", model.recommend(1, top_n=3))
    except Exception as e:
        print("RBM run skipped:", e)


def run_geomlc():
    try:
        from corerec.engines.unionizedFilterEngine.geomlc import GeoMLC
    except Exception as e:
        print("GeoMLC not available:", e)
        return
    users = [1, 1, 2, 3]
    items = [10, 20, 10, 30]
    ratings = [1.0, 1.0, 1.0, 1.0]
    model = GeoMLC(n_factors=4, n_epochs=1, batch_size=2, device='cpu')
    model.fit(users, items, ratings)
    print("GeoMLC recs:", model.recommend(1, top_n=2))


if __name__ == "__main__":
    run_fast()
    run_fast_recommender()
    run_sar()
    run_rlrmc()
    run_rbm()
    run_geomlc() 