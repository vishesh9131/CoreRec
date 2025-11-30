#!/usr/bin/env python3
# quickstart for core deep engines with tiny data

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from importlib import import_module


def try_import(module_path, cls_name):
    try:
        mod = import_module(module_path)
        return getattr(mod, cls_name)
    except Exception as e:
        print(f"Skip {cls_name}: {e}")
        return None


def run_dcn():
    DCN = try_import("corerec.engines.dcn", "DCN")
    if not DCN:
        return
    model = DCN(
        embedding_dim=8, num_cross_layers=1, deep_layers=[8], epochs=1, batch_size=8, device="cpu"
    )
    users = [1, 2, 1, 3]
    items = [10, 10, 20, 30]
    ratings = [1, 0, 1, 0]
    try:
        model.fit(users, items, ratings)
        print("DCN recs:", model.recommend(1, top_n=3, exclude_seen=False))
    except Exception as e:
        print("DCN run skipped:", e)


def run_deepfm():
    DeepFM = try_import("corerec.engines.deepfm", "DeepFM")
    if not DeepFM:
        return
    model = DeepFM(embedding_dim=8, hidden_layers=[8], epochs=1, batch_size=8, device="cpu")
    users = [1, 2, 1, 3]
    items = [10, 10, 20, 30]
    ratings = [1, 0, 1, 0]
    try:
        model.fit(users, items, ratings)
        print("DeepFM recs:", model.recommend(1, top_n=3, exclude_seen=False))
    except Exception as e:
        print("DeepFM run skipped:", e)


def run_gnnrec():
    GNNRec = try_import("corerec.engines.gnnrec", "GNNRec")
    if not GNNRec:
        return
    model = GNNRec(embedding_dim=16, num_gnn_layers=1, epochs=1, batch_size=4, device="cpu")
    users = [1, 2, 1, 3]
    items = [10, 10, 20, 30]
    ratings = [1, 1, 1, 0]
    model.fit(users, items, ratings)
    print("GNNRec recs:", model.recommend(1, top_n=3, exclude_seen=False))


def run_mind():
    MIND = try_import("corerec.engines.mind", "MIND")
    if not MIND:
        return
    model = MIND(
        embedding_dim=16, num_interests=2, epochs=1, batch_size=8, max_seq_length=5, device="cpu"
    )
    users = [1, 1, 1, 2]
    items = [10, 20, 30, 10]
    ts = [1, 2, 3, 1]
    model.fit(users, items, ts)
    print("MIND recs:", model.recommend(1, top_n=3))


def run_nasrec():
    NASRec = try_import("corerec.engines.nasrec", "NASRec")
    if not NASRec:
        return
    model = NASRec(embedding_dim=16, hidden_dims=[16], epochs=1, batch_size=8, device="cpu")
    users = [1, 2, 1, 3]
    items = [10, 10, 20, 30]
    ratings = [1, 0, 1, 0]
    model.fit(users, items, ratings)
    print("NASRec recs:", model.recommend(1, top_n=3, exclude_seen=False))


def run_sasrec():
    SASRec = try_import("corerec.engines.sasrec", "SASRec")
    if not SASRec:
        return
    from scipy.sparse import csr_matrix

    user_ids = [1, 2]
    item_ids = [10, 20, 30]
    dense = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
    inter = csr_matrix(dense)
    model = SASRec(
        hidden_units=16,
        num_blocks=1,
        num_heads=1,
        num_epochs=1,
        batch_size=4,
        max_seq_length=5,
        device="cpu",
    )
    model.fit(inter, user_ids, item_ids)
    print("SASRec recs:", model.recommend(1, top_n=3))


if __name__ == "__main__":
    run_dcn()
    run_deepfm()
    run_gnnrec()
    run_mind()
    run_nasrec()
    run_sasrec()
