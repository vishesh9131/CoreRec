"""Fairness check: does CoreRec DCN produce a usable ranking when trained on
BINARY implicit labels (rating>=4 -> 1, else 0) instead of raw 1-5 ratings?
If yes, the quickstart's raw-rating usage is the problem; if no, the model is.
"""
import os
import sys
import time

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import datautil
import metrics as M

data = datautil.load_split()
tr = data["train"]
seen = datautil.train_seen(data)
rel = datautil.test_relevant(data)
eu = datautil.eval_users(data, n=300, seed=42)
n_items = data["n_items"]

from corerec.engines import DCN

labels = (tr["rating"].values >= 4).astype(np.float32)
print(f"binary labels: positives={labels.sum():.0f}/{len(labels)} ({labels.mean():.3f})")

m = DCN(embedding_dim=32, epochs=20, learning_rate=0.001, verbose=False, device="cpu")
t0 = time.perf_counter()
m.fit(user_ids=tr["uidx"].values, item_ids=tr["iidx"].values, ratings=labels)
print(f"fit {time.perf_counter()-t0:.1f}s")

# score variance
s0 = np.array(m.batch_predict([(int(eu[0]), int(i)) for i in range(n_items)]), float)
print(f"user {eu[0]} scores: min={s0.min():.4f} max={s0.max():.4f} std={s0.std():.4f} #unique={len(np.unique(np.round(s0,4)))}")

def score_fn(u):
    return np.array(m.batch_predict([(int(u), int(i)) for i in range(n_items)]), float)

r = M.ranking_metrics(score_fn, n_items, seen, rel, k=10, user_subset=eu)
print("BINARY-label DCN ranking:", {k: round(v,4) if isinstance(v,float) else v for k,v in r.items()})
