"""Diagnostic: is CoreRec DCN's per-item scoring meaningful for ranking, and what
scale does predict() output? Confirms whether the weak ranking/RMSE numbers are
real model behaviour vs a harness artifact.
"""
import os
import sys

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import datautil

data = datautil.load_split()
tr = data["train"]
from corerec.engines import DCN

m = DCN(embedding_dim=32, epochs=20, learning_rate=0.001, verbose=False, device="cpu")
m.fit(user_ids=tr["uidx"].values, item_ids=tr["iidx"].values,
      ratings=tr["rating"].astype(float).values)

n_items = data["n_items"]
seen = datautil.train_seen(data)
rel = datautil.test_relevant(data)

# pick 3 users that have relevant test items
users = [u for u in sorted(rel.keys())][:3]
print("predict() sample (true rating scale is 1-5):")
for u in users[:1]:
    for i in list(rel[u])[:5]:
        print(f"  u={u} i={i} predict={m.predict(user_id=int(u), item_id=int(i)):.4f}")

print("\nper-item score distribution via batch_predict (per user):")
for u in users:
    scores = np.array(m.batch_predict([(int(u), int(i)) for i in range(n_items)]), dtype=float)
    print(f"  u={u}: min={scores.min():.4f} max={scores.max():.4f} "
          f"mean={scores.mean():.4f} std={scores.std():.4f} "
          f"#unique={len(np.unique(np.round(scores,4)))}")
    # native recommend vs our argsort
    masked = scores.copy()
    for i in seen.get(u, ()):
        masked[i] = -np.inf
    top10_scores = set(np.argsort(-masked)[:10].tolist())
    try:
        native = m.recommend(user_id=int(u), top_k=10)
        native = set(int(x[0]) if isinstance(x,(tuple,list)) else int(x) for x in native)
        print(f"        native recommend ∩ our-top10 = {len(top10_scores & native)}/10")
    except Exception as e:
        print("        recommend err", e)
