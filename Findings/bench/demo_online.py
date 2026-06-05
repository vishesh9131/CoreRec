"""Demonstrate corerec.serving.OnlineRecommender at million-item scale.

Trains native BPR embeddings (GPU), builds a FAISS ANN index, then reports:
  - accuracy (NDCG@10 / Recall@10) of the ONLINE ANN serving path
  - serving latency (p50/p99) and single-thread throughput (QPS)
  - index build time and peak memory
  - freshness: incremental add_items + fold_in_user without retraining

Usage: python demo_online.py --dataset gowalla --dim 64 --epochs 15 --device cuda
"""
import argparse
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import datautil
import metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gowalla")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--index", default="hnsw")
    ap.add_argument("--eval_users", type=int, default=1000)
    args = ap.parse_args()

    d = datautil.load_dataset(args.dataset)
    tr, te = d["train"], d["test"]
    print(f"[{args.dataset}] users={d['n_users']} items={d['n_items']} "
          f"train={len(tr)} test={len(te)}")

    df = pd.DataFrame({"user_id": tr["uidx"].values, "item_id": tr["iidx"].values})

    from corerec.serving import OnlineRecommender

    t0 = time.perf_counter()
    rec = OnlineRecommender.from_interactions(
        df, dim=args.dim, epochs=args.epochs, device=args.device,
        index_type=args.index, metric="cosine", verbose=True)
    build_s = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"build (train embeddings + ANN index): {build_s:.1f}s  peak_mem={peak_mb:.0f} MB")

    # ---- accuracy of the ONLINE serving path ----
    rel = datautil.test_relevant(d)
    eu = datautil.eval_users(d, n=args.eval_users, seed=42)
    ndcgs, recalls, hrs = [], [], []
    for u in eu:
        truth = rel.get(u)
        if not truth:
            continue
        recs = rec.recommend(u, top_k=10)
        hits = np.array([1.0 if i in truth else 0.0 for i in recs])
        nrel = len(truth)
        idcg = M._dcg(np.ones(min(nrel, 10)))
        ndcgs.append(M._dcg(hits) / idcg if idcg > 0 else 0.0)
        recalls.append(hits.sum() / nrel)
        hrs.append(1.0 if hits.sum() > 0 else 0.0)
    print(f"ONLINE accuracy: NDCG@10={np.mean(ndcgs):.4f}  "
          f"Recall@10={np.mean(recalls):.4f}  HitRate@10={np.mean(hrs):.4f}  "
          f"(n={len(ndcgs)})")

    # ---- serving latency + throughput (single thread) ----
    probe = eu[:2000] if len(eu) >= 2000 else (eu * (2000 // len(eu) + 1))[:2000]
    t0 = time.perf_counter()
    for u in probe:
        rec.recommend(u, top_k=10)
    dt = time.perf_counter() - t0
    s = rec.stats()
    print(f"serving: p50={s['latency_ms_p50']:.4f} ms  p99={s['latency_ms_p99']:.4f} ms  "
          f"throughput={len(probe)/dt:,.0f} req/s (1 thread)")

    # ---- ANN vs EXACT: does approximate retrieval preserve accuracy? ----
    from corerec.serving import OnlineRecommender as OR
    exact = OR.from_embeddings(
        item_ids=rec._item_ids, item_emb=rec._item_emb,
        user_ids=list(rec._user_emb), user_emb=np.stack(list(rec._user_emb.values())),
        seen={u: rec._seen.get(u, set()) for u in rec._user_emb},
        index_type="flat", metric="cosine")
    en = []
    for u in eu:
        truth = rel.get(u)
        if not truth:
            continue
        recs = exact.recommend(u, top_k=10)
        hits = np.array([1.0 if i in truth else 0.0 for i in recs])
        idcg = M._dcg(np.ones(min(len(truth), 10)))
        en.append(M._dcg(hits) / idcg if idcg > 0 else 0.0)
    es = exact.stats()
    print(f"ANN vs EXACT: NDCG@10 ann={np.mean(ndcgs):.4f} exact={np.mean(en):.4f} "
          f"(recall of exact = {np.mean(ndcgs)/max(np.mean(en),1e-9):.1%}); "
          f"latency ann p50={s['latency_ms_p50']:.3f}ms exact p50={es['latency_ms_p50']:.3f}ms "
          f"(speedup {es['latency_ms_p50']/max(s['latency_ms_p50'],1e-9):.0f}x)")

    # ---- freshness without retraining ----
    new_emb = np.random.RandomState(7).randn(5, args.dim).astype("float32")
    n_before = rec.stats()["n_items"]
    rec.add_items([f"NEW_{i}" for i in range(5)], new_emb)
    some_user = eu[0]
    rec.fold_in_user("BRAND_NEW_USER",
                     item_ids=list(rec._seen.get(some_user, []))[:10])
    print(f"freshness: add_items {n_before}->{rec.stats()['n_items']} (no retrain); "
          f"fold-in new user -> {rec.recommend('BRAND_NEW_USER', top_k=5)[:5]}")
    print(f"cold-start unknown user -> popularity fallback: "
          f"{rec.recommend('totally_unknown', top_k=5)}")


if __name__ == "__main__":
    main()
