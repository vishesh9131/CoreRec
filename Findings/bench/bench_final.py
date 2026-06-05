"""Final head-to-head: CoreRec native LightGCN/BPR vs all competitors at the
standard cutoffs (NDCG@10/@20, Recall@10/@20), full-ranking, on ML-1M, Gowalla
and Yelp2018. CoreRec embeddings are scored by EXACT full ranking (U @ V^T) so the
accuracy comparison is apples-to-apples with the competitors (the ANN-serving
approximation is measured separately in the online-serving section).

Usage: python bench_final.py --dataset gowalla --device cuda
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import datautil
import metrics as M
import runner as R


def corerec_native(model_name, data, device, epochs):
    from corerec.serving.online import _train_lightgcn_embeddings, _train_bpr_embeddings
    tr = data["train"]
    u = tr["uidx"].to_numpy(); i = tr["iidx"].to_numpy()
    nu, ni = data["n_users"], data["n_items"]
    t0 = time.perf_counter()
    if model_name == "LightGCN":
        U, V = _train_lightgcn_embeddings(u, i, nu, ni, dim=64, epochs=epochs, device=device)
    else:
        U, V = _train_bpr_embeddings(u, i, nu, ni, dim=64, epochs=epochs, device=device)
    fit_t = time.perf_counter() - t0
    Vt = V.T.copy()

    def score_fn(uidx):
        return U[uidx] @ Vt
    return score_fn, fit_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--eval_users", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = datautil.load_dataset(args.dataset)
    seen = datautil.train_seen(data)
    rel = datautil.test_relevant(data)
    eu = datautil.eval_users(data, n=args.eval_users, seed=42)
    n_items = data["n_items"]
    print(f"[{args.dataset}] users={data['n_users']} items={n_items} "
          f"train={len(data['train'])} eval_users={len(eu)}")

    specs = [
        ("implicit", "ALS"), ("implicit", "BPR"), ("implicit", "ItemKNN"),
        ("lightfm", "WARP"), ("cornac", "BPR"), ("cornac", "MF"),
        ("corerec_native", "BPR"), ("corerec_native", "LightGCN"),
    ]
    rows = []
    for fw, md in specs:
        try:
            if fw == "corerec_native":
                R.DEVICE = args.device
                score_fn, fit_t = corerec_native(md, data, args.device, args.epochs)
            else:
                R.DEVICE = "cpu"
                score_fn, _, fit_t = R.DISPATCH[fw](md, data)
            m10 = M.ranking_metrics(score_fn, n_items, seen, rel, k=10, user_subset=eu)
            m20 = M.ranking_metrics(score_fn, n_items, seen, rel, k=20, user_subset=eu)
            row = {"framework": fw, "model": md, "fit_s": round(fit_t, 1),
                   "NDCG@10": m10["NDCG@10"], "Recall@10": m10["Recall@10"],
                   "NDCG@20": m20["NDCG@20"], "Recall@20": m20["Recall@20"]}
        except Exception as e:
            row = {"framework": fw, "model": md, "error": str(e)[:100]}
        rows.append(row)
        print(f"  {fw:16} {md:10} " + (
            f"N@10={row.get('NDCG@10',0):.4f} N@20={row.get('NDCG@20',0):.4f} "
            f"R@20={row.get('Recall@20',0):.4f} fit={row.get('fit_s','-')}s"
            if "error" not in row else f"FAILED {row['error']}"))

    rows.sort(key=lambda r: -(r.get("NDCG@20") or -1))
    print(f"\n=== {args.dataset}: ranked by NDCG@20 ===")
    for r in rows:
        if "error" in r:
            continue
        star = " <-- CoreRec" if r["framework"] == "corerec_native" else ""
        print(f"  {r['framework']+' '+r['model']:26} NDCG@20={r['NDCG@20']:.4f} "
              f"Recall@20={r['Recall@20']:.4f} NDCG@10={r['NDCG@10']:.4f}{star}")

    out = args.out or os.path.join(os.path.dirname(__file__), "results",
                                   f"final_{args.dataset}.json")
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset, "rows": rows}, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
