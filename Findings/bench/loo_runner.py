"""Second standard protocol: leave-one-out with 99 sampled negatives (He et al.
NeuMF), reporting HR@10 and NDCG@10. Reuses the same framework adapters as
runner.py, so the only thing that changes is the evaluation protocol.

Usage: python loo_runner.py --framework cornac --model BPR --size 100000 \
           --out results/loo/cornac_BPR.json
"""
import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import datautil
import metrics as M
import runner as R

K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=100000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval_users", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    R.DEVICE = args.device
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    data = datautil.load_ml1m_loo(n_interactions=args.size)
    cand = data["candidates"]
    users = sorted(cand.keys())
    if args.eval_users < len(users):
        import numpy as np
        users = sorted(np.random.RandomState(42).choice(users, args.eval_users, replace=False).tolist())

    score_fn, _, fit_t = R.DISPATCH[args.framework](args.model, data)
    res = M.loo_metrics(score_fn, cand, k=K, user_subset=users)

    out = {
        "framework": args.framework, "model": args.model, "protocol": "LOO+99neg",
        "dataset": "ml1m", "size": args.size,
        "n_users": data["n_users"], "n_items": data["n_items"],
        "fit_time_s": round(fit_t, 4),
        "peak_mem_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        f"HR@{K}": round(res[f"HR@{K}"], 5),
        f"NDCG@{K}": round(res[f"NDCG@{K}"], 5),
        "n_eval_users": res["n_eval_users"],
        "device": args.device,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
