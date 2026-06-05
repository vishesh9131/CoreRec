"""Run ONE RecBole model on the SAME ML-1M split CoreRec uses (exported to
RecBole atomic files), reporting standard Recall/NDCG/Hit@10. Lets us ask the
same-model question: does CoreRec's DCN/DeepFM/SASRec/... match RecBole's
reference implementation of the same model?

NumPy-2.0 compat shim applied first (RecBole 1.2.1 references removed aliases).

Usage: python recbole_runner.py --model DCN --size 100000 --out results/same/recbole_DCN.json
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
for _a, _b in [("bool8", "bool_"), ("float_", "float64"), ("int_", "int64"),
               ("unicode_", "str_"), ("complex_", "complex128")]:
    if not hasattr(np, _a):
        setattr(np, _a, getattr(np, _b))

# scipy >=1.14 removed dok_matrix._update, which RecBole 1.2.1 graph models call.
import scipy.sparse as _sp
if not hasattr(_sp.dok_matrix, "_update"):
    _sp.dok_matrix._update = lambda self, data: dict.update(self, data)

sys.path.insert(0, os.path.dirname(__file__))
import datautil

K = 10
EPOCHS = 20

# RecBole model_type: general (CF), context-aware (CTR), sequential
SEQUENTIAL = {"SASRec", "BERT4Rec", "GRU4Rec", "Caser", "NARM"}
CONTEXT = {"DCN", "DeepFM", "xDeepFM", "AutoInt", "WideDeep", "DCNV2", "FM"}


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def write_atomic(data, name, root):
    ddir = os.path.join(root, name)
    os.makedirs(ddir, exist_ok=True)
    header = "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"
    tr = data["train"]
    # carve a small validation slice from train for early stopping
    n = len(tr)
    rng = np.random.RandomState(0)
    vmask = rng.rand(n) < 0.1
    parts = {"train": tr[~vmask], "valid": tr[vmask], "test": data["test"]}
    for split, df in parts.items():
        with open(os.path.join(ddir, f"{name}.{split}.inter"), "w") as f:
            f.write(header)
            for u, i, r, t in zip(df["uidx"], df["iidx"], df["rating"], df["ts"]):
                f.write(f"{int(u)}\t{int(i)}\t{float(r)}\t{int(t)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=100000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = datautil.load_dataset("ml1m", n_interactions=args.size)

    import tempfile
    root = tempfile.mkdtemp(prefix="rb_")
    name = "mlsplit"
    write_atomic(data, name, root)

    # PyTorch 2.6+ flipped torch.load(weights_only) to True; RecBole 1.2.1 saves
    # full checkpoints, so restore the old default for its internal reloads.
    import torch
    _orig_load = torch.load
    def _patched_load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)
    torch.load = _patched_load

    from recbole.quick_start import run_recbole

    mtype = ("sequential" if args.model in SEQUENTIAL
             else "context" if args.model in CONTEXT else "general")
    cfg = {
        "data_path": root,
        "benchmark_filename": ["train", "valid", "test"],
        "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": "rating", "TIME_FIELD": "timestamp",
        "topk": [K], "metrics": ["Recall", "NDCG", "Hit"],
        "valid_metric": f"NDCG@{K}", "epochs": EPOCHS,
        "train_batch_size": 2048, "eval_batch_size": 4096,
        "device": args.device, "use_gpu": args.device == "cuda",
        "show_progress": False, "verbose": False, "state": "ERROR",
        "checkpoint_dir": os.path.join(root, "ckpt"),
    }
    if mtype == "context":
        # CTR models need labels; threshold rating>=4 -> positive
        cfg["threshold"] = {"rating": 4}
        cfg["eval_args"] = {"split": {"RS": [8, 1, 1]}, "order": "RO",
                            "mode": "labeled", "group_by": "user"}
    if mtype == "sequential":
        cfg["MAX_ITEM_LIST_LENGTH"] = 50
        cfg["eval_args"] = {"split": {"RS": [8, 1, 1]}, "order": "TO",
                            "mode": "full", "group_by": "user"}

    t0 = time.perf_counter()
    res = run_recbole(model=args.model, dataset=name, config_dict=cfg)
    elapsed = time.perf_counter() - t0

    tr = res.get("test_result", {}) or {}
    def g(*names):
        for n in names:
            if n in tr:
                return float(tr[n])
        return None

    out = {
        "framework": "recbole",
        "model": args.model,
        "dataset": "ml1m",
        "size": args.size,
        "n_users": data["n_users"], "n_items": data["n_items"],
        "fit_time_s": round(elapsed, 4),          # end-to-end (train+eval)
        "recommend_latency_ms_per_user": None,
        "peak_mem_mb": round(peak_mem_mb(), 1),
        f"Recall@{K}": g(f"recall@{K}"),
        f"NDCG@{K}": g(f"ndcg@{K}"),
        f"HitRate@{K}": g(f"hit@{K}"),
        "device": args.device,
        "RMSE": None, "MAE": None,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
