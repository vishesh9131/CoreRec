"""Generative recommendation benchmark: HSTU against SASRec on MovieLens-1M.

Protocol (the one the HSTU paper's public ML-1M numbers use):
  - data: all 1,000,209 ML-1M ratings with timestamps; each user's latest
    rating is the test target (the NCF authors' leave-latest-out split,
    downloaded once from their repository);
  - every rating counts as an interaction; history is ordered by timestamp;
  - full ranking over all 3,706 items, items already in the history removed;
  - HR@K and NDCG@K for K = 10 and 50, over all 6,040 users;
  - no early stopping on the test target: every run trains a fixed number
    of epochs and is scored once at the end.

Models:
  hstu           corerec.engines.HSTU (encoder="hstu")
  sasrec-ssm     corerec.engines.HSTU(encoder="sasrec"): the SASRec block under
                 the identical sampled-softmax recipe -- isolates the architecture
  sasrec-legacy  corerec.engines.SASRec, CoreRec's existing implementation
                 (binary cross-entropy, one negative) -- what users got before

Usage:
  python generative_bench.py --model hstu --epochs 100 --seed 1 --out results/generative/hstu_s1.json
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DATA_URL = "https://raw.githubusercontent.com/hexiangnan/neural_collaborative_filtering/master/Data/"
CACHE = os.path.expanduser(os.environ.get("CORERec_ML1M_DIR", "~/.cache/corerec/ml-1m-ncf"))


def load_ml1m():
    os.makedirs(CACHE, exist_ok=True)
    frames = {}
    for split in ("train", "test"):
        path = os.path.join(CACHE, f"ml-1m.{split}.rating")
        if not os.path.exists(path):
            urllib.request.urlretrieve(DATA_URL + f"ml-1m.{split}.rating", path)
        frames[split] = np.loadtxt(path, dtype=np.int64, delimiter="\t")  # user, item, rating, ts
    return frames["train"], frames["test"]


def evaluate(model, users, targets, ks=(10, 50)):
    """Full-ranking HR@K / NDCG@K; history removed. Ties count against the target."""
    hits = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    for s in range(0, len(users), 512):
        batch = users[s:s + 512]
        scores = model.score_users(batch)
        for row, u in enumerate(batch):
            sc = scores[row]
            sc[model.user_sequences[u] - 1] = -np.inf
            t = model.item_to_index[targets[u]] - 1
            rank = int((sc > sc[t]).sum() + (sc == sc[t]).sum() - 1)  # 0-based, pessimistic on ties
            for k in ks:
                if rank < k:
                    hits[k] += 1
                    ndcg[k] += 1.0 / np.log2(rank + 2)
    n = len(users)
    out = {}
    for k in ks:
        out[f"HR@{k}"] = round(hits[k] / n, 5)
        out[f"NDCG@{k}"] = round(ndcg[k] / n, 5)
    return out


class LegacySASRecAdapter:
    """Gives corerec.engines.SASRec the score_users/user_sequences surface used above."""

    def __init__(self, model):
        import torch
        self.m, self.torch = model, torch
        self.item_to_index = {it: idx for it, idx in model.item_to_index.items()}
        self.user_sequences = {u: np.asarray(s, dtype=np.int64) for u, s in model.user_sequences.items()}

    def score_users(self, users):
        torch, m = self.torch, self.m
        n = m.max_seq_length
        X = np.zeros((len(users), n), dtype=np.int64)
        for r, u in enumerate(users):
            seq = m.user_sequences[u][-n:]
            X[r, n - len(seq):] = seq  # SASRec left-pads
        m.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(X, device=m.device)
            h = m.model(x, x == 0)[:, -1, :]
            scores = m._score_items(h)[:, 1:]
        return scores.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["hstu", "sasrec-ssm", "sasrec-legacy"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=0,
                    help="also score every N epochs, to record the learning curve (the headline is the final epoch)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)

    train, test = load_ml1m()
    users = train[:, 0].tolist()
    items = train[:, 1].tolist()
    times = train[:, 3].astype(np.float64)
    targets = {int(u): int(i) for u, i in zip(test[:, 0], test[:, 1])}

    curve, eval_s = [], [0.0]
    t0 = time.time()
    if args.model in ("hstu", "sasrec-ssm"):
        from corerec.engines import HSTU
        model = HSTU(encoder="hstu" if args.model == "hstu" else "sasrec", epochs=args.epochs,
                     max_seq_length=args.max_len, seed=args.seed, device="cpu")
        def checkpoint(epoch, m):
            if args.eval_every and epoch % args.eval_every == 0 and epoch < args.epochs:
                te = time.time()
                ev = [u for u in sorted(targets) if u in m.user_sequences and targets[u] in m.item_to_index]
                curve.append({"epoch": epoch, **evaluate(m, ev, targets)})
                eval_s[0] += time.time() - te
                print("curve", curve[-1], flush=True)
        model.fit(users, items, timestamps=times, on_epoch_end=checkpoint)
        scorer = model
    else:
        from corerec.engines import SASRec
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # Same size and history length as the HSTU runs; SASRec's own loss and sampler.
        order = np.argsort(times, kind="stable")
        uit = {}
        for r in order:
            uit.setdefault(users[r], []).append((items[r], times[r]))
        model = SASRec(hidden_units=50, num_blocks=2, num_heads=1, dropout_rate=0.2,
                       max_seq_length=args.max_len, epochs=args.epochs, batch_size=128,
                       learning_rate=1e-3, device=torch.device("cpu"), verbose=False)
        model.fit(user_ids=users, item_ids=items, ratings=[1.0] * len(users), user_item_timestamps=uit)
        scorer = LegacySASRecAdapter(model)
    fit_s = time.time() - t0 - eval_s[0]

    eval_users = [u for u in sorted(targets) if u in scorer.user_sequences and targets[u] in scorer.item_to_index]
    t1 = time.time()
    metrics = evaluate(scorer, eval_users, targets)
    out = {
        "model": args.model, "dataset": "ml-1m", "protocol": "leave-latest-out, full ranking, history removed",
        "epochs": args.epochs, "seed": args.seed, "max_len": args.max_len,
        "n_eval_users": len(eval_users), "fit_time_s": round(fit_s, 1), "eval_time_s": round(time.time() - t1, 1),
        **metrics,
    }
    if curve:
        out["curve"] = curve
    if hasattr(model, "history") and isinstance(model.history, list) and model.history:
        out["final_train_loss"] = round(float(model.history[-1]), 4)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
