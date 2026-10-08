"""Where does each model stop working as data grows?

One model x one size per subprocess, so a crash or OOM in one doesn't take the
rest down. Reports fit seconds, peak RSS and recommend() latency.

    python Findings/scale/probe.py --sizes S M --models ALS EASE

Heavy: size M and up wants a workstation or cloud box, not a laptop.
"""
import argparse, json, os, resource, subprocess, sys, time, warnings

SIZES = {  # users, items, interactions
    "S": (10_000, 5_000, 200_000),
    "M": (100_000, 20_000, 2_000_000),
    "L": (1_000_000, 100_000, 10_000_000),
}
KW = {  # one epoch: we want per-epoch cost, not accuracy
    "TwoTower": dict(embedding_dim=32, epochs=1, verbose=False),
    "SASRec": dict(hidden_units=32, num_blocks=1, epochs=1, max_seq_length=50, verbose=False),
    "HSTU": dict(embedding_dim=32, epochs=1),
    "DCN": dict(embedding_dim=16, epochs=1), "DeepFM": dict(embedding_dim=16, epochs=1),
    "LightGCN": dict(epochs=1), "MultVAE": dict(epochs=1), "MultiDAE": dict(epochs=1),
    "ALS": dict(epochs=5), "Item2Vec": dict(epochs=1),
}


def data(size, seed=0):
    import numpy as np
    n_u, n_i, n = SIZES[size]
    rng = np.random.default_rng(seed)
    users = rng.integers(0, n_u, n)
    items = np.minimum(rng.zipf(1.3, n) - 1, n_i - 1)  # long-tail popularity
    ts = rng.integers(0, 1_000_000, n)
    import pandas as pd
    df = pd.DataFrame({"user_id": users, "item_id": items, "rating": 1.0, "timestamp": ts})
    return df.drop_duplicates(["user_id", "item_id"])


def one(model, size):
    warnings.filterwarnings("ignore")
    import numpy as np
    import corerec.engines as E
    df = data(size)
    m = getattr(E, model)(**KW.get(model, {}))
    t = time.perf_counter()
    m.fit(df)
    fit_s = time.perf_counter() - t
    us = df["user_id"].drop_duplicates().sample(100, random_state=0).tolist()
    t = time.perf_counter()
    for u in us:
        m.recommend(u, top_k=10)
    rec_ms = (time.perf_counter() - t) * 10  # per call
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss / 1e9 if sys.platform == "darwin" else rss / 1e6
    print(json.dumps(dict(model=model, size=size, rows=len(df), fit_s=round(fit_s, 1),
                          rec_ms=round(rec_ms, 1), peak_gb=round(rss_gb, 2))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=2)
    ap.add_argument("--sizes", nargs="+", default=["S", "M"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    if a.one:
        one(*a.one); sys.exit()
    import corerec.engines as E
    models = a.models or [m for m in E.MODELS if m != "TFIDFRecommender"]
    for size in a.sizes:
        for model in models:
            env = dict(os.environ, OMP_NUM_THREADS="8")
            cmd = [sys.executable, __file__, "--one", model, size]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout, env=env)
                line = [l for l in p.stdout.splitlines() if l.startswith("{")]
                if line:
                    print(line[-1], flush=True)
                else:
                    err = (p.stderr.strip().splitlines() or ["killed"])[-1][:160]
                    print(json.dumps(dict(model=model, size=size, error=err)), flush=True)
            except subprocess.TimeoutExpired:
                print(json.dumps(dict(model=model, size=size, error=f"timeout {a.timeout}s")), flush=True)
