"""Shared, deterministic data layer for the cross-framework benchmark.

Uses the canonical MovieLens-100K split (u1.base / u1.test) so every framework
sees byte-identical train/test data. All id spaces are built from TRAIN only;
test rows referencing unseen users/items are dropped (standard practice).
"""
import os
import numpy as np
import pandas as pd

COLS = ["user", "item", "rating", "ts"]
POS_THRESHOLD = 4.0  # rating >= 4 counts as a relevant/positive item

# Default layout when someone dropped GroupLens ml-100k under the old
# cr_learn_setup tree. Override with COREC_ML100K_DIR if the files live elsewhere.
_DEFAULT_ML100K = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "cr_learn_setup", "cr_learn", "CRDS", "ml_100k",
)


def ml100k_dir():
    """Directory that should contain u1.base / u1.test.

    COREC_ML100K_DIR wins when set, so CI can unpack GroupLens anywhere without
    reconstructing the cr_learn_setup path.
    """
    override = os.environ.get("COREC_ML100K_DIR")
    if override:
        return os.path.abspath(override)
    return os.path.abspath(_DEFAULT_ML100K)


def ml100k_available():
    d = ml100k_dir()
    return os.path.isfile(os.path.join(d, "u1.base")) and os.path.isfile(
        os.path.join(d, "u1.test"))


# kept for callers that imported the constant; always points at the default
# layout, not the env override -- prefer ml100k_dir() for new code.
ML100K_DIR = _DEFAULT_ML100K


def _read(path):
    return pd.read_csv(path, sep="\t", names=COLS, engine="python")


def _index_from_train(train, test):
    """Build contiguous ids from train; drop cold test rows + train/test overlap.

    Cold users/items (not in train) get NaN on map and are dropped. Same (user,
    item) pairs that somehow landed in both splits are stripped from test so a
    masked train item can't also count as a hit.
    """
    users = sorted(train["user"].unique())
    items = sorted(train["item"].unique())
    u2i = {u: k for k, u in enumerate(users)}
    i2i = {it: k for k, it in enumerate(items)}

    train = train.copy()
    test = test.copy()
    for df in (train, test):
        df["uidx"] = df["user"].map(u2i)
        df["iidx"] = df["item"].map(i2i)

    # cold-start rows: unseen user or item in test
    test = test.dropna(subset=["uidx", "iidx"]).copy()
    test["uidx"] = test["uidx"].astype(int)
    test["iidx"] = test["iidx"].astype(int)
    train["uidx"] = train["uidx"].astype(int)
    train["iidx"] = train["iidx"].astype(int)

    # drop leakage: (uidx, iidx) already observed in train
    train_pairs = set(zip(train["uidx"].tolist(), train["iidx"].tolist()))
    keep = [
        (u, i) not in train_pairs
        for u, i in zip(test["uidx"].tolist(), test["iidx"].tolist())
    ]
    test = test.loc[keep].reset_index(drop=True)

    return {
        "train": train,
        "test": test,
        "n_users": len(users),
        "n_items": len(items),
        "u2i": u2i,
        "i2i": i2i,
    }


def load_split():
    d = ml100k_dir()
    base = os.path.join(d, "u1.base")
    testp = os.path.join(d, "u1.test")
    if not (os.path.isfile(base) and os.path.isfile(testp)):
        raise FileNotFoundError(
            "MovieLens-100K u1.base/u1.test not found under %s. "
            "Set COREC_ML100K_DIR or unpack GroupLens ml-100k there "
            "(see BENCHMARKS.md)." % d
        )
    train = _read(base)
    test = _read(testp)
    return _index_from_train(train, test)


def train_seen(data):
    """dict uidx -> set(iidx) of items seen in train (to mask at ranking time)."""
    seen = {}
    for u, i in zip(data["train"]["uidx"].values, data["train"]["iidx"].values):
        seen.setdefault(int(u), set()).add(int(i))
    return seen


def test_relevant(data):
    """dict uidx -> set(iidx) of relevant test items (rating >= threshold)."""
    rel = {}
    t = data["test"]
    mask = t["rating"].values >= POS_THRESHOLD
    for u, i in zip(t["uidx"].values[mask], t["iidx"].values[mask]):
        rel.setdefault(int(u), set()).add(int(i))
    return rel


ML1M_RATINGS = os.path.expanduser("~/.cache/crlearn/datasets/ml_1m/ratings.dat")


def load_ml1m(n_interactions=None, seed=42, test_frac=0.2):
    """Load MovieLens-1M from the cr_learn cache, optionally subsample to
    n_interactions, then random per-interaction split. Returns the same dict
    shape as load_split() so the runner adapters work unchanged.

    Used for the scalability sweep (10k -> 1M from a single source dataset).
    """
    df = pd.read_csv(ML1M_RATINGS, sep="::", engine="python",
                     names=["user", "item", "rating", "ts"])
    rng = np.random.RandomState(seed)
    if n_interactions is not None and n_interactions < len(df):
        idx = rng.choice(len(df), size=n_interactions, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    # random per-interaction train/test split
    mask = rng.rand(len(df)) >= test_frac
    train = df[mask].copy()
    test = df[~mask].copy()
    return _index_from_train(train, test)


def load_ml1m_loo(n_interactions=None, seed=42, n_neg=99):
    """Leave-one-out split (He et al. NeuMF protocol): each user's most recent
    interaction is the test positive; the rest are train. Also returns, per test
    user, a candidate list of [positive] + n_neg sampled negatives for ranking.
    """
    df = pd.read_csv(ML1M_RATINGS, sep="::", engine="python",
                     names=["user", "item", "rating", "ts"])
    rng = np.random.RandomState(seed)
    if n_interactions is not None and n_interactions < len(df):
        idx = rng.choice(len(df), size=n_interactions, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    df = df.sort_values(["user", "ts"])
    # last per user -> test, rest -> train; keep users with >=2 interactions
    is_last = df.groupby("user").cumcount(ascending=False) == 0
    counts = df.groupby("user")["item"].transform("size")
    keep = counts >= 2
    train = df[keep & ~is_last].copy()
    test = df[keep & is_last].copy()

    out = _index_from_train(train, test)
    train, test = out["train"], out["test"]

    seen = {}
    for u, i in zip(train["uidx"].values, train["iidx"].values):
        seen.setdefault(int(u), set()).add(int(i))

    n_items = out["n_items"]
    candidates = {}  # uidx -> (pos_iidx, np.array of n_neg negatives)
    for u, pos in zip(test["uidx"].values, test["iidx"].values):
        u = int(u)
        pos = int(pos)
        s = seen.get(u, set())
        negs = []
        # rejection sample; rare to loop much with n_neg=99 on these catalogues
        while len(negs) < n_neg:
            c = int(rng.randint(0, n_items))
            if c != pos and c not in s:
                negs.append(c)
        candidates[u] = (pos, np.array(negs))

    out["seen"] = seen
    out["candidates"] = candidates
    return out


_LIGHTGCN_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_lightgcn_format(name, n_interactions=None, seed=42):
    """Load a dataset in the canonical LightGCN/NGCF adjacency-list format
    (train.txt/test.txt; each line: 'user item item ...'). These are the standard
    large implicit-feedback benchmarks used in graph-recsys papers (Gowalla,
    Yelp2018, Amazon-Book). All interactions are implicit; we tag them rating=5.0
    so the shared relevance threshold (>=4) treats every test item as relevant.
    """
    ddir = os.path.join(_LIGHTGCN_DIR, name)

    def _read_adj(fn):
        rows = []
        with open(os.path.join(ddir, fn)) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                u = int(parts[0])
                for it in parts[1:]:
                    rows.append((u, int(it)))
        return rows

    train_rows = _read_adj("train.txt")
    test_rows = _read_adj("test.txt")

    rng = np.random.RandomState(seed)
    if n_interactions is not None and n_interactions < len(train_rows):
        idx = rng.choice(len(train_rows), size=n_interactions, replace=False)
        train_rows = [train_rows[i] for i in idx]

    train = pd.DataFrame(train_rows, columns=["user", "item"])
    test = pd.DataFrame(test_rows, columns=["user", "item"])
    train["rating"] = 5.0
    test["rating"] = 5.0
    train["ts"] = 0
    test["ts"] = 0
    return _index_from_train(train, test)


def load_dataset(name, n_interactions=None, seed=42):
    if name == "ml100k":
        return load_split()
    if name == "ml1m":
        return load_ml1m(n_interactions=n_interactions, seed=seed)
    if name in ("gowalla", "yelp2018", "amazon-book"):
        return load_lightgcn_format(name, n_interactions=n_interactions, seed=seed)
    raise ValueError(name)


def eval_users(data, n=300, seed=42):
    """Deterministic subset of test users with >=1 relevant item.

    The SAME subset is used for every framework so ranking means are comparable;
    capping keeps the per-item scoring of neural models tractable.
    """
    rel = test_relevant(data)
    cand = sorted(rel.keys())
    rng = np.random.RandomState(seed)
    if n >= len(cand):
        return cand
    return sorted(rng.choice(cand, size=n, replace=False).tolist())
