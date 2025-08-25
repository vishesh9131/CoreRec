import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data"


def _guess_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str], Optional[str]]:
    user_candidates = ["user_id", "user", "uid", "UserId", "userid"]
    item_candidates = ["item_id", "item", "iid", "movie_id", "video_id", "content_id"]
    rating_candidates = ["rating", "score", "rating_value", "y"]
    ts_candidates = ["timestamp", "ts", "time", "datetime"]

    def pick(cols: List[str], df_cols: List[str]) -> Optional[str]:
        low = {c.lower(): c for c in df_cols}
        for c in cols:
            if c in low:
                return low[c]
        return None

    user_col = pick(user_candidates, list(df.columns)) or df.columns[0]
    item_col = pick(item_candidates, list(df.columns)) or df.columns[1]
    rating_col = pick(rating_candidates, list(df.columns))
    ts_col = pick(ts_candidates, list(df.columns))
    return user_col, item_col, rating_col, ts_col


def _load_crlearn_ijcai() -> Optional[Dict[str, List]]:
    """
    Try loading interactions from cr_learn.ijcai.load().
    Returns dict with keys: users, items, ratings, timestamps
    """
    try:
        from cr_learn.ijcai import load as cr_load  # type: ignore
    except Exception:
        return None

    try:
        data = cr_load()
    except Exception:
        return None

    users: List = []
    items: List = []
    ratings: List[float] = []
    timestamps: List[int] = []

    # Preferred source: explicit interactions mapping
    if isinstance(data, dict) and "user_merchant_interaction" in data:
        um = data["user_merchant_interaction"]
        # um: Dict[user_id, List[merchant_id]]
        t = 0
        for u, merch_list in um.items():
            for m in merch_list:
                users.append(str(u))
                items.append(str(m))
                ratings.append(1.0)
                timestamps.append(t)
                t += 1
        if users:
            return {"users": users, "items": items, "ratings": ratings, "timestamps": timestamps}

    # Fallback: merchant_train dataframe with columns user_id, merchant_id
    for key in ("merchant_train", "train", "interactions", "df"):
        if key in data and hasattr(data[key], "__class__") and hasattr(data[key], "columns"):
            df = data[key]
            cols = [c.lower() for c in df.columns]
            if "user_id" in cols and "merchant_id" in cols:
                user_col = df.columns[cols.index("user_id")]
                item_col = df.columns[cols.index("merchant_id")]
                users = df[user_col].astype(str).tolist()
                items = df[item_col].astype(str).tolist()
                ratings = [1.0] * len(users)
                timestamps = list(range(len(users)))
                return {"users": users, "items": items, "ratings": ratings, "timestamps": timestamps}

    return None


def load_interactions(prefer: str = "crlearn") -> Dict[str, List]:
    """
    Load a dataset for examples.
    Preferences:
    - "crlearn": use cr_learn.ijcai if available
    - "netflix"|"youtube"|"spotify": use sample_data CSVs
    Returns dict with users, items, ratings, timestamps
    """
    if prefer == "crlearn":
        payload = _load_crlearn_ijcai()
        if payload is not None:
            return payload
        # if crlearn unavailable, fall back to netflix
        prefer = "netflix"

    candidates: List[Path] = []
    if prefer == "netflix":
        candidates = [SAMPLE / "netflix_demo.csv", SAMPLE / "youtube_demo.csv", SAMPLE / "spotify_demo.csv"]
    elif prefer == "youtube":
        candidates = [SAMPLE / "youtube_demo.csv", SAMPLE / "netflix_demo.csv", SAMPLE / "spotify_demo.csv"]
    else:
        candidates = [SAMPLE / "spotify_demo.csv", SAMPLE / "netflix_demo.csv", SAMPLE / "youtube_demo.csv"]

    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        # fallback: synthesize small dataset
        users = [1, 1, 2, 2, 3]
        items = [10, 20, 20, 30, 40]
        ratings = [1.0] * len(users)
        ts = list(range(len(users)))
        return {"users": users, "items": items, "ratings": ratings, "timestamps": ts}

    df = pd.read_csv(path)
    user_col, item_col, rating_col, ts_col = _guess_columns(df)

    users = df[user_col].astype(str).tolist()
    items = df[item_col].astype(str).tolist()
    if rating_col and rating_col in df:
        # robust conversion; if non-numeric (e.g., age ratings like 'NC-17'), fallback to 1.0
        try:
            ratings = pd.to_numeric(df[rating_col], errors="coerce").fillna(1.0).astype(float).tolist()
        except Exception:
            ratings = [1.0] * len(df)
    else:
        ratings = [1.0] * len(df)

    if ts_col and ts_col in df:
        ts = pd.to_datetime(df[ts_col], errors="coerce").astype("int64", errors="ignore").tolist()
        # if parse failed to numeric, fallback to positional ordering
        if any(isinstance(t, str) for t in ts):
            ts = list(range(len(df)))
    else:
        ts = list(range(len(df)))

    return {"users": users, "items": items, "ratings": ratings, "timestamps": ts}


def build_csr_from_interactions(users: List, items: List, ratings: Optional[List] = None) -> Tuple[csr_matrix, List, List]:
    """
    Build a csr matrix and return (matrix, unique_users, unique_items) with original IDs.
    """
    if ratings is None:
        ratings = [1.0] * len(users)
    uvals = list(dict.fromkeys(users))
    ivals = list(dict.fromkeys(items))
    uidx = {u: i for i, u in enumerate(uvals)}
    iidx = {it: i for i, it in enumerate(ivals)}
    rows = [uidx[u] for u in users]
    cols = [iidx[i] for i in items]
    data = np.array(ratings, dtype=float)
    mat = csr_matrix((data, (rows, cols)), shape=(len(uvals), len(ivals)))
    return mat, uvals, ivals 