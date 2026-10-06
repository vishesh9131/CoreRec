"""From an interactions file to a live recommendation API.

This is what ``corerec train`` and ``corerec serve`` run:

    corerec serve events.csv                 # train, report, serve on :8000
    corerec train events.csv -o artifacts/m  # train, report, save
    corerec serve artifacts/m                # serve a saved artifact

The same steps are available from Python::

    from corerec.serving.from_csv import train_from_csv, save_artifact, build_server

    result = train_from_csv("events.csv", model="ALS")
    print(result.report())
    save_artifact(result, "artifacts/m")
    build_server(result.model, result.manifest()).start()

The file needs one row per interaction with a user column and an item column.
A rating/weight column and a timestamp column are used when present. Column
names are detected from common spellings (``user_id``, ``userId``, ``customer``,
``item_id``, ``movie_id``, ``product_id``, ``rating``, ``timestamp`` ...) or set
explicitly.
"""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

ARTIFACT_VERSION = 1
MANIFEST = "manifest.json"
MODEL_FILE = "model"
DEFAULT_MODEL = "ALS"

# Normalized (lowercase, alphanumerics only) spellings, in priority order.
_CANDIDATES = {
    "user": ["userid", "user", "uid", "customerid", "customer", "visitorid", "visitor",
             "memberid", "member", "accountid", "account", "clientid", "sessionid"],
    "item": ["itemid", "item", "iid", "productid", "product", "movieid", "movie",
             "songid", "song", "trackid", "track", "videoid", "video", "articleid",
             "article", "contentid", "content", "bookid", "book", "sku", "asin"],
    "rating": ["rating", "score", "weight", "value", "count", "playcount", "plays",
               "clicks", "quantity", "qty", "stars", "implicit"],
    "timestamp": ["timestamp", "ts", "time", "datetime", "date", "eventtime",
                  "createdat", "unixtime", "epoch"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def detect_columns(columns: Sequence[str], **explicit: Optional[str]) -> Dict[str, Optional[str]]:
    """Map the roles user/item/rating/timestamp onto *columns*.

    Explicit names (``user=...``, ``item=...``) win. User and item are
    required; rating and timestamp are optional and come back as ``None``.
    """
    by_norm = {}
    for c in columns:
        by_norm.setdefault(_norm(c), c)
    found: Dict[str, Optional[str]] = {}
    taken = set()
    for role in ("user", "item", "rating", "timestamp"):
        name = explicit.get(role)
        if name:
            if name not in columns:
                raise ValueError(f"{role} column {name!r} is not in the file; columns are {list(columns)}")
            found[role] = name
        else:
            found[role] = next(
                (by_norm[c] for c in _CANDIDATES[role] if c in by_norm and by_norm[c] not in taken),
                None,
            )
        if found[role]:
            taken.add(found[role])
    missing = [r for r in ("user", "item") if not found[r]]
    if missing:
        flags = " ".join(f"--{r}-col NAME" for r in missing)
        raise ValueError(
            f"Could not find the {' and '.join(missing)} column in {list(columns)}. "
            f"Name it explicitly with {flags}."
        )
    return found


def read_interactions(
    path: Union[str, Path],
    user_col: Optional[str] = None,
    item_col: Optional[str] = None,
    rating_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
):
    """Read a CSV/TSV/Parquet file into a frame with columns user, item, rating[, timestamp].

    Duplicate (user, item) rows are merged: weights are summed (so repeated
    plays or clicks count up), and the latest timestamp is kept.

    Returns ``(frame, columns)`` where *columns* maps each role to the source
    column name it came from.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        raw = pd.read_parquet(path)
    else:
        sep = "\t" if suffix in (".tsv", ".tab") else None
        raw = pd.read_csv(path, sep=sep, engine="python")
    cols = detect_columns(list(raw.columns), user=user_col, item=item_col,
                          rating=rating_col, timestamp=timestamp_col)

    df = pd.DataFrame({"user": raw[cols["user"]], "item": raw[cols["item"]]})
    if cols["rating"]:
        df["rating"] = pd.to_numeric(raw[cols["rating"]], errors="coerce")
    else:
        df["rating"] = 1.0
    if cols["timestamp"]:
        ts = raw[cols["timestamp"]]
        if not pd.api.types.is_numeric_dtype(ts):
            # Works at any datetime resolution (pandas 3 may infer seconds or us).
            ts = (pd.to_datetime(ts, errors="coerce") - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
        df["timestamp"] = pd.to_numeric(ts, errors="coerce")
    df = df.dropna(subset=["user", "item", "rating"])
    if df.empty:
        raise ValueError(f"{path} has no usable rows after dropping missing values")

    agg = {"rating": "sum"}
    if "timestamp" in df:
        agg["timestamp"] = "max"
    df = df.groupby(["user", "item"], as_index=False, sort=False).agg(agg)
    return df.reset_index(drop=True), cols


def holdout_split(df: pd.DataFrame, test_fraction: float = 0.2, seed: int = 42):
    """Hold out a share of each user's interactions for evaluation.

    With timestamps, each user's most recent interactions are held out (the
    model must predict the future, not fill gaps in the past). Without them,
    a seeded random share is. Users with a single interaction stay in train.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    if "timestamp" in df:
        order = df.sort_values(["user", "timestamp"], kind="mergesort")
    else:
        order = df.iloc[rng.permutation(len(df))].sort_values("user", kind="mergesort")
    position = order.groupby("user").cumcount(ascending=False)  # 0 = last
    size = order.groupby("user")["item"].transform("size")
    n_test = np.floor(size * test_fraction).astype(int).clip(lower=(size > 1).astype(int))
    is_test = position < n_test
    return order[~is_test].reset_index(drop=True), order[is_test].reset_index(drop=True)


def popular_items(df: pd.DataFrame, n: int = 100) -> List[Any]:
    """Items ordered by how many users interacted with them."""
    return df["item"].value_counts().index[:n].tolist()


class PopularityBaseline:
    """Recommends the most popular items to everyone. The bar a model must clear."""

    def __init__(self, items: List[Any]):
        self.items = items

    def recommend(self, user_id: Any, top_k: int = 10, **_: Any) -> List[Any]:
        return self.items[:top_k]


def _model_class(name: str):
    import corerec.engines as engines

    if name not in engines.MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose one of: {', '.join(engines.list_models())}")
    if engines.MODELS[name][1] == "content":
        raise ValueError(f"{name} recommends from item text, not interactions; it cannot train on this file")
    return getattr(engines, name)


def fit_model(name: str, df: pd.DataFrame, params: Optional[Dict[str, Any]] = None):
    """Construct model *name* with *params* and fit it on a user/item/rating frame."""
    model = _model_class(name)(**(params or {}))
    users, items, ratings = df["user"].tolist(), df["item"].tolist(), df["rating"].astype(float).tolist()
    if name == "SAR":
        model.fit_from_lists(users, items, ratings)
    else:
        model.fit(user_ids=users, item_ids=items, ratings=ratings)
    return model


@dataclass
class TrainResult:
    model: Any
    model_name: str
    params: Dict[str, Any]
    columns: Dict[str, Optional[str]]
    stats: Dict[str, int]
    popular: List[Any]
    fit_seconds: float
    k: int = 10
    metrics: Optional[Dict[str, float]] = None
    baseline: Optional[Dict[str, float]] = None
    source: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> Dict[str, Any]:
        return {
            "artifact_version": ARTIFACT_VERSION,
            "model": self.model_name,
            "params": self.params,
            "columns": self.columns,
            "stats": self.stats,
            "k": self.k,
            "metrics": self.metrics,
            "baseline": self.baseline,
            "fit_seconds": round(self.fit_seconds, 3),
            "source": self.source,
            "popular_items": [_json_id(i) for i in self.popular],
        }

    def report(self) -> str:
        s = self.stats
        cols = ", ".join(f"{role}={name}" for role, name in self.columns.items() if name)
        lines = [
            f"Data      {s['rows']:,} interactions, {s['users']:,} users, {s['items']:,} items ({cols})",
            f"Model     {self.model_name} {self.params or ''}".rstrip(),
            f"Trained   in {self.fit_seconds:.1f}s",
        ]
        if self.metrics is not None:
            key = f"NDCG@{self.k}"
            rec = f"Recall@{self.k}"
            m, b = self.metrics, self.baseline
            lines += [
                f"Holdout   {s['test_rows']:,} interactions from {m['n_users']:,} users",
                f"          {'':14}{key:>10}{rec:>11}",
                f"          {self.model_name:<14}{m[key]:>10.4f}{m[rec]:>11.4f}",
                f"          {'most popular':<14}{b[key]:>10.4f}{b[rec]:>11.4f}",
            ]
            if m[key] <= b[key]:
                lines.append("          Warning: the model does not beat recommending the most popular "
                             "items. Try another --model or more data.")
        return "\n".join(lines)


def _json_id(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def train_from_csv(
    path: Union[str, Path],
    model: str = DEFAULT_MODEL,
    params: Optional[Dict[str, Any]] = None,
    evaluate: bool = True,
    test_fraction: float = 0.2,
    k: int = 10,
    seed: int = 42,
    refit: bool = True,
    **columns: Optional[str],
) -> TrainResult:
    """Read *path*, measure *model* on a holdout, then train it on everything.

    The holdout score is compared with recommending the most popular items,
    which is the baseline any model has to beat to be worth serving. With
    ``refit=True`` (the default) the served model is retrained on all rows, so
    no data is thrown away; the reported metrics come from the holdout run.
    """
    from corerec.evaluation.evaluate import evaluate as run_eval

    _model_class(model)  # fail on a bad name before reading a large file
    params = dict(params or {})
    df, cols = read_interactions(path, **{f"{r}_col": v for r, v in columns.items()})
    stats = {"rows": len(df), "users": int(df["user"].nunique()), "items": int(df["item"].nunique()),
             "test_rows": 0}

    metrics = baseline = None
    if evaluate:
        train, test = holdout_split(df, test_fraction=test_fraction, seed=seed)
        stats["test_rows"] = len(test)
        if test.empty:
            raise ValueError("Too little data to evaluate: every user has a single interaction. "
                             "Pass --no-eval to train without a holdout.")
        start = time.perf_counter()
        held_out = fit_model(model, train, params)
        fit_seconds = time.perf_counter() - start
        eval_args = dict(test_interactions=test, train_interactions=train, k=k,
                         user_col="user", item_col="item", rating_col="rating")
        metrics = run_eval(held_out, **eval_args)
        # Popularity from the train split only, so the baseline does not peek at the test set.
        baseline = run_eval(PopularityBaseline(popular_items(train, n=k + 1000)), **eval_args)

    if refit or not evaluate:
        start = time.perf_counter()
        final = fit_model(model, df, params)
        fit_seconds = time.perf_counter() - start
    else:
        final = held_out

    return TrainResult(model=final, model_name=model, params=params, columns=cols, stats=stats,
                       popular=popular_items(df), fit_seconds=fit_seconds, k=k,
                       metrics=metrics, baseline=baseline, source=str(path))


def save_artifact(result: TrainResult, out_dir: Union[str, Path]) -> Path:
    """Write the model and a manifest.json describing it to *out_dir*."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.model.save(str(out / MODEL_FILE))
    (out / MANIFEST).write_text(json.dumps(result.manifest(), indent=2, default=str))
    return out


def is_artifact(path: Union[str, Path]) -> bool:
    return (Path(path) / MANIFEST).is_file()


def load_artifact(path: Union[str, Path]):
    """Load ``(model, manifest)`` from a directory written by :func:`save_artifact`.

    Some classic models persist with pickle, so only load artifacts you made
    or trust.
    """
    path = Path(path)
    manifest = json.loads((path / MANIFEST).read_text())
    model = _model_class(manifest["model"]).load(str(path / MODEL_FILE))
    return model, manifest


def build_server(model, manifest: Optional[Dict[str, Any]] = None, host: str = "0.0.0.0", port: int = 8000):
    """A :class:`~corerec.serving.ModelServer` that answers unknown users with popular items."""
    from corerec.serving.model_server import ModelServer

    manifest = manifest or {}
    return ModelServer(model, host=host, port=port, metadata=manifest,
                       fallback_items=manifest.get("popular_items"))


def parse_params(pairs: Sequence[str]) -> Dict[str, Any]:
    """Turn ``["factors=64", "reg=0.1", "name=x"]`` into a kwargs dict."""
    params: Dict[str, Any] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(f"--param expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            params[key.strip()] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            params[key.strip()] = raw
    return params
