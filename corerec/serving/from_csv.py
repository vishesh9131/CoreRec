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
import inspect
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
        return model
    extra = {}
    # Sequential models read the order of events; hand them the clock when the file has one.
    if "timestamp" in df and "timestamps" in inspect.signature(model.fit).parameters:
        extra["timestamps"] = df["timestamp"].astype(float).tolist()
    model.fit(user_ids=users, item_ids=items, ratings=ratings, **extra)
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
            # newest event the model learned from; corerec retrain treats later rows as new
            "trained_through": self.extra.get("trained_through"),
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

    extra = {"trained_through": float(df["timestamp"].max())} if "timestamp" in df else {}
    return TrainResult(model=final, model_name=model, params=params, columns=cols, stats=stats,
                       popular=popular_items(df), fit_seconds=fit_seconds, k=k,
                       metrics=metrics, baseline=baseline, source=str(path), extra=extra)


class _Served:
    """A model as the server answers: popular items when it can't (unknown user, empty)."""

    def __init__(self, model, popular: List[Any]):
        self.model, self.popular = model, popular

    def recommend(self, user_id: Any, top_k: int = 10, **kwargs: Any) -> List[Any]:
        try:
            recs = self.model.recommend(user_id, top_k=top_k)
        except Exception:
            recs = []
        return recs or self.popular[:top_k]


def retrain_artifact(
    artifact: Union[str, Path],
    data: Optional[Union[str, Path]] = None,
    feedback: Optional[Union[str, Path]] = None,
    tolerance: float = 0.0,
    k: int = 10,
    min_new_rows: int = 100,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Retrain the model in *artifact* on new data; replace it only if it isn't worse.

    *data* defaults to the file the artifact was trained on, re-read so rows
    appended since count. Clicks/purchases from a *feedback* log are added as
    interactions. Rows newer than the artifact's ``trained_through`` are the new
    data, split in time into two halves:

    - candidate: trained on everything old plus the earlier half
    - both models are scored on the later half, which neither has seen, served
      as ModelServer serves them (popular items for users a model can't answer)

    A random holdout can't judge the deployed model: it was trained on those
    rows, and its exclude-seen filter removes exactly the held-out items.

    The candidate is promoted when its NDCG@k >= current - ``tolerance``; it is
    then refit on all rows and saved, with the old artifact in ``previous/``.
    """
    import shutil

    from corerec.evaluation.evaluate import evaluate as run_eval

    artifact = Path(artifact)
    current, manifest = load_artifact(artifact)
    source = data or manifest.get("source")
    if not source:
        raise ValueError("the artifact doesn't record its training file; pass data=")
    cutoff = manifest.get("trained_through")
    if cutoff is None:
        raise ValueError("retrain needs to know which rows are new, and this artifact has no "
                         "trained_through: train it from a file with a timestamp column "
                         "(corerec 0.7.1+)")
    cols = manifest.get("columns") or {}
    df, cols = read_interactions(source, user_col=cols.get("user"), item_col=cols.get("item"),
                                 rating_col=cols.get("rating"), timestamp_col=cols.get("timestamp"))
    if "timestamp" not in df:
        raise ValueError(f"{source} has no timestamp column; retrain can't tell new rows from old")
    n_feedback = 0
    if feedback:
        from corerec.serving.feedback import FeedbackLog

        fb = FeedbackLog(feedback).to_events().rename(columns={"user_id": "user", "item_id": "item"})
        n_feedback = len(fb)
        df = pd.concat([df, fb[["user", "item", "rating", "timestamp"]]], ignore_index=True)

    name, params = manifest["model"], manifest.get("params") or {}
    old, new = df[df["timestamp"] <= cutoff], df[df["timestamp"] > cutoff].sort_values("timestamp")
    decision = {"promoted": False, "would_promote": False, "model": name, "metric": f"NDCG@{k}",
                "rows": len(df), "new_rows": len(new), "feedback_rows": n_feedback,
                "data": str(source), "candidate": None, "current": None, "tolerance": tolerance}
    if len(new) < min_new_rows:
        decision["reason"] = f"only {len(new)} new rows since the last training (need {min_new_rows})"
        return decision

    def merged(frame):
        agg = {"rating": "sum", "timestamp": "max"}
        return frame.groupby(["user", "item"], as_index=False, sort=False).agg(agg)

    half = len(new) // 2
    train = merged(pd.concat([old, new.iloc[:half]]))
    test = new.iloc[half:]
    test = test.merge(train[["user", "item"]], on=["user", "item"], how="left", indicator=True)
    test = merged(test[test["_merge"] == "left_only"].drop(columns="_merge"))
    start = time.perf_counter()
    candidate = fit_model(name, train, params)
    popular = popular_items(train, n=k + 1000)
    eval_args = dict(test_interactions=test, train_interactions=train, k=k,
                     user_col="user", item_col="item", rating_col="rating")
    key = f"NDCG@{k}"
    cand = run_eval(_Served(candidate, popular), **eval_args)
    curr = run_eval(_Served(current, popular), **eval_args)
    promote = cand[key] >= curr[key] - tolerance
    decision.update(candidate=cand[key], current=curr[key], test_rows=len(test),
                    would_promote=bool(promote), promoted=bool(promote and not dry_run))
    if promote and not dry_run:
        everything = merged(df)
        final = fit_model(name, everything, params)  # ship the model trained on everything
        backup = artifact / "previous"
        shutil.rmtree(backup, ignore_errors=True)
        backup.mkdir()
        for f in artifact.iterdir():
            if f.is_file():
                shutil.move(str(f), backup / f.name)
        stats = {"rows": len(everything), "users": int(everything["user"].nunique()),
                 "items": int(everything["item"].nunique()), "test_rows": len(test)}
        result = TrainResult(model=final, model_name=name, params=params, columns=cols,
                             stats=stats, popular=popular_items(everything),
                             fit_seconds=time.perf_counter() - start, k=k, metrics=cand,
                             baseline=None, source=str(source),
                             extra={"trained_through": float(everything["timestamp"].max())})
        save_artifact(result, artifact)
        m = json.loads((artifact / MANIFEST).read_text())
        m["retrain"] = {k_: v for k_, v in decision.items() if k_ != "promoted"}
        (artifact / MANIFEST).write_text(json.dumps(m, indent=2, default=str))
    return decision


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


def build_server(model, manifest: Optional[Dict[str, Any]] = None, host: str = "0.0.0.0",
                 port: int = 8000, feedback_log: Optional[Union[str, Path]] = None,
                 challenger: Optional[Any] = None, challenger_share: float = 0.1,
                 artifact: Optional[Union[str, Path]] = None):
    """A :class:`~corerec.serving.ModelServer` that answers unknown users with popular items.

    feedback_log: JSONL path; turns on /feedback and /metrics.
    challenger: a second model for an A/B test, given ``challenger_share`` of users.
    artifact: the directory *model* came from; enables POST /reload after a retrain.
    """
    from corerec.serving.model_server import ModelServer

    manifest = manifest or {}
    models, traffic = model, None
    if challenger is not None:
        models = {"control": model, "challenger": challenger}
        traffic = {"control": 1 - challenger_share, "challenger": challenger_share}
    reload_fn = (lambda: load_artifact(artifact)[0]) if artifact and challenger is None else None
    return ModelServer(models, host=host, port=port, metadata=manifest,
                       fallback_items=manifest.get("popular_items"), feedback_log=feedback_log,
                       traffic=traffic, reload_fn=reload_fn)


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
