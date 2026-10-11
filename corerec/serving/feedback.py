"""
Feedback logging, online metrics, A/B comparison and drift checks.

ModelServer writes one JSON line per event to a FeedbackLog:

    {"type": "impression", "request_id", "ts", "user_id", "items": [...], "variant", "source"}
    {"type": "feedback",   "request_id", "ts", "user_id", "item_id", "event": "click"}

Everything below reads that file, so it works the same on a live server's log
or a copied one:

    log = FeedbackLog("feedback.jsonl")
    log.metrics()      # CTR, MRR, coverage, fallback share -- per variant
    log.compare("control", "treatment")   # CTR difference and its p-value
    log.drift()        # alerts when recent traffic moved away from the baseline
    log.to_events()    # clicks as a user_id/item_id/rating/timestamp frame for retraining
"""

import json
import math
import threading
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def _safe(v: Any) -> Any:
    return v.item() if hasattr(v, "item") and not isinstance(v, (list, dict, str)) else v


class FeedbackLog:
    """Append-only JSONL log of impressions and feedback. Thread-safe within a process."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # joined impressions, built incrementally: /metrics only parses the
        # bytes appended since the last call instead of the whole file
        self._read_lock = threading.Lock()
        self._offset = 0
        self._imps: Dict[str, Dict[str, Any]] = {}
        self._orphans: Dict[str, List[str]] = defaultdict(list)  # clicks seen before their impression

    # -- writing -------------------------------------------------------- #
    def _write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")

    def impression(self, user_id: Any, items: List[Any], variant: str = "default",
                   source: str = "model") -> str:
        request_id = uuid.uuid4().hex
        self._write({"type": "impression", "request_id": request_id, "ts": time.time(),
                     "user_id": _safe(user_id), "items": [_safe(i) for i in items],
                     "variant": variant, "source": source})
        return request_id

    def feedback(self, user_id: Any, item_id: Any, event: str = "click",
                 request_id: Optional[str] = None) -> None:
        self._write({"type": "feedback", "request_id": request_id, "ts": time.time(),
                     "user_id": _safe(user_id), "item_id": _safe(item_id), "event": event})

    # -- reading -------------------------------------------------------- #
    def records(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def _click(imp, key):
        if key not in imp["clicked"]:
            imp["clicked"].add(key)
            imp["hits"].update(k for k, i in enumerate(imp["keys"]) if i == key)

    def _ingest(self, r):
        if r["type"] == "impression":
            imp = dict(r, clicked=set(), hits=set(),
                       keys=[json.dumps(i, default=str) for i in r["items"]])
            self._imps[r["request_id"]] = imp
            for key in self._orphans.pop(r["request_id"], ()):
                self._click(imp, key)
        elif r["type"] == "feedback" and r.get("request_id") is not None:
            key = json.dumps(r["item_id"], default=str)
            imp = self._imps.get(r["request_id"])
            if imp is None:
                self._orphans[r["request_id"]].append(key)
            else:
                self._click(imp, key)

    def _joined(self) -> List[Dict[str, Any]]:
        """Impressions in log order, each with the items clicked from it.

        Reads only what was appended since the last call; a file that shrank
        (rotated or rewritten) is read again from the start.
        """
        with self._read_lock:
            size = self.path.stat().st_size if self.path.exists() else 0
            if size < self._offset:
                self._offset, self._imps, self._orphans = 0, {}, defaultdict(list)
            if size > self._offset:
                with open(self.path, "rb") as f:
                    f.seek(self._offset)
                    chunk = f.read(size - self._offset)
                end = chunk.rfind(b"\n") + 1  # leave a half-written last line for next time
                for line in chunk[:end].splitlines():
                    if line.strip():
                        self._ingest(json.loads(line))
                self._offset += end
            return list(self._imps.values())

    @staticmethod
    def _summary(imps) -> Dict[str, Any]:
        shown = clicks = requests_clicked = fallback = 0
        rr, distinct = [], set()
        for imp in imps:
            shown += len(imp["keys"])
            distinct.update(imp["keys"])
            hit = imp["hits"]
            clicks += len(hit)
            requests_clicked += bool(hit)
            rr.append(1.0 / (min(hit) + 1) if hit else 0.0)
            fallback += imp.get("source") == "fallback"
        n = len(imps)
        return {
            "requests": n,
            "impressions": shown,
            "clicks": clicks,
            "ctr": clicks / shown if shown else float("nan"),
            "requests_with_click": requests_clicked / n if n else float("nan"),
            "mrr": sum(rr) / n if n else float("nan"),
            "distinct_items_shown": len(distinct),
            "fallback_share": fallback / n if n else float("nan"),
        }

    def metrics(self, since: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """Online metrics per variant (and "all"), from impressions after ``since`` (epoch s).

        ctr: clicks on shown items / items shown. mrr: mean of 1/rank of the
        first clicked item per request (0 if none). Feedback without a
        request_id is kept for retraining but can't be attributed here.
        """
        imps = [i for i in self._joined() if since is None or i["ts"] >= since]
        by_variant = defaultdict(list)
        for imp in imps:
            by_variant[imp.get("variant", "default")].append(imp)
        out = {v: self._summary(group) for v, group in sorted(by_variant.items())}
        out["all"] = self._summary(imps)
        return out

    def compare(self, a: str, b: str) -> Dict[str, Any]:
        """CTR of variant ``b`` against ``a`` with a two-proportion z-test."""
        m = self.metrics()
        if a not in m or b not in m:
            raise ValueError(f"need traffic on both variants; have {sorted(set(m) - {'all'})}")
        ca, na, cb, nb = m[a]["clicks"], m[a]["impressions"], m[b]["clicks"], m[b]["impressions"]
        pa, pb = ca / na, cb / nb
        pooled = (ca + cb) / (na + nb)
        se = math.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb)) if 0 < pooled < 1 else 0.0
        z = (pb - pa) / se if se else 0.0
        p = math.erfc(abs(z) / math.sqrt(2))
        return {"ctr_" + a: pa, "ctr_" + b: pb, "lift": (pb - pa) / pa if pa else float("nan"),
                "z": z, "p_value": p, "significant_at_0.05": p < 0.05,
                "impressions": {a: na, b: nb}}

    def drift(self, recent: int = 1000, ctr_drop: float = 0.3, fallback_rise: float = 0.1,
              shift: float = 0.2) -> Dict[str, Any]:
        """Compare the last ``recent`` requests with everything before them.

        Alerts when CTR fell by more than ``ctr_drop`` (relative), the
        unknown-user fallback share rose by more than ``fallback_rise``
        (absolute), or the distribution of clicked items moved by more than
        ``shift`` (total variation distance, 0 = same, 1 = disjoint) and by
        more than sampling noise: the 99th percentile of the TVD between random
        splits of the same clicks, reported as ``click_shift_noise``.
        """
        imps = sorted(self._joined(), key=lambda i: i["ts"])
        if len(imps) < 2 * recent:
            return {"alerts": [], "note": f"need {2 * recent} requests, have {len(imps)}"}
        base, now = self._summary(imps[:-recent]), self._summary(imps[-recent:])
        alerts = []
        if base["ctr"] > 0 and (base["ctr"] - now["ctr"]) / base["ctr"] > ctr_drop:
            alerts.append(f"CTR fell from {base['ctr']:.4f} to {now['ctr']:.4f}")
        if now["fallback_share"] - base["fallback_share"] > fallback_rise:
            alerts.append(f"fallback (unknown-user) share rose from {base['fallback_share']:.1%} "
                          f"to {now['fallback_share']:.1%}")

        tvd, noise = _click_shift([i for imp in imps[:-recent] for i in imp["clicked"]],
                                  [i for imp in imps[-recent:] for i in imp["clicked"]])
        if tvd > max(shift, noise):
            alerts.append(f"clicked-item distribution shifted (TVD {tvd:.2f}, noise {noise:.2f})")
        return {"alerts": alerts, "baseline": base, "recent": now, "click_shift_tvd": tvd,
                "click_shift_noise": noise}

    def to_events(self, events=("click", "purchase")):
        """Positive feedback as a user_id/item_id/rating/timestamp DataFrame."""
        import pandas as pd

        rows = [(r["user_id"], r["item_id"], 1.0, r["ts"]) for r in self.records()
                if r["type"] == "feedback" and r.get("event") in events]
        return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def _click_shift(a: List[Any], b: List[Any], n_perm: int = 200) -> Tuple[float, float]:
    """TVD between two samples of clicked items, and what noise alone gives.

    Two small samples over a large catalogue are far apart in TVD even when
    they come from one distribution (0.22 on stationary ML-1M traffic), so
    the observed TVD is compared with the 99th percentile over random splits
    of the pooled clicks into groups of the same sizes (#207).
    """
    import numpy as np

    if not a or not b:
        return 0.0, 0.0
    index: Dict[Any, int] = {}
    codes = np.array([index.setdefault(i, len(index)) for i in a + b])

    def tvd(x, y):
        px = np.bincount(x, minlength=len(index)) / len(x)
        return 0.5 * float(np.abs(px - np.bincount(y, minlength=len(index)) / len(y)).sum())

    rng = np.random.default_rng(0)  # fixed, so /metrics doesn't flicker between calls
    null = []
    for _ in range(n_perm):
        p = rng.permutation(codes)
        null.append(tvd(p[:len(a)], p[len(a):]))
    return tvd(codes[:len(a)], codes[len(a):]), float(np.quantile(null, 0.99))


def assign_variant(user_id: Any, traffic: Dict[str, float], salt: str = "corerec") -> str:
    """Sticky assignment: the same user always lands in the same variant."""
    import hashlib

    h = int(hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()[:15], 16) / 16 ** 15
    total, acc = sum(traffic.values()), 0.0
    for name, share in traffic.items():
        acc += share / total
        if h < acc:
            return name
    return name
