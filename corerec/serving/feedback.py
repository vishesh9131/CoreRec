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
from typing import Any, Dict, List, Optional, Union


def _safe(v: Any) -> Any:
    return v.item() if hasattr(v, "item") and not isinstance(v, (list, dict, str)) else v


class FeedbackLog:
    """Append-only JSONL log of impressions and feedback. Thread-safe within a process."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

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
    def _join(records):
        """Impressions keyed by request_id, each with the set of items clicked from it."""
        imps = {r["request_id"]: dict(r, clicked=set()) for r in records if r["type"] == "impression"}
        for r in records:
            if r["type"] == "feedback" and r.get("request_id") in imps:
                imps[r["request_id"]]["clicked"].add(json.dumps(r["item_id"], default=str))
        return list(imps.values())

    @staticmethod
    def _summary(imps) -> Dict[str, Any]:
        shown = clicks = requests_clicked = fallback = 0
        rr, distinct = [], set()
        for imp in imps:
            items = [json.dumps(i, default=str) for i in imp["items"]]
            shown += len(items)
            distinct.update(items)
            hit = [k for k, i in enumerate(items) if i in imp["clicked"]]
            clicks += len(hit)
            requests_clicked += bool(hit)
            rr.append(1.0 / (hit[0] + 1) if hit else 0.0)
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
        imps = [i for i in self._join(self.records()) if since is None or i["ts"] >= since]
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
        ``shift`` (total variation distance, 0 = same, 1 = disjoint).
        """
        imps = sorted(self._join(self.records()), key=lambda i: i["ts"])
        if len(imps) < 2 * recent:
            return {"alerts": [], "note": f"need {2 * recent} requests, have {len(imps)}"}
        base, now = self._summary(imps[:-recent]), self._summary(imps[-recent:])
        alerts = []
        if base["ctr"] > 0 and (base["ctr"] - now["ctr"]) / base["ctr"] > ctr_drop:
            alerts.append(f"CTR fell from {base['ctr']:.4f} to {now['ctr']:.4f}")
        if now["fallback_share"] - base["fallback_share"] > fallback_rise:
            alerts.append(f"fallback (unknown-user) share rose from {base['fallback_share']:.1%} "
                          f"to {now['fallback_share']:.1%}")

        def clicked(group):
            c = Counter(i for imp in group for i in imp["clicked"])
            total = sum(c.values())
            return {k: v / total for k, v in c.items()} if total else {}

        p, q = clicked(imps[:-recent]), clicked(imps[-recent:])
        tvd = 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q)) if p and q else 0.0
        if tvd > shift:
            alerts.append(f"clicked-item distribution shifted (TVD {tvd:.2f})")
        return {"alerts": alerts, "baseline": base, "recent": now, "click_shift_tvd": tvd}

    def to_events(self, events=("click", "purchase")):
        """Positive feedback as a user_id/item_id/rating/timestamp DataFrame."""
        import pandas as pd

        rows = [(r["user_id"], r["item_id"], 1.0, r["ts"]) for r in self.records()
                if r["type"] == "feedback" and r.get("event") in events]
        return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


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
