#!/usr/bin/env python3
"""Production-style stress probes for CoreRec (run: conda activate corerec && python scripts/stress_test_production.py)."""
from __future__ import annotations

import gc
import inspect
import json
import os
import pickle
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    name: str
    category: str
    passed: bool
    duration_ms: float
    detail: str = ""
    severity: str = "info"  # info | warn | fail | critical


@dataclass
class StressReport:
    results: List[ProbeResult] = field(default_factory=list)

    def add(self, **kwargs) -> None:
        self.results.append(ProbeResult(**kwargs))

    def summary(self) -> Dict[str, Any]:
        by_cat: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            by_cat.setdefault(r.category, {"pass": 0, "fail": 0, "warn": 0})
            if r.passed:
                by_cat[r.category]["pass"] += 1
            elif r.severity in ("warn",):
                by_cat[r.category]["warn"] += 1
            else:
                by_cat[r.category]["fail"] += 1
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed and r.severity == "fail"),
            "warnings": sum(1 for r in self.results if not r.passed and r.severity == "warn"),
            "critical": sum(1 for r in self.results if r.severity == "critical"),
            "by_category": by_cat,
            "failures": [
                {"name": r.name, "category": r.category, "detail": r.detail, "severity": r.severity}
                for r in self.results
                if not r.passed
            ],
        }


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000


def make_tiny_data():
    import numpy as np
    import pandas as pd

    users = [0, 0, 1, 1, 2, 2, 3, 3]
    items = [10, 11, 10, 12, 11, 13, 12, 13]
    ratings = [5.0, 4.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0]
    df = pd.DataFrame({"user_id": users, "item_id": items, "rating": ratings})
    sar_df = df.rename(columns={"user_id": "userID", "item_id": "itemID", "rating": "rating"})
    n_users, n_items = 4, 4
    mat = np.zeros((n_users, n_items))
    for u, i, r in zip(users, items, ratings):
        mat[u, i - 10] = r
    return df, sar_df, users, items, ratings, mat


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def probe_imports(report: StressReport) -> None:
    cases = [
        ("corerec", lambda: __import__("corerec")),
        ("corerec.engines.DCN", lambda: __import__("corerec.engines.dcn", fromlist=["DCN"])),
        ("corerec.engines.collaborative.SAR", lambda: __import__("corerec.engines.collaborative", fromlist=["SAR"])),
        ("corerec.api.base_recommender", lambda: __import__("corerec.api.base_recommender", fromlist=["BaseRecommender"])),
        ("corerec.pipelines.orchestrator", lambda: __import__("corerec.pipelines.orchestrator")),
        ("corerec.serving.model_server", lambda: __import__("corerec.serving.model_server")),
        ("corerec.serialization.serializer", lambda: __import__("corerec.serialization.serializer")),
        ("corerec.sandbox", lambda: __import__("corerec.sandbox")),
    ]
    for name, fn in cases:
        try:
            _, ms = timed(fn)
            report.add(name=f"import:{name}", category="imports", passed=True, duration_ms=ms)
        except Exception as e:
            report.add(
                name=f"import:{name}",
                category="imports",
                passed=False,
                duration_ms=0,
                detail=str(e),
                severity="fail",
            )


def probe_api_uniformity(report: StressReport) -> None:
    from corerec.api.base_recommender import BaseRecommender

    models = _load_production_models()
    for label, cls in models:
        sig = inspect.signature(cls.recommend)
        params = list(sig.parameters.keys())
        uses_top_n = "top_n" in params and "top_k" not in params
        uses_top_k = "top_k" in params
        inherits = issubclass(cls, BaseRecommender)
        if not inherits:
            report.add(
                name=f"inheritance:{label}",
                category="api_uniformity",
                passed=False,
                duration_ms=0,
                detail=f"{label} does not inherit BaseRecommender",
                severity="critical",
            )
        elif uses_top_n and not uses_top_k:
            report.add(
                name=f"recommend_sig:{label}",
                category="api_uniformity",
                passed=False,
                duration_ms=0,
                detail=f"uses top_n instead of BaseRecommender top_k: {params}",
                severity="warn",
            )
        else:
            report.add(
                name=f"recommend_sig:{label}",
                category="api_uniformity",
                passed=True,
                duration_ms=0,
                detail=str(params),
            )


def probe_unfitted_errors(report: StressReport) -> None:
    from corerec.api.exceptions import ModelNotFittedError

    classes, factories = _production_model_registry()
    for label, cls in classes:
        try:
            m = factories[label]()
            try:
                m.predict(0, 0)
                report.add(
                    name=f"unfitted_predict:{label}",
                    category="error_handling",
                    passed=False,
                    duration_ms=0,
                    detail="predict succeeded without fit()",
                    severity="critical",
                )
            except ModelNotFittedError:
                report.add(name=f"unfitted_predict:{label}", category="error_handling", passed=True, duration_ms=0)
            except (RuntimeError, ValueError) as e:
                report.add(
                    name=f"unfitted_predict:{label}",
                    category="error_handling",
                    passed=False,
                    duration_ms=0,
                    detail=f"wrong exception type: {type(e).__name__}: {e}",
                    severity="warn",
                )
            except Exception as e:
                report.add(
                    name=f"unfitted_predict:{label}",
                    category="error_handling",
                    passed=False,
                    duration_ms=0,
                    detail=f"unexpected: {type(e).__name__}: {e}",
                    severity="fail",
                )
        except Exception as e:
            report.add(
                name=f"unfitted_init:{label}",
                category="error_handling",
                passed=False,
                duration_ms=0,
                detail=str(e),
                severity="fail",
            )


def probe_invalid_inputs(report: StressReport) -> None:
    """After fit, bad IDs should fail gracefully."""
    from corerec.engines.collaborative import SAR

    df, sar_df, *_ = make_tiny_data()
    m = SAR()
    m.fit(sar_df)
    for uid in [99999, None, "", -1]:
        try:
            m.recommend(uid, top_k=3)
            report.add(
                name=f"invalid_user:{repr(uid)}",
                category="input_validation",
                passed=False,
                duration_ms=0,
                detail="recommend returned without error for unknown user",
                severity="warn",
            )
        except Exception:
            report.add(name=f"invalid_user:{repr(uid)}", category="input_validation", passed=True, duration_ms=0)


def probe_persistence(report: StressReport) -> None:
    classes, factories = _production_model_registry()
    df, sar_df, users, items, ratings, mat = make_tiny_data()

    for label, cls in classes:
        t0 = time.perf_counter()
        try:
            m = _fit_model(label, factories[label], df, sar_df, users, items, ratings, mat)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / f"{label}.artifact"
                m.save(str(path))
                loaded = cls.load(str(path))
                score_before = m.predict(_first_user(label, users), _first_item(label, items))
                score_after = loaded.predict(_first_user(label, users), _first_item(label, items))
                ok = abs(score_before - score_after) < 1e-4 or (score_before == score_after)
                report.add(
                    name=f"save_load:{label}",
                    category="persistence",
                    passed=ok,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    detail="" if ok else f"score drift {score_before} -> {score_after}",
                    severity="fail" if not ok else "info",
                )
            del m
            gc.collect()
        except Exception as e:
            report.add(
                name=f"save_load:{label}",
                category="persistence",
                passed=False,
                duration_ms=(time.perf_counter() - t0) * 1000,
                detail=traceback.format_exc(limit=2),
                severity="fail",
            )


def probe_concurrent_recommend(report: StressReport) -> None:
    """Basic thread-safety smoke: parallel recommend after fit."""
    import threading

    from corerec.engines.collaborative import SAR

    _, sar_df, *_ = make_tiny_data()
    m = SAR()
    m.fit(sar_df)
    errors: List[str] = []

    def worker(uid):
        try:
            for _ in range(20):
                m.recommend(uid, top_k=2)
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ms = (time.perf_counter() - t0) * 1000
    report.add(
        name="concurrent_recommend:SAR",
        category="concurrency",
        passed=len(errors) == 0,
        duration_ms=ms,
        detail="; ".join(errors[:3]),
        severity="fail" if errors else "info",
    )


def probe_pickle_security(report: StressReport) -> None:
    """Document pickle RCE risk — load should not execute arbitrary code from malicious pickle."""
    import io

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo STRESS_TEST_PICKLE_RCE",))

    payload = pickle.dumps(Evil())
    report.add(
        name="pickle_rce_awareness",
        category="security",
        passed=False,
        duration_ms=0,
        detail="Models use pickle/torch.load — untrusted artifacts are unsafe (framework-wide risk)",
        severity="warn",
    )
    # Do NOT actually unpickle Evil in CI; just note the pattern exists
    _ = len(payload)


def probe_cr_learn(report: StressReport) -> None:
    try:
        from cr_learn import ml_1m

        t0 = time.perf_counter()
        data = ml_1m.load()
        ms = (time.perf_counter() - t0) * 1000
        keys = sorted(data.keys()) if isinstance(data, dict) else []
        ratings = data.get("ratings") if isinstance(data, dict) else None
        n = len(ratings) if ratings is not None else 0
        report.add(
            name="cr_learn:ml_1m.load",
            category="datasets",
            passed=n > 0,
            duration_ms=ms,
            detail=f"keys={keys}, ratings_rows={n}",
        )
    except ImportError:
        report.add(
            name="cr_learn:ml_1m.load",
            category="datasets",
            passed=False,
            duration_ms=0,
            detail="cr_learn not installed — docs examples won't run out of the box",
            severity="warn",
        )
    except Exception as e:
        report.add(
            name="cr_learn:ml_1m.load",
            category="datasets",
            passed=False,
            duration_ms=0,
            detail=str(e),
            severity="fail",
        )


def probe_serving_deps(report: StressReport) -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401

        report.add(name="serving:deps_installed", category="serving", passed=True, duration_ms=0)
    except ImportError as e:
        report.add(
            name="serving:deps_installed",
            category="serving",
            passed=False,
            duration_ms=0,
            detail=f"FastAPI stack missing: {e} — serving module not production-ready without extras",
            severity="warn",
        )


def probe_pipeline_e2e(report: StressReport) -> None:
    try:
        from corerec.pipelines.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator()
        report.add(
            name="pipeline:orchestrator_init",
            category="pipelines",
            passed=True,
            duration_ms=0,
            detail=str(type(orch)),
        )
    except Exception as e:
        report.add(
            name="pipeline:orchestrator_init",
            category="pipelines",
            passed=False,
            duration_ms=0,
            detail=str(e),
            severity="fail",
        )


def probe_memory_leak_smoke(report: StressReport) -> None:
    """Repeated fit/recommend cycles — rough memory stability signal."""
    import tracemalloc

    from corerec.engines.collaborative import SAR

    _, sar_df, *_ = make_tiny_data()
    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    for _ in range(10):
        m = SAR()
        m.fit(sar_df)
        m.recommend(0, top_k=2)
        del m
    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    diff = snap1.compare_to(snap0, "lineno")
    total_kb = sum(s.size_diff for s in diff[:20]) / 1024
    report.add(
        name="memory:10x_SAR_fit",
        category="performance",
        passed=total_kb < 50_000,  # 50MB threshold — generous smoke test
        duration_ms=0,
        detail=f"top-20 delta ~{total_kb:.1f} KB",
        severity="warn" if total_kb >= 50_000 else "info",
    )


def probe_doc_import_paths(report: StressReport) -> None:
    """Tutorial import paths that commonly break."""
    broken = []
    cases = [
        ("engines.afm.AFM", "corerec.engines.afm", "AFM"),
        ("engines.bpr", "corerec.engines.bpr", None),
        ("engines.dcn", "corerec.engines.dcn", "DCN"),
    ]
    for label, mod, attr in cases:
        try:
            m = __import__(mod, fromlist=[attr] if attr else [])
            if attr and not hasattr(m, attr):
                broken.append(f"{label}: module exists but no {attr}")
            else:
                report.add(name=f"doc_path:{label}", category="docs_accuracy", passed=True, duration_ms=0)
        except Exception as e:
            broken.append(f"{label}: {e}")
            report.add(
                name=f"doc_path:{label}",
                category="docs_accuracy",
                passed=False,
                duration_ms=0,
                detail=str(e),
                severity="warn",
            )


# ---------------------------------------------------------------------------
# Model fixtures (mirror test_all_production_models)
# ---------------------------------------------------------------------------

def _production_model_registry():
    """Return list of (label, class) and dict label -> factory callable."""
    from corerec.engines.bert4rec import BERT4Rec
    from corerec.engines.collaborative import FAST, LightGCN, NCF, SAR
    from corerec.engines.collaborative.fast_recommender import FASTRecommender
    from corerec.engines.content_based import TFIDFRecommender
    from corerec.engines.dcn import DCN
    from corerec.engines.deepfm import DeepFM
    from corerec.engines.gnnrec import GNNRec
    from corerec.engines.mind import MIND
    from corerec.engines.nasrec import NASRec
    from corerec.engines.sasrec import SASRec
    from corerec.engines.two_tower import TwoTower

    classes = [
        ("DCN", DCN),
        ("DeepFM", DeepFM),
        ("GNNRec", GNNRec),
        ("MIND", MIND),
        ("NASRec", NASRec),
        ("SASRec", SASRec),
        ("TwoTower", TwoTower),
        ("BERT4Rec", BERT4Rec),
        ("SAR", SAR),
        ("NCF", NCF),
        ("FAST", FAST),
        ("FASTRecommender", FASTRecommender),
        ("LightGCN", LightGCN),
        ("TFIDFRecommender", TFIDFRecommender),
    ]
    factories = {
        "DCN": lambda: DCN(embedding_dim=8, num_cross_layers=1, deep_layers=[16], num_epochs=1, batch_size=4),
        "DeepFM": lambda: DeepFM(embedding_dim=8, hidden_dims=[16], num_epochs=1, batch_size=4),
        "GNNRec": lambda: GNNRec(embedding_dim=8, num_layers=1, num_epochs=1, batch_size=4),
        "MIND": lambda: MIND(embedding_dim=8, num_interests=2, num_epochs=1, batch_size=4),
        "NASRec": lambda: NASRec(embedding_dim=8, num_epochs=1, batch_size=4),
        "SASRec": lambda: SASRec(hidden_units=16, num_blocks=1, num_heads=1, num_epochs=1, batch_size=4),
        "TwoTower": lambda: TwoTower(embedding_dim=8, num_epochs=1, batch_size=4),
        "BERT4Rec": lambda: BERT4Rec(embedding_dim=16, num_layers=1, num_heads=2, num_epochs=1, batch_size=4),
        "SAR": SAR,
        "NCF": lambda: NCF(embedding_dim=8, hidden_dims=[16]),
        "FAST": FAST,
        "FASTRecommender": lambda: FASTRecommender(factors=8, iterations=2, seed=42),
        "LightGCN": lambda: LightGCN(embedding_dim=8, num_layers=1, num_epochs=1),
        "TFIDFRecommender": TFIDFRecommender,
    }
    return classes, factories


def _load_production_models():
    classes, _ = _production_model_registry()
    return classes


def _fit_model(label, factory, df, sar_df, users, items, ratings, mat):
    if label == "SAR":
        m = factory()
        m.fit(sar_df)
        return m
    if label == "NCF":
        m = factory()
        m.fit(df)
        return m
    if label == "TFIDFRecommender":
        docs = ["action movie", "romance film", "action hero", "love story"]
        m = factory()
        m.fit([10, 11, 12, 13], docs)
        return m
    if label in ("SASRec", "BERT4Rec", "TwoTower"):
        m = factory()
        m.fit(users, items, mat)
        return m
    if label == "LightGCN":
        m = factory()
        m.fit(users, items, ratings)
        return m
    if label in ("FAST", "FASTRecommender"):
        m = factory()
        m.fit(users, items, ratings)
        return m
    m = factory()
    m.fit(user_ids=users, item_ids=items, ratings=ratings)
    return m


def _first_user(label, users):
    return users[0]


def _first_item(label, items):
    return items[0]


def main():
    report = StressReport()
    print("CoreRec production stress test\n" + "=" * 50)
    probe_imports(report)
    probe_api_uniformity(report)
    probe_unfitted_errors(report)
    probe_invalid_inputs(report)
    probe_persistence(report)
    probe_concurrent_recommend(report)
    probe_pickle_security(report)
    probe_cr_learn(report)
    probe_serving_deps(report)
    probe_pipeline_e2e(report)
    probe_memory_leak_smoke(report)
    probe_doc_import_paths(report)

    summary = report.summary()
    out_path = ROOT / "scripts" / "stress_test_report.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nReport written to {out_path}")
    return 0 if summary["failed"] == 0 and summary["critical"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
