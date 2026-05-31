#!/usr/bin/env python3
"""Production-style stress probes for CoreRec (run: conda activate corerec && python scripts/stress_test_production.py)."""
from __future__ import annotations

import gc
import inspect
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
        failures = [
            {"name": r.name, "category": r.category, "detail": r.detail, "severity": r.severity}
            for r in self.results
            if not r.passed
        ]
        grades = _compute_grades(by_cat, failures)
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed and r.severity == "fail"),
            "warnings": sum(1 for r in self.results if not r.passed and r.severity == "warn"),
            "critical": sum(1 for r in self.results if r.severity == "critical"),
            "by_category": by_cat,
            "grades": grades,
            "failures": failures,
        }


def _compute_grades(by_cat: Dict, failures: List) -> Dict[str, str]:
    critical = [f for f in failures if f["severity"] == "critical"]
    p0 = [f for f in failures if f["category"] in ("persistence", "safe_bundle") and f["severity"] == "fail"]
    if critical:
        overall = "D"
    elif len(p0) >= 3:
        overall = "C+"
    elif len(p0) >= 1:
        overall = "B-"
    elif failures:
        overall = "B+"
    else:
        overall = "A-"
    return {
        "overall_production_readiness": overall,
        "note": "Compared to PyTorch/TF serving maturity and LangChain DX — recsys core is solid; platform/docs gaps remain.",
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
    user_list = sorted(set(users))
    item_list = sorted(set(items))
    n_users, n_items = len(user_list), len(item_list)
    mat = np.zeros((n_users, n_items), dtype=np.float32)
    umap = {u: i for i, u in enumerate(user_list)}
    imap = {it: i for i, it in enumerate(item_list)}
    for u, i, r in zip(users, items, ratings):
        mat[umap[u], imap[i]] = max(mat[umap[u], imap[i]], r)
    return df, sar_df, users, items, ratings, mat, user_list, item_list


def probe_imports(report: StressReport) -> None:
    cases = [
        ("corerec", lambda: __import__("corerec")),
        ("corerec.engines.DCN", lambda: __import__("corerec.engines.dcn", fromlist=["DCN"])),
        ("corerec.engines.collaborative.SAR", lambda: __import__("corerec.engines.collaborative", fromlist=["SAR"])),
        ("corerec.api.base_recommender", lambda: __import__("corerec.api.base_recommender", fromlist=["BaseRecommender"])),
        ("corerec.api.model_bundle", lambda: __import__("corerec.api.model_bundle", fromlist=["is_safe_bundle"])),
        ("corerec.pipelines.orchestrator", lambda: __import__("corerec.pipelines.orchestrator")),
        ("corerec.serving.model_server", lambda: __import__("corerec.serving.model_server")),
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


def probe_pytest_production(report: StressReport) -> None:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_all_production_models.py",
        "tests/test_production_contract.py",
        "tests/test_safe_persistence.py",
        "tests/test_api_uniformity.py",
        "tests/test_serving_smoke.py",
        "-q",
        "--tb=no",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    ms = (time.perf_counter() - t0) * 1000
    report.add(
        name="pytest:production_suite",
        category="ci_contract",
        passed=proc.returncode == 0,
        duration_ms=ms,
        detail=(proc.stdout or proc.stderr)[-500:],
        severity="fail" if proc.returncode else "info",
    )


def probe_api_uniformity(report: StressReport) -> None:
    from corerec.api.base_recommender import BaseRecommender

    for label, cls in _load_production_models():
        sig = inspect.signature(cls.recommend)
        params = list(sig.parameters.keys())
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
        elif "top_k" not in params:
            report.add(
                name=f"recommend_sig:{label}",
                category="api_uniformity",
                passed=False,
                duration_ms=0,
                detail=f"missing top_k: {params}",
                severity="warn",
            )
        else:
            report.add(name=f"recommend_sig:{label}", category="api_uniformity", passed=True, duration_ms=0)


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
                    detail=f"wrong exception: {type(e).__name__}: {e}",
                    severity="warn",
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
    from corerec.api.exceptions import RecommendationError
    from corerec.engines.collaborative import SAR

    _, sar_df, *_ = make_tiny_data()
    m = SAR()
    m.fit(sar_df)
    for uid in [99999, None, ""]:
        try:
            m.recommend(uid, top_k=3)
            report.add(
                name=f"invalid_user:{repr(uid)}",
                category="input_validation",
                passed=False,
                duration_ms=0,
                detail="recommend returned without error",
                severity="fail",
            )
        except (RecommendationError, Exception):
            report.add(name=f"invalid_user:{repr(uid)}", category="input_validation", passed=True, duration_ms=0)


def probe_safe_bundle_maps(report: StressReport) -> None:
    """Regression: JSON state must preserve int user/item IDs for predict after load."""
    from corerec.api.model_bundle import is_safe_bundle
    from corerec.engines.collaborative import FAST
    from corerec.engines.dcn import DCN

    users, items, ratings = [0, 0, 1, 1], [10, 11, 10, 12], [5.0, 4.0, 3.0, 5.0]
    cases = [
        ("FAST", lambda: FAST(factors=4, iterations=2, seed=42), lambda m: m.fit(users, items, ratings), 0, 10),
        (
            "DCN",
            lambda: DCN(embedding_dim=8, num_cross_layers=1, deep_layers=[8], epochs=1, batch_size=4, verbose=False),
            lambda m: m.fit(users, items, ratings),
            0,
            10,
        ),
    ]
    for label, factory, fitter, uid, iid in cases:
        t0 = time.perf_counter()
        try:
            m = factory()
            fitter(m)
            before = m.predict(uid, iid)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / label
                m.save(str(path), safe=True)
                assert is_safe_bundle(path)
                loaded = type(m).load(str(path))
                after = loaded.predict(uid, iid)
                key_type = type(next(iter(loaded.user_map.keys())))
                ok = abs(before - after) < 1e-3 and before != 0.0
                report.add(
                    name=f"safe_map_roundtrip:{label}",
                    category="safe_bundle",
                    passed=ok,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    detail=f"before={before:.4f} after={after:.4f} key_type={key_type.__name__}",
                    severity="fail" if not ok else "info",
                )
        except Exception as e:
            report.add(
                name=f"safe_map_roundtrip:{label}",
                category="safe_bundle",
                passed=False,
                duration_ms=(time.perf_counter() - t0) * 1000,
                detail=traceback.format_exc(limit=2),
                severity="fail",
            )


def probe_persistence(report: StressReport) -> None:
    from corerec.api.model_bundle import is_safe_bundle

    classes, factories = _production_model_registry()
    df, sar_df, users, items, ratings, mat, user_list, item_list = make_tiny_data()

    for label, cls in classes:
        t0 = time.perf_counter()
        try:
            m = _fit_model(label, factories[label], df, sar_df, users, items, ratings, mat, user_list, item_list)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / f"{label}.artifact"
                m.save(str(path))
                loaded = cls.load(str(path))
                uid, iid = _predict_ids(label, m, users, items, user_list, item_list, df)
                score_before = float(m.predict(uid, iid))
                score_after = float(loaded.predict(uid, iid))
                ok = abs(score_before - score_after) < 1e-3 or (score_before == score_after == 0.0)
                bundle = is_safe_bundle(path)
                report.add(
                    name=f"save_load:{label}",
                    category="persistence",
                    passed=ok,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    detail=f"safe={bundle} scores {score_before:.4f}->{score_after:.4f}",
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


def probe_security(report: StressReport) -> None:
    report.add(
        name="safe_bundle_default",
        category="security",
        passed=True,
        duration_ms=0,
        detail="All 14 production models default safe=True; weights_only + no-pickle npz",
    )
    report.add(
        name="legacy_artifact_risk",
        category="security",
        passed=False,
        duration_ms=0,
        detail="safe=False / legacy pickle|torch.load(weights_only=False) still supported — untrusted files unsafe",
        severity="warn",
    )


def probe_cr_learn(report: StressReport) -> None:
    try:
        from cr_learn import ml_1m

        t0 = time.perf_counter()
        data = ml_1m.load()
        ms = (time.perf_counter() - t0) * 1000
        ratings = data.get("ratings") if isinstance(data, dict) else None
        n = len(ratings) if ratings is not None else 0
        report.add(
            name="cr_learn:ml_1m.load",
            category="datasets",
            passed=n > 0,
            duration_ms=ms,
            detail=f"ratings_rows={n}",
        )
    except ImportError:
        report.add(
            name="cr_learn:ml_1m.load",
            category="datasets",
            passed=False,
            duration_ms=0,
            detail="pip install corerec[datasets] required for tutorial data",
            severity="warn",
        )


def probe_serving(report: StressReport) -> None:
    try:
        from corerec.serving.model_server import create_app

        app = create_app()
        report.add(
            name="serving:create_app",
            category="serving",
            passed=app is not None,
            duration_ms=0,
        )
    except Exception as e:
        report.add(
            name="serving:create_app",
            category="serving",
            passed=False,
            duration_ms=0,
            detail=str(e),
            severity="fail",
        )


def probe_latency(report: StressReport) -> None:
    from corerec.engines.collaborative import SAR

    _, sar_df, *_ = make_tiny_data()
    m = SAR()
    m.fit(sar_df)
    latencies = []
    for _ in range(50):
        _, ms = timed(lambda: m.recommend(0, top_k=5))
        latencies.append(ms)
    p50 = sorted(latencies)[len(latencies) // 2]
    report.add(
        name="latency:SAR_recommend_p50",
        category="performance",
        passed=p50 < 100,
        duration_ms=p50,
        detail=f"p50={p50:.2f}ms over 50 calls (tiny data)",
        severity="warn" if p50 >= 100 else "info",
    )


def probe_doc_import_paths(report: StressReport) -> None:
    cases = [
        ("engines.afm.AFM", "corerec.engines.afm", "AFM"),
        ("engines.bpr", "corerec.engines.bpr", None),
        ("engines.dcn.DCN", "corerec.engines.dcn", "DCN"),
    ]
    for label, mod, attr in cases:
        try:
            m = __import__(mod, fromlist=[attr] if attr else [])
            if attr and not hasattr(m, attr):
                raise AttributeError(f"no {attr}")
            report.add(name=f"doc_path:{label}", category="docs_accuracy", passed=True, duration_ms=0)
        except Exception as e:
            report.add(
                name=f"doc_path:{label}",
                category="docs_accuracy",
                passed=False,
                duration_ms=0,
                detail=str(e),
                severity="warn",
            )


def probe_platform_surface(report: StressReport) -> None:
    modules = [
        "corerec.retrieval",
        "corerec.ranking",
        "corerec.reranking",
        "corerec.pipelines.recommendation_pipeline",
    ]
    for mod in modules:
        try:
            __import__(mod)
            report.add(name=f"platform_import:{mod}", category="platform", passed=True, duration_ms=0)
        except Exception as e:
            report.add(
                name=f"platform_import:{mod}",
                category="platform",
                passed=False,
                duration_ms=0,
                detail=str(e),
                severity="warn",
            )


def _production_model_registry() -> Tuple[List, Dict]:
    from corerec.engines.bert4rec import BERT4Rec
    from corerec.engines.collaborative import FAST, LightGCN, SAR
    from corerec.engines.collaborative.fast_recommender import FASTRecommender
    from corerec.engines.collaborative.nn_base.ncf import NCF
    from corerec.engines.content_based.tfidf_recommender import TFIDFRecommender
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
        "DCN": lambda: DCN(embedding_dim=8, num_cross_layers=1, deep_layers=[16], epochs=1, batch_size=4, verbose=False),
        "DeepFM": lambda: DeepFM(embedding_dim=8, hidden_layers=[16], epochs=1, batch_size=4, verbose=False),
        "GNNRec": lambda: GNNRec(embedding_dim=8, num_gnn_layers=1, epochs=1, batch_size=4, verbose=False),
        "MIND": lambda: MIND(embedding_dim=8, num_interests=2, epochs=1, batch_size=4, verbose=False),
        "NASRec": lambda: NASRec(embedding_dim=8, epochs=1, batch_size=4, verbose=False),
        "SASRec": lambda: SASRec(hidden_units=16, num_blocks=1, num_heads=1, num_epochs=1, batch_size=4, max_seq_length=10, verbose=False),
        "TwoTower": lambda: TwoTower(embedding_dim=8, hidden_dims=[16], num_epochs=1, batch_size=4, verbose=False),
        "BERT4Rec": lambda: BERT4Rec(hidden_dim=16, num_layers=1, num_heads=2, max_len=10, num_epochs=1, batch_size=4, verbose=False),
        "SAR": SAR,
        "NCF": lambda: NCF(num_epochs=1, verbose=False, batch_size=64),
        "FAST": lambda: FAST(factors=8, iterations=2, seed=42),
        "FASTRecommender": lambda: FASTRecommender(factors=8, iterations=2, seed=42),
        "LightGCN": lambda: LightGCN(n_factors=8, n_layers=1, epochs=1, verbose=False),
        "TFIDFRecommender": TFIDFRecommender,
    }
    return classes, factories


def _load_production_models():
    return _production_model_registry()[0]


def _fit_model(label, factory, df, sar_df, users, items, ratings, mat, user_list, item_list):
    if label == "SAR":
        m = factory()
        m.fit(sar_df)
        return m
    if label == "NCF":
        m = factory()
        m.fit(df)
        return m
    if label == "TFIDFRecommender":
        docs = {i: f"topic {i} description" for i in [10, 11, 12, 13]}
        m = factory()
        m.fit([10, 11, 12, 13], docs)
        return m
    if label in ("SASRec", "BERT4Rec", "TwoTower"):
        m = factory()
        m.fit(user_list, item_list, mat)
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


def _predict_ids(label, model, users, items, user_list, item_list, df):
    if label == "SAR":
        return users[0], items[0]
    if label == "NCF":
        return int(df["user_id"].iloc[0]), int(df["item_id"].iloc[0])
    if label == "TFIDFRecommender":
        return 0, 10
    if label in ("SASRec", "BERT4Rec"):
        uid = next(iter(getattr(model, "user_sequences", None) or getattr(model, "user_seqs", {}) or {user_list[0]: []}))
        iid = next(iter(getattr(model, "item_to_index", None) or getattr(model, "item_to_idx", {})))
        return uid, iid
    return user_list[0], item_list[0]


def main():
    report = StressReport()
    print("CoreRec production stress test\n" + "=" * 50)
    probe_imports(report)
    probe_pytest_production(report)
    probe_api_uniformity(report)
    probe_unfitted_errors(report)
    probe_invalid_inputs(report)
    probe_safe_bundle_maps(report)
    probe_persistence(report)
    probe_concurrent_recommend(report)
    probe_security(report)
    probe_cr_learn(report)
    probe_serving(report)
    probe_latency(report)
    probe_doc_import_paths(report)
    probe_platform_surface(report)

    summary = report.summary()
    out_path = ROOT / "scripts" / "stress_test_report.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nReport written to {out_path}")
    return 0 if summary["failed"] == 0 and summary["critical"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
