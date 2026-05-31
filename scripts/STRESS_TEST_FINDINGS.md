# CoreRec Production Stress Test — Engineering Findings

**Date:** 2026-06-01  
**Environment:** `conda activate corerec` · Python 3.10 · PyTorch 2.2.2 · CoreRec 0.5.2  
**Method:** Automated probes (`scripts/stress_test_production.py`) + pytest production/CI suites + manual parity checks  
**Benchmark bar:** PyTorch / TensorFlow (ML platform maturity) · LangChain (DX + production ops)  
**Report artifact:** `scripts/stress_test_report.json`

---

## Executive Summary

| Dimension | Grade | vs PyTorch/TF | vs LangChain |
|-----------|-------|---------------|--------------|
| **Production recommender core (14 models)** | **A** | Narrower scope (recsys only) but unified `fit/predict/recommend/save/load` | N/A — not an agent framework |
| **API consistency** | **A** | All 14 inherit `BaseRecommender`; `top_k` on recommend; `ModelNotFittedError` 14/14 | Better typed errors than typical LangChain chains |
| **Persistence / serialization** | **A-** | Safe bundle default; int ID map roundtrip fixed; predict parity in CI | LangChain had similar pickle legacy issues (improving) |
| **Testing & CI** | **A-** | **68/68** production tests green; platform smoke in CI; **57%** repo coverage | Stronger than most OSS recsys libs |
| **Serving & platform** | **B-** | FastAPI `ModelServer` works in CI; retrieval/ranking/reranking smoke only | No hosted/cloud story |
| **Documentation / DX** | **B** | Safe bundle docs shipped; legacy sandbox paths still drift | LangChain docs volume still ahead |
| **Security / ops** | **B** | Safe default on save; legacy `safe=False` paths remain; 92 Dependabot alerts open | Same artifact-trust class of issues |
| **Overall production readiness** | **A-** | Ready for **internal batch + API serving** of the 14 production models | Good library, not a platform |

**Overall grade: A-** (up from B+ pre-P0). Core recsys path is production-capable; platform layer and test debt are the remaining gaps.

---

## Probe Results (58 checks)

| Category | Pass | Warn | Fail |
|----------|------|------|------|
| imports | 8 | 0 | 0 |
| api_uniformity | 14 | 0 | 0 |
| error_handling (ModelNotFittedError) | 14 | 0 | 0 |
| input_validation (SAR) | 3 | 0 | 0 |
| safe_bundle (FAST/DCN map roundtrip) | 2 | 0 | 0 |
| persistence (via pytest parity) | 1 | 0 | 0 |
| concurrency (SAR 4×20 threads) | 1 | 0 | 0 |
| performance (SAR p50 < 100ms) | 1 | 0 | 0 |
| platform imports | 4 | 0 | 0 |
| ci_contract (production pytest) | 1 | 0 | 0 |
| security | 1 | 1 | 0 |
| docs_accuracy | 1 | 3 | 0 |
| serving | 0 | 0 | 1* |
| legacy test collection | 0 | 1 | 0 |

\* Stress script calls non-existent `create_app()`; real API is `ModelServer(model).app` — **CI serving smoke passes**.

---

## What Passed (Production-Grade)

### 1. Contract tests — **68/68 green**

```bash
python -m pytest tests/test_all_production_models.py \
  tests/test_production_contract.py tests/test_safe_persistence.py \
  tests/test_api_uniformity.py tests/test_serving_smoke.py \
  tests/test_platform_stages.py tests/test_pipeline_integration.py -q
```

- All **14 production models**: import → fit → predict → recommend → save/load
- **Predict score parity** after reload on every model (`abs(before - after) < 1e-3`)
- All inherit `BaseRecommender`
- `ModelNotFittedError` on unfitted `predict()` for all 14 models
- `top_k` unified across recommend APIs

### 2. Safe serialization (`corerec_safe_v1`) — P0 fixed

- **Format:** `{base}.meta.json` + `.weights.pt` (torch) or `.arrays.npz` (numpy/sparse)
- **Security:** `weights_only=True` on torch load; `allow_pickle=False` on npz
- **Int ID maps:** DCN, FAST, FASTRecommender, DeepFM use `save_map_state()` / `load_map_state()` + `coerce_id()`

**Manual verification:**
```
FAST  predict(0,10): 4.238 → 4.238  user_map keys: int
DCN   predict(0,10): 0.508 → 0.508  user_map keys: int
```

### 3. Input validation (SAR)
- Unknown users (`99999`), `None`, `""` raise errors — no silent garbage lists

### 4. Concurrency
- 4 threads × 20 SAR `recommend()` calls — no races (read-only inference)

### 5. Datasets
- `cr_learn.ml_1m.load()` → **1,000,209** ratings when `[datasets]` extra installed

### 6. Serving stack
- `ModelServer(SAR).app` + FastAPI TestClient → `/recommend` returns JSON (CI green)
- NumPy int64 JSON-safe in responses

### 7. Platform stage smoke (new)
- `PopularityRetriever`, `PointwiseRanker`, `DiversityReranker` — basic retrieve → rank → rerank path
- `RecommendationPipeline` orchestrator — mock integration test green

### 8. CI pipeline
- Python 3.9 / 3.10 / 3.11 matrix
- Production + contract + safe persistence + platform smoke
- Ruff syntax gate (E9/F63/F7/F82)
- Dependabot configured (pip weekly + github-actions)

### 9. Performance (tiny data)
- SAR `recommend(top_k=5)` p50 **~0.09 ms**, p95 **~0.10 ms** on 4×4 matrix

---

## Warnings (Non-Blocking but Real)

### P1 — Legacy artifact paths still unsafe (by design)

- `safe=False` → pickle / `torch.load(weights_only=False)` still supported on several models
- Untrusted artifacts = RCE risk (same class as pre-2024 LangChain loaders)
- **Mitigation shipped:** safe default + migration docs at `docs/source/user_guide/safe_bundle_persistence.md`

### P1 — Full test suite collection broken (13 modules)

CI step `pytest tests/ --ignore=tests/unionizedFilterEngine` **cannot collect** due to:

```
tests/test_FM_base.py, test_FFM_base.py, test_GNN_base.py, …
tests/contentFilterEngine/all_algorithms_test.py
tests/test_integration.py
```

Root causes: missing `corerec.engines.contentFilterEngine.*`, broken sandbox imports, syntax error in `unionizedFilterEngine/nn_base_import_test.py`.

**Impact:** CI “full suite” job may fail on collection even when production tests pass. Production path is isolated and green.

### P1 — Documentation integrity tests

- `tests/test_docs.py` fails locally without `pip install corerec[docs]` (mkdocs missing)
- AFM/BPR modules (`corerec.engines.afm`, `corerec.engines.bpr`) **do not exist** — sandbox-only; stress probe flags as doc drift

### P2 — Sandbox surface

```
Warning: Could not import some sandbox modules: No module named 'corerec.engines.collaborative.mf_base'
```

Sandbox is research/experimental — not production-proven.

### P2 — Platform layer coverage still thin

| Module | Coverage | CI proof |
|--------|----------|----------|
| retrieval (popularity) | ~20% | 1 smoke test |
| ranking (pointwise) | ~20% | 1 smoke test |
| reranking (diversity) | ~15% | 1 smoke test |
| vector_store, dssm, model_retriever | **0%** | none |
| pipelines orchestrator | partial | mock integration |

Compare to PyTorch: `torch.distributed` has extensive integration tests. CoreRec platform layer is **beta**.

### P2 — Dependency hygiene

- GitHub reports **92 vulnerabilities** on default branch (4 critical) — Dependabot config added; resolution via PR backlog
- `httpx` / Starlette deprecation warning in FastAPI TestClient

### P2 — No export path for serving at scale

- No first-class ONNX / TorchScript export for TwoTower, DCN, etc.
- PyTorch/TF standard: Triton, TF Serving, ONNX Runtime — CoreRec stops at FastAPI wrapper

### P3 — Stress script operational issues

- `probe_persistence()` retrains all 14 models sequentially — **>15 min**, appears hung after NCF on some hardware
- `probe_serving()` calls wrong API (`create_app` vs `ModelServer`) — false fail in JSON report
- **Workaround:** rely on pytest save/load parity tests (authoritative)

### P3 — API rough edges

- FAST legacy `.npy` path mental model vs safe bundle base path
- SASRec / BERT4Rec still support legacy pickle-instance fallback when safe bundle missing
- BERT4Rec attn_mask deprecation warning from PyTorch

---

## Comparison to Industry Frameworks

### vs PyTorch / TensorFlow

| Capability | PyTorch/TF | CoreRec |
|------------|------------|---------|
| Model zoo + training | ✅ Massive | ✅ 14 production rec models |
| Unified save/load security | ✅ `weights_only` (recent) | ✅ Safe bundle default |
| Distributed training | ✅ | ❌ Single-process only |
| Serving (Triton, TF Serving) | ✅ | ⚠️ Basic FastAPI wrapper |
| Export (ONNX, SavedModel) | ✅ | ❌ |
| Test coverage culture | ✅ Extensive | ⚠️ 57% repo; platform thin |
| Benchmark suite | ✅ | ❌ No MLPerf-style recsys bench |

**Verdict:** CoreRec is a **specialized recsys layer on PyTorch**. Compare to **Microsoft Recommenders / RecBole**, not raw PyTorch.

### vs LangChain

| Capability | LangChain | CoreRec |
|------------|-----------|---------|
| Agent / chain abstraction | ✅ | ❌ |
| Production observability | ⚠️ Improving | ⚠️ Basic logging |
| Serialization safety | ⚠️ Historically pickle-heavy | ✅ Safe bundle default |
| Documentation volume | ✅ Huge | ⚠️ Prod tutorials good; sandbox drift |
| Error types | ⚠️ Generic | ✅ `ModelNotFittedError`, `RecommendationError` |
| Pipeline orchestration | ✅ LCEL / Runnable | ⚠️ `RecommendationPipeline` alpha |

**Verdict:** Different domains. CoreRec **wins on recsys API clarity and typed errors**; LangChain wins on ecosystem, examples, and observability tooling.

---

## Recommended Roadmap

| Priority | Item | Status |
|----------|------|--------|
| P0 | Int ID map roundtrip (DCN/FAST/DeepFM) | ✅ Done |
| P1 | Predict parity in every `test_save_load` | ✅ Done |
| P1 | Safe bundle spec + migration docs | ✅ Done |
| P2 | Platform stage CI smoke | ✅ Done |
| P2 | Dependabot config + baseline dep bumps | ✅ Done (alert backlog open) |
| **P1** | Fix CI full-suite collection (ignore or repair 13 broken test modules) | 🔲 Open |
| **P1** | Fix stress script (`ModelServer`, skip redundant persistence retrain) | 🔲 Open |
| **P2** | Expand platform tests (vector store, ensemble retriever, DSSM) | 🔲 Open |
| **P2** | Resolve Dependabot alert backlog (92) | 🔲 Open |
| **P3** | Benchmark suite (MovieLens-1M train time + serving latency SLA) | 🔲 Open |
| **P3** | ONNX export for TwoTower / DCN | 🔲 Open |

---

## How to Reproduce

```bash
conda activate corerec
cd CoreRec

# Production contract (must pass)
python -m pytest tests/test_all_production_models.py \
  tests/test_production_contract.py tests/test_safe_persistence.py \
  tests/test_api_uniformity.py tests/test_serving_smoke.py \
  tests/test_platform_stages.py tests/test_pipeline_integration.py -q

# Automated stress probes (~40s fast path; full script can take 15+ min)
python scripts/stress_test_production.py
# → scripts/stress_test_report.json

# P0 manual spot-check
python -c "
from corerec.engines.collaborative import FAST
import tempfile, os
m=FAST(factors=4,iterations=2,seed=42)
m.fit([0,0,1],[10,11,10],[5.,4.,3.])
print('before', m.predict(0,10))
p=os.path.join(tempfile.mkdtemp(),'f'); m.save(p)
print('after ', FAST.load(p).predict(0,10))
"

# Full repo coverage (excludes broken legacy if you add ignores)
python -m pytest tests/ --ignore=tests/unionizedFilterEngine \
  --cov=corerec --cov-fail-under=40 -q
```

---

## Bottom Line

CoreRec is **production-ready for the 14 documented models** in batch and lightweight API serving scenarios. The P0 safe-bundle ID map bug is **fixed and regression-tested**. CI production suite is **68/68 green** with predict parity.

**Ship today:** SAR, NCF, LightGCN, GNNRec, MIND, NASRec, TwoTower, BERT4Rec, SASRec, DCN, DeepFM, FAST, TFIDF — all with safe bundles.

**Before FAANG-scale multi-tenant serving:** expand platform test coverage, add benchmarks, consider ONNX export, and burn down legacy test/doc debt.

**Overall: A-** for recsys library production use · **B** for full-platform maturity vs PyTorch/LangChain.
