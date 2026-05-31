# CoreRec Production Stress Test — Engineering Findings

**Date:** 2026-05-31  
**Environment:** `conda activate corerec` · Python 3.10 · PyTorch 2.2.2 · CoreRec 0.5.2  
**Method:** Production test suite + targeted probes (API, persistence, security, serving, datasets)  
**Benchmark bar:** PyTorch / TensorFlow (ML platform maturity) · LangChain (DX + production ops)

---

## Executive Summary

| Dimension | Grade | vs PyTorch/TF | vs LangChain |
|-----------|-------|---------------|--------------|
| **Production recommender core (14 models)** | **A-** | Narrower scope (recsys only) but unified `fit/predict/recommend/save/load` | N/A — not an agent framework |
| **API consistency** | **A** | Comparable contract discipline for model classes | Better typed errors than typical LangChain chains |
| **Persistence / serialization** | **A-** | Safe bundle + int ID map roundtrip fixed (P0) | LangChain has similar pickle legacy issues |
| **Testing & CI** | **A-** | 64+ production tests green; platform stage smoke in CI; 53% cov | Stronger than many OSS recsys libs |
| **Serving & platform** | **C+** | FastAPI smoke works; retrieval/ranking/reranking largely untested | No hosted/cloud story |
| **Documentation / DX** | **B** | Quickstart + 14 prod tutorials fixed; sandbox docs still drift | LangChain docs volume still ahead |
| **Security / ops** | **B+** | Safe default on save; migration docs shipped; Dependabot configured | Same class of artifact-trust issues |
| **Overall production readiness** | **A-** | Ready for **internal batch recsys**; platform layer smoke-tested | Good library, not a platform |

---

## What Passed (Production-Grade)

### 1. Contract tests — **64/64 green**
```bash
python -m pytest tests/test_all_production_models.py \
  tests/test_production_contract.py tests/test_safe_persistence.py \
  tests/test_api_uniformity.py tests/test_serving_smoke.py -q
```
- All **14 production models**: import → fit → predict → recommend → save/load
- All inherit `BaseRecommender`
- `ModelNotFittedError` on unfitted `predict()` for covered models
- `top_k` unified across recommend APIs

### 2. Safe serialization shipped (`corerec_safe_v1`)
- **Format:** `{base}.meta.json` + `.weights.pt` (torch) or `.arrays.npz` (numpy/sparse)
- **Security:** `weights_only=True` on torch load; `allow_pickle=False` on npz
- **Coverage gate:** 50% on production paths (40% CI minimum)

### 3. Input validation (SAR exemplar)
- Unknown users raise `RecommendationError` (not silent garbage)
- `None` / empty user_id rejected

### 4. Concurrency smoke
- 4 threads × 20 SAR `recommend()` calls — no races observed (read-only inference path)

### 5. Dataset integration
- `cr_learn.ml_1m.load()` works when `[datasets]` extra installed (~1M ratings)

### 6. Serving stack
- `corerec[serving]` → FastAPI app constructs; JSON-safe numpy int64 in responses

### 7. CI pipeline
- Python 3.9 / 3.10 / 3.11
- Production + contract + safe persistence + platform smoke
- Ruff syntax gate

---

## Critical Findings — Resolved (2026-05-31)

### ✅ P0 — Safe bundle int ID maps (DCN, FAST, FASTRecommender, DeepFM)

**Fix:** Models now use `save_map_state()` / `load_map_state()` with `*_pairs` lists and `coerce_id()` on restore. Covered by `test_safe_persistence.py` (FAST/DCN roundtrip) and predict-score parity in all 14 `test_save_load` tests.

### ✅ P1 — Predict score parity after save/load

All production model `test_save_load` methods assert `abs(predict_before - predict_after) < 1e-3`.

### ✅ P1 — Safe bundle spec + legacy migration docs

See `docs/source/user_guide/safe_bundle_persistence.md` and updated `model_persistence.md`.

### ✅ P2 — Platform stage CI smoke

`tests/test_platform_stages.py` — PopularityRetriever, PointwiseRanker, DiversityReranker.

### ✅ P2 — Dependabot cleanup (config + baseline deps)

`.github/dependabot.yml` (pip weekly + github-actions); `requests>=2.32.0`, `urllib3>=2.2.0` in `requirements.txt`. Remaining GitHub alert resolution follows Dependabot PRs.

---

## Previously Critical (for reference)

### P0 — Safe bundle breaks integer user/item IDs (FIXED)

**Repro:**
```python
from corerec.engines.collaborative import FAST
model.fit([0,0,1,1], [10,11,10,12], [5.,4.,3.,5.])
s0 = model.predict(0, 10)          # e.g. 4.24
loaded = FAST.load(path)           # safe bundle
s1 = loaded.predict(0, 10)        # 0.0 — silent failure
list(loaded.user_map.keys())       # ['0', '1'] — str keys, not int
```

Same issue on **DCN** (`user_map` / `item_map` stored as raw JSON dicts).

**Root cause:** `_jsonify()` stringifies dict keys; restore does not coerce back to int.

**Impact:** Production save/load **silently breaks inference** for numeric IDs. Current unit tests only assert `is_fitted` or `user_factors is not None` — they do **not** assert score parity.

**Models using safe `*_pairs` format:** SAR, GNNRec, MIND, NASRec, NCF, LightGCN, BERT4Rec, TwoTower, SASRec, TFIDF — **lower risk**.

**Models still storing raw dicts in JSON state:** DCN, DeepFM (feature maps), FAST, FASTRecommender — **high risk** for int IDs.

**Fix:** Use `save_map_state()` / `pairs()` everywhere; add `test_predict_score_parity_after_load` to all production models.

---

### P1 — Tests green but persistence not fully validated

| Model | Current save/load test | Gap |
|-------|------------------------|-----|
| FAST / FASTRecommender | `user_factors is not None` | No predict parity |
| DCN / DeepFM | `is_fitted` only | No recommend after load |
| Most torch models | `is_fitted` | No score drift check |

**Recommendation:** Add one assertion per model: `abs(predict_before - predict_after) < 1e-3`.

---

### P1 — Legacy artifact paths still unsafe (by design, but document loudly)

- `safe=False` → pickle / full `torch.load(weights_only=False)`
- Untrusted artifacts = RCE risk (same as pickle-based LangChain loaders pre-2024)

Safe path mitigates when used; **migration guide needed** for existing `.pkl` / `.pt` deployments.

---

## Warnings (Non-Blocking but Real)

### Documentation drift
- Tutorials reference `corerec.engines.afm`, `corerec.engines.bpr` — **modules do not exist** (sandbox-only or removed)
- 52 sandbox tutorials fixed to `corerec.sandbox.*` but AFM/BPR production paths still broken in older docs

### Sandbox / research surface
- `corerec.sandbox` import warns: `No module named 'corerec.engines.collaborative.mf_base'`
- Full `tests/` collection: **13 import errors** (contentFilterEngine / FM_base paths) — CI ignores `tests/unionizedFilterEngine` but other broken tests remain

### Platform modules (retrieval, ranking, reranking, pipelines)
- Import successfully but **~0–15% test coverage**
- `RecommendationPipeline`, vector store, ensemble retriever — **not production-proven** in CI
- Compare to PyTorch: entire `torch.distributed` tested at scale; CoreRec platform layer is **alpha**

### Performance characteristics (tiny-data probe)
- SAR `recommend(top_k=5)` p50 **< 5ms** on 4×4 data — fine for small models
- No benchmark suite for 1M-user SASRec / LightGCN training time
- No ONNX / TorchScript export path (PyTorch/TF serving standard)

### Dependency hygiene
- GitHub reports **92 vulnerabilities** on default branch (4 critical)
- `httpx` deprecation warning in FastAPI TestClient

### API rough edges
- FAST `load()` appends `.npy` when path lacks extension — conflicts mentally with safe bundle base path
- SASRec / BERT4Rec still support legacy **pickle entire instance** fallback
- DeepFM `recommend` historically had `top_n` param — verify deprecation shim in docs

---

## Comparison to Industry Frameworks

### vs PyTorch / TensorFlow
| Capability | PyTorch/TF | CoreRec |
|------------|------------|---------|
| Model zoo + training | ✅ Massive | ✅ 14 production rec models |
| Unified save/load security | ⚠️ `weights_only` recent | ✅ Safe bundle default (with P0 bug) |
| Distributed training | ✅ | ❌ Single-process only |
| Serving (TF Serving, Triton) | ✅ | ⚠️ Basic FastAPI wrapper |
| Export (ONNX, SavedModel) | ✅ | ❌ |
| Test coverage culture | ✅ Extensive | ⚠️ 50% scoped, platform untested |

**Verdict:** CoreRec is a **specialized recsys layer on PyTorch**, not a full ML platform. Appropriate comparison is **Microsoft Recommenders / RecBole**, not raw PyTorch.

### vs LangChain
| Capability | LangChain | CoreRec |
|------------|-----------|---------|
| Agent / chain abstraction | ✅ | ❌ |
| Production observability | ⚠️ Improving | ⚠️ Basic logging |
| Serialization safety | ⚠️ Historically pickle-heavy | ✅ Safe bundle (fix P0) |
| Documentation volume | ✅ Huge | ⚠️ Good prod tutorials; sandbox drift |
| Error types | ⚠️ Generic | ✅ `ModelNotFittedError`, `RecommendationError` |

**Verdict:** Different problem domain. CoreRec **wins on recsys API clarity**; LangChain wins on ecosystem and examples.

---

## Recommended Roadmap (Production Engineer Priority)

1. ~~**P0:** Fix int ID map roundtrip~~ ✅ Done  
2. ~~**P1:** Predict-score parity in every `test_save_load`~~ ✅ Done  
3. ~~**P1:** Document `corerec_safe_v1` + migration~~ ✅ Done  
4. ~~**P2:** Retrieval/ranking/reranking CI smoke~~ ✅ Done  
5. ~~**P2:** Dependabot config + baseline dep bumps~~ ✅ Done (alert backlog via Dependabot PRs)  
6. **P3:** Benchmark suite (MovieLens-1M, latency SLA for SAR/TwoTower serving)  
7. **P3:** Optional ONNX export for TwoTower / DCN serving

---

## How to Reproduce

```bash
conda activate corerec
cd CoreRec

# Production contract (must pass)
python -m pytest tests/test_all_production_models.py \
  tests/test_production_contract.py tests/test_safe_persistence.py -q

# Stress probes (updated script)
python scripts/stress_test_production.py
# → scripts/stress_test_report.json

# Manual P0 repro
python -c "
from corerec.engines.collaborative import FAST
import tempfile, os
m=FAST(factors=4,iterations=2,seed=42)
m.fit([0,0,1],[10,11,10],[5.,4.,3.])
print('before', m.predict(0,10))
p=os.path.join(tempfile.mkdtemp(),'f'); m.save(p)
print('after ', FAST.load(p).predict(0,10))
"
```

---

## Bottom Line

CoreRec has crossed from **research prototype → production-capable recsys library** for the **14 documented production models**, with CI, safe serialization intent, and a coherent API. It is **not yet** at PyTorch/TensorFlow platform maturity or LangChain ecosystem scale.

**Ship internally today:** All 14 production models with safe bundles (including DCN, FAST, DeepFM).  
**Overall grade: A-** after P0–P2 fixes.
