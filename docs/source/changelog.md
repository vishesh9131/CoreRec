# Changelog

All notable changes to CoreRec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.4] - 2026-06-05

### Added
- **Model zoo expanded 14 -> 41 production models**, all conforming to the unified
  contract (`fit`/`predict`/`recommend`/`save`/`load` + collapse guard), built on
  shared GPU-capable bases and covered by 112 CI-gated contract tests:
  - Deep CTR / feature-interaction (12): FM, AFM, NFM, DeepFM, DCN, AutoInt,
    xDeepFM, FiBiNet, PNN, WideDeep, GMF, MLP (`corerec.engines.deep_ctr`).
  - Sequential / session (6): GRU4Rec, Caser, BST, DIN, DIEN, NARM
    (`corerec.engines.sequential`).
  - Classic CF (4): ItemKNN, UserKNN, EASE, SLIM (`corerec.engines.classic_cf`).
  - Auto-encoder CF (2): MultVAE, MultiDAE (`corerec.engines.vae_cf`).
  - Graph + native MF (3): NGCF (`corerec.engines.graph_cf`), ALS/WMF and Item2Vec
    (`corerec.engines.matrix_factorization`).
  - With the larger zoo CoreRec's best model leads three of the four standard
    benchmarks (ML-100K via SLIM/UserKNN, Gowalla and Yelp2018 via LightGCN).
- `corerec.serving.OnlineRecommender` — ANN-backed online serving engine
  (sub-millisecond retrieval at million-item scale via FAISS HNSW/IVF), with
  incremental `add_items` / `fold_in_user` (freshness without retraining) and a
  graceful popularity fallback for cold-start users. See the Online Serving guide.
- Native GPU sparse trainers: an efficient sparse LightGCN and a strengthened BPR
  (`OnlineRecommender.from_interactions(df, model="lightgcn"|"bpr")`). CoreRec's
  LightGCN ranks #1 on the Gowalla and Yelp2018 large-scale graph benchmarks.
- `corerec.evaluation.evaluate(model, test, train, k=...)` — one-call standard
  top-K evaluation (NDCG/MAP/MRR/Precision/Recall/HitRate@K).
- `BaseRecommender.card()` — reproducibility/audit metadata (params + versions).
- `task` argument (`implicit`/`rating`) with negative sampling for `DCN`, `DeepFM`,
  `GNNRec`.

### Fixed
- Deep-model output collapse: `DCN`/`DeepFM`/`GNNRec` trained on observed-only data
  (all-positive labels) collapsed to a constant. Negative sampling + the `task`
  contract restore them to parity with reference implementations (e.g. DCN
  NDCG@10 0.005 → 0.296 on ML-100K). A post-fit guard warns on score collapse.
- Vectorized full-ranking inference for deep models (recommend latency
  ~236 ms → ~1.5 ms/user); removed NCF's per-row `iterrows` data prep.
- `VectorIndex` now supports inner-product HNSW, so dot-product models (MF/ALS)
  are served correctly rather than distorted by cosine normalization.
- Lazy `torch` import in `corerec.utils.seed` — pure collaborative-filtering usage
  (e.g. SAR) no longer pulls in PyTorch (~754 MB → ~124 MB resident).
- `task="rating"` models now round-trip correctly through `save`/`load`.

## [0.5.3] - 2026-03-31

### Added
- Safe bundle persistence (`corerec_safe_v1`) for all 14 production models
- Production platform tests (retrieval, ranking, reranking, serving, pipelines)
- Model catalog documentation pages

### Fixed
- Safe bundle ID map round-trip for integer keys
- Predict score parity on save/load for all production models
- Documentation alignment (QuickStart, tutorials, `top_k`, artifact paths)

### Changed
- Unified `recommend(..., top_k=)` API across production models

## [Unreleased - historical]

### Added
- LICENSE file (MIT License)
- Proper packaging configuration (pyproject.toml, setup.py in root)
- Comprehensive requirements files (requirements.txt, requirements-dev.txt, requirements-test.txt)
- CI/CD workflows using GitHub Actions (test.yml, lint.yml, docs.yml)
- CONTRIBUTING.md with detailed contribution guidelines
- Pre-commit hooks configuration (.pre-commit-config.yaml)
- Improved .gitignore to exclude build artifacts
- CRITICAL_ISSUES_ANALYSIS.md - comprehensive codebase analysis
- QUICKSTART_FIXES.md - guide for fixing critical issues
- CHANGELOG.md (this file)
- CLI tool with tab completion (`corerec` command)

### Changed
- Moved setup.py from production/ to root directory
- Updated .gitignore to follow Python best practices
- Improved package structure for better import consistency

### Fixed
- Build artifacts (*.o, *.pyc, __pycache__) cleanup from repository
- Package installation issues (setup.py now in correct location)

### Security
- Added Bandit security scanning to CI/CD
- Added Safety dependency vulnerability scanning

## [0.5.1] - 2024-10-XX

### Added
- Multiple recommendation engines (DCN, DeepFM, GNNRec, MIND, NASRec, SASRec)
- Unionized Filter Engine (collaborative filtering algorithms)
- Content Filter Engine (content-based filtering algorithms)
- Comprehensive API with BaseRecommender base class
- Extensive examples directory

### Known Issues
- Two base classes exist (BaseRecommender and BaseCorerec) - consolidation needed
- Circular import risks in core_rec.py
- Incomplete type hints
- C++ extensions not properly configured in setup
- Missing comprehensive test coverage reports

## [0.5.0] - 2024-09-XX

### Added
- Initial release of CoreRec
- Basic recommendation models
- Core API structure

---

For detailed changes, see the [GitHub Releases](https://github.com/vishesh9131/CoreRec/releases) page.

