# Changelog

All notable changes to CoreRec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-08-07

### Fixed

- **`pip install corerec` produced an install that failed on use.** `pyyaml` was
  declared in neither `pyproject.toml` nor `requirements.txt`, and `matplotlib`
  and `networkx` were missing from `pyproject.toml`. Since
  `corerec/utils/__init__.py` imports `.config`, which imports `yaml`, any
  `from corerec.utils...` raised `ModuleNotFoundError` on a clean environment.
  This is the main reason to upgrade.
- `BaseRecommender.batch_predict` was a list comprehension over `predict()`,
  meaning one forward pass per pair for a torch model. Scoring a 1650-item
  catalogue for one user took 262ms. `NCF` now overrides it with a single
  batched pass: 262ms to 6.8ms, scores identical to 6e-08.
- `ModelServer`'s `/recommend` passes `exclude_items`, which `TwoTower` and
  `BERT4Rec` did not accept, so every HTTP request against those models
  returned 500. Both now implement it, and `TwoTower.recommend(exclude_seen=)`
  actually excludes (it was accepted and never read).
- `LightGCN` sampled negatives and shuffled epochs against numpy's global RNG
  with nothing seeding it, so identical code on identical data produced
  different models. It now takes `seed` (default 42) and persists it.
- `SAR(similarity_type=...)` returned near-random rankings for `lift`,
  `mutual_information` and `inclusion_index` at the default `threshold=1`
  (NDCG@10 0.0007, 0.0015, 0.0541 against jaccard's 0.3730). Constructing one
  of those now warns, naming the cause and the remedy.
- `corerec/data/__init__.py` was empty while `data` was advertised in
  `corerec.__all__`; it now exports its 11 dataset types.
- `GNNRec` rebuilt a dense identity matrix on every mini-batch; the graph is
  now built once. It remains impractically slow -- see BENCHMARKS.md.

### Changed

- **Removed ~159k lines of vendored third-party source** (`torch_nn`,
  `torch_utils`, `cr_boosters`, `cr_pkg`, `cr_utility`, `sandbox`), which were
  byte-identical copies of PyTorch modules redistributed inside an MIT package.
  Call sites now import `torch.nn` / `torch.optim` / `torch.utils.data`
  directly. A clone is 30MB rather than 1.6GB.
- `corerec.__all__` went from 74 names to 39. The 35 removed all resolved to
  `None` because the modules behind them do not exist; the lazy loader now
  raises `AttributeError` naming the failed import instead of caching `None`.
- `corerec.vish_graphs` no longer injects 20 misspelled aliases of
  `scale_and_save_matrices` into its namespace. `vish_graphs`, `visualization`
  and `metrics` declare `__all__`, so a star-import no longer pulls torch,
  numpy, pandas, matplotlib, csv or multiprocessing into the caller's scope.
- `TwoTower.fit` and `BERT4Rec.fit` accept the `fit(user_ids, item_ids,
  ratings)` triple the rest of the zoo uses, alongside the interaction matrix
  they took before.
- `corerec.core_rec` no longer re-exports `GATConv`/`GCNConv`/`HANConv`/
  `SAGEConv`. They required `torch_geometric`, which was never a declared
  dependency, so importing the module failed on a clean install.

### Added

- `BENCHMARKS.md`: CoreRec against `implicit` on MovieLens-100K, with the
  losses included. CoreRec's ALS beats implicit's on quality (NDCG@10 0.4168 vs
  0.4100) and is ~9x slower to fit; its cosine SAR beats implicit's cosine
  ItemKNN (0.3955 vs 0.3858). Every figure has its raw JSON checked in.
- `examples/train_and_serve.py`: interactions to a live HTTP endpoint in one
  runnable file, exercised by `tests/test_train_and_serve.py` so it cannot rot.
- `tests/test_model_contract.py`: drives every production model through the
  same fit/recommend/exclude_items/save-load calls, with divergences recorded
  explicitly rather than tolerated silently.

### Known issues

- `GNNRec` does not finish training on MovieLens-100K within an hour on one
  core; `LightGCN` takes 151s on the same data.
- `bert4rec`, `two_tower` and `sasrec` are not reproducible run to run.
- Several documentation pages contain code examples importing modules that
  were removed; see `KNOWN_STALE` in `tests/test_docs_imports.py`.

## [0.5.3] - 2026-03-31

### Added
- Safe bundle persistence (`corerec_safe_v1`) for all 14 production models
- Production platform tests (retrieval, ranking, reranking, serving, pipelines)
- Model catalog documentation pages

### Fixed
- Safe bundle ID map round-trip for int keys (DCN, FAST, DeepFM, etc.)
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

## [0.5.0] and earlier

See git history for previous changes.

---

## Version Number Guide

- MAJOR version: Incompatible API changes
- MINOR version: Added functionality (backwards-compatible)
- PATCH version: Bug fixes (backwards-compatible)

[Unreleased]: https://github.com/vishesh9131/CoreRec/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/vishesh9131/CoreRec/releases/tag/v0.5.1
