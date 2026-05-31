# CoreRec API Versioning Policy

CoreRec follows **semantic versioning** (`MAJOR.MINOR.PATCH`) with explicit deprecation
cycles, similar to PyTorch and TensorFlow.

## Current API version

- Package version: see `corerec.__version__`
- Unified recommender API: **v1.0** (`BaseRecommender`)

## Stability tiers

| Tier | Path | Guarantee |
|------|------|-----------|
| **Production** | `corerec.engines.*` | Stable `fit` / `predict` / `recommend` / `save` / `load` |
| **Platform** | `corerec.pipelines`, `corerec.serving` | Beta — tested in CI with `[serving]` extra |
| **Sandbox** | `corerec.sandbox.*` | Experimental — no stability guarantee |

## Unified public contract

All **14 production models** implement:

```python
model.fit(...)                          # or fit(RecommenderDataset)
score = model.predict(user_id, item_id)
recs = model.recommend(user_id, top_k=10, exclude_items=None)
model.save(path)
loaded = ModelClass.load(path)
```

### Deprecated (removed in CoreRec 1.0)

| Legacy | Replacement |
|--------|-------------|
| `recommend(..., top_n=10)` | `recommend(..., top_k=10)` |
| `items_to_ignore` | `exclude_items` |
| `from corerec.engines.collaborative import FastRecommender` (broken) | Works; aliases `FASTRecommender` |
| Pickle-only artifacts (untrusted) | Use `safe_persistence.save_artifact` / torch `state_dict` |

Deprecation warnings use Python's `DeprecationWarning` and can be surfaced in CI:

```bash
PYTHONWARNINGS=default python -m pytest tests/test_api_uniformity.py
```

## RecommenderDataset (unified training)

```python
from corerec.api.dataset import RecommenderDataset

ds = RecommenderDataset.from_triplet(user_ids, item_ids, ratings)
model.fit(ds)

ds = RecommenderDataset.from_dataframe(train_df)
sar.fit(ds)
```

## Release checklist

1. All production model tests pass (`tests/test_all_production_models.py`)
2. API uniformity tests pass (`tests/test_api_uniformity.py`)
3. Coverage ≥ 40% (`--cov-fail-under=40`)
4. No new undocumented breaking changes without deprecation period
