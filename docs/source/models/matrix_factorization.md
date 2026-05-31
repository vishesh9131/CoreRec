# Matrix Factorization & Collaborative Filtering

Classic and neural collaborative filtering models that learn user/item representations from interactions.

## Production models (CI-tested)

| Model | Type | Import | Tutorial |
|-------|------|--------|----------|
| **SAR** | Item similarity / co-occurrence | `from corerec.engines.collaborative import SAR` | [SAR](../tutorials/sar_tutorial.md) |
| **NCF** | Neural CF (GMF + MLP) | `from corerec.engines.collaborative import NCF` | [NCF](../tutorials/ncf_tutorial.md) |
| **FAST** | Fast similarity CF | `from corerec.engines.collaborative import FAST` | [FAST](../tutorials/fast_tutorial.md) |
| **FASTRecommender** | Extended FAST | `from corerec.engines.collaborative import FASTRecommender` | [FASTRecommender](../tutorials/fast_recommender_tutorial.md) |

### SAR (DataFrame API)

SAR expects Microsoft Recommenders-style column names: `userID`, `itemID`, `rating`.

```python
from corerec.engines.collaborative import SAR
import pandas as pd

df = pd.DataFrame({
    "userID": [0, 0, 1, 1],
    "itemID": [10, 11, 10, 12],
    "rating": [5, 4, 3, 5],
})
model = SAR(similarity_type="jaccard")
model.fit(df)
recs = model.recommend(user_id=0, top_k=10)
```

### NCF (DataFrame API)

```python
from corerec.engines.collaborative import NCF

model = NCF(num_epochs=10, verbose=True)
model.fit(df)  # columns: user_id, item_id, rating
recs = model.recommend(user_id=0, top_k=10)
```

### FAST / FASTRecommender (triplet API)

```python
from corerec.engines.collaborative import FAST

model = FAST(factors=64, iterations=15, seed=42)
model.fit(user_ids, item_ids, ratings)
score = model.predict(user_id=0, item_id=10)
```

## Sandbox models (experimental)

| Model | Import | Tutorial |
|-------|--------|----------|
| ALS | `corerec.sandbox.collaborative_full.mf_base.als_base` | [ALS](../tutorials/als_tutorial.md) |
| SVD | `corerec.sandbox.collaborative_full.mf_base.svd_base` | [SVD](../tutorials/svd_tutorial.md) |
| A2SVD | `corerec.sandbox.collaborative_full.mf_base.a2svd_base` | [A2SVD](../tutorials/a2svd_tutorial.md) |
| MF-Base | `corerec.sandbox.collaborative_full.mf_base.matrix_factorization_base` | [MF Base](../tutorials/mf_base_tutorial.md) |
| FM-Base | `corerec.sandbox.collaborative_full.nn_base.FM_base` | [FM Base](../tutorials/fm_base_tutorial.md) |
| Matrix Factorization | `corerec.sandbox.collaborative_full.mf_base.matrix_factorization_base` | [Matrix Factorization](../tutorials/matrixfactorization_tutorial.md) |
| User-Based CF | `corerec.sandbox.collaborative_full.mf_base.user_based_base` | [User-Based](../tutorials/userbased_tutorial.md) |

```{admonition} Legacy paths
:class: note
Old docs referenced `corerec.engines.unionizedFilterEngine.*` — that module layout no longer exists. Use production imports above or sandbox paths from tutorials.
```

## When to use

| Scenario | Start with |
|----------|------------|
| Fast baseline, implicit feedback | **SAR** |
| Neural pairwise ranking | **NCF** |
| Low-latency similarity CF | **FAST** |
| Research / classic MF | Sandbox ALS, SVD |

## See also

- [Model tiers](index.md#model-tiers)
- [Graph-based models](graph_based.md) (LightGCN)
- [Tutorials](../tutorials/index.md)
