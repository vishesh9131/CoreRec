# Matrix Factorization & Collaborative Filtering

Classic and neural collaborative filtering models that learn user/item representations from interactions.

## Production models (CI-tested)

| Model | Type | Import | Tutorial |
|-------|------|--------|----------|
| **SAR** | Item similarity / co-occurrence | `from corerec.engines.collaborative import SAR` | [SAR](../tutorials/sar_tutorial.md) |
| **ALS** | Implicit matrix factorization | `from corerec.engines import ALS` | below |
| **EASE** | Closed-form item-item linear model | `from corerec.engines import EASE` | below |

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

### ALS (triplet API)

```python
from corerec.engines import ALS

user_ids = [0, 0, 1, 1, 2]
item_ids = [10, 11, 10, 12, 11]
ratings = [5.0, 4.0, 3.0, 5.0, 2.0]

model = ALS(factors=64, iterations=15)
model.fit(user_ids, item_ids, ratings)
recs = model.recommend(user_id=0, top_k=10)
score = model.predict(user_id=0, item_id=10)
```

### EASE (triplet API)

```python
from corerec.engines import EASE

model = EASE(reg=250.0)
model.fit(user_ids, item_ids, ratings)
recs = model.recommend(user_id=0, top_k=10)
```

NCF and FAST were removed in 0.7.0; ALS and EASE cover the same ground and are
CI-tested.

## Sandbox models (experimental)

| Model | Import | Tutorial |
|-------|--------|----------|
| ALS | `corerec.sandbox.collaborative_full.mf_base.als_base` | [ALS](../tutorials/removed_models.md) |
| SVD | `corerec.sandbox.collaborative_full.mf_base.svd_base` | [SVD](../tutorials/removed_models.md) |
| A2SVD | `corerec.sandbox.collaborative_full.mf_base.a2svd_base` | [A2SVD](../tutorials/removed_models.md) |
| MF-Base | `corerec.sandbox.collaborative_full.mf_base.matrix_factorization_base` | [MF Base](../tutorials/removed_models.md) |
| FM-Base | `corerec.sandbox.collaborative_full.nn_base.FM_base` | [FM Base](../tutorials/removed_models.md) |
| Matrix Factorization | `corerec.sandbox.collaborative_full.mf_base.matrix_factorization_base` | [Matrix Factorization](../tutorials/removed_models.md) |
| User-Based CF | `corerec.sandbox.collaborative_full.mf_base.user_based_base` | [User-Based](../tutorials/removed_models.md) |

```{admonition} Legacy paths
:class: note
Old docs referenced `corerec.engines.unionizedFilterEngine.*` — that module layout no longer exists. Use production imports above or sandbox paths from tutorials.
```

## When to use

| Scenario | Start with |
|----------|------------|
| Fast baseline, implicit feedback | **SAR** |
| Matrix factorization, implicit feedback | **ALS** |
| Strong item-item baseline, small/medium catalogues | **EASE** |
| Research / classic MF | Sandbox SVD, A2SVD |

## See also

- [Model tiers](index.md#model-tiers)
- [Graph-based models](graph_based.md) (LightGCN)
- [Tutorials](../tutorials/index.md)
