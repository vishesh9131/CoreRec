# Graph-Based Models

LightGCN learns user and item embeddings by propagating them over a bipartite
interaction graph. It uses implicit feedback: a positive rating means an observed
interaction, rather than a target star rating.

## Train and recommend

```python
from corerec.engines import LightGCN

users = [1, 1, 2, 2, 3, 3]
items = [10, 20, 20, 30, 10, 30]
model = LightGCN(n_factors=8, n_layers=1, epochs=2, batch_size=4,
                 device="cpu", verbose=False)
model.fit(users, items)
assert model.recommend(1, top_k=1) == [30]
```

Training stores the interaction graph sparsely, while user and item embeddings
use memory proportional to `(users + items) * n_factors`. Increasing the number
of graph layers increases propagation work. Start with one layer and a small
embedding size before measuring a larger catalog.

## Historical implementations

GNNRec and the former sandbox graph models are unavailable in this release.
Their older examples cannot be imported. See [removed models](../tutorials/removed_models.md).

## See also

- [LightGCN tutorial](../tutorials/lightgcn_tutorial.md)
- [Matrix factorization](matrix_factorization.md)
- [Model index](models_index.md)
