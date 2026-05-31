# Graph-Based Models

Models that treat users and items as nodes in a bipartite interaction graph.

## Production models (CI-tested)

| Model | Import | Tutorial |
|-------|--------|----------|
| **GNNRec** | `from corerec.engines.gnnrec import GNNRec` | [GNNRec](../tutorials/gnnrec_tutorial.md) |
| **LightGCN** | `from corerec.engines.collaborative import LightGCN` | [LightGCN](../tutorials/lightgcn_tutorial.md) |

### GNNRec

Graph neural message passing on the user–item bipartite graph. Uses BCE loss — pass **0/1 labels** (or normalize explicit ratings to [0, 1]).

```python
from corerec.engines.gnnrec import GNNRec

model = GNNRec(
    embedding_dim=64,
    num_gnn_layers=3,
    epochs=20,
    batch_size=256,
    verbose=True,
)
model.fit(user_ids=user_ids, item_ids=item_ids, ratings=binary_ratings)
recs = model.recommend(user_id=1, top_k=10)
```

### LightGCN

Simplified graph convolution for collaborative filtering.

```python
from corerec.engines.collaborative import LightGCN

model = LightGCN(n_factors=64, n_layers=3, epochs=100, verbose=True)
model.fit(user_ids, item_ids, ratings)  # triplet implicit feedback
recs = model.recommend(user_id=1, top_k=10)
```

## Sandbox models (experimental)

| Model | Import | Tutorial |
|-------|--------|----------|
| GeoIMC | sandbox graph_based | [GeoIMC](../tutorials/geoimc_tutorial.md) |
| LightGCN-Base | `corerec.sandbox.collaborative_full.graph_based_base.lightgcn_base` | [LightGCN Base](../tutorials/lightgcn_base_tutorial.md) |
| GNN-Base | `corerec.sandbox.collaborative_full.graph_based_base.GNN_base` | [GNN Base](../tutorials/gnn_base_tutorial.md) |

## When to use

- Rich user–item interaction graphs
- Social or knowledge-graph extensions (sandbox)
- When CF matrix methods underperform on link structure

## See also

- [Deep learning models](deep_learning.md) (GNNRec)
- [Matrix factorization](matrix_factorization.md)
- [Model index](models_index.md)
