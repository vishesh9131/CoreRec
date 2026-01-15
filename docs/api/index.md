# API Reference

CoreRec reference documentation is organized by functional module.

## Core Modules

*   [**Base Recommender**](base-recommender.md): The abstract base class `BaseRecommender` that all models inherit from.
*   [**Engines**](../engines/index.md):
    *   `corerec.engines.collaborative`
    *   `corerec.engines.content_based`
    *   `corerec.engines` (Deep Learning models)

## Utilities

*   [**VishGraphs**](../utilities/visualization.md): Graph generation and plotting.
*   [**Metrics**](../utilities/evaluation-metrics.md): `aaj_accuracy`, `precision_at_k`, etc.

## Model Interface

All models in CoreRec follow a standard interface:

```python
class Model(BaseRecommender):
    def fit(self, interaction_matrix, user_ids, item_ids):
        ...
    
    def recommend(self, user_id, top_n=10):
        ...
```
