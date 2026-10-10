# Advanced Usage Examples

Run the setup first, then the examples below in the same Python session.
The small dataset makes each example runnable without a download.

```python
import numpy as np
from corerec.engines import ItemKNN

users = np.array([1, 1, 2, 2, 3, 3])
items = np.array([10, 20, 20, 30, 10, 30])
model = ItemKNN().fit(users, items)
```

## Model Ensembles

Combine retrieval sources using reciprocal rank fusion. Fit each source before
combining them; this merges candidate rankings rather than neural model weights.

```python
from corerec.retrieval import CollaborativeRetriever, EnsembleRetriever, PopularityRetriever

collaborative = CollaborativeRetriever(model=model)
popular = PopularityRetriever().fit([10, 20, 30], interaction_counts=[2, 2, 2])
ensemble = EnsembleRetriever([("collaborative", collaborative, 1.),
                              ("popular", popular, .5)], strategy="rrf").fit()
result = ensemble.retrieve(1, top_k=2)
assert len(result.candidates) == 2
```

Popularity may return items already seen by a known user. Apply your application's
exclusions when mixing sources if that policy is required.

## Hyperparameter Tuning

Keep validation interactions out of training. This tiny split has one held-out
item per user; use a chronological split and a larger catalog for an actual
model comparison.

```python
from sklearn.model_selection import ParameterGrid

validation = {1: {30}, 2: {10}, 3: {20}}
def hit_rate(candidate):
    return np.mean([bool(set(candidate.recommend(user, top_k=1)) & relevant)
                    for user, relevant in validation.items()])

best_score, best_params = -np.inf, None
for params in ParameterGrid({"top_k_neighbors": [1, 2], "shrink": [0., 1.]}):
    candidate = ItemKNN(**params).fit(users, items)
    score = hit_rate(candidate)
    if score > best_score:
        best_score, best_params = score, params
assert best_params is not None
```

## Cross-Validation

Splitting random rows can leak future interactions into training. For rating
prediction, keep a validation split separate; for sequential recommendation,
split each user's history chronologically. The example above defines explicit
held-out items rather than evaluating the interactions used to fit the model.

## Custom Evaluation Metrics

`hit_rate()` above measures whether the top recommendation contains a held-out
item. It weights each user once. Count unique users when aggregating metrics;
iterating interaction rows would overweight users with longer histories.

## Batch Processing

`batch_predict()` takes `(user, item)` pairs. Batch recommendation returns one
entry per user.

```python
scores = model.batch_predict([(1, 30), (2, 10)])
recommendations = model.batch_recommend([1, 2], top_k=1)
assert len(scores) == 2
assert len(recommendations) == 2
```

## Model Persistence

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from corerec.serving import ModelLoader

with TemporaryDirectory() as directory:
    path = Path(directory) / "itemknn"
    model.save(path)
    restored = ModelLoader().load(path)
    assert restored.recommend(1, top_k=1) == model.recommend(1, top_k=1)
```

See [model persistence](../user_guide/model_persistence.md) for trusted legacy
migration and custom neural modules.

## Handling Cold Start

Unknown users have no collaborative history. Select a popularity fallback
explicitly, using counts from training data:

```python
user_id = 999
if user_id in model.user_map:
    recommendations = model.recommend(user_id, top_k=2)
else:
    recommendations = [candidate.item_id for candidate
                       in popular.retrieve(user_id, top_k=2).candidates]
assert len(recommendations) == 2
```

## Performance Optimization

Measure training and serving separately. For neural models, choose a device
through the model constructor:

```python
from corerec.engines import DCN
from corerec.device import resolve_device

neural = DCN(embedding_dim=8, deep_layers=[8], epochs=1,
             device=str(resolve_device("auto")))
neural.fit(users, items, np.ones(len(users)))
```

Keep concurrent training jobs within the memory available on the chosen device.
Separate processes each allocate their own model and data; more workers can
increase memory use without improving throughput.

## See Also

- [Basic usage](basic_usage.md)
- [Production deployment](production_deployment.md)
- [Tutorials](../tutorials/index.md)
