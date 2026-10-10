# Evaluation

Comprehensive metrics and model evaluation for recommendation systems.

## Quick Start: one-call `evaluate`

`corerec.evaluation.evaluate` scores a fitted model on held-out interactions via
the model's own `recommend` path and returns the standard top-K ranking metrics:

```python
from corerec.evaluation import evaluate

results = evaluate(
    model,
    test_interactions=test_df,        # DataFrame or (user, item[, rating]) tuples
    train_interactions=train_df,      # excluded from each user's recommendations
    k=10,
    relevance_threshold=4.0,          # rating >= 4 counts as relevant (optional)
)
# {'NDCG@10': ..., 'MAP@10': ..., 'MRR@10': ...,
#  'Precision@10': ..., 'Recall@10': ..., 'HitRate@10': ..., 'n_users': ...}
```

For finer control (cross-validation, custom metric sets) use the `Evaluator` class:

```python
from corerec.evaluation import Evaluator

evaluator = Evaluator(metrics=["NDCG@10", "MAP@10", "Recall@20"])
results = evaluator.evaluate(model, test_data)   # test_data: {user: [items]}
```

`Evaluator` calls `evaluate()` underneath, so the numbers match what
`corerec train`, `serve` and `retrain` report. Pass
`train_interactions=` to remove seen items the same way; without it, that is
left to the model. Results use the keys above (`NDCG@10`, ...), and also the
spelling you asked for, so `"ndcg@10"` works too.

## API Reference

```{eval-rst}
.. automodule:: corerec.evaluation
   :members:
   :show-inheritance:
```

### Metrics

```{eval-rst}
.. automodule:: corerec.evaluation.metrics
   :members:
```

### Evaluator

```{eval-rst}
.. automodule:: corerec.evaluation.evaluator
   :members:
   :show-inheritance:
```
