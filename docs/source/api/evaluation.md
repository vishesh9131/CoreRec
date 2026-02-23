# Evaluation

Comprehensive metrics and model evaluation for recommendation systems.

## Quick Start

```python
from corerec.evaluation import Evaluator

evaluator = Evaluator(metrics=["ndcg@10", "map@10", "recall@20"])
results = evaluator.evaluate(model, test_data)
```

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
