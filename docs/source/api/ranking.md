# Ranking

Stage 2 of the recommendation pipeline. Rankers score candidates with precision, narrowing thousands to tens.

## Quick Start

```python
from corerec.ranking import PointwiseRanker

ranker = PointwiseRanker(score_fn=my_score_fn, feature_extractor=my_extractor)
ranker.fit()
result = ranker.rank(candidates, context)
```

## API Reference

```{eval-rst}
.. automodule:: corerec.ranking.base
   :members:
   :show-inheritance:
```

### Pointwise Ranker

```{eval-rst}
.. automodule:: corerec.ranking.pointwise
   :members:
   :show-inheritance:
```

### Pairwise Ranker

```{eval-rst}
.. automodule:: corerec.ranking.pairwise
   :members:
   :show-inheritance:
```

### Feature Cross Ranker

```{eval-rst}
.. automodule:: corerec.ranking.feature_cross
   :members:
   :show-inheritance:
```
