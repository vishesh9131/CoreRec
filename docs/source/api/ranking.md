# Ranking

Stage 2 of the recommendation pipeline. Rankers score candidates with precision, narrowing thousands to tens.

## Quick Start

```python
from corerec.ranking import PointwiseRanker
from corerec.retrieval.base import Candidate

# score_fn gets each candidate's features ("retrieval_score" plus whatever a
# feature_extractor(candidate, context) adds) and returns a float
ranker = PointwiseRanker(score_fn=lambda f: f.get("retrieval_score", 0)).fit()
candidates = [Candidate(item_id=1, score=0.2), Candidate(item_id=2, score=0.9)]
result = ranker.rank(candidates, {"user_id": 7})
print(result)   # item 2 first
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
