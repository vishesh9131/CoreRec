# Reranking

Stage 3 of the recommendation pipeline. Rerankers apply business rules, diversity constraints, and fairness adjustments.

## Quick Start

```python
from corerec.reranking import DiversityReranker, BusinessRulesReranker

diversity = DiversityReranker(lambda_=0.7)
business = BusinessRulesReranker()
business.add_boost(item_id=42, multiplier=2.0)
business.add_blocklist([999, 998])
```

## API Reference

```{eval-rst}
.. automodule:: corerec.reranking.base
   :members:
   :show-inheritance:
```

### Diversity Reranker

```{eval-rst}
.. automodule:: corerec.reranking.diversity
   :members:
   :show-inheritance:
```

### Fairness Reranker

```{eval-rst}
.. automodule:: corerec.reranking.fairness
   :members:
   :show-inheritance:
```

### Business Rules Reranker

```{eval-rst}
.. automodule:: corerec.reranking.business
   :members:
   :show-inheritance:
```
