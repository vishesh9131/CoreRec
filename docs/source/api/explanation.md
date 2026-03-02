# Explanation

Recommendation explainability for debugging and user trust.

## Quick Start

```python
from corerec.explanation import FeatureExplainer

explainer = FeatureExplainer(
    item_features=item_features,
    user_preferences=user_prefs,
    templates={"category": "Matches your interest in {value}"}
)
explanation = explainer.explain(item_id=42, context={"user_id": 5})
print(explanation.text)
```

## API Reference

```{eval-rst}
.. automodule:: corerec.explanation.base
   :members:
   :show-inheritance:
```

### Feature-Based Explainer

```{eval-rst}
.. automodule:: corerec.explanation.feature_based
   :members:
   :show-inheritance:
```

### Generative Explainer

```{eval-rst}
.. automodule:: corerec.explanation.generative
   :members:
   :show-inheritance:
```
