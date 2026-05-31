# Bayesian Models

Probabilistic and ranking-oriented matrix factorization methods.

```{admonition} Sandbox only
:class: warning
All Bayesian models in CoreRec are **sandbox / experimental**. There is no production-tier Bayesian model in the 14-model CI suite.
```

## Sandbox models

| Model | Import | Tutorial |
|-------|--------|----------|
| **BPR** | `from corerec.sandbox.collaborative_full.cornac_bpr import BPR` | [BPR](../tutorials/bpr_tutorial.md) |
| **BPR-MF** | `from corerec.sandbox.collaborative_full.bayesian_method_base.bprmf_base import BPRMF_base` | [BPR-MF](../tutorials/bprmf_tutorial.md) |
| **VMF** | `from corerec.sandbox.collaborative_full.mf_base.vmf_base import VMF_base` | [VMF](../tutorials/vmf_tutorial.md) |

### Example (BPR)

```python
from corerec.sandbox.collaborative_full.cornac_bpr import BPR

model = BPR(k=50, max_iter=100, learning_rate=0.01, lambda_reg=0.01, verbose=True)
model.fit(user_ids, item_ids)  # implicit feedback pairs
```

```{admonition} Legacy paths
:class: note
Do **not** use `corerec.engines.unionizedFilterEngine.*` or `corerec.engines.bpr` — those modules do not exist. Import from `corerec.sandbox.*` as shown in each tutorial.
```

## When to use

- Implicit feedback (clicks, views)
- Pairwise ranking optimization (BPR)
- Research prototypes requiring uncertainty (VMF)

## See also

- [Matrix factorization](matrix_factorization.md)
- [Model tiers](index.md#sandbox-models-experimental)
