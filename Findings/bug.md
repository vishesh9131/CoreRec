# CoreRec integration bugs

Found by *using* the library across layer boundaries, not by reading it. See
`BUGHUNT_PROMPT.md` for the loop that produces these.

Report only. Fixes happen in a separate session where they can be reviewed.

---

## Combinations exercised

| # | Combination | Date | Result |
|---|---|---|---|
| 1 | models × serving — ALS, Item2Vec, LightGCN, TwoTower through `/recommend`, `/predict`, `/batch/recommend` | 2026-08-08 | **1 bug** (#1); ALS/Item2Vec/LightGCN clean on all three endpoints |

---

## Open

### 1. `fit(..., ratings=...)` raises TypeError on TwoTower and BERT4Rec

**Layers:** models × (any caller using the documented API)
**Severity:** breaks-on-use
**Found:** 2026-08-08

The README and `docs/` document one calling convention:

```python
model.fit(user_ids, item_ids, ratings)
```

It works positionally on every model. **By keyword it fails on two of them**,
because the third parameter is named differently:

| Model | third parameter |
|---|---|
| `ALS`, `LightGCN`, `NCF`, `DCN`, `DeepFM` | `ratings` |
| `TwoTower`, `BERT4Rec` | `interactions` |

Reproduce:

```python
import numpy as np
from corerec.engines import TwoTower

rng = np.random.default_rng(0)
U = rng.integers(0, 40, 300).tolist()
I = rng.integers(0, 60, 300).tolist()
R = rng.uniform(1, 5, 300).tolist()

TwoTower(embedding_dim=16, num_epochs=3, verbose=False).fit(
    user_ids=U, item_ids=I, ratings=R
)
# TypeError: TwoTower.fit() got an unexpected keyword argument 'ratings'
```

**Expected:** the documented keyword form works on every model, as the
positional form already does.
**Actual:** `TypeError` on `TwoTower` and `BERT4Rec`.

**Root cause:** `corerec/engines/two_tower.py` and `corerec/engines/bert4rec.py`
name the third parameter `interactions`. Both were changed to accept the triple
*positionally* via `normalize_interactions()`, but the parameter kept its
original name, so keyword callers still hit the old signature.

**Why no test caught it:** `tests/test_model_contract.py` calls
`model.fit(users, items, ratings)` positionally, which passes. The contract test
should exercise the keyword form too, since that is what the documentation shows
and what a caller building kwargs will use.

**Suspected blast radius:** anything that builds `fit` arguments as a dict —
config-driven training, hyperparameter sweeps, `fit(**params)` — breaks on these
two models while working on the rest. A caller cannot write one code path over
the model zoo.

**Suggested fix:** accept `ratings` as an alias on both, keeping `interactions`
working, and add a keyword-form case to the contract test.

---

## Fixed

_(nothing yet — entries move here with the commit that fixes them)_
