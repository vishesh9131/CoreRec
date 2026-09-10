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

_(none)_

---

## Fixed

### 1. `fit(..., ratings=...)` raises TypeError on TwoTower and BERT4Rec

**Fixed in:** zoo API parity audit (TwoTower/BERT4Rec accept `ratings=` alias;
SASRec also takes the event-list + keyword form via `normalize_interactions`).

**Layers:** models × (any caller using the documented API)
**Severity:** breaks-on-use
**Found:** 2026-08-08

The README and `docs/` document one calling convention:

```python
model.fit(user_ids, item_ids, ratings)
```

It works positionally on every model. **By keyword it failed on TwoTower and
BERT4Rec**, because the third parameter was named `interactions`. SASRec also
refused the event-list triple and required a dense matrix.

**Root cause:** parameter naming / matrix-only fit path.
**Fix:** `ratings=` alias on TwoTower and BERT4Rec; SASRec routes triples
through `normalize_interactions` and accepts keyword `user_ids`/`item_ids`/
`ratings`.
