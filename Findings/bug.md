# CoreRec integration bugs

Found by *using* the library across layer boundaries, not by reading it. See
`BUGHUNT_PROMPT.md` for the loop that produces these.

Report only. Fixes happen in a separate session where they can be reviewed.

---

## Combinations exercised

| # | Combination | Date | Result |
|---|---|---|---|
| 1 | models × serving — ALS, Item2Vec, LightGCN, TwoTower through `/recommend`, `/predict`, `/batch/recommend` | 2026-08-08 | **1 bug** (#1); ALS/Item2Vec/LightGCN clean on all three endpoints |
| 2 | models × serving.BatchInferenceEngine — ALS, Item2Vec, ItemKNN, EASE, TwoTower through `batch_predict` and `batch_recommend` | 2026-08-08 | clean on all 5; DCN/DeepFM construction surfaced bug #2 |
| 3 | models × persist — ALS, Item2Vec, ItemKNN, EASE, DCN, DeepFM, TwoTower, BERT4Rec: fit → save → load → predict/recommend, checked round-trip equality | 2026-08-08 | clean on all 8; SASRec fit surfaced bug #3 |
| 4 | pipeline — CollaborativeRetriever(ALS) → PointwiseRanker(Item2Vec) → {DiversityReranker, FairnessReranker, BusinessRulesReranker} | 2026-08-08 | retrieval + ranking + Diversity clean; **bug #4** (BusinessRulesReranker silently ignores `top_k`); FairnessReranker needs `group_fn`, EnsembleRetriever untested (PopularityRetriever fit signature mismatch — separate lead, not chased) |
| 5 | models × eval — ALS, Item2Vec, ItemKNN, EASE, TwoTower, DCN through `Evaluator.evaluate` with ndcg/recall/hit_rate | 2026-08-08 | Evaluator ran on all 6 without exception; **bug #5** (per-user errors are print-and-continue, so all-broken models score 0.0 indistinguishably from real 0.0) |
| 6 | data × models — feed `corerec.data.{RecommendationDataset, ContextualDataset, GraphDataset}` (and a raw DataFrame) into `ALS.fit` | 2026-08-08 | **bug #6** — no dataset class can be handed to any model; the two modules are entirely disconnected |
| 7 | models × serving.OnlineRecommender — ALS, Item2Vec, TwoTower through `from_model` → `recommend` / `add_items` / `fold_in_user` | 2026-08-08 | **bug #7** — `from_model` raises `NotImplementedError` on all 3; TwoTower's `get_user_embedding` (singular) misses the extractor's `get_user_embeddings` (plural); ALS/Item2Vec store factors under different names than the extractor scans |
| 8 | models × persist × serving.ModelLoader — save(path), then `ModelLoader.load(path)`, then predict/recommend | 2026-08-08 | **bug #8** — DCN/TwoTower.save(path) writes `.weights.pt`+`.meta.json` neighbours instead of `path`; ALS/Item2Vec/ItemKNN/EASE saves are state dicts, so `ModelLoader.load` returns a `dict` with no `.predict`/`.recommend` |
| 9 | eval.CrossValidator × models — try the one workflow the docstring shows (`cv.cross_validate(model, data, metric=...)`) | 2026-08-08 | `.split()` works and integrates fine with `ALS + Evaluator` in a manual loop; **bug #9** — the documented `cross_validate` method doesn't exist on the class |
| 10 | corerec.hybrid.RetrievalThenRerank × pipeline — construct with `CollaborativeRetriever + PointwiseRanker` and call `recommend()` | 2026-08-08 | **bug #10** — `RetrievalThenRerank` silently becomes `None` because `corerec.hybrid` swallows the `ModuleNotFoundError` from `corerec.ranking.base_ranker` (wrong path); PromptReranker imports fine |
| 11 | SemanticRetriever × pipeline — pre-computed embeddings → PointwiseRanker → DiversityReranker, plus edge cases (top_k>N, batch, no-encoder text query) | 2026-08-08 | clean; all edge cases handled with sensible errors |
| 12 | EnsembleRetriever × mixed retrievers — CollaborativeRetriever(ALS) + PopularityRetriever + SemanticRetriever, all three fusion strategies (`union`, `rrf`, `weighted`) | 2026-08-08 | fusion arithmetic works, `union` unsurprisingly favours whichever retriever has the largest raw scores; **bug #11** — a failing child retriever is silently dropped with no log and no signal, so a broken retriever looks the same as one that returned nothing |
| 13 | content_based × pipeline — TFIDFRecommender, Word2VecRecommender, DSSM, YoutubeDNN through `CollaborativeRetriever` and by direct call | 2026-08-08 | TFIDF composes with `CollaborativeRetriever` (10 candidates); DSSM / YoutubeDNN import; **`Word2VecRecommender is None`** — same shape as bug #10 (silent import swallow in `content_based/__init__.py:66-70`: imports `Word2VecRecommender` from `.word2vec` but the module only defines `WORD2VEC`); **bug #12** — TFIDF's `recommend_by_text(top_k=...)` blows up because that method only accepts `top_n`, while its sibling `recommend()` accepts both |
| 14 | reranker chain — DiversityReranker / FairnessReranker / BusinessRulesReranker composed pairwise and as a triple | 2026-08-08 | rerankers don't actually chain: **bug #13** — DiversityReranker and FairnessReranker reorder items but leave `RankedCandidate.score` at the original relevance value, so the next reranker's default `sort(key=score)` reverts to the pre-reranking order; also re-confirms bug #4 (BusinessRules still ignores `top_k` inside a chain) |
| 15 | serving.OnlineRecommender.from_interactions (LightGCN + BPR) → recommend / add_items / fold_in_user / popularity fallback | 2026-08-08 | clean end-to-end; unknown-user recommend returns exactly the top-10 most-interacted items; from_interactions is the working workaround for the broken `from_model` (bug #7) |
| 16 | persist × pipeline — ALS/Item2Vec/EASE/TwoTower: fit → save → `cls.load(path)` → CollaborativeRetriever → PointwiseRanker(reloaded model) → DiversityReranker | 2026-08-08 | clean on all 4 with class-specific loader; `is_fitted=True` survives round-trip; class-specific `.load` composes with the pipeline where `ModelLoader.load` cannot (bug #8) |

---

## Open


### 2. Constructor training-length kwarg is three different names across the zoo

**Layers:** models × (any caller constructing models from a shared config)
**Severity:** breaks-on-use
**Found:** 2026-08-08

The same conceptual argument — "how long to train" — is spelled three ways
depending on which model class you pick, with no accepted alias:

| kwarg | models |
|---|---|
| `epochs` (26) | `DCN`, `DeepFM`, `GNNRec`, `MIND`, `NASRec`, `GRU4Rec`, `Caser`, `BST`, `DIN`, `DIEN`, `NARM`, `MultVAE`, `MultiDAE`, `NGCF`, `FM`, `AFM`, `NFM`, `DeepFMCTR`, `DCNCTR`, `AutoInt`, `xDeepFM`, `FiBiNet`, `PNN`, `WideDeep`, `GMF`, `MLP` |
| `num_epochs` (3) | `TwoTower`, `BERT4Rec`, `SASRec` |
| `iterations` (2) | `ALS`, `Item2Vec` |
| (none) | `ItemKNN`, `UserKNN`, `EASE`, `SLIM` |

Reproduce (exercises the failure a user hits when standardising on one name):

```python
from corerec.engines import DCN, TwoTower, ALS

# Pick any one kwarg and try to use it across models
TwoTower(embedding_dim=8, num_epochs=2)       # OK
DCN(embedding_dim=8, num_epochs=2)            # TypeError: unexpected 'num_epochs'
ALS(factors=8, epochs=2)                      # TypeError: unexpected 'epochs'
```

**Expected:** one canonical name (or accepted aliases) so a caller can write a
single `MODEL_CLASSES[name](**common_kwargs)` factory or config-driven trainer.
**Actual:** `TypeError` on any generic path; each model class must be
special-cased.

**Root cause:** each model was written independently. `docs/CORRECT_PARAMETERS.md`
already documents this as a known gotcha ("Some use `epochs`, others use
`num_epochs`"), which suggests the maintainers know but users still have to
work around it — the "gotcha" isn't a fix, it's paperwork over the bug.

**Why no test caught it:** every existing test either constructs one model at
a time with its native kwarg or reads per-model configs. No test iterates a
list of model classes with a shared kwarg dict, which is the case that fails.

**Suspected blast radius:** the same as bug #1 (config-driven training,
hyperparameter sweeps, model factories, benchmarking harnesses). This bug
prevents the fix suggested there — `fit(**params)` cannot be made uniform
until `__init__(**params)` is too.

**Suggested fix:** pick one name (`epochs` is the plurality) and accept the
other two as deprecated aliases on the minority classes; extend the contract
test to construct every registered model from `{"epochs": 2}`.

---


### 5. `Evaluator.evaluate` reports 0.0 for a totally broken model

**Layers:** models × eval
**Severity:** silent-wrong-result
**Found:** 2026-08-08

`Evaluator.evaluate` wraps the per-user loop in `try/except Exception`, prints
the error, and continues. If every user raises, the metric list stays empty
and the final average is coerced to `0.0` — the same value a legitimately bad
model would produce.

Reproduce:

```python
from corerec.evaluation import Evaluator

class BrokenModel:
    def recommend(self, user_id, top_k=20):
        raise RuntimeError("kaboom")

gt = {1: [10, 20, 30], 2: [11, 21, 31], 3: [12, 22, 32]}
print(Evaluator(metrics=["ndcg@10", "recall@10"]).evaluate(BrokenModel(), gt))
# Error evaluating user 1: kaboom
# Error evaluating user 2: kaboom
# Error evaluating user 3: kaboom
# {'ndcg@10': 0.0, 'recall@10': 0.0}
```

**Expected:** either raise, or return something a downstream harness can
detect (e.g. `NaN`, or a companion `errors` count). Anything that lets a
benchmark script know "this model produced zero valid predictions".
**Actual:** returns `0.0` for every metric. A benchmark leaderboard, a CI
regression check, or a `compare_models` table all treat this as "the model
scored 0" — a plausible-looking bad-but-valid result. The stdout print goes
to noise in any pipe/aggregator setup.

**Root cause:** `corerec/evaluation/evaluator.py:65-100` — bare
`except Exception as e: print(...); continue` in the per-user loop, and
`np.mean(v) if v else 0.0` in the aggregation. Both together turn every
error into a silent zero.

**Suspected blast radius:**
- NOT `BENCHMARKS.md`. Verified: `Findings/bench/runner.py` uses its own
  `Findings/bench/metrics.py` and never imports `corerec.evaluation`, so the
  published numbers are unaffected. (An earlier draft of this entry claimed
  otherwise and was wrong.) SAR's 0.0007 was a genuine measurement, not a
  swallowed error.
- Anyone evaluating with the library's own `Evaluator` rather than the bench
  harness — which is what the docs point users at;
- `compare_models` (line 105) — a broken new model looks like a losing entry
  rather than a bug to fix;
- CI regression checks that key on "did the metric drop" cannot distinguish
  "regression" from "model broke entirely".

**Suggested fix:** count errors, expose them alongside metrics, and coerce
empty metric lists to `NaN` rather than `0.0`. Optionally add a `strict=True`
flag that re-raises. Ideally, refuse to average when errors > threshold.

---

### 6. `corerec.data` dataset classes cannot be handed to any model

**Layers:** data × models
**Severity:** breaks-on-use
**Found:** 2026-08-08

`corerec.data` exports 11 dataset classes (`RecommendationDataset`,
`ContextualDataset`, `GraphDataset`, `SequentialRecommendationDataset`, …)
and every model documents `model.fit(user_ids, item_ids, ratings)`. There is
no wiring between the two: no model accepts a dataset object, and no dataset
exposes the triple as arrays.

Reproduce:

```python
import pandas as pd, numpy as np
from corerec.data import RecommendationDataset, ContextualDataset, GraphDataset
from corerec.engines import ALS
from scipy.sparse import csr_matrix

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "user_id": rng.integers(0, 40, 300),
    "item_id": rng.integers(0, 60, 300),
    "rating":  rng.uniform(1, 5, 300),
})

for name, ds in [
    ("RecommendationDataset", RecommendationDataset(df)),
    ("ContextualDataset",     ContextualDataset(
        [(int(r.user_id), int(r.item_id), float(r.rating), {})
         for r in df.itertuples()], {})),
    ("GraphDataset",          GraphDataset(
        [(int(r.user_id), int(r.item_id), float(r.rating))
         for r in df.itertuples()], csr_matrix((40, 60)))),
]:
    try:
        ALS(factors=8, iterations=3).fit(ds)
        print(name, "OK")
    except TypeError as e:
        print(name, "FAIL:", e)

ALS(factors=8, iterations=3).fit(df)  # also fails with the same TypeError
```

Every dataset fails with:
```
TypeError: _EmbeddingCFBase.fit() missing 1 required positional argument: 'item_ids'
```
Passing the raw DataFrame, `ds.samples`, or `ds.data` all fail the same way.

**Expected:** either `model.fit(dataset)` works, or the docs are honest that
users must unpack the dataset back into three lists themselves. A namespace
called `corerec.data` next to `corerec.engines` sets the expectation of
composition.
**Actual:** the two namespaces are entirely disconnected. Every model needs
three parallel lists/arrays; every dataset stores data as `List[Tuple]`
(behind `.data`) or a `DataFrame` (behind `.interactions`) or PyTorch
`__getitem__` samples. A user has to write the adapter for every combination.

**Root cause:** no shared contract. `corerec/engines/matrix_factorization.py`
(and every other model) requires positional `(user_ids, item_ids, ratings)`;
none of the `Dataset` classes surface those. The `RecommendationDataset` even
has a `.samples` list of tuples, but no model reads it.

**Suspected blast radius:** the entire `corerec.data` module is a dead-end —
users construct datasets, then have to unpack them right back to feed a
model. The situation looks worse against the docs: `corerec.data` is
prominently exported and appears in tutorials, so users write pipeline code
around it and then discover it can't drive `fit()`.

**Suggested fix:** either give `BaseRecommender.fit` a `dataset=`  dispatch
that unpacks any known dataset type into arrays, or give every dataset a
`.to_triples()` method that returns `(user_ids, item_ids, ratings)`; add a
test that fits every model type from every dataset type.

---

### 7. `OnlineRecommender.from_model` fails on every CoreRec model that has embeddings

**Layers:** models × serving
**Severity:** breaks-on-use
**Found:** 2026-08-08

`OnlineRecommender.from_model` is documented as the entry point for turning
a trained CoreRec model into a served ANN index. Its docstring says it
"supports models that expose factors via common attributes/methods
(item_factors/user_factors, get_item_embeddings, embeddings)". None of
CoreRec's own embedding-based models expose the exact names it scans for,
so every call raises `NotImplementedError`.

Reproduce:

```python
import numpy as np
from corerec.serving import OnlineRecommender
from corerec.engines import ALS, Item2Vec, TwoTower

rng = np.random.default_rng(0)
U = rng.integers(0, 40, 300).tolist()
I = rng.integers(0, 60, 300).tolist()
R = rng.uniform(1, 5, 300).tolist()

for cls, kw in [(ALS, {"factors": 8, "iterations": 3}),
                (Item2Vec, {"embedding_dim": 8, "epochs": 2, "verbose": False}),
                (TwoTower, {"embedding_dim": 8, "num_epochs": 2, "verbose": False})]:
    m = cls(**kw); m.fit(U, I, R)
    try:
        OnlineRecommender.from_model(m)
        print(cls.__name__, "OK")
    except NotImplementedError as e:
        print(cls.__name__, "FAIL:", e)
```

Output:
```
ALS FAIL: ALS does not expose embeddings; use OnlineRecommender.from_interactions or from_embeddings instead.
Item2Vec FAIL: Item2Vec does not expose embeddings; ...
TwoTower FAIL: TwoTower does not expose embeddings; ...
```

**Expected:** any trained CoreRec model with a factor/embedding representation
can be handed to `from_model` — that's what the docstring promises and the
whole point of the constructor.
**Actual:** every embedding-based model raises. The error message insists the
model "does not expose embeddings" while `TwoTower` in particular has both
`get_user_embedding()` and `get_item_embeddings()`.

**Root cause:** `corerec/serving/online.py:399-423`. The extractor scans:

- attributes `user_factors + item_factors` or `user_embeddings + item_embeddings`
- methods   `get_user_embeddings + get_item_embeddings` (both plural)

But actual model surfaces are:

| model    | what it has                                                 |
|----------|-------------------------------------------------------------|
| ALS      | factors stored internally, not under `user_factors`/`item_factors` |
| Item2Vec | same                                                        |
| TwoTower | `get_user_embedding` (singular) + `get_item_embeddings` (plural) |

For TwoTower the mismatch is literally one `s`: the extractor asks for
`get_user_embeddings`, the model provides `get_user_embedding`. Same shape as
the ranker/predict bug in commit `540c644` and the `exclude_items` bug — two
sides written independently to slightly different names.

**Suspected blast radius:** `from_model` is the only ergonomic path from
trained CoreRec model → online serving; users are pushed onto
`from_interactions` (which retrains a fresh LightGCN/BPR from raw data,
ignoring their trained model) or `from_embeddings` (which requires them to
pull out the arrays themselves). The advertised composition path doesn't
work.

**Suggested fix:** either accept both singular/plural method names and add
the actual attribute names ALS/Item2Vec use, or define one contract method
(e.g. `BaseRecommender.extract_embeddings() -> (item_ids, V, users, U)`) and
have every embedding model implement it. Add a contract test that runs
`from_model` on every model with an embedding surface.

---

### 8. `model.save(path)` doesn't create `path`, and generic loaders can't find/reconstruct it

**Layers:** models × persist × serving (ModelLoader)
**Severity:** breaks-on-use
**Found:** 2026-08-08

Two related failures on the same combination — a saved model cannot be loaded
back with `serving.ModelLoader`, the documented generic loader.

**A. `save(path)` writes to a different location than requested.** DCN, TwoTower
and other torch-based models take the argument as a *prefix*, not a path:

```python
import os, tempfile, numpy as np
from corerec.engines import DCN

rng = np.random.default_rng(0)
U = rng.integers(0,40,300).tolist()
I = rng.integers(0,60,300).tolist()
R = rng.uniform(1,5,300).tolist()

m = DCN(embedding_dim=8, epochs=2, verbose=False); m.fit(U, I, R)
tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "DCN.model")
m.save(path)
print("exists(path):", os.path.exists(path))          # False
print("dir contents:", os.listdir(tmp))               # ['DCN.weights.pt', 'DCN.meta.json']
```

**B. `ModelLoader.load(path)` returns a raw `dict` for MF-family saves.** ALS,
Item2Vec, ItemKNN, EASE all pickle a state dict, not `self`. The documented
generic loader unpickles that dict and hands it back:

```python
from corerec.serving import ModelLoader
from corerec.engines import ALS

m = ALS(factors=8, iterations=3); m.fit(U, I, R)
path = os.path.join(tempfile.mkdtemp(), "als.pkl"); m.save(path)
loaded = ModelLoader().load(path)
print(type(loaded).__name__)          # dict
loaded.predict(0, 5)                  # AttributeError: 'dict' object has no attribute 'predict'
```

**Expected:** `save(path)` produces a file at `path`. `ModelLoader.load(path)`
returns something with the same `predict`/`recommend` surface the model had.
The docstring example (`loader.load('models/ncf_v1.pkl')`) shows one path in
and one usable model out.
**Actual:**
- Torch-based models silently write to `{path.stem}.weights.pt` and
  `{path.stem}.meta.json` alongside the requested path; `os.path.exists(path)`
  is False and `ModelLoader` gets `FileNotFoundError`.
- MF-family models write a state dict via pickle; `ModelLoader.load` returns a
  `dict` with no method surface.

Only the class-specific loader (`DCN.load(path)`, `ALS.load(path)`) knows the
per-model convention — but that requires the caller to already know which
class produced the file, defeating the purpose of `ModelLoader`.

**Root cause:**
- `corerec/serving/model_loader.py:60-80` uses either
  `corerec.serialization.load_from_file` (which rejects `.model`) or bare
  `pickle.load` (which returns whatever was pickled). Neither reconstructs.
- Each model's `save()` writes whatever internal format it wants and encodes
  the reconstruction logic in its own `load()`. There's no shared metadata
  telling ModelLoader which class to instantiate.

**Suspected blast radius:** any deployment path that saves a model in one
process and loads it in another — CI packaging, canary rollout, the whole
serving story. Users who round-trip via `cls.load(path)` in the same process
never hit this; anyone using `ModelLoader` or reading `path` from a manifest
does.

**Suggested fix:** either (a) have every `save()` write a single archive at
the exact path given, containing a class marker so a generic loader can
reconstruct, or (b) drop `ModelLoader.load` in favour of a factory that reads
the class name from a manifest. Add a contract test:
`save → ModelLoader.load → predict/recommend` on every model.

---

### 9. `CrossValidator.cross_validate` is documented but not implemented

**Layers:** eval × models
**Severity:** breaks-on-use
**Found:** 2026-08-08

`CrossValidator`'s class docstring shows exactly one usage example:

```
class CrossValidator:
    """
    Cross-validation utilities.

    Example::

        cv = CrossValidator(n_folds=5)
        avg_score = cv.cross_validate(model, data, metric='ndcg@10')
    """
```

That method doesn't exist. Only `split()` and the constructor do.

Reproduce:

```python
import pandas as pd
from corerec.evaluation import CrossValidator

cv = CrossValidator(n_folds=3)
cv.cross_validate(None, pd.DataFrame({"a": [1, 2, 3]}), metric="ndcg@10")
# AttributeError: 'CrossValidator' object has no attribute 'cross_validate'
```

**Expected:** the documented one-liner runs — either `cv.cross_validate(...)`
returns a mean score, or the docs no longer promise it.
**Actual:** `AttributeError`. Users have to hand-roll the fold loop and wire
in `Evaluator` themselves, which works but is nothing like the advertised API.

**Root cause:** `corerec/evaluation/evaluator.py:181-240`. The class was left
as a `split()`-only stub; the docstring never got trimmed to match. Same
shape as the `test_docs.py` finding — docs promise a call path that no
implementation ever backed.

**Suspected blast radius:** anyone reading the docstring, following the
example, and hitting `AttributeError` at first use. Not directly used by
other components (so it doesn't cascade), but the class exists specifically
to be called via `cross_validate`, so the only failure mode is 100% failure.

**Suggested fix:** implement `cross_validate(model_factory, data, metric)`
that (a) splits, (b) refits a fresh model per fold, (c) evaluates each fold
with `Evaluator`, (d) returns per-fold + mean. Or delete the class and
document the manual `Evaluator` + `split` loop as the CV recipe. Either way,
the docstring and the class body must agree.

---

### 10. `RetrievalThenRerank` is silently `None` — broken import path is swallowed

**Layers:** pipeline × hybrid × models
**Severity:** confusing-error
**Found:** 2026-08-08

`corerec.hybrid.RetrievalThenRerank` is documented as the two-stage
recommender that composes any retriever with any ranker. The module-level
export unconditionally swallows import failures and rebinds the name to
`None`, so the class silently doesn't exist. First construction raises a
generic `TypeError: 'NoneType' object is not callable` that says nothing
about the real cause.

Reproduce:

```python
from corerec.hybrid import RetrievalThenRerank

print(RetrievalThenRerank)                            # None
RetrievalThenRerank(name="rr", config={}, retriever=None, reranker=None)
# TypeError: 'NoneType' object is not callable
```

Or import directly to see the real error:

```python
from corerec.hybrid.retrieval_then_rerank import RetrievalThenRerank
# ModuleNotFoundError: No module named 'corerec.ranking.base_ranker'
```

**Expected:** `RetrievalThenRerank(retriever, reranker)` returns a working
two-stage model, per the class's docstring and its position in
`corerec.hybrid.__all__`.
**Actual:** `RetrievalThenRerank is None`. Any attempt to construct it
raises a `TypeError` from the interpreter, with no hint that the actual
problem is a broken import path (`corerec.ranking.base_ranker` — the real
module is `corerec.ranking.base`).

**Root cause:**
- `corerec/hybrid/retrieval_then_rerank.py:19` imports
  `from corerec.ranking.base_ranker import BaseRanker`. That module doesn't
  exist; the file is `corerec/ranking/base.py`.
- `corerec/hybrid/__init__.py:7-10` wraps the import in
  `try / except ImportError: RetrievalThenRerank = None`. The exception is
  discarded (no logging, no re-raise, no reason string). Compare with
  commit `7dcf838` "Name the extra when an optional submodule cannot
  import" — that fix was applied elsewhere but not here.

**Suspected blast radius:** anything documented under `corerec.hybrid`, plus
any user code that follows the "retrieve, then rerank" narrative in the
README/docs. The `try/except ImportError = None` pattern is likely repeated
in other lazy-load submodules — every occurrence turns a real import bug
into a `NoneType` mystery at call time. Same shape as the test_docs.py
finding that revealed 74 broken documented paths.

**Suggested fix:** fix the import path
(`from corerec.ranking.base import BaseRanker`), then delete the
`try/except ImportError` swallow (or, if kept for optional torch/faiss
deps, apply the `7dcf838` "name the extra" pattern so the failure tells
the user what to install).

---

### 11. `EnsembleRetriever` silently swallows a broken child retriever

**Layers:** pipeline (retrieval) — cross-retriever composition
**Severity:** silent-wrong-result
**Found:** 2026-08-08

`EnsembleRetriever.retrieve` wraps each child call in a bare
`try/except Exception: pass`. A retriever that raises contributes 0
candidates and no signal — no log, no warning, no `errors` field on the
result. The code even comments `# in production you'd log this` but doesn't
log.

Reproduce:

```python
import numpy as np
from collections import Counter
from corerec.engines import ALS
from corerec.retrieval import (
    EnsembleRetriever, CollaborativeRetriever, SemanticRetriever,
)

rng = np.random.default_rng(0)
U = rng.integers(0, 40, 300).tolist()
I = rng.integers(0, 60, 300).tolist()
R = rng.uniform(1, 5, 300).tolist()

als = ALS(factors=8, iterations=3); als.fit(U, I, R)
collab = CollaborativeRetriever(model=als).fit()
sem = SemanticRetriever().fit(item_ids=list(range(60)),
                              item_embeddings=rng.normal(size=(60, 16)))

ens = EnsembleRetriever(
    retrievers=[("collab", collab, 1.0), ("sem", sem, 1.0)],
    strategy="rrf",
)
res = ens.retrieve(query=3, top_k=10)   # int query — semantic needs an embedding
print({c.source for c in res.candidates})
# {'ensemble(collab)'} — no hint that 'sem' errored out
```

`sem.retrieve(query=3, top_k=10)` on its own raises
`ValueError: matmul: Input operand 1 does not have enough dimensions ...` —
a real bug that the ensemble hides.

**Expected:** either the ensemble surfaces failures (result carries an
`errors: {retriever_name: exception}` field, or a logger.warning fires, or
a `strict=True` toggle re-raises), or the docstring warns that a broken
retriever will be silently ignored.
**Actual:** the failure is discarded. A misconfigured or degraded retriever
looks the same as one that legitimately produced no candidates. In
production this means an SRE cannot see that "the semantic retriever has
been erroring for 3 hours" from the response payload.

**Root cause:** `corerec/retrieval/ensemble.py:120-126`:

```python
try:
    result = retriever.retrieve(query, top_k=k_each, **kwargs)
    all_results.append((name, weight, result))
except Exception as e:
    # one retriever failing shouldn't kill the ensemble
    # in production you'd log this
    pass
```

The rationale (survive one dead retriever) is fine; the implementation
(swallow silently, no log) is exactly the anti-pattern bug #5 flagged in
`Evaluator.evaluate`. Same shape, different file.

**Suspected blast radius:** production canaries — a new retriever variant
that silently 500s looks like "we deployed the new retriever and it got 0%
selection share, so it must be bad" rather than "the new retriever is
crashing on every call". Also A/B tests: the winning ensemble is judged
against a silently-broken baseline.

**Suggested fix:** at minimum, call `logging.warning(...)` with the
exception and the retriever name. Better: attach an `errors` dict on
`RetrievalResult` so downstream code can act on it. Optionally support
`strict=True` on the ensemble constructor to re-raise instead of swallow.

---

### 12. `TFIDFRecommender` — `recommend(top_k=…)` works, `recommend_by_text(top_k=…)` doesn't

**Layers:** models (single class, cross-method drift)
**Severity:** breaks-on-use
**Found:** 2026-08-08

Two methods on the same class disagree on the pagination parameter name:

```
TFIDFRecommender.recommend        (user_id_or_indices=None, top_k=10, top_n=None, ...)
TFIDFRecommender.recommend_by_text(query_text: str,       top_n=10)
```

`recommend` accepts both `top_k` (the CoreRec convention) and legacy
`top_n`. `recommend_by_text` accepts only `top_n`. Same class, same
concept, different spelling.

Reproduce:

```python
from corerec.engines.content_based import TFIDFRecommender

items = list(range(20))
docs  = {i: f"item {i} topic {i % 5}" for i in items}
m = TFIDFRecommender(); m.fit(items, docs)

m.recommend(user_id=0, top_k=5)               # OK
m.recommend_by_text("topic 3", top_k=5)       # TypeError: unexpected keyword 'top_k'
```

**Expected:** the same `top_k` kwarg works on every `recommend*` method of
one class. That's the surface every other recommender in CoreRec presents,
and it's what `recommend` on this same class advertises.
**Actual:** `TypeError` on the text variant. A user writing a helper like
`fn = m.recommend_by_text if using_text else m.recommend; fn(..., top_k=5)`
crashes on the text branch only.

**Root cause:** `recommend_by_text` was left on the old `top_n` name when
the rest of the API migrated to `top_k`. The `recommend` method got a
`top_n` alias to soften the transition; `recommend_by_text` didn't.

**Suspected blast radius:** anything that switches between the two
recommend paths (text vs id) with a shared kwargs dict — search-driven UIs,
A/B harnesses that swap the query type, wrappers that expose one
`top_k` argument to the outside world. The same shape as bug #2
(epochs/num_epochs) and bug #1 (ratings/interactions): parameter name
drift inside code that's supposed to be uniform.

**Suggested fix:** accept `top_k` as an alias on `recommend_by_text`
(keeping `top_n` for compatibility), or standardise on one name across
every `recommend*` in the codebase.

---


## Fixed

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

**Status:** fixed upstream in 33911a3 (zoo API parity, #29); verified `fit(user_ids=, item_ids=, ratings=)` on TwoTower, BERT4Rec, SASRec.

---

### 3. `SASRec.fit(user_ids, item_ids, ratings)` rejects the documented triple

**Layers:** models × (any caller using the documented `fit` triple)
**Severity:** breaks-on-use
**Found:** 2026-08-08

Every model in the zoo documents `model.fit(user_ids, item_ids, ratings)` with
three 1-D sequences of the same length (README lines 76, 183, 274; quickstart;
per-model docs). SASRec accepts the call but interprets the third argument as
a **2-D dense interaction matrix** and raises when it isn't:

```python
import numpy as np
from corerec.engines import SASRec

rng = np.random.default_rng(0)
U = rng.integers(0, 40, 300).tolist()
I = rng.integers(0, 60, 300).tolist()
R = rng.uniform(1, 5, 300).tolist()

SASRec(num_epochs=1, verbose=False).fit(U, I, R)
# ValueError: interaction_matrix must be 2D
```

**Expected:** the same triple works on SASRec as it does on every other model.
**Actual:** `ValueError: interaction_matrix must be 2D`. The message names the
internal variable, not the API mismatch, so a user has no hint that SASRec
wants `fit(users, items, csr_matrix)` in a completely different shape.

**Root cause:** `corerec/engines/sasrec.py:582-610`. `fit()` has two branches
— legacy `(interaction_matrix, users, items)` and "standard" `(users, items,
interaction_matrix)` — but both name the third argument `interaction_matrix`
and require it to be 2D. The zoo-standard triple (three 1-D sequences) is not
handled at all; the parameter renamed to `ratings` on the docs never got
plumbed through here. `tests/engines_models_smoke_test.py:184` uses the legacy
matrix form so the standard form is untested.

**Suspected blast radius:** anything iterating models with a shared call —
benchmarks, hyperparameter sweeps, the same "one factory over the zoo" pattern
that #1 and #2 already block. Compounds with bug #1 (TwoTower/BERT4Rec's
`interactions` kwarg): a config-driven caller now needs three special cases
for one API.

**Suggested fix:** detect 1-D `arg3` and either build the interaction matrix
via `normalize_interactions()` (as TwoTower/BERT4Rec do) or emit an error that
names the expected shape. Add a triple-form case to the smoke test.

**Status:** fixed upstream in 33911a3 (zoo API parity, #29); verified `fit(user_ids=, item_ids=, ratings=)` on TwoTower, BERT4Rec, SASRec.

---

### 4. `BusinessRulesReranker.rerank(top_k=N)` silently ignores `top_k`

**Layers:** pipeline (reranking) — cross-reranker consistency
**Severity:** silent-wrong-result
**Found:** 2026-08-08

The other two rerankers accept `top_k` and truncate to that many items:

```
DiversityReranker.rerank(ranked, context=None, top_k=None, **kwargs)
FairnessReranker.rerank (ranked, context=None, top_k=None, **kwargs)
BusinessRulesReranker.rerank(ranked, context=None,          **kwargs)  # no top_k
```

Because `BusinessRulesReranker.rerank` accepts `**kwargs`, `top_k` is
swallowed without error and the full ranked list is returned.

Reproduce:

```python
from corerec.ranking.base import RankedCandidate, RankingResult
from corerec.reranking import BusinessRulesReranker, DiversityReranker

ranked = RankingResult(
    candidates=[RankedCandidate(item_id=i, score=1.0/(i+1)) for i in range(20)],
    ranker_name="test",
)

d = DiversityReranker(lambda_=1.0).rerank(ranked, top_k=5)
b = BusinessRulesReranker().rerank(ranked, top_k=5)

print(len(d.candidates), len(b.candidates))  # 5 20
```

**Expected:** `top_k=5` returns 5 items regardless of which reranker
implementation is used — the docstring of `BaseReranker.rerank` says
"reranker-specific parameters" go through `**kwargs`, but a caller wiring the
reranking stage generically has every reason to assume `top_k` is honored
uniformly.
**Actual:** `BusinessRulesReranker` returns the full list; `DiversityReranker`
returns 5. Swapping rerankers silently changes the response size.

**Root cause:** `corerec/reranking/business.py:96-101` — `rerank()` never
reads `top_k` and never truncates. The other two rerankers implement it.

**Suspected blast radius:** any pipeline that alternates rerankers via
config, A/B tests one against another, or feeds the reranked list into a
paginated response — the pagination cursor and the response size will
disagree when the business-rules variant is active.

**Suggested fix:** accept `top_k: Optional[int] = None` and truncate `result`
before returning; add a contract test that asserts every reranker honors it.

**Status:** fixed 2026-10-06 in `corerec/reranking/business.py`; covered by `tests/test_reranker_chaining.py`.

---

### 13. Rerankers don't chain — a follow-up reranker undoes the previous one

**Layers:** pipeline (reranking) — cross-reranker composition
**Severity:** silent-wrong-result
**Found:** 2026-08-08

`DiversityReranker` and `FairnessReranker` both reorder candidates but write
the **original relevance score** into `RankedCandidate.score`, not the
adjusted score they used to pick the new order. Downstream, any reranker
that sorts by `.score` — e.g. `BusinessRulesReranker` — reverts to the
pre-reranking order. The user's fairness/diversity intent is silently
discarded.

Reproduce:

```python
from corerec.ranking.base import RankedCandidate, RankingResult
from corerec.reranking import FairnessReranker, BusinessRulesReranker

cands = [
    RankedCandidate(item_id=1, score=1.0, features={"g": "A"}),
    RankedCandidate(item_id=2, score=0.9, features={"g": "A"}),
    RankedCandidate(item_id=3, score=0.8, features={"g": "A"}),
    RankedCandidate(item_id=4, score=0.5, features={"g": "B"}),
    RankedCandidate(item_id=5, score=0.4, features={"g": "B"}),
    RankedCandidate(item_id=6, score=0.3, features={"g": "B"}),
]
r = RankingResult(candidates=cands, ranker_name="fake")

group = lambda i: "A" if i in (1, 2, 3) else "B"
after_fair = FairnessReranker(group_fn=group, objective="equal",
                              fairness_weight=0.9).rerank(r)
after_biz  = BusinessRulesReranker().rerank(after_fair)   # NO rules added

print([c.item_id for c in after_fair.candidates])
# [1, 4, 5, 2, 6, 3]   ← Fairness interleaves groups A and B
print([c.item_id for c in after_biz.candidates])
# [1, 2, 3, 4, 5, 6]   ← BusinessRules sorts by score → back to original order
```

**Expected:** chaining rerankers preserves each stage's contribution.
Calling `BusinessRulesReranker().rerank(x)` with no rules should be a
no-op — the docstring frames it as "apply configurable business rules,"
and no rules means no changes. Even with rules, the fairness ordering
should hold on ties.
**Actual:** BusinessRules unconditionally re-sorts by `score`. Because
Fairness (and Diversity) leave `.score` at the original relevance value,
every reranker that runs after them re-derives the original ranking.

**Root cause:**
- `corerec/reranking/fairness.py:147-152` and
  `corerec/reranking/diversity.py:120-127` both build the output as
  `RankedCandidate(..., score=rc.score, ...)` — preserving the *input*
  score. Neither stores the adjusted score they used to pick the new order.
  Order lives only in the *sequence* of `candidates`, not in a numeric field.
- `corerec/reranking/business.py:127-128` sorts on `.score` on every call,
  regardless of whether any rules are configured, so it happily reorders
  a list that arrived pre-ordered.

**Suspected blast radius:** any user following the "chain rerankers"
narrative in `corerec/reranking/__init__.py` docstring. The user gets a
fairness/diversity-aware first pass followed by a business-rules stage —
the natural production ordering. Silently, the business stage nukes the
fairness/diversity work. There is no error, no warning; the response
looks reasonable but has none of the guarantees the earlier stage
promised.

**Suggested fix:**
- Rerankers that reorder must write the adjusted score into `.score`
  (and can keep the original in `.retrieval_score` or a new
  `.relevance_score` for later stages that need it).
- `BusinessRulesReranker` should preserve incoming order when it has no
  ordering rules of its own — only apply `sort` when a boost/pin changed
  something. Or use a stable sort keyed on both the boosted score and the
  incoming rank.
- Add a chain-invariance test: for any pair `(A, B)`, if `B` has no
  configured rules that reorder, `B(A(x))` must equal `A(x)`.

**Status:** fixed 2026-10-06 in `corerec/reranking/business.py`; covered by `tests/test_reranker_chaining.py`.

How: unboosted items now keep exactly the order the previous stage produced; each boosted item is reinserted ahead of the first item its boosted score beats. On relevance-ordered input that is identical to the old sort, so single-stage behaviour is unchanged.

---

### 13b. `BusinessRulesReranker` mutated its input when boosting

Found while fixing #13. Boosts were applied as `c.score *= multiplier` on the
caller's own `RankedCandidate` objects, so reranking the same result twice
compounded the boost: an item at 0.25 with a 10x boost read 250.0 after three
calls. Boosted and unboosted candidates are now copied with
`dataclasses.replace` before anything is changed.

**Status:** fixed 2026-10-06 alongside #13; `test_rerank_does_not_mutate_its_input`.
