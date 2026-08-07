# Learning CoreRec

Every snippet here was executed against CoreRec 0.6.0 before being written down.
Where something does not work, that is stated rather than omitted.

---

## The mental model

CoreRec is three layers. Most libraries give you the first one only; the second
is what CoreRec is actually *for*.

```
  models        →  one algorithm, one fit(), one recommend()
  pipeline      →  retrieval → ranking → reranking
  serving       →  the same object behind an HTTP endpoint
```

You can stop at layer 1 and it behaves like `implicit` or `surprise`. The reason
to keep going is that a real recommender is never one model — it is a cheap wide
retriever, a precise ranker over its output, and business rules on top.

---

## Layer 1 — models

### Where they live

Import location matters and is not always where the name suggests:

| Module | Models |
|---|---|
| `corerec.engines` | `DCN`, `DeepFM`, `SASRec`, `BERT4Rec`, `MIND`, `NASRec`, `GNNRec`, `TwoTower` |
| `corerec.engines.collaborative` | `NCF`, `SAR`, `LightGCN`, `FAST`, `TwoTower` |
| `corerec.engines.matrix_factorization` | `ALS`, `Item2Vec` |
| `corerec.engines.content_based` | `TFIDFRecommender`, `DSSM`, `YoutubeDNN`, `BERT4Rec` |

`corerec.engines` also lazily resolves `AutoInt`, `AFM`, `DIN`, `DIEN`, `NFM`,
`PNN`, `FiBiNet`, `xDeepFM`, `WideDeep`, `Caser`, `MultVAE`, `MultiDAE`. They do
not appear in `dir()` because of the lazy `__getattr__`, but they import.

34 models are importable in total.

### The six calls

```python
import numpy as np
from corerec.engines.matrix_factorization import ALS

rng = np.random.default_rng(0)
users   = rng.integers(0, 50, 400).tolist()
items   = rng.integers(0, 80, 400).tolist()
ratings = rng.uniform(1, 5, 400).tolist()

model = ALS(factors=32, iterations=20)
model.fit(user_ids=users, item_ids=items, ratings=ratings)

model.predict(users[0], items[0])                       # -> 0.2117
model.recommend(users[0], top_k=5)                      # -> [10, 73, 70, 22, 31]
model.recommend(users[0], top_k=5, exclude_items=[10])  # drops item 10
model.batch_predict([(users[0], i) for i in items[:4]]) # -> [0.212, 0.426, ...]

model.save("artifacts/als")                             # base path, no extension
model = ALS.load("artifacts/als")                       # recommendations identical
```

`save` writes a *bundle*: `artifacts/als.meta.json` + `artifacts/als.weights.pt`.
Pass the base path to both `save` and `load`, not a `.pkl` filename.

### Two models take a DataFrame instead

`SAR` and `NCF` predate the triple convention:

```python
import pandas as pd
from corerec.engines.collaborative import SAR

df = pd.DataFrame({"userID": users, "itemID": items, "rating": ratings})
sar = SAR(similarity_type="cosine")
sar.fit(df)                                    # DataFrame, not three lists
# sar.fit_from_lists(users, items, ratings)    # if you prefer the triple
```

`tests/test_model_contract.py` lists these in `KNOWN_DIVERGENT` so the exception
is tracked rather than discovered.

### Which model to pick

From `BENCHMARKS.md` (MovieLens-100K, NDCG@10):

| | |
|---|---|
| **Start here** | `ALS` — 0.4168, best single model measured |
| Fast baseline | `SAR(similarity_type="cosine")` — 0.3955 in 0.35s |
| Best result | fuse two models (see below) — 0.4493 |
| Avoid for small data | `NCF` 0.3359, `LightGCN` 0.3360 — worse than a 0.08s ItemKNN and 300x slower |
| Do not use | `GNNRec` — does not finish ML-100K in an hour |

Use `similarity_type="cosine"` for SAR. `jaccard` is the default and scores
0.3730; `lift`, `mutual_information` and `inclusion_index` are near-random at the
default threshold and now warn when constructed.

---

## Layer 2 — the pipeline

Three stages, each independently swappable.

```python
from corerec.retrieval import CollaborativeRetriever
from corerec.ranking   import PointwiseRanker
from corerec.reranking import DiversityReranker

USER = users[0]

# 1. retrieval — cheap, wide, recall-oriented
cands = CollaborativeRetriever(model=model).retrieve(USER, top_k=20)

# 2. ranking — expensive, precise, over ~20 items instead of the whole catalogue
ranker = PointwiseRanker(model=model).fit()
ranked = ranker.rank(cands, context={"user_id": USER})

# 3. reranking — business rules; nothing to do with model quality
final = DiversityReranker(lambda_=0.7).rerank(ranked, top_k=5)

[(c.item_id, c.score) for c in final.candidates]
# [(0, 0.548), (34, 0.319), (35, 0.318), (26, 0.309), (46, 0.3)]
```

Every stage returns a result object with a `.candidates` list, not a bare list:

| Object | Fields |
|---|---|
| `RetrievalResult` | `candidates`, `query_id`, `retriever_name`, `timing_ms` |
| `Candidate` | `item_id`, `score`, `source`, `metadata` |
| `RankingResult` | `candidates`, `query_id`, `ranker_name`, `timing_ms` |
| `RankedCandidate` | `item_id`, `score`, `retrieval_score`, `rank`, `predictions`, `features` |

### Three things that will trip you

1. **Rankers need `.fit()`** even when there is nothing to train. `rank()` raises
   `RuntimeError: must be fitted before ranking` otherwise.

2. **Pass `context={"user_id": ...}` when using `model=`.** A recommender
   scores a (user, item) pair and the user lives in the context, not in the
   candidate. Omit it and you get an error saying exactly that.

3. **`score_fn` sees `item_id` by default.** No `feature_extractor` needed
   unless you want extra features.

### Why bother with three stages

Scoring a 1M-item catalogue with a deep model per request is impossible. Retrieval
narrows 1M to 500 with something cheap; ranking spends real compute on those 500;
reranking applies diversity, freshness and business rules that have nothing to do
with predicted relevance. `implicit` and `surprise` have no equivalent — that
layer is CoreRec's actual differentiator.

Available stages: `PopularityRetriever`, `CollaborativeRetriever`,
`EnsembleRetriever`; `PointwiseRanker`, `PairwiseRanker`, `FeatureCrossRanker`;
`DiversityReranker`, `BusinessRulesReranker`, `FairnessReranker`.

---

## Layer 3 — serving

```python
from corerec.serving import ModelServer          # pip install corerec[serving]

server = ModelServer(model)
server.start()          # POST /recommend on :8000
```

```bash
curl -X POST localhost:8000/recommend \
     -H 'Content-Type: application/json' \
     -d '{"user_id": 1, "top_k": 5}'
```

Endpoints: `/predict`, `/recommend`, `/batch/predict`, `/batch/recommend`,
`/health`, `/info`, and `/docs` for the OpenAPI page.

`server.app` is the FastAPI object, so you can test without binding a port:

```python
from fastapi.testclient import TestClient
client = TestClient(ModelServer(model).app)
client.post("/recommend", json={"user_id": 1, "top_k": 5}).json()
```

`examples/train_and_serve.py` is this whole path in one runnable file, and
`tests/test_train_and_serve.py` runs it on every commit.

For approximate nearest-neighbour serving there is `OnlineRecommender`, which
supports incremental `add_items` and `fold_in_user` without retraining.

---

## Things worth knowing

**Seeds.** `LightGCN` takes `seed=` (default 42) and is reproducible.
`BERT4Rec`, `TwoTower` and `SASRec` draw from numpy's global RNG with no seeding
— two runs of identical code give different models. Recorded in
`tests/test_benchmark_bugfixes.py::KNOWN_NONREPRODUCIBLE`.

**Single runs of sampled-negative models mean little.** Measured spread across
5 seeds: `NCF_binary` 13.1%, `LightGCN` 6.0%, `implicit`'s BPR 5.1%. Report
mean ± std, or use a deterministic model.

**`batch_predict` batches on NCF only.** The base implementation is a list
comprehension over `predict()` — fine for classical models, one forward pass per
pair for a torch model. If you add a neural model, override it.

**Optional extras.** `corerec[serving]` for FastAPI, `corerec[transformers]` for
encoder models, `corerec[datasets]` for `cr_learn`. A missing extra now raises an
error naming the extra rather than "no attribute".

---

## Where to look next

| | |
|---|---|
| `BENCHMARKS.md` | numbers, method, and the comparisons CoreRec loses |
| `examples/train_and_serve.py` | the shortest complete path |
| `tests/test_model_contract.py` | the API every model is held to |
| `Findings/bench/runner.py` | how to benchmark a new model |
