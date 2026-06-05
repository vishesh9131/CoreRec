# Online Serving at Scale

CoreRec is not only an offline trainer: `corerec.serving.OnlineRecommender` turns a
trained model (or raw interactions) into an **online recommendation tier** that is
sub-millisecond, fresh, and cold-start-graceful at million-item scale.

It addresses the three things a production feed needs that a plain `model.recommend`
does not:

- **Sub-linear retrieval** — ANN over the item catalogue (FAISS HNSW/IVF via
  `corerec.retrieval.VectorIndex`) instead of an O(catalogue) full scan.
- **Freshness without retraining** — add new items and fold in new/returning users
  incrementally.
- **Graceful cold-start** — unknown users fall back to a popularity ranking instead
  of raising.

## Quick start: train and serve in one call

```python
import pandas as pd
from corerec.serving import OnlineRecommender

# interactions: any DataFrame with user/item columns
df = pd.DataFrame({"user_id": users, "item_id": items})

# trains native embeddings (LightGCN by default) AND builds the ANN index
rec = OnlineRecommender.from_interactions(
    df, model="lightgcn", dim=64, epochs=200, device="cuda",
)

rec.recommend(user_id=42, top_k=10)        # sub-ms ANN retrieval
rec.recommend_batch([1, 2, 3], top_k=10)   # batched
```

`model="lightgcn"` (default) is the strongest option on large graph datasets;
`model="bpr"` is a lighter matrix-factorization alternative. Both train on GPU when
available. For a ~7x faster fit at a small accuracy cost, pass a large `batch`
(e.g. `from_interactions(..., )` with the trainer's `batch=1_000_000`).

## Serving an existing model's embeddings

If you already trained an embedding model, serve its factors directly:

```python
rec = OnlineRecommender.from_embeddings(
    item_ids=item_ids, item_emb=item_vectors,
    user_ids=user_ids, user_emb=user_vectors,
    index_type="hnsw", metric="ip",   # use "ip" for dot-product models (MF/ALS)
)
```

```{note}
Choose `metric="ip"` for dot-product models (matrix factorization, ALS, LightGCN)
and `metric="cosine"` for normalized/embedding models. CoreRec's `VectorIndex`
supports inner-product HNSW, so dot-product rankings are preserved rather than
distorted by cosine normalization.
```

`OnlineRecommender.from_model(model)` extracts embeddings automatically from models
that expose `user_factors`/`item_factors` or `get_user_embeddings()`/
`get_item_embeddings()`.

## Freshness: updating without retraining

A live feed changes constantly. Update the served index in place:

```python
# new content arrives -> add it to the live index (no retrain)
rec.add_items(item_ids=["post_991", "post_992"], item_emb=new_item_vectors)

# a new or returning user, described by their latest interactions ->
# fold them in (ridge solve over item factors, no retrain)
rec.fold_in_user("user_777", item_ids=["post_12", "post_44", "post_991"])
rec.recommend("user_777", top_k=10)
```

## Cold-start

Unknown users do not raise; they receive a popularity-based fallback ranking:

```python
rec.recommend("brand_new_user", top_k=10)   # -> popular items, not an error
```

## Monitoring

```python
rec.stats()
# {'n_items': ..., 'n_users': ..., 'index_type': 'hnsw', 'faiss': True,
#  'queries_served': ..., 'latency_ms_p50': ..., 'latency_ms_p99': ...}
```

## Scale and performance

On Gowalla (29,858 users, 40,981 items, ~1.0M interactions) the engine builds in
seconds, serves at **~0.09 ms/user (p50), ~10,000 req/s single-thread**, and holds
**~1.5 GB** of memory — versus tens of GB for dense co-occurrence models. FAISS
HNSW scales to a **1M-item index searched in ~0.02 ms/query**, and the approximate
ANN preserves 88–99% of exact-search accuracy.

## Installing the ANN backend

```bash
pip install faiss-cpu   # or faiss-gpu
```

Without FAISS, `VectorIndex` falls back to brute-force NumPy (correct but
O(catalogue)); install FAISS for sub-linear serving.

## What still belongs to infrastructure

`OnlineRecommender` is the retrieval/ranking engine. A full production feed still
wraps it with a hardened RPC server, a feature store, and a streaming source that
calls `add_items`/`fold_in_user` — those are operational layers, not modelling.
```
