# Model Serving

Production serving for recommendation models: an ANN-backed online recommendation
engine for high-scale, always-fresh feeds, plus a REST/batch server.

## Online serving (recommended for production)

`OnlineRecommender` serves top-K via ANN (sub-millisecond at million-item scale),
supports incremental freshness, and falls back gracefully for unknown users. See
the [Online Serving guide](../user_guide/online_serving.md) for the full walkthrough.

```python
from corerec.serving import OnlineRecommender

rec = OnlineRecommender.from_interactions(df, model="lightgcn", device="cuda")
rec.recommend(user_id=42, top_k=10)              # sub-ms ANN
rec.add_items(new_ids, new_vectors)              # freshness, no retrain
rec.fold_in_user("new_user", item_ids=[...])     # new user, no retrain
```

## REST / batch server

```python
from corerec.serving import ModelServer

server = ModelServer(model=my_model, host="0.0.0.0", port=8000)
server.start()
```

## API Endpoints

- `POST /predict` — Single prediction
- `POST /recommend` — Single user recommendations
- `POST /batch/predict` — Batch predictions
- `POST /batch/recommend` — Batch recommendations
- `GET /health` — Health check
- `GET /info` — Model information

## API Reference

```{eval-rst}
.. automodule:: corerec.serving
   :members:
   :show-inheritance:
```
