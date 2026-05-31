# Production Deployment Guide

This guide covers best practices for deploying CoreRec models in production environments.

```{admonition} Recommended: CoreRec ModelServer
:class: tip
For production REST serving, use the built-in **ModelServer** instead of rolling your own Flask/FastAPI app. See {doc}`../api/serving` and the example below.
```

## Model Serialization

Production models default to the **safe bundle** format (`corerec_safe_v1`). Pass a **base path** (not a `.pkl` filename):

```python
from corerec.engines.dcn import DCN

model = DCN(embedding_dim=64, epochs=20)
model.fit(user_ids, item_ids, ratings)

model.save("artifacts/production_dcn")  # writes .meta.json + .weights.pt
loaded = DCN.load("artifacts/production_dcn")
```

See {doc}`../user_guide/safe_bundle_persistence` for layout and legacy migration.

## Serving with ModelServer (recommended)

```python
from corerec.serving import ModelServer
from corerec.engines.dcn import DCN

model = DCN.load("artifacts/production_dcn")
server = ModelServer(model, host="0.0.0.0", port=8000)
server.start()

# Endpoints: POST /predict, POST /recommend, POST /batch/predict,
#            POST /batch/recommend, GET /health, GET /info
```

Install serving extras: `pip install "corerec[serving]"`.

Example request:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_k": 10}'
```

Full API: {doc}`../api/serving`.

## API Deployment (custom Flask / FastAPI)

Use this only when you need custom middleware or auth beyond `ModelServer`.

### Flask API Example

```python
from flask import Flask, request, jsonify
from corerec.engines.dcn import DCN

app = Flask(__name__)
model = DCN.load("artifacts/production_dcn")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    score = model.predict(user_id=data['user_id'], item_id=data['item_id'])
    return jsonify({'score': float(score)})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    recs = model.recommend(user_id=data['user_id'], top_k=data.get('top_k', 10))
    return jsonify({'recommendations': recs})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from corerec.engines.dcn import DCN

app = FastAPI()
model = DCN.load("artifacts/production_dcn")

class PredictRequest(BaseModel):
    user_id: int
    item_id: int

class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        score = model.predict(user_id=request.user_id, item_id=request.item_id)
        return {"score": float(score)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend")
def recommend(request: RecommendRequest):
    try:
        recs = model.recommend(user_id=request.user_id, top_k=request.top_k)
        return {"recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

## Performance Optimization

### Caching Recommendations

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_recommendations(user_id, top_k):
    return model.recommend(user_id=user_id, top_k=top_k)
```

### Batch Processing

```python
def batch_recommend(user_ids, top_k=10):
    return {uid: model.recommend(user_id=uid, top_k=top_k) for uid in user_ids}
```

## Monitoring and Logging

```python
import logging
import time

logger = logging.getLogger(__name__)

def predict_with_timing(user_id, item_id):
    t0 = time.time()
    score = model.predict(user_id=user_id, item_id=item_id)
    logger.info("predict user=%s item=%s score=%s ms=%.2f", user_id, item_id, score, (time.time()-t0)*1000)
    return score
```

## Error Handling

```python
from corerec.api.exceptions import ModelNotFittedError, RecommendationError

def safe_predict(user_id, item_id):
    try:
        return model.predict(user_id=user_id, item_id=item_id)
    except ModelNotFittedError:
        logger.error("Model not fitted")
        return None
    except RecommendationError as e:
        logger.error("Recommendation error: %s", e)
        return None
```

## Model Versioning

```python
from pathlib import Path

MODEL_DIR = Path("artifacts/models")

def save_versioned_model(model, version):
    path = MODEL_DIR / f"model_v{version}"
    model.save(str(path))
    return path

def load_latest_model():
    dirs = sorted(MODEL_DIR.glob("model_v*"))
    return DCN.load(str(dirs[-1])) if dirs else None
```

## Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install "corerec[serving]"
COPY serve.py .
COPY artifacts/production_dcn/ ./artifacts/production_dcn/
CMD ["python", "serve.py"]
```

`serve.py`:

```python
from corerec.serving import ModelServer
from corerec.engines.dcn import DCN

model = DCN.load("artifacts/production_dcn")
ModelServer(model, host="0.0.0.0", port=8000).start()
```

## See Also

- [ModelServer API](../api/serving.md)
- [Safe bundle persistence](../user_guide/safe_bundle_persistence.md)
- [Basic Usage](basic_usage.md)
- [Pipeline tutorial](../tutorials/pipeline_tutorial.md)
