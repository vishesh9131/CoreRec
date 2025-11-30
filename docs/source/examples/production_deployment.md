# Production Deployment Guide

This guide covers best practices for deploying CoreRec models in production environments.

## Model Serialization

### Saving Models for Production

```python
from corerec.engines.dcn import DCN

# Train model
model = DCN(embedding_dim=64, epochs=20)
model.fit(user_ids, item_ids, ratings)

# Save for production
model.save('production_model.pkl')

# Include metadata
metadata = {
    'version': '1.0.0',
    'training_date': '2024-01-01',
    'dataset': 'production_data',
    'performance_metrics': {
        'rmse': 0.85,
        'mae': 0.65
    }
}
model.save('production_model.pkl', metadata=metadata)
```

### Loading Models in Production

```python
from corerec.engines.dcn import DCN

# Load model
model = DCN.load('production_model.pkl')

# Verify model is ready
assert model.is_fitted, "Model must be fitted"

# Use for predictions
score = model.predict(user_id=1, item_id=10)
```

## API Deployment

### Flask API Example

```python
from flask import Flask, request, jsonify
from corerec.engines.dcn import DCN

app = Flask(__name__)

# Load model at startup
model = DCN.load('production_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_id = data['user_id']
    item_id = data['item_id']
    
    try:
        score = model.predict(user_id=user_id, item_id=item_id)
        return jsonify({'score': float(score)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data['user_id']
    top_k = data.get('top_k', 10)
    
    try:
        recommendations = model.recommend(user_id=user_id, top_k=top_k)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from corerec.engines.dcn import DCN

app = FastAPI()

# Load model at startup
model = DCN.load('production_model.pkl')

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
        recommendations = model.recommend(
            user_id=request.user_id,
            top_k=request.top_k
        )
        return {"recommendations": recommendations}
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

# Use cached version
recommendations = get_cached_recommendations(user_id=1, top_k=10)
```

### Batch Processing

```python
# Process multiple requests at once
def batch_recommend(user_ids, top_k=10):
    results = {}
    for user_id in user_ids:
        results[user_id] = model.recommend(user_id=user_id, top_k=top_k)
    return results
```

## Monitoring and Logging

### Logging Predictions

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_with_logging(user_id, item_id):
    score = model.predict(user_id=user_id, item_id=item_id)
    logger.info(f"Prediction: user={user_id}, item={item_id}, score={score}")
    return score
```

### Performance Monitoring

```python
import time

def predict_with_timing(user_id, item_id):
    start_time = time.time()
    score = model.predict(user_id=user_id, item_id=item_id)
    elapsed = time.time() - start_time
    
    logger.info(f"Prediction took {elapsed:.4f}s")
    return score
```

## Error Handling

### Robust Error Handling

```python
from corerec.api.exceptions import (
    ModelNotFittedError,
    RecommendationError,
    InvalidDataError
)

def safe_predict(user_id, item_id):
    try:
        return model.predict(user_id=user_id, item_id=item_id)
    except ModelNotFittedError:
        logger.error("Model not fitted")
        return None
    except RecommendationError as e:
        logger.error(f"Recommendation error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None
```

## Model Versioning

### Version Management

```python
import os
from pathlib import Path

MODEL_DIR = Path('models')
VERSION = '1.0.0'

def save_versioned_model(model, version):
    model_path = MODEL_DIR / f'model_v{version}.pkl'
    model.save(str(model_path))
    return model_path

def load_latest_model():
    versions = sorted([f for f in MODEL_DIR.glob('model_v*.pkl')])
    if versions:
        latest = versions[-1]
        return DCN.load(str(latest))
    return None
```

## Docker Deployment

### Dockerfile Example

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY production_model.pkl .

# Run application
CMD ["python", "app.py"]
```

## See Also

- [Basic Usage](basic_usage.md) - Basic examples
- [Advanced Usage](advanced_usage.md) - Advanced patterns
- [Tutorials](../tutorials/index.md) - Model-specific tutorials

