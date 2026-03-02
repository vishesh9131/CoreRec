# Model Serving

Production REST API and batch serving for recommendation models.

## Quick Start

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
