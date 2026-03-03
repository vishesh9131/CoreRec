# DCN_base Tutorial: Deep & Cross Network Base

```{admonition} Sandbox Model — Experimental
:class: warning
This model is part of the CoreRec **Sandbox** — an experimental collection of algorithms for research and exploration. Sandbox models are **not production-tested** and may have incomplete implementations. For production-ready models, see the [Production Models](../models/index.md#production-models-tested--stable).
```

## Introduction

**DCN_base** is a Neural Network Models model for recommendation systems. This model implements Deep & Cross Network Base.

## How DCN_base Works

### Architecture

DCN_base uses a sophisticated architecture for recommendation tasks.

### Mathematical Foundation

The model learns user and item representations for prediction.

## Reference Implementation

```{admonition} Implementation Note
:class: info
The code examples below are **reference implementations** for learning purposes. This is a sandbox model and has not been production-tested. Verify behavior thoroughly before using in production.
```


### Step 1: Import and Load Data

```python
from corerec.engines.neural_network.dcn_base import DCN_base
import cr_learn
import numpy as np

# Load dataset
data = cr_learn.load_dataset('movielens-100k')
print(f"Loaded {len(data.ratings)} ratings")

# Split data
train_data, test_data = data.train_test_split(test_size=0.2)
```

### Step 2: Initialize Model

```python
model = DCN_base(
    name="DCN_base_Model",
    embedding_dim=64,
    epochs=20,
    batch_size=256,
    learning_rate=0.001,
    verbose=True
)

print(f"Initialized {model.name}")
```

### Step 3: Train

```python
model.fit(
    user_ids=train_data.user_ids,
    item_ids=train_data.item_ids,
    ratings=train_data.ratings
)

print("Training complete!")
```

### Step 4: Predict

```python
# Single prediction
score = model.predict(user_id=1, item_id=100)
print(f"Predicted score: {score:.3f}")

# Batch predictions
pairs = [(1, 100), (2, 200), (3, 300)]
scores = model.batch_predict(pairs)
for (uid, iid), s in zip(pairs, scores):
    print(f"User {uid}, Item {iid}: {s:.3f}")
```

### Step 5: Recommend

```python
# Get top-10 recommendations
recommendations = model.recommend(
    user_id=1,
    top_k=10,
    exclude_items=train_data.get_user_items(1)
)

print(f"Top-10 recommendations for User 1:")
for rank, item_id in enumerate(recommendations, 1):
    print(f"  {rank}. Item {item_id}")
```

### Step 6: Evaluate

```python
from corerec.metrics import rmse, ndcg_at_k

# Rating prediction
predictions = [model.predict(u, i) for u, i, r in test_data]
test_rmse = rmse(test_data.ratings, predictions)
print(f"Test RMSE: {test_rmse:.4f}")

# Ranking quality
ndcg = ndcg_at_k(model, test_data, k=10)
print(f"NDCG@10: {ndcg:.4f}")
```

### Step 7: Save & Load

```python
# Save model
model.save('dcn_base_model.pkl')

# Load model
loaded = DCN_base.load('dcn_base_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use DCN_base

✅ Best for datasets with neural network models characteristics

### Best Practices

1. Start with default parameters\n2. Tune embedding_dim based on data\n3. Use early stopping\n4. Monitor validation metrics

## Scaling & Production Considerations

If you plan to move this sandbox model toward production use, consider:

- **Data Volume**: Test with realistic dataset sizes; sandbox implementations may not be optimized for large-scale data.
- **Distributed Training**: For datasets exceeding single-machine memory, consider wrapping the model in distributed training frameworks (e.g., PyTorch DDP, Horovod).
- **Hyperparameter Tuning**: Use systematic search (Optuna, Ray Tune) rather than manual tuning. The defaults in sandbox models are not tuned for any specific domain.
- **Serving Latency**: Profile inference time. Some architectures (e.g., attention-based) may need quantization or ONNX export for low-latency serving.
- **Monitoring**: Implement A/B testing and online metrics (CTR, engagement, diversity) before replacing an existing system.
- **Validation**: Write comprehensive unit tests and integration tests before deploying. Sandbox models do not have CI coverage.

## Further Reading

