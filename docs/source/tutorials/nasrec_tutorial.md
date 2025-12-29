# NASRec Tutorial: NASRec

## Introduction

**NASRec** is a recommendation model

## How NASRec Works

### Architecture

NASRec (Neural Architecture Search for Recommendations) automatically discovers optimal neural architectures using reinforcement learning-based search.

**Search Space:**
1. **Operation Types**: Convolution, LSTM, Attention, MLP, Skip connections
2. **Layer Configurations**: Hidden sizes, activation functions
3. **Connection Patterns**: Sequential, residual, dense

**Search Process:**
1. Controller RNN proposes architectures
2. Train candidate on validation set
3. Use validation performance as reward
4. Update controller with policy gradient

### Mathematical Foundation

**Architecture Encoding:**
```
arch = Controller_RNN(random_state)
arch = [layer_1_type, layer_1_config, ..., layer_n_type, layer_n_config]
```

**Reward Function:**
```
R = α · NDCG@10 - β · latency - γ · params
where α, β, γ are balancing coefficients
```

**Controller Update:**
```
∇L = E[R(arch) · ∇log P(arch|θ)]
θ ← θ + η · ∇L
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.nasrec import NASRec
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
model = NASRec(
    name="NASRec_Model",
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
model.save('nasrec_model.pkl')

# Load model
loaded = NASRec.load('nasrec_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Advanced Usage

### Feature Engineering

Add model-specific advanced usage here.

## Key Takeaways

### When to Use NASRec

✅ **Ideal For:**
- Novel recommendation domains without established architectures
- Performance-critical applications
- Research and experimentation
- When you have significant compute budget

❌ **Not Ideal For:**
- Quick prototyping (search is slow)
- Limited compute resources
- Well-understood domains (use proven architectures)
- Production systems without retraining

### Best Practices

1. **Search Budget**: Minimum 50-100 architecture evaluations
2. **Early Stopping**: Stop unpromising architectures at 5 epochs
3. **Warm Start**: Initialize with known good architectures  
4. **Constrained Search**: Limit search space to reduce time
5. **Multi-Objective**: Balance performance,latency, model size

### Performance Comparison

Compare NASRec with similar models on your dataset.

## Further Reading

- [NASRec API Reference](../api/engines.rst#nasrec)
- Paper: See original paper for details
- [Code Examples](../examples/nasrec_advanced.md)
