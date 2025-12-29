# SASRec Tutorial: SASRec

## Introduction

**SASRec** is a recommendation model

## How SASRec Works

### Architecture

SASRec (Self-Attentive Sequential Recommendation) uses self-attention mechanism to model item-item transitions in user sequences.

**Core Innovation:** Replaces RNN/CNN with self-attention blocks for better long-range dependencies.

**Architecture:**
```
Item Sequence → Embedding → Positional Encoding → 
Multi-Head Self-Attention Blocks (×L) → Prediction Layer
```

**Multi-Head Attention Block:**
1. Self-attention with causal masking
2. Point-wise feed-forward network
3. Layer normalization
4. Residual connections

### Mathematical Foundation

**Self-Attention:**
```
Q = E · W^Q, K = E · W^K, V = E · W^V
Attention(Q,K,V) = softmax(QK^T / √d_k) · V
```

**Causal Masking** (prevents future leakage):
```
M_ij = {0 if i ≥ j, -∞ if i < j}
Attention = softmax((QK^T + M) / √d_k) · V  
```

**Position Encoding:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Prediction:**
```
r_i = [r_i^1; r_i^2; ...; r_i^h]  # concat heads
y_i,t = E_t^T · FFN(LN(r_i + E_i))
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.sasrec import SASRec
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
model = SASRec(
    name="SASRec_Model",
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
model.save('sasrec_model.pkl')

# Load model
loaded = SASRec.load('sasrec_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Advanced Usage

### Feature Engineering

Add model-specific advanced usage here.

## Key Takeaways

### When to Use SASRec

✅ **Ideal For:**
- Sequential user behavior (browsing, listening, watching)
- Session-based recommendations  
- Long sequences (50-200 items)
- Capturing long-range dependencies
- E-commerce, streaming platforms

❌ **Not Ideal For:**
- Very short sequences (<5 items)
- Static user-item ratings
- Graph-structured data
- When interpretability is critical

### Best Practices

1. **Sequence Length**: 50-200 items optimal
2. **Attention Blocks**: 2-4 blocks sufficient
3. **Attention Heads**: 2-4 heads
4. **Hidden Size**: 50-100 dimensions
5. **Dropout**: 0.2-0.5 for regularization
6. **Positional Encoding**: Essential for sequence order
7. **Learning Rate**: 0.001 with warmup (1000 steps)

### Performance Comparison

Compare SASRec with similar models on your dataset.

## Further Reading

- [SASRec API Reference](../api/engines.rst#sasrec)
- Paper: See original paper for details
- [Code Examples](../examples/sasrec_advanced.md)
