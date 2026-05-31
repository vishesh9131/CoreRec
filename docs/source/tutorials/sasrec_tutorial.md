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

SASRec needs a user–item **interaction matrix** (not raw rating triplets).

```python
from corerec.engines.sasrec import SASRec
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split
import numpy as np

data = ml_1m.load()
ratings_df = data['ratings']
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Build interaction matrix for sequential training
user_list = sorted(train_df['user_id'].unique())
item_list = sorted(train_df['movie_id'].unique())
user_idx = {u: i for i, u in enumerate(user_list)}
item_idx = {it: j for j, it in enumerate(item_list)}

train_mat = np.zeros((len(user_list), len(item_list)), dtype=np.float32)
for _, row in train_df.iterrows():
    train_mat[user_idx[row['user_id']], item_idx[row['movie_id']]] = 1.0

print(f"Loaded {len(ratings_df)} ratings → {len(user_list)} users, {len(item_list)} items")
```

### Step 2: Initialize Model

```python
model = SASRec(
    hidden_units=64,
    num_blocks=2,
    num_heads=2,
    max_seq_length=50,
    num_epochs=5,
    batch_size=256,
    learning_rate=0.001,
    verbose=True
)
```

### Step 3: Train

```python
model.fit(user_list, item_list, train_mat)

print("Training complete!")
```

### Step 4: Predict

```python
sample_user = user_list[0]
sample_item = item_list[0]
score = model.predict(sample_user, sample_item)
print(f"Predicted score: {score:.3f}")
```

### Step 5: Recommend

```python
recommendations = model.recommend(user_id=sample_user, top_k=10)

print(f"Top-10 recommendations for User {sample_user}:")
for rank, item_id in enumerate(recommendations, 1):
    print(f"  {rank}. Item {item_id}")
```

### Step 6: Evaluate

```python
from sklearn.metrics import mean_squared_error

# Sample evaluation on test pairs
test_users = test_df['user_id'].values[:100]
test_items = test_df['movie_id'].values[:100]
test_ratings = test_df['rating'].values[:100]
test_pred = [model.predict(u, i) for u, i in zip(test_users, test_items)]
rmse = np.sqrt(mean_squared_error(test_ratings, test_pred))
print(f"Sample Test RMSE: {rmse:.4f}")
```

### Step 7: Save & Load

```python
model.save('artifacts/sasrec_model')

loaded = SASRec.load('artifacts/sasrec_model')
recs = loaded.recommend(sample_user, top_k=5)
print(f"Loaded model recommendations: {recs}")
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

- [SASRec API Reference](../api/engines.md)
- Paper: See original paper for details
