# SASRec Tutorial: Self-Attentive Sequential Recommendation

## Introduction

**SASRec** uses self-attention mechanism to model sequential user behavior, capturing both short and long-term patterns.

**Paper**: Kang & McAuley 2018 - Self-Attentive Sequential Recommendation

## How SASRec Works

### Architecture

**Transformer-Based Sequential Modeling:**

1. **Input**: Sequence of user's items [i₁, i₂, ..., iₙ]
2. **Embedding Layer**: Item embeddings + positional encoding
3. **Self-Attention Blocks** (stacked L times):
   - Multi-head self-attention with causal masking
   - Point-wise feed-forward network
   - Layer normalization + Residual connections
4. **Prediction**: Final item representation

**Key Innovation**: Self-attention captures long-range dependencies better than RNN/CNN

**Architecture:**
```
Items → Embed + Position → Self-Attention×L → Predict Next
```

### Mathematical Foundation

**Self-Attention:**
```
Q = EW^Q, K = EW^K, V = EW^V
Attention(Q,K,V) = softmax(QK^T/√d_k + M) V
```

**Causal Mask M** (prevent attending to future):
```
M_ij = { 0    if i ≥ j
       {-∞    if i < j
```

**Multi-Head Attention:**
```
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)  
MultiHead = Concat(head_1, ..., head_h)W^O
```

**Positional Encoding:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Feed-Forward:**
```
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.sasrec import SASRec
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset with CORRECT API
data = ml_1m.load()  # Returns dict with 'ratings', 'users', 'movies'
ratings_df = data['ratings']  # DataFrame with user_id, movie_id, rating, timestamp

print(f"Loaded {len(ratings_df)} ratings")

# Split data
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Extract arrays for model
train_users = train_df['user_id'].values
train_items = train_df['movie_id'].values
train_ratings = train_df['rating'].values

test_users = test_df['user_id'].values
test_items = test_df['movie_id'].values
test_ratings = test_df['rating'].values
```

### Step 2: Initialize Model

```python
model = SASRec(
    name="SASRec_Model",
    hidden_units=64,           # Embedding dimension (NOT embedding_dim!)
    num_blocks=2,              # Number of transformer blocks
    num_heads=2,               # Number of attention heads
    max_seq_length=50,         # Maximum sequence length
    dropout_rate=0.2,          # Dropout rate
    position_encoding="learned",
    batch_size=128,
    num_epochs=20,
    learning_rate=0.001,
    verbose=True
)

print(f"Initialized {model.name}")
```

### Step 3: Train

```python
model.fit(
    user_ids=train_users,
    item_ids=train_items,
    ratings=train_ratings
)

print("Training complete!")
```

### Step 4: Predict

```python
# Single prediction
score = model.predict(user_id=1, item_id=100)
print(f"Predicted score: {score:.3f}")

# Batch predictions
test_predictions = model.batch_predict(list(zip(test_users[:100], test_items[:100])))
```

### Step 5: Recommend

```python
# Get top-10 recommendations for user
user_id = 1
recommendations = model.recommend(
    user_id=user_id,
    top_k=10
)

print(f"Top-10 recommendations for User {user_id}:")
for rank, item_id in enumerate(recommendations, 1):
    print(f"  {rank}. Item {item_id}")
```

### Step 6: Evaluate

```python
from sklearn.metrics import mean_squared_error
import numpy as np

# Predict all test ratings
test_pred = [model.predict(u, i) for u, i in zip(test_users, test_items)]
rmse = np.sqrt(mean_squared_error(test_ratings, test_pred))
print(f"Test RMSE: {rmse:.4f}")
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

## Key Takeaways

### When to Use SASRec

✅ **Perfect For:**
- Sequential user behavior (browsing, watching, listening)
- Session-based recommendations
- Long sequences (50-200 items)
- Capturing long-range dependencies
- Next-item prediction
- E-commerce, streaming platforms, news

❌ **Not Suitable For:**
- Very short sequences (<5 items)
- Static ratings (no sequence)
- Graph-structured data
- When interpretability is critical
- Memory-constrained systems

### Best Practices

1. **Sequence Length**: 50-200 items optimal (truncate longer)
2. **Attention Blocks**: 2-4 blocks (L=2 often sufficient)
3. **Attention Heads**: 1-4 heads (h=2 works well)
4. **Hidden Dimension**: 50-100 (d model)
5. **Dropout**: 0.2-0.5 for regularization
6. **Positional Encoding**: ESSENTIAL (order matters!)
7. **Learning Rate**: 0.001 with linear warmup (1000 steps)
8. **Batch Size**: Large (128-512) with padding
9. **Negative Sampling**: Sample popular items as negatives

## Further Reading

- Paper: Kang & McAuley 2018 - Self-Attentive Sequential Recommendation
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
