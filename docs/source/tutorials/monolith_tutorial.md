# Monolith Tutorial: Monolith: Real Time Recommendation System

## Introduction

**Monolith** designed for online training with collisionless embedding tables and dynamic feature eviction.

**Paper**: Liu et al. 2022 - Monolith: Real Time Recommendation System With Collisionless Embedding Table (ByteDance)

## How Monolith Works

### Architecture

**Online Learning Architecture:**

1. **Collisionless Embedding**: Hash-free, direct mapping
2. **Dynamic Eviction**: Remove stale features to save memory
3. **Online Training**: Updates in real-time streaming
4. **Parameter Server**: Distributed parameter synchronization

**Key Innovation**: Handling non-stationary distribution with real-time updates

**Architecture:**
```
Stream -> [Collisionless Embed] -> [Deep Network] -> Update
                  ↑
           [Eviction Policy]
```

### Mathematical Foundation

**Collisionless Hash:**
```
idx = Map(feature_value)  # No modulo collision
```

**Frequency Filter:**
Only create embedding if `count(feature) > threshold`.

**Adaptive Learning Rate:**
```
lr_t = lr_0 / sqrt(Σ g_t^2)
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.monolith import Monolith
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
model = Monolith(
    name="Monolith_Model",
    embedding_dim=64,
    collisionless=True,
    eviction_threshold=10,
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
model.save('monolith_model.pkl')

# Load model
loaded = Monolith.load('monolith_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use Monolith

✅ **Perfect For:**
- Real-time recommendation (TikTok scale)
- Streaming data
- Non-stationary user interests
- Massive sparse features
- Low-latency updates

❌ **Not For:**
- Batch processing
- Static datasets
- Small scale systems

### Best Practices

1. **Eviction**: Bloom filter for frequency
2. **Sync**: Async parameter updates
3. **Batch Size**: Small for online (or mini-batch)
4. **Fault Tolerance**: Snapshotting

## Further Reading

- Paper: Liu et al. 2022 - Monolith: Real Time Recommendation System With Collisionless Embedding Table (ByteDance)
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
