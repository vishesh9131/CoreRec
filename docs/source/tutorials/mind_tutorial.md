# MIND Tutorial: MIND

## Introduction

**MIND** is a recommendation model

## How MIND Works

### Architecture

MIND (Multi-Interest Network with Dynamic Routing) uses a multi-interest extraction layer with dynamic routing mechanism to capture diverse user interests.

**Key Components:**
1. **Multi-Interest Extractor Layer**: Uses capsule network with dynamic routing
2. **Label-Aware Attention**: Attends to relevant interests for target item
3. **Interest Aggregation**: Combines multiple interest representations

**Architecture Flow:**
```
User Behavior → Embedding → Multi-Interest Capsules → Label-Aware Attention → Prediction
```

### Mathematical Foundation

**Multi-Interest Extraction:**
```
e_i = Embed(item_i)
interests = Capsule([e_1, e_2, ..., e_n])  # B × K × d
where K = number of interests
```

**Dynamic Routing:**
```
c_ij = exp(b_ij) / Σ_k exp(b_ik)  # routing coefficients
s_j = Σ_i c_ij * u_i             # interest capsule
v_j = squash(s_j)                # activation
```

**Label-Aware Attention:**
```
a_i = softmax(e_target^T · interest_i)
user_repr = Σ a_i · interest_i
score = σ(user_repr^T · e_target)
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.mind import MIND
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split

data = ml_1m.load()
ratings_df = data['ratings']
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

train_users = train_df['user_id'].values.tolist()
train_items = train_df['movie_id'].values.tolist()
train_ratings = train_df['rating'].values.tolist()
timestamps = train_df['timestamp'].values.tolist() if 'timestamp' in train_df else None

print(f"Loaded {len(ratings_df)} ratings")
```

### Step 2: Initialize Model

```python
model = MIND(
    embedding_dim=64,
    num_interests=4,
    epochs=5,
    batch_size=256,
    learning_rate=0.001,
    verbose=True
)
```

### Step 3: Train

```python
model.fit(
    user_ids=train_users,
    item_ids=train_items,
    ratings=train_ratings,
    timestamps=timestamps
)

print("Training complete!")
```

### Step 4: Predict

```python
sample_user = train_users[0]
sample_item = train_items[0]
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
import numpy as np

test_users = test_df['user_id'].values[:100]
test_items = test_df['movie_id'].values[:100]
test_ratings = test_df['rating'].values[:100]
test_pred = [model.predict(u, i) for u, i in zip(test_users, test_items)]
rmse = np.sqrt(mean_squared_error(test_ratings, test_pred))
print(f"Sample Test RMSE: {rmse:.4f}")
```

### Step 7: Save & Load

```python
model.save('artifacts/mind_model')

loaded = MIND.load('artifacts/mind_model')
recs = loaded.recommend(sample_user, top_k=5)
print(f"Loaded model recommendations: {recs}")
```

## Advanced Usage

### Feature Engineering

Add model-specific advanced usage here.

## Key Takeaways

### When to Use MIND

✅ **Ideal For:**
- E-commerce with diverse user interests (fashion, electronics, books)
- Multi-category recommendations
- Users with varied browsing patterns
- Capturing interest evolution over time

❌ **Not Ideal For:**
- Single-domain recommendations
- Very sparse data (<100 items per user)
- Real-time systems (slower than simpler models)

### Best Practices

1. **Number of Interests (K)**: Start with K=4, increase for diverse catalogs
2. **Routing Iterations**: 3 iterations sufficient for most cases
3. **Sequence Length**: Use 20-50 recent items
4. **Interest Regularization**: Add diversity loss to prevent collapsed interests
5. **Training**: Use auxiliary losses for each interest capsule

### Performance Comparison

Compare MIND with similar models on your dataset.

## Further Reading

- [MIND API Reference](../api/engines.md)
- Paper: See original paper for details
