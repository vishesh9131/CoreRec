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
model = MIND(
    name="MIND_Model",
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
model.save('mind_model.pkl')

# Load model
loaded = MIND.load('mind_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
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

- [MIND API Reference](../api/engines.rst#mind)
- Paper: See original paper for details
- [Code Examples](../examples/mind_advanced.md)
