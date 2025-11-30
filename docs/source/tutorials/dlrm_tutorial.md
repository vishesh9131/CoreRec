# DLRM Tutorial: Deep Learning Recommendation Model

## Introduction

**DLRM** separates dense and sparse feature processing, using explicit pairwise dot product interactions between embeddings.

**Paper**: Naumov et al. 2019 - Deep Learning Recommendation Model (Facebook)

## How DLRM Works

### Architecture

**Parallel Dense/Sparse Processing:**

1. **Bottom MLP**: Process continuous features
2. **Embedding Layers**: Parallel lookup for categorical features
3. **Explicit Interactions**: Dot products between all embedding pairs
4. **Top MLP**: Process concatenated features + interactions

**Key Innovation**: Scalable architecture for production (Facebook, Pinterest)

**Architecture:**
```
Dense → Bottom MLP →                        [Dot Products] → Concat → Top MLP → Pred
Sparse → Embed×K → /
```

### Mathematical Foundation

**Dense Processing:**
```
z_dense = MLP_bottom(x_dense)
```

**Embedding Interactions** (all pairs):
```
I = {<e_i, e_j> : for all i < j}
where <·,·> is dot product
```

**Concatenation:**
```
z = [z_dense; e_1; e_2; ...; e_K; I]
```

**Final Prediction:**
```
ŷ = σ(MLP_top(z))
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.dlrm import DLRM
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
model = DLRM(
    name="DLRM_Model",
    bottom_mlp=[512, 256, 64],
    top_mlp=[512, 512, 256, 1],
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
model.save('dlrm_model.pkl')

# Load model
loaded = DLRM.load('dlrm_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use DLRM

✅ **Ideal For:**
- Large-scale CTR prediction (billions of users)
- Production systems (Facebook, Pinterest scale)
- Mix of dense & sparse features
- Parallelizable infrastructure
- High-throughput serving

❌ **Not For:**
- Small datasets (<100K)
- Pure collaborative filtering
- Sequential patterns
- Limited compute/memory

### Best Practices

1. **Bottom MLP**: [512, 256, 64]
2. **Top MLP**: [512, 512, 256, 1]
3. **Embedding Dim**: 16-128 (by cardinality)
4. **Batch Size**: 2048-8192 (very large!)
5. **Mixed Precision**: Use FP16 for speed
6. **Parallelization**: Embeddings in parallel
7. **Caching**: Cache popular item embeddings
8. **Hardware**: Multi-GPU or TPU

## Further Reading

- Paper: Naumov et al. 2019 - Deep Learning Recommendation Model (Facebook)
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
