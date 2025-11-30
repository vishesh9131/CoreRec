# LightGCN Tutorial: Light Graph Convolutional Network

## Introduction

**LightGCN** simplifies graph convolutional networks by removing feature transformation and non-linear activation, keeping only neighborhood aggregation for collaborative filtering.

**Paper**: He et al. 2020 - LightGCN: Simplifying and Powering Graph Convolution Network

## How LightGCN Works

### Architecture

**Simplified GCN for Collaborative Filtering:**

1. **Graph Construction**: User-item bipartite graph
2. **Light Graph Convolution** (L layers):
   - Pure neighborhood aggregation
   - NO feature transformation
   - NO non-linear activation
3. **Layer Combination**: Weighted sum of all layers
4. **Prediction**: Inner product of final embeddings

**Key Innovation**: Shows that feature transformation and activation hurt CF performance

**Architecture:**
```
Graph → Embed → Aggregate(1) → ... → Aggregate(L) → Combine Layers → Predict
                   ↓                                        ↓
           (no transformation)                    (weighted average)
```

### Mathematical Foundation

**Light Graph Convolution Layer:**
```
e_u^(k+1) = Σ_{i ∈ N_u} (1/√|N_u||N_i|) · e_i^(k)
e_i^(k+1) = Σ_{u ∈ N_i} (1/√|N_i||N_u|) · e_u^(k)
```

**NO transformation matrix W**
**NO activation function σ**

**Layer Combination:**
```
e_u = Σ_{k=0}^K α_k · e_u^(k)
e_i = Σ_{k=0}^K α_k · e_i^(k)
```
where α_k = 1/(K+1) (uniform weighting)

**Prediction:**
```
ŷ_ui = e_u^T · e_i
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.lightgcn import LightGCN
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
model = LightGCN(
    name="LightGCN_Model",
    embedding_dim=64,
    num_layers=3,
    reg_weight=1e-4,
    negative_samples=1000,
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
model.save('lightgcn_model.pkl')

# Load model
loaded = LightGCN.load('lightgcn_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use LightGCN

✅ **Perfect For:**
- Large-scale collaborative filtering (millions of users/items)
- Implicit feedback data
- When you want GNN benefits with MF simplicity
- Sparse bipartite graphs
- Top-N recommendation at scale

❌ **Avoid When:**
- Have rich node features → use full GCN
- Heterogeneous graphs with multiple edge types
- Need to model side information
- Graph structure is not beneficial

### Best Practices

1. **Number of Layers**: 2-4 layers (3 often optimal)
2. **Embedding Dimension**: 64 standard, 128-256 for large datasets
3. **Layer Combination**: Uniform weights work well (α_k = 1/(K+1))
4. **Negative Sampling**: Sample 1000-2000 negatives per positive (more than NCF!)
5. **Dropout**: Not needed (architecture is simple enough)
6. **Learning Rate**: 0.001 with decay
7. **Batch Size**: Large (2048-8192) for stability
8. **Regularization**: L2=1e-4 on embeddings

## Further Reading

- Paper: He et al. 2020 - LightGCN: Simplifying and Powering Graph Convolution Network
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
