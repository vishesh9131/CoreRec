# AFM Tutorial: Attentional Factorization Machine

## Introduction

**AFM** adds attention mechanism to FM to learn the importance of different feature interactions automatically.

**Paper**: Xiao et al. 2017 - Attentional Factorization Machines

## How AFM Works

### Architecture

**FM + Attention:**

1. **Feature Embeddings**: v_i for each feature
2. **Element-wise Products**: v_i ⊙ v_j for all pairs
3. **Attention Network**: Learn weights for each interaction
4. **Weighted Sum**: Combine interactions by importance

**Key Innovation**: Not all feature interactions equally important

**Architecture:**
```
Features → Embed → [Pairwise Products] → Attention → Sum → Linear → Pred
```

### Mathematical Foundation

**Attention-Based Pooling:**
```
α_ij = h^T ReLU(W(v_i ⊙ v_j) + b)
a_ij = exp(α_ij) / Σ exp(α_kl)
```

**AFM Formula:**
```
ŷ = w_0 + Σ w_i·x_i + p^T Σ_{i,j} a_ij(v_i ⊙ v_j)x_i x_j
```
where:
- ⊙ is element-wise product
- a_ij is attention weight for (i,j) interaction
- p is projection vector

**vs FM:**
```
FM:  Σ <v_i, v_j>  # equal weight
AFM: Σ a_ij(v_i ⊙ v_j)  # learned weight
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.afm import AFM
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
model = AFM(
    name="AFM_Model",
    embedding_dim=10,
    attention_dim=64,
    dropout=0.6,
    reg=1e-6,
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
model.save('afm_model.pkl')

# Load model
loaded = AFM.load('afm_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use AFM

✅ **Perfect For:**
- Sparse feature interactions
- CTR prediction
- When interactions vary in importance
- Interpretable recommendations
- Feature engineering insights

❌ **Not For:**
- Need very deep interactions (use DeepFM)
- Sequential patterns (use RNN)
- Dense feature spaces
- Very large scale (slower than FM)

### Best Practices

1. **Embedding Dim**: 8-16 (smaller than FM)
2. **Attention Dim**: 32-64
3. **Dropout**: 0.5-0.7 on attention layer
4. **Regularization**: L2 = 1e-6 to 1e-4
5. **Learning Rate**: 0.001-0.01
6. **Batch Normalization**: After attention
7. **Activation**: ReLU in attention network
8. **Visualization**: Analyze learned attention weights

## Further Reading

- Paper: Xiao et al. 2017 - Attentional Factorization Machines
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
