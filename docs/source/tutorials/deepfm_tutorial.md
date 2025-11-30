# DeepFM Tutorial: Factorization Machine with Deep Learning

## Introduction

**DeepFM** combines Factorization Machines for 2nd-order interactions with Deep Neural Networks for high-order interactions, sharing the same embedding layer.

**Paper**: Guo et al. 2017 - DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

## How DeepFM Works

### Architecture

**Shared Embedding + Two Components:**

1. **FM Component**: Models 2nd-order feature interactions
   - Linear part: first-order features
   - Interaction part: pairwise interactions via dot products
   - Same embeddings as DNN (parameter sharing!)

2. **Deep Component**: Models high-order interactions  
   - Stacked fully-connected layers
   - Learns complex non-linear combinations

3. **Shared Embedding Layer**: Both components use same embeddings
   - Reduces parameters
   - FM improves DNN's embedding learning

**Flow:**
```
Raw Features → Embedding Layer → [FM Component]                                                → Sum → Prediction
                      ↓            [DNN Component]/
                  (shared)
```

### Mathematical Foundation

**FM Component:**
```
y_FM = w_0 + Σ_i w_i·x_i + Σ_i Σ_j <v_i, v_j>·x_i·x_j
```
where `<v_i, v_j>` is dot product of embeddings

**DNN Component:**
```
a^(0) = [e_1, e_2, ..., e_m]  # concatenated embeddings
a^(l+1) = σ(W^(l) · a^(l) + b^(l))
y_DNN = W^(out) · a^(L) + b^(out)
```

**Final Prediction:**
```
ŷ = sigmoid(y_FM + y_DNN)
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.deepfm import DeepFM
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
model = DeepFM(
    name="DeepFM_Model",
    embedding_dim=32,
    deep_layers=[256, 128, 64],
    dropout=0.3,
    use_bn=True,
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
model.save('deepfm_model.pkl')

# Load model
loaded = DeepFM.load('deepfm_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use DeepFM

✅ **Ideal For:**
- CTR prediction in advertising
- Sparse high-dimensional features (categorical data)
- Click-through rate estimation  
- App recommendation
- When you need both low-order and high-order interactions
- Large-scale industrial applications

❌ **Avoid When:**
- Dense feature spaces (images, text)
- Sequential patterns dominate → use RNN/Transformer
- Graph structure important → use GNN  
- Very small datasets (<10K) → use simpler models

### Best Practices

1. **Embedding Dimension**: 8-32 for sparse data, 64-128 for rich features
2. **DNN Architecture**: [256, 128, 64] for complex patterns
3. **Dropout**: 0.3-0.5 to prevent overfitting
4. **Batch Normalization**: Apply after each dense layer
5. **Negative Sampling**: Essential for implicit feedback
6. **Batch Size**: 512-2048 for large datasets
7. **Optimizer**: Adam with lr=0.001
8. **Feature Hashing**: Use for very high-cardinality features

## Further Reading

- Paper: Guo et al. 2017 - DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
