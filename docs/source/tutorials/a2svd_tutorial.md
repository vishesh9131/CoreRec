# A2SVD Tutorial: Adaptive Singular Value Decomposition

## Introduction

**A2SVD** extends SVD with adaptive regularization and neighborhood-based refinements for better generalization.

**Paper**: Koren 2008 - Factorization Meets Neighborhood

## How A2SVD Works

### Architecture

**Enhanced SVD with Adaptivity:**

1. **Base SVD**: Standard latent factor decomposition
2. **Adaptive Regularization**: Per-user/item regularization
3. **Neighborhood Integration**: Add implicit feedback signals
4. **Bias Modeling**: Global, user, item, and temporal biases

**Key Innovation**: Adaptive λ based on user/item activity

**Architecture:**
```
SVD Factors + Neighborhood Implicit + Temporal Bias → Prediction
```

### Mathematical Foundation

**Adaptive Prediction:**
```
r̂_ui = μ + b_u(t) + b_i(t) + q_i^T(p_u + |N(u)|^(-0.5) Σ_{j∈N(u)} y_j)
```
where:
- μ: global mean
- b_u(t), b_i(t): time-dependent biases
- p_u: user factors
- q_i: item factors
- y_j: implicit factors from neighbors

**Adaptive Regularization:**
```
λ_u = λ_0 + β/√|R_u|
λ_i = λ_0 + β/√|R_i|
```
Lower regularization for active users/items

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.a2svd import A2SVD
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
model = A2SVD(
    name="A2SVD_Model",
    n_factors=150,
    base_reg=0.02,
    adaptive_beta=0.5,
    n_neighbors=50,
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
model.save('a2svd_model.pkl')

# Load model
loaded = A2SVD.load('a2svd_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use A2SVD

✅ **Excellent For:**
- Explicit ratings with temporal dynamics
- Users/items with varying activity levels
- When simple SVD underfits active users
- Netflix-style datasets
- Combining explicit + implicit signals

❌ **Not For:**
- Pure implicit feedback (use BPR)
- Need real-time updates
- Very small datasets (regularization complex)
- When simple SVD works fine

### Best Practices

1. **Base Factors**: k=100-200
2. **Adaptive λ**: β=0.1-1.0
3. **Neighborhood Size**: 50-100 similar items
4. **Temporal Bins**: Week or month-level
5. **Learning Rate**: 0.001-0.01
6. **Epochs**: 30-50
7. **Combine Signals**: Weight implicit feedback 0.1-0.3
8. **Bias Priority**: Optimize biases first

## Further Reading

- Paper: Koren 2008 - Factorization Meets Neighborhood
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
