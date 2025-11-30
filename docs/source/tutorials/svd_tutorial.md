# SVD Tutorial: Singular Value Decomposition

## Introduction

**SVD** decomposes the user-item rating matrix into user and item latent factor matrices using gradient descent optimization.

**Paper**: Simon Funk 2006 - Netflix Prize; Koren 2008 - Factorization Meets Neighborhood

## How SVD Works

### Architecture

**Classic Matrix Factorization:**

1. **Decomposition**: R ≈ U × V^T
   - U: user latent factors (m × k)
   - V: item latent factors (n × k)
2. **Prediction**: Rating = dot(user_factors, item_factors) + biases
3. **Optimization**: Minimize squared error with regularization

**Key Innovation**: Efficient factorization for sparse matrices

**Architecture:**
```
User u → [user_vector_u]                           dot product + biases → Prediction
Item i → [item_vector_i] /
```

### Mathematical Foundation

**Prediction Formula:**
```
r̂_ui = μ + b_u + b_i + q_i^T · p_u
```
where:
- μ: global mean rating
- b_u: user bias
- b_i: item bias
- p_u: user latent factors (k-dim)
- q_i: item latent factors (k-dim)

**Loss Function:**
```
L = Σ_{(u,i)∈R} (r_ui - r̂_ui)² + λ(||p_u||² + ||q_i||² + b_u² + b_i²)
```

**Gradient Updates:**
```
e_ui = r_ui - r̂_ui
p_u ← p_u + α(e_ui · q_i - λ · p_u)
q_i ← q_i + α(e_ui · p_u - λ · q_i)
b_u ← b_u + α(e_ui - λ · b_u)
b_i ← b_i + α(e_ui - λ · b_i)
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.svd import SVD
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
model = SVD(
    name="SVD_Model",
    n_factors=100,
    learning_rate=0.005,
    reg=0.02,
    n_epochs=30,
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
model.save('svd_model.pkl')

# Load model
loaded = SVD.load('svd_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use SVD

✅ **Excellent For:**
- Explicit ratings (1-5 stars)
- Baseline collaborative filtering
- Medium-scale datasets (10K-10M ratings)
- Rating prediction tasks
- Well-understood, interpretable results

❌ **Avoid When:**
- Implicit feedback (use BPR instead)
- Need deep interactions (use neural models)
- Sequential patterns (use RNNs)
- Very large scale (>100M ratings) - use ALS instead

### Best Practices

1. **Factors (k)**: Start with 20-100, more for complex patterns
2. **Learning Rate**: 0.005-0.01 typical
3. **Regularization**: 0.02-0.1 to prevent overfitting
4. **Initialization**: Random normal (0, 0.1)
5. **Biases**: Always include global and user/item biases!
6. **Convergence**: Monitor RMSE on validation set
7. **Early Stopping**:Stop when validation RMSE increases
8. **Epochs**: 20-50 usually sufficient

## Further Reading

- Paper: Simon Funk 2006 - Netflix Prize; Koren 2008 - Factorization Meets Neighborhood
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
