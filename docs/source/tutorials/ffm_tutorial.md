# FFM Tutorial: Field-aware Factorization Machine

## Introduction

**FFM** extends FM with field-aware latent vectors where each feature has different embeddings for interacting with different fields.

**Paper**: Juan et al. 2016 - Field-aware Factorization Machines for CTR Prediction

## How FFM Works

### Architecture

**Field-Aware Interactions:**

1. **Feature Fields**: Group features (e.g., User, Item, Context)
2. **Field-Specific Embeddings**: v_{i,f_j} for feature i w.r.t. field j
3. **Pairwise Interactions**: Use appropriate field-specific vectors

**Key Innovation**: Different latent factors per field pair

**Example:**
- Publisher × Advertiser uses v_{pub,adv} and v_{adv,pub}
- Publisher × Gender uses v_{pub,gender} and v_{gender,pub}

### Mathematical Foundation

**FFM Formula:**
```
ŷ = w_0 + Σ_i w_i·x_i + Σ_i Σ_{j>i} <v_{i,f_j}, v_{j,f_i}>·x_i·x_j
```
where:
- v_{i,f_j}: latent vector for feature i w.r.t. field of feature j
- f_j: field of feature j

**vs Standard FM:**
```
FM:  <v_i, v_j>  # same vector always
FFM: <v_{i,f_j}, v_{j,f_i}>  # field-aware
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.ffm import FFM
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
model = FFM(
    name="FFM_Model",
    n_factors=4,
    learning_rate=0.2,
    reg=0.00002,
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
model.save('ffm_model.pkl')

# Load model
loaded = FFM.load('ffm_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use FFM

✅ **Perfect For:**
- CTR prediction with field structure
- Sparse categorical features
- When feature fields matter (User/Item/Context)
- Kaggle competitions
- Display advertising

❌ **Not For:**
- No natural field structure
- Dense numerical features
- Need deep interactions
- Very large scale (>10M features)

### Best Practices

1. **Latent Factors**: k=4-8 (smaller than FM!)
2. **Optimizer**: AdaGrad or FTRL
3. **Learning Rate**: 0.1-0.2 (higher than FM)
4. **Regularization**: λ = 0.00002 typical
5. **Early Stopping**: Monitor validation AUC
6. **Normalization**: Normalize continuous features
7. **Field Definition**: Carefully design fields
8. **Memory**: k×fields larger than FM

## Further Reading

- Paper: Juan et al. 2016 - Field-aware Factorization Machines for CTR Prediction
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
