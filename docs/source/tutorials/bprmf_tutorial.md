# BPRMF Tutorial: Bayesian Personalized Ranking Matrix Factorization

## Introduction

**BPRMF** combines BPR ranking optimization with matrix factorization for implicit feedback collaborative filtering.

**Paper**: Rendle et al. 2009 - BPR-MF

## How BPRMF Works

### Architecture

**Ranking-Optimized MF:**

1. **Matrix Factorization**: User × Item latent factors
2. **Pairwise Ranking**: BPR loss on (user, pos_item, neg_item) triplets
3. **Sampling**: Bootstrap sampling for efficiency
4. **Optimization**: SGD with ranking gradients

**Key Innovation**: Standard MF with BPR loss instead of pointwise loss

**Architecture:**
```
[U×I Factors] → <p_u, q_i> - <p_u, q_j> → BPR Loss
where i = positive, j = negative
```

### Mathematical Foundation

**Matrix Factorization:**
```
x̂_ui = <p_u, q_i> = Σ_f p_uf · q_if
```

**BPR-MF Loss:**
```
L = -Σ_{(u,i,j)} log σ(x̂_ui - x̂_uj) + λ||Θ||²
```

**Gradients:**
```
∂L/∂p_u = σ(-x̂_uij)(q_j - q_i) + λ·p_u
∂L/∂q_i = σ(-x̂_uij)·p_u + λ·q_i
∂L/∂q_j = σ(-x̂_uij)·(-p_u) + λ·q_j
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.bprmf import BPRMF
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
model = BPRMF(
    name="BPRMF_Model",
    n_factors=64,
    learning_rate=0.05,
    reg=0.01,
    num_negatives=3,
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
model.save('bprmf_model.pkl')

# Load model
loaded = BPRMF.load('bprmf_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use BPRMF

✅ **Perfect For:**
- Implicit feedback ranking
- Top-N recommendation
- When you want simple MF with ranking
- Cold-start items (better than pointwise)
- Medium-scale CF (100K-10M interactions)

❌ **Not For:**
- Explicit ratings (use SVD)
- Need features (use neural models)
- Very sparse data
- Sequential patterns (use RNN)

### Best Practices

1. **Factors**: 20-100 dimensions
2. **Learning Rate**: 0.05-0.1 (higher than SVD)
3. **Regularization**: 0.01-0.001
4. **Negative Samples**: 1-5 per positive
5. **Sampling**: Uniform or popularity-based
6. **Update Frequency**: Shuffle every epoch
7. **Initialization**: Small random ~N(0, 0.01)
8. **Convergence**: Monitor AUC, not loss

## Further Reading

- Paper: Rendle et al. 2009 - BPR-MF
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
