# BPR Tutorial: Bayesian Personalized Ranking

## Introduction

**BPR** optimizes for pairwise ranking using a Bayesian approach, assuming users prefer observed items over unobserved ones.

**Paper**: Rendle et al. 2009 - BPR: Bayesian Personalized Ranking from Implicit Feedback

## How BPR Works

### Architecture

**Pairwise Ranking Optimization:**

1. **Assumption**: User prefers observed item i over unobserved item j
2. **Pairwise Comparisons**: For each user, create positive-negative pairs
3. **Matrix Factorization**: Learn user/item embeddings
4. **Ranking Loss**: Optimize probability that positive > negative

**Key Innovation**: Learns to rank rather than predict ratings

**Architecture:**
```
User u, Item i (pos), Item j (neg) → Embeddings → 
    Score(u,i) - Score(u,j) → Sigmoid → Loss
```

### Mathematical Foundation

**Scoring Function:**
```
x̂_uij = x̂_ui - x̂_uj
where x̂_ui = <p_u, q_i>  # dot product
```

**BPR-Opt Criterion:**
```
BPR-OPT = Σ_{(u,i,j)} ln σ(x̂_uij) - λ||Θ||²
```
where σ is sigmoid, Θ are parameters

**Gradient Update:**
```
∂BPR/∂θ = Σ_{(u,i,j)} σ(-x̂_uij) · ∂x̂_uij/∂θ - λθ
```

**Sampling Strategy:**
- For each user u
- Sample observed item i (positive)
- Sample unobserved item j (negative)
- Update to maximize x̂_ui - x̂_uj

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.bpr import BPR
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
model = BPR(
    name="BPR_Model",
    embedding_dim=50,
    learning_rate=0.05,
    reg=0.01,
    num_negatives=5,
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
model.save('bpr_model.pkl')

# Load model
loaded = BPR.load('bpr_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use BPR

✅ **Perfect For:**
- Implicit feedback (clicks, views, purchases)
- Top-N recommendation rankings
- When you care about order, not exact ratings
- Large item catalogs (need to rank many items)
- Personalized rankings

❌ **Not For:**
- Explicit ratings (use rating prediction models)
- Need probability calibration
- Very sparse data (try simpler MF first)
- Sequential patterns (use SASRec)

### Best Practices

1. **Negative Sampling**: Sample 1-10 negatives per positive
2. **Learning Rate**: 0.01-0.05 typical (higher than supervised)
3. **Regularization**: λ = 0.01-0.001
4. **Sampling Strategy**: Uniform or popularity-based for negatives
5. **Batch Size**: 256-1024 for stability
6. **Convergence**: Monitor pairwise accuracy, not loss
7. **Embedding Dim**: 20-100 usually sufficient
8. **Update Frequency**: Bootstrap sampling each epoch

## Further Reading

- Paper: Rendle et al. 2009 - BPR: Bayesian Personalized Ranking from Implicit Feedback
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
