# MIND Tutorial: Multi-Interest Network with Dynamic Routing

## Introduction

**MIND** captures diverse user interests using a multi-interest extraction layer with capsule networks and dynamic routing.

**Paper**: Li et al. 2019 - Multi-Interest Network with Dynamic Routing

## How MIND Works

### Architecture

**Capsule-Based Multi-Interest Extraction:**

1. **Behavior Embedding**: Embed user's historical items
2. **Multi-Interest Capsules**: Extract K diverse interests via capsule network
3. **Dynamic Routing**: Route item embeddings to interest capsules
4. **Label-Aware Attention**: Attend to relevant interests for target item
5. **Aggregation**: Combine attended interests for prediction

**Key Innovation**: Models users as having multiple interests rather than single preference vector

**Architecture Flow:**
```
Items → Embed → Capsule(K interests) → Label Attention → Predict
                     ↓
              Dynamic Routing (3 iterations)
```

### Mathematical Foundation

**Interest Capsule Extraction:**
```
S_j = Σ_i c_ij · û_i|j    # weighted sum
v_j = squash(S_j)         # capsule activation
where û_i|j = W · e_i     # transformed embeddings
```

**Dynamic Routing (3 iterations):**
```
b_ij ← b_ij + û_i|j · v_j     # update routing logits
c_ij = softmax_j(b_ij)         # routing coefficients
```

**Label-Aware Attention:**
```
score_k = softmax(e_target^T · interest_k / √d)
user_vector = Σ_k score_k · interest_k
ŷ = σ(user_vector^T · e_target)
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.mind import MIND
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
model = MIND(
    name="MIND_Model",
    embedding_dim=64,
    num_interests=4,
    routing_iterations=3,
    seq_len=50,
    use_interest_regularization=True,
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
model.save('mind_model.pkl')

# Load model
loaded = MIND.load('mind_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use MIND

✅ **Perfect For:**
- E-commerce with diverse catalogs (fashion + electronics + books)
- Users with multi-faceted interests
- Capturing interest drift/evolution
- Session-based recommendations with multiple intent
- Personalized diverse recommendations

❌ **Avoid For:**
- Single-domain recommendations (music only, movies only)
- Very sparse data (<50 items per user)
- Users with narrow interests
- Real-time serving (more complex than single-vector models)

### Best Practices

1. **Number of Interests (K)**: 4-8 typical, more for very diverse catalogs
2. **Routing Iterations**: 3 iterations standard (more doesn't help much)
3. **Capsule Dimension**: Same as embedding (64-128)
4. **Sequence Length**: Use 20-100 recent items
5. **Interest Regularization**: Add orthogonality loss to prevent collapse
6. **Auxiliary Losses**: Train each interest capsule independently
7. **Serving**: Cache interests, only attend at inference
8. **Hard Negative Mining**: Sample from different interest clusters

## Further Reading

- Paper: Li et al. 2019 - Multi-Interest Network with Dynamic Routing
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
