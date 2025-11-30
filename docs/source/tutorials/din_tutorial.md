# DIN Tutorial: Deep Interest Network

## Introduction

**DIN** uses local activation (attention) to adaptively learn user interests from historical behaviors based on the candidate item.

**Paper**: Zhou et al. 2018 - Deep Interest Network for Click-Through Rate Prediction

## How DIN Works

### Architecture

**Adaptive Interest Activation:**

1. **Behavior Representation**: Embed user's historical items
2. **Local Activation Unit**:
   - Calculates attention weights between each historical item and target item
   - Relevant items get higher weights
3. **Weighted Pooling**: Sum historical embeddings weighted by attention
4. **Final MLP**: Combines activated interest with other features

**Key Innovation**: Not all behaviors are equally relevant - activate based on target item

**Architecture:**
```
Historical Items → Embed → Attention(target) → Weighted Sum →                                                               Concat → MLP → Prediction
Target Item → Embed ────────────────────────────────────────→ /

Other Features ──────────────────────────────────────────────→ /
```

### Mathematical Foundation

**Attention Weight Calculation:**
```
a(e_i, e_target) = MLP([e_i; e_target; e_i ⊙ e_target; e_i - e_target])
α_i = exp(a_i) / Σ_j exp(a_j)
```

**Activated User Representation:**
```
Interest = Σ_i α_i · e_i
```

**Final Prediction:**
```
ŷ = σ(MLP([Interest; User_features; Item_features; Context]))
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.din import DIN
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
model = DIN(
    name="DIN_Model",
    item_embed_dim=16,
    attention_mlp=[64, 32],
    deep_mlp=[256, 128, 64, 32],
    activation='dice',
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
model.save('din_model.pkl')

# Load model
loaded = DIN.load('din_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use DIN

✅ **Perfect For:**
- CTR prediction with user history
- Diverse user behaviors (browsing different categories)
- When relevance to target matters
- Short-to-medium sequences (5-30 items)
- Display advertising, e-commerce

❌ **Not Suitable For:**
- No user history available
- All items equally relevant
- Very long sequences (use interest evolution like DIEN)
- Static collaborative filtering

### Best Practices

1. **Sequence Length**: 5-30 items (shorter than DIEN/SASRec)
2. **Embedding Dimension**: 8-32 for items, 4-16 for features
3. **Attention MLP**: [64, 32] typically sufficient
4. **Main MLP**: [256, 128, 64, 32]
5. **Activation**: PReLU (from paper) or Dice activation
6. **Batch Normalization**: Use adaptive BN across mini-batches
7. **Regularization**: Dropout (0.1-0.2) + L2 on embeddings
8. **Negative Sampling**: Important for training efficiency

## Further Reading

- Paper: Zhou et al. 2018 - Deep Interest Network for Click-Through Rate Prediction
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
