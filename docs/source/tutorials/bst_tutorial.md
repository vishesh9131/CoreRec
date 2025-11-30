# BST Tutorial: Behavior Sequence Transformer

## Introduction

**BST** applies transformer architecture to user behavior sequences with target item attention for personalized recommendation.

**Paper**: Chen et al. 2019 - Behavior Sequence Transformer for E-commerce Recommendation

## How BST Works

### Architecture

**Transformer for E-commerce:**

1. **Behavior Sequence**: User's historical actions
2. **Multi-Head Self-Attention**: Capture item dependencies
3. **Target Attention**: Attend based on candidate item
4. **Position Encoding**: Capture sequence order
5. **Embedding Concat**: Combine sequence + features

**Key Innovation**: Target-aware attention (like DIN but with Transformer)

**Architecture:**
```
[Seq] → Transformer → Target Attn → [Other Features] → MLP → Pred
                          ↑
                     Candidate Item
```

### Mathematical Foundation

**Self-Attention:**
```
H = MultiHead(V_seq)
```

**Target Attention:**
```
α_i = softmax(h_i^T W_q e_target / √d)
v_user = Σ α_i · h_i
```

**Final Prediction:**
```
features = [v_user; e_target; context]
ŷ = MLP(features)
```

**Positional Encoding:**
```
PE(pos) = LearnableEmbed(pos)
```
Learned, not sinusoidal

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.bst import BST
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
model = BST(
    name="BST_Model",
    seq_len=20,
    hidden_dim=64,
    num_layers=2,
    num_heads=2,
    use_target_attention=True,
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
model.save('bst_model.pkl')

# Load model
loaded = BST.load('bst_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use BST

✅ **Perfect For:**
- E-commerce with rich sequences
- CTR prediction with behavior history
- Target-aware recommendations
- Medium-length sequences (10-50 items)
- When you need attention visualization

❌ **Not For:**
- Very short sequences (<5 items)
- Pure collaborative filtering
- Real-time constraints (slower than GRU)
- When SASRec already works

### Best Practices

1. **Sequence Length**: 10-50 recent items
2. **Transformer Layers**: 1-2 layers (shallow!)
3. **Attention Heads**: 1-2 heads
4. **Hidden Dim**: 64-128
5. **Target Attention**: Essential for performance
6. **Dropout**: 0.1-0.3
7. **Position Encoding**: Learnable works better
8. **Features**: Combine with user/item features

## Further Reading

- Paper: Chen et al. 2019 - Behavior Sequence Transformer for E-commerce Recommendation
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
