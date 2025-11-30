# Caser Tutorial: Convolutional Sequence Embedding Recommendation

## Introduction

**Caser** treats user sequence as an "image" applying CNNs with horizontal (recent patterns) and vertical (skip patterns) convolutional filters.

**Paper**: Tang & Wang 2018 - Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding

## How Caser Works

### Architecture

**CNN for Sequential Patterns:**

1. **Sequence Embedding Matrix**: L×d (sequence × embedding)
2. **Horizontal Convolution**: Capture recent sequential patterns
3. **Vertical Convolution**: Capture skip patterns (point-level)
4. **Pooling**: Max pooling over all filters
5. **Fully Connected**: Combine features for prediction

**Key Innovation**: CNN treats sequence as "image"

**Architecture:**
```
Sequence Matrix → [Horiz Conv] → Pool →                                          Concat → FC → Predict
                  [Vert Conv]  → Pool → /
```

### Mathematical Foundation

**Horizontal Convolution** (height h):
```
c^h = ReLU(W^h * E_{i:i+h} + b^h)
```
Captures last h items together

**Vertical Convolution** (width 1):
```
c^v = ReLU(W^v * E_i + b^v)
```
Captures single-point patterns

**Output:**
```
z = [max(c^h₁); ...; max(c^hₙ); max(c^v₁); ...; max(c^vₘ)]
y = FC(z)
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.caser import Caser
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
model = Caser(
    name="Caser_Model",
    seq_len=10,
    horiz_filters=[2, 3, 4],
    num_filters=16,
    dropout=0.5,
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
model.save('caser_model.pkl')

# Load model
loaded = Caser.load('caser_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use Caser

✅ **Ideal For:**
- Short-medium sequences (5-50 items)
- Local sequential patterns
- Recent behavior matters most
- E-commerce browsing sessions
- When RNN/Transformer is overkill

❌ **Not For:**
- Very long sequences (>100 items)
- Global long-range dependencies
- When position encoding crucial
- Sparse interaction data

### Best Practices

1. **Sequence Length**: L = 5-20 recent items
2. **Horizontal Filters**: heights = [2, 3, 4, 5]
3. **Vertical Filters**: 4-16 filters
4. **Filter Numbers**: 16-64 per size
5. **Dropout**: 0.5 on FC layer
6. **Embedding Dim**: 50-100
7. **Activation**: ReLU standard
8. **Pooling**: Max pooling works best

## Further Reading

- Paper: Tang & Wang 2018 - Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
