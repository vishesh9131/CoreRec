# YouTubeDNN Tutorial: YouTube Deep Neural Network

## Introduction

**YouTubeDNN** uses a two-stage architecture: candidate generation (retrieval) followed by a deep ranking model.

**Paper**: Covington et al. 2016 - Deep Neural Networks for YouTube Recommendations

## How YouTubeDNN Works

### Architecture

**Two-Stage Pipeline:**

1. **Candidate Generation (Retrieval)**:
   - Input: User history, search tokens
   - Output: Hundreds of candidates
   - Model: Extreme Multiclass Classification (Softmax)
   - Approximated by Nearest Neighbor Search in embedding space

2. **Ranking**:
   - Input: Candidates + rich features
   - Output: Precise score (watch time)
   - Model: Deep MLP with calibrated output

**Key Innovation**: Scalable deep learning for billions of videos

**Architecture:**
```
User History -> Average Embed -> MLP -> Softmax (Classify Video ID)
```

### Mathematical Foundation

**Candidate Generation (Softmax):**
```
P(w_t = i | U, C) = exp(v_i u) / Σ_j exp(v_j u)
```
where u is user embedding (from MLP), v_i is video embedding.

**Ranking (Weighted Logistic):**
Predict expected watch time using weighted logistic regression.
```
E[T] ≈ exp(w^T x)
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.youtubednn import YouTubeDNN
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
model = YouTubeDNN(
    name="YouTubeDNN_Model",
    embedding_dim=64,
    hidden_units=[1024, 512, 256],
    n_classes=10000,
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
model.save('youtubednn_model.pkl')

# Load model
loaded = YouTubeDNN.load('youtubednn_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use YouTubeDNN

✅ **Perfect For:**
- Massive scale (millions/billions of items)
- Video recommendation
- Two-stage systems (Retrieval -> Ranking)
- Implicit feedback (watch history)
- Handling fresh content (age feature)

❌ **Not For:**
- Small datasets
- Explicit ratings
- Simple ranking tasks

### Best Practices

1. **Example Age**: Crucial feature for freshness
2. **Negative Sampling**: Importance sampling for Softmax
3. **Embeddings**: Average user's watch history embeddings
4. **Hidden Layers**: ReLU "tower" structure [1024, 512, 256]
5. **Input Features**: Normalize continuous features

## Further Reading

- Paper: Covington et al. 2016 - Deep Neural Networks for YouTube Recommendations
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
