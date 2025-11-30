# TDM Tutorial: Tree-based Deep Model

## Introduction

**TDM** uses a hierarchical tree structure to index items, allowing logarithmic time retrieval with deep learning models.

**Paper**: Zhu et al. 2018 - Learning Tree-based Deep Model for Recommender Systems (Alibaba)

## How TDM Works

### Architecture

**Tree Indexing:**

1. **Item Tree**: Items are leaf nodes of a balanced tree
2. **Beam Search**: Traverse tree level-by-level
3. **Node Prediction**: Predict probability of user liking child nodes
4. **Deep Network**: Scores user-node pairs

**Key Innovation**: O(log N) retrieval with deep models (breaking vector search limit)

**Architecture:**
```
Level 1 -> Top-K Nodes -> Level 2 -> Top-K Nodes ... -> Items
```

### Mathematical Foundation

**Probability:**
```
P(u likes node n) = softmax(DNN(u, n))
```

**Max-Heap Property:**
```
P(parent) >= P(child)
```
Approximated by training.

**Retrieval:**
Beam search with width K at each level.

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.tdm import TDM
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
model = TDM(
    name="TDM_Model",
    tree_depth=10,
    beam_size=20,
    node_dim=64,
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
model.save('tdm_model.pkl')

# Load model
loaded = TDM.load('tdm_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use TDM

✅ **Perfect For:**
- Massive item corpus (billions)
- Replacing ANN (Approximate Nearest Neighbor)
- Full corpus retrieval
- E-commerce (Alibaba scale)

❌ **Not For:**
- Small item sets
- High-frequency updates to item set (tree rebuild)
- Simple ranking

### Best Practices

1. **Tree Construction**: K-Means clustering on item embeddings
2. **Beam Size**: 20-50
3. **Tree Depth**: log_2(Items)
4. **Negative Sampling**: Sample from same level

## Further Reading

- Paper: Zhu et al. 2018 - Learning Tree-based Deep Model for Recommender Systems (Alibaba)
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
