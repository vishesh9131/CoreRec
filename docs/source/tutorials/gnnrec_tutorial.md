# GNNRec Tutorial: Graph Neural Network Recommender

## Introduction

**GNNRec** leverages graph neural networks to learn user and item representations by aggregating information from neighboring nodes in the interaction graph.

**Paper**: Hamilton et al. 2017 - Inductive Representation Learning on Large Graphs

## How GNNRec Works

### Architecture

**Graph-Based Learning:**

1. **Graph Construction**: 
   - Nodes: Users and Items
   - Edges: Interactions (ratings, clicks, purchases)
   - Bipartite graph structure

2. **Message Passing**: L layers of neighborhood aggregation
   - Aggregate neighbor information
   - Update node representations
   - Stack multiple layers for multi-hop propagation

3. **Aggregation Functions**:
   - Mean: `h_v = σ(W · MEAN{h_u : u ∈ N(v)})`
   - Sum: `h_v = σ(W · SUM{h_u : u ∈ N(v)})`
   - Attention: `h_v = σ(Σ α_uv · h_u)`

**Flow:**
```
Graph → Embed → Aggregate^(1) → ... → Aggregate^(L) → Predict
```

### Mathematical Foundation

**Message Passing Layer l:**
```
h_v^(l) = σ(W^(l) · AGG({h_u^(l-1) : u ∈ N(v)}) + b^(l))
```

**Mean Aggregator:**
```
AGG = (1/|N(v)|) · Σ_{u ∈ N(v)} h_u^(l-1)
```

**Attention Aggregator:**
```
α_uv = exp(LeakyReLU(a^T [W·h_u || W·h_v])) / Z
h_v = σ(Σ_{u ∈ N(v)} α_uv · W · h_u)
```

**Prediction:**
```
score(u,i) = h_u^(L)^T · h_i^(L)
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.gnnrec import GNNRec
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
model = GNNRec(
    name="GNNRec_Model",
    embedding_dim=128,
    num_layers=3,
    aggregator='mean',
    dropout=0.1,
    negative_samples=5,
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
model.save('gnnrec_model.pkl')

# Load model
loaded = GNNRec.load('gnnrec_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use GNNRec

✅ **Best For:**
- Social networks with user connections (trust, follow, friendship)
- Multi-hop relationships matter (friend-of-friend)
- Cold-start users (can leverage network)
- Heterogeneous graphs (users, items, categories, attributes)
- When graph structure provides signal

❌ **Not For:**
- No graph structure available → use collaborative filtering
- Very sparse graphs (avg degree < 2) → not enough signal
- Real-time constraints (GNN can be slow)
- Pure sequential patterns → use RNN

### Best Practices

1. **Number of Layers**: 2-3 layers optimal (more causes over-smoothing)
2. **Aggregator Choice**: Mean for homogeneous, Attention for heterogeneous
3. **Embedding Dimension**: 64-128 for most graphs
4. **Neighborhood Sampling**: Sample 10-25 neighbors per node (for scalability)
5. **Mini-batch Training**: Use GraphSAINT or cluster-GCN for large graphs
6. **Dropout**: 0.1-0.3 on edges/messages
7. **Skip Connections**: Add for deep networks (>3 layers)
8. **Negative Sampling**: 5-10 negatives per positive edge

## Further Reading

- Paper: Hamilton et al. 2017 - Inductive Representation Learning on Large Graphs
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
