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

**Flow:**
```
Graph → Embed → Aggregate^(1) → ... → Aggregate^(L) → Predict
```

### Mathematical Foundation

**Message Passing Layer l:**
```
h_v^(l) = σ(W^(l) · AGG({h_u^(l-1) : u ∈ N(v)}) + b^(l))
```

**Prediction:**
```
score(u,i) = h_u^(L)^T · h_i^(L)
```

```{admonition} Rating range
:class: important
GNNRec trains with **binary cross-entropy**. Use **0/1 labels** (implicit feedback) or normalize explicit ratings to **[0, 1]** before calling `fit()`.
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.gnnrec import GNNRec
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split
import numpy as np

data = ml_1m.load()
ratings_df = data['ratings']

print(f"Loaded {len(ratings_df)} ratings")

train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

train_users = train_df['user_id'].values
train_items = train_df['movie_id'].values
# BCE loss: binarize explicit ratings (watched/liked = 1.0)
train_ratings = (train_df['rating'].values >= 1.0).astype(np.float32)

test_users = test_df['user_id'].values
test_items = test_df['movie_id'].values
test_ratings = test_df['rating'].values  # keep original for RMSE comparison if desired
```

### Step 2: Initialize Model

```python
model = GNNRec(
    name="GNNRec_Model",
    embedding_dim=128,
    num_gnn_layers=3,
    dropout=0.1,
    epochs=20,
    batch_size=256,
    learning_rate=0.001,
    verbose=True,
)

print(f"Initialized {model.name}")
```

### Step 3: Train

```python
model.fit(
    user_ids=train_users,
    item_ids=train_items,
    ratings=train_ratings,
)

print("Training complete!")
```

### Step 4: Predict

```python
score = model.predict(user_id=1, item_id=100)
print(f"Predicted score: {score:.3f}")

test_predictions = model.batch_predict(list(zip(test_users[:100], test_items[:100])))
```

### Step 5: Recommend

```python
user_id = 1
recommendations = model.recommend(user_id=user_id, top_k=10)

print(f"Top-10 recommendations for User {user_id}:")
for rank, item_id in enumerate(recommendations, 1):
    print(f"  {rank}. Item {item_id}")
```

### Step 6: Evaluate

```python
from sklearn.metrics import mean_squared_error
import numpy as np

test_pred = [model.predict(u, i) for u, i in zip(test_users, test_items)]
rmse = np.sqrt(mean_squared_error(test_ratings, test_pred))
print(f"Test RMSE (on original 1–5 scale): {rmse:.4f}")
```

### Step 7: Save & Load

```python
model.save('artifacts/gnnrec')
loaded = GNNRec.load('artifacts/gnnrec')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use GNNRec

✅ **Best For:**
- User–item interaction graphs with enough density
- Multi-hop collaborative signal on bipartite graphs
- When graph convolution fits your retrieval/ranking stack

❌ **Not For:**
- Pure sequential next-item modeling → use SASRec or BERT4Rec
- Very sparse graphs with almost no edges
- Raw 1–5 star labels without binarization or normalization

### Best Practices

1. **Layers**: `num_gnn_layers=2–3` is a good default
2. **Labels**: Use 0/1 implicit feedback or scale ratings to [0, 1]
3. **Embedding dimension**: 64–128 for most datasets
4. **Batch size**: 256–1024 depending on GPU memory
5. **Save path**: Use a base path (`artifacts/gnnrec`) for the safe bundle

## Further Reading

- Paper: Hamilton et al. 2017 - Inductive Representation Learning on Large Graphs
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
