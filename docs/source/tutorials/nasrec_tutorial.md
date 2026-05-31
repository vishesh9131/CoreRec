# NASRec Tutorial: NASRec

## Introduction

**NASRec** is a recommendation model

## How NASRec Works

### Architecture

NASRec (Neural Architecture Search for Recommendations) automatically discovers optimal neural architectures using reinforcement learning-based search.

**Search Space:**
1. **Operation Types**: Convolution, LSTM, Attention, MLP, Skip connections
2. **Layer Configurations**: Hidden sizes, activation functions
3. **Connection Patterns**: Sequential, residual, dense

**Search Process:**
1. Controller RNN proposes architectures
2. Train candidate on validation set
3. Use validation performance as reward
4. Update controller with policy gradient

### Mathematical Foundation

**Architecture Encoding:**
```
arch = Controller_RNN(random_state)
arch = [layer_1_type, layer_1_config, ..., layer_n_type, layer_n_config]
```

**Reward Function:**
```
R = α · NDCG@10 - β · latency - γ · params
where α, β, γ are balancing coefficients
```

**Controller Update:**
```
∇L = E[R(arch) · ∇log P(arch|θ)]
θ ← θ + η · ∇L
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.nasrec import NASRec
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split

data = ml_1m.load()
ratings_df = data['ratings']
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

train_users = train_df['user_id'].values.tolist()
train_items = train_df['movie_id'].values.tolist()
train_ratings = train_df['rating'].values.tolist()

print(f"Loaded {len(ratings_df)} ratings")
```

### Step 2: Initialize Model

```python
model = NASRec(
    embedding_dim=64,
    hidden_dims=[128, 64],
    num_cells=3,
    epochs=5,
    batch_size=256,
    learning_rate=0.001,
    verbose=True
)
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
sample_user = train_users[0]
sample_item = train_items[0]
score = model.predict(sample_user, sample_item)
print(f"Predicted score: {score:.3f}")
```

### Step 5: Recommend

```python
recommendations = model.recommend(user_id=sample_user, top_k=10)

print(f"Top-10 recommendations for User {sample_user}:")
for rank, item_id in enumerate(recommendations, 1):
    print(f"  {rank}. Item {item_id}")
```

### Step 6: Evaluate

```python
from sklearn.metrics import mean_squared_error
import numpy as np

test_users = test_df['user_id'].values[:100]
test_items = test_df['movie_id'].values[:100]
test_ratings = test_df['rating'].values[:100]
test_pred = [model.predict(u, i) for u, i in zip(test_users, test_items)]
rmse = np.sqrt(mean_squared_error(test_ratings, test_pred))
print(f"Sample Test RMSE: {rmse:.4f}")
```

### Step 7: Save & Load

```python
model.save('artifacts/nasrec_model')

loaded = NASRec.load('artifacts/nasrec_model')
recs = loaded.recommend(sample_user, top_k=5)
print(f"Loaded model recommendations: {recs}")
```

## Advanced Usage

### Feature Engineering

Add model-specific advanced usage here.

## Key Takeaways

### When to Use NASRec

✅ **Ideal For:**
- Novel recommendation domains without established architectures
- Performance-critical applications
- Research and experimentation
- When you have significant compute budget

❌ **Not Ideal For:**
- Quick prototyping (search is slow)
- Limited compute resources
- Well-understood domains (use proven architectures)
- Production systems without retraining

### Best Practices

1. **Search Budget**: Minimum 50-100 architecture evaluations
2. **Early Stopping**: Stop unpromising architectures at 5 epochs
3. **Warm Start**: Initialize with known good architectures  
4. **Constrained Search**: Limit search space to reduce time
5. **Multi-Objective**: Balance performance,latency, model size

### Performance Comparison

Compare NASRec with similar models on your dataset.

## Further Reading

- [NASRec API Reference](../api/engines.md)
- Paper: See original paper for details
