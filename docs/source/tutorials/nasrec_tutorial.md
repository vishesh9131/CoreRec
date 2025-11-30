# NASRec Tutorial: Neural Architecture Search for Recommendation

## Introduction

**NASRec** automatically discovers optimal neural network architectures using reinforcement learning or evolutionary algorithms.

**Paper**: Various NAS papers applied to RecSys

## How NASRec Works

### Architecture

**Automated Architecture Discovery:**

1. **Search Space Definition**:
   - Operations: Conv, LSTM, Attention, MLP, Pooling
   - Connections: Skip, Residual, Dense
   - Hyperparameters: Hidden sizes, activations

2. **Search Strategy**:
   - Controller: RNN or Evolution Algorithm
   - Proposes candidate architectures
   - Evaluates on validation set
   - Updates based on performance

3. **Architecture Encoding**:
   - Sequence of layer types and configurations
   - Example: [LSTM-64, Attention-32, MLP-128, ...]

**Search Process:**
```
Random → Controller → Candidate Arch → Train → Evaluate
  ↑                                                ↓
  └──────────── Update Controller ←───────────────┘
```

### Mathematical Foundation

**Architecture Sampling:**
```
arch ~ Controller(θ)
arch = [layer_1, ..., layer_n]
where layer_i = (type_i, config_i)
```

**Reward Function:**
```
R(arch) = α · Metric(arch) - β · Latency(arch) - γ · Params(arch)
```
Balances accuracy, speed, and model size

**Controller Update (REINFORCE):**
```
∇_θ J = E_{arch~p(·|θ)}[R(arch) · ∇_θ log p(arch|θ)]
θ ← θ + η · ∇_θ J
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.nasrec import NASRec
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
model = NASRec(
    name="NASRec_Model",
    search_space='wide',  # 'narrow', 'wide', 'full'
    search_iterations=200,
    epochs_per_arch=10,
    max_params=5e6,
    max_latency_ms=50,
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
model.save('nasrec_model.pkl')

# Load model
loaded = NASRec.load('nasrec_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use NASRec

✅ **Ideal For:**
- Novel domains without established architectures
- Research and experimentation
- When you have significant compute budget (100+ GPU hours)
- Performance-critical applications worth the search cost
- AutoML platforms

❌ **Not For:**
- Quick prototyping (search takes days/weeks)
- Limited compute (<10 GPUs)
- Well-solved domains (just use proven architectures)
- Frequently changing data (search doesn't transfer)
- Production with tight constraints

### Best Practices

1. **Search Budget**: Minimum 100-500 architecture evaluations  
2. **Early Stopping**: Stop bad architectures at epoch 5
3. **Warm Start**: Initialize with known good architectures
4. **Constrained Search**: Limit latency/params to feasible range
5. **Transfer Learning**: Fine-tune found architecture
6. **Multi-Objective**: Use Pareto frontier for accuracy-latency trade-off
7. **Supernet Training**: Train once, search multiple times
8. **Progressive Search**: Start simple, add complexity gradually

## Further Reading

- Paper: Various NAS papers applied to RecSys
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
