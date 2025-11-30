# MMoE Tutorial: Multi-gate Mixture-of-Experts

## Introduction

**MMoE** uses a Mixture-of-Experts (MoE) architecture with task-specific gating networks to learn task relationships automatically.

**Paper**: Ma et al. 2018 - Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (Google)

## How MMoE Works

### Architecture

**Multi-Task Learning with Gates:**

1. **Shared Experts**: Multiple expert networks share input
2. **Task-Specific Gates**: Learn how to weigh experts for each task
3. **Task Towers**: Specific networks for each objective (e.g., CTR, CVR)

**Key Innovation**: Solves "seesaw" problem where improving one task hurts another

**Architecture:**
```
Input -> [Expert 1] [Expert 2] ... [Expert N]
            |          |              |
         [Gate A]   [Gate B]       [Gate C]
            ↓          ↓              ↓
         [Tower A]  [Tower B]      [Tower C]
```

### Mathematical Foundation

**Mixture of Experts:**
```
f(x) = Σ_{i=1}^n g(x)_i · E_i(x)
```

**Gating Network (Task k):**
```
g^k(x) = softmax(W_g^k x)
```

**Final Output (Task k):**
```
y_k = h^k(f^k(x))
```
where h^k is the task tower.

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.mmoe import MMoE
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
model = MMoE(
    name="MMoE_Model",
    num_experts=4,
    expert_dim=64,
    task_names=['income', 'marital'],
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
model.save('mmoe_model.pkl')

# Load model
loaded = MMoE.load('mmoe_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use MMoE

✅ **Perfect For:**
- Multi-objective optimization (e.g., Clicks & Conversions)
- When tasks have complex relationships (correlated or conflicting)
- Large-scale production systems
- Reducing parameter count vs separate models

❌ **Not For:**
- Single task learning
- Small datasets
- When tasks are completely unrelated

### Best Practices

1. **Number of Experts**: 4-8 usually sufficient
2. **Expert Size**: Smaller than single model
3. **Gate Bias**: Initialize to uniform
4. **Task Weights**: Tune loss weights (e.g., 1.0 for CTR, 2.0 for CVR)

## Further Reading

- Paper: Ma et al. 2018 - Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (Google)
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
