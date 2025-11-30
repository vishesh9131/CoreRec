# PLE Tutorial: Progressive Layered Extraction

## Introduction

**PLE** improves MMoE by explicitly separating shared and task-specific experts to avoid negative transfer.

**Paper**: Tang et al. 2020 - Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations (Tencent)

## How PLE Works

### Architecture

**CGC (Customized Gate Control):**

1. **Task-Specific Experts**: Only feed into specific task
2. **Shared Experts**: Feed into all tasks
3. **Multi-Level**: Stacked CGC modules for deep extraction

**Key Innovation**: Eliminates negative transfer by isolating task-specific knowledge

**Architecture:**
```
[Specific Exp A] [Shared Exp] [Specific Exp B]
       |              |              |
    [Gate A]          |           [Gate B]
       ↓              |              ↓
    [Tower A]         |           [Tower B]
```

### Mathematical Foundation

**CGC Aggregation:**
```
E^k = [SpecificExperts^k, SharedExperts]
y^k = Gate(x) · E^k(x)
```
Gate only selects from relevant experts + shared experts.

**Loss:**
```
L = Σ_k w_k L_k(y_k, label_k)
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.ple import PLE
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
model = PLE(
    name="PLE_Model",
    num_shared_experts=2,
    num_specific_experts=2,
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
model.save('ple_model.pkl')

# Load model
loaded = PLE.load('ple_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use PLE

✅ **Perfect For:**
- Complex multi-task scenarios
- When MMoE suffers from negative transfer
- Highly conflicting tasks (e.g., Like vs Share)
- Deep multi-level representation learning

❌ **Not For:**
- Simple tasks
- Limited computational budget (more params than MMoE)

### Best Practices

1. **Levels**: 1-2 CGC layers
2. **Expert Split**: 50% shared, 50% specific
3. **Gate Activation**: Softmax
4. **Gradient Analysis**: Monitor gradient conflicts

## Further Reading

- Paper: Tang et al. 2020 - Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations (Tencent)
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
